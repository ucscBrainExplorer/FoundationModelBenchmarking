#!/usr/bin/env python3
"""
Predict cell types using FAISS k-nearest-neighbor majority voting.

Given a FAISS index, reference annotations, and test embeddings, predicts
cell types without requiring ground truth labels.

Usage:
  python3 predict.py \
    --index indices/index_ivfflat.faiss \
    --ref_annot reference_data/prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --embeddings test_data/dataset.npy \
    --metadata test_data/dataset_prediction_obs.tsv \
    --k 30 \
    --output predictions/output.tsv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from data_loader import load_faiss_index, load_reference_annotations
from prediction_module import execute_query, vote_neighbors, distance_weighted_knn_vote
from obo_parser import parse_obo_names, parse_obo_replacements


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict cell types using FAISS nearest-neighbor majority voting"
    )
    parser.add_argument("--index", type=str, required=True,
                        help="Path to FAISS index file (.faiss)")
    parser.add_argument("--ref_annot", type=str, required=True,
                        help="Reference annotation TSV (needs cell_type_ontology_term_id)")
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file (CL ID -> canonical name mapping)")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Test embeddings file (.npy)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Optional test metadata TSV. All columns will be included in output.")
    parser.add_argument("--method", type=str, default="distance_weighted_knn",
                        choices=["majority_voting", "distance_weighted_knn", "both"],
                        help="Voting method (default: distance_weighted_knn)")
    parser.add_argument("--k", type=int, default=30,
                        help="Number of nearest neighbors (default: 30)")
    parser.add_argument("--output", type=str, default="predictions.tsv",
                        help="Output TSV path (default: predictions.tsv)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Validate output path is writable (fail fast, no side effects)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        # Check if parent directory exists so we can create output_dir later
        parent_dir = os.path.dirname(output_dir)
        if parent_dir and not os.path.exists(parent_dir):
            print(f"Error: Parent directory '{parent_dir}' does not exist for output path '{args.output}'")
            sys.exit(1)

    # 2. Load reference annotations (fast, catches column errors early)
    print(f"Loading reference annotations from {args.ref_annot}...")
    ref_df = load_reference_annotations(args.ref_annot)
    print(f"  Loaded {len(ref_df)} reference cells")

    # 3. Parse OBO file -> {CL_id: canonical_name} dict (fast)
    print(f"Parsing OBO file {args.obo}...")
    cl_names = parse_obo_names(args.obo)
    print(f"  Parsed {len(cl_names)} ontology terms")

    # 3a. Resolve obsolete CL terms in reference annotations
    cl_replacements = parse_obo_replacements(args.obo)
    print(f"  Found {len(cl_replacements)} obsolete terms with replacements")
    term_col = 'cell_type_ontology_term_id'
    ref_terms = ref_df[term_col]
    obsolete_mask = ref_terms.isin(cl_replacements)
    n_replaced = int(obsolete_mask.sum())
    # Count obsolete terms that have NO replacement (in OBO but not in replacements dict)
    all_obsolete_ids = set(ref_terms) - set(cl_names)
    n_unresolvable = len(all_obsolete_ids - set(cl_replacements))
    if n_replaced > 0:
        ref_df[term_col] = ref_terms.replace(cl_replacements)
        print(f"  Resolved {n_replaced} obsolete reference cells to current terms")
    if n_unresolvable > 0:
        print(f"  {n_unresolvable} obsolete terms have no replacement and will remain as-is")
    obsolete_comment = f"# Obsolete terms resolved: {n_replaced} replaced, {n_unresolvable} unresolvable (no replacement in OBO)\n"

    # 4. Load embeddings (medium cost)
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    n_cells = embeddings.shape[0]
    print(f"  Loaded {n_cells} embeddings (dimension: {embeddings.shape[1]})")

    # 5. Load metadata if provided (fast) - all columns will be included in output
    metadata_df = None
    if args.metadata:
        print(f"Loading metadata from {args.metadata}...")
        metadata_df = pd.read_csv(args.metadata, sep='\t')

        # Validate metadata has same number of rows as embeddings
        if len(metadata_df) != n_cells:
            print(f"  Warning: metadata has {len(metadata_df)} rows but embeddings has {n_cells} rows.")
            if len(metadata_df) > n_cells:
                print(f"  Using first {n_cells} rows of metadata.")
                metadata_df = metadata_df.iloc[:n_cells].copy()
            else:
                print(f"  Error: metadata has fewer rows than embeddings. Cannot proceed.")
                sys.exit(1)
        print(f"  Loaded metadata with {len(metadata_df.columns)} columns: {list(metadata_df.columns)}")

    # 6. Load FAISS index (expensive - only after all validations pass)
    print(f"Loading FAISS index from {args.index}...")
    index = load_faiss_index(args.index)
    print(f"  Index loaded (dimension: {index.d}, vectors: {index.ntotal})")

    # 6a. Validate embedding dimension matches FAISS index dimension
    embedding_dim = embeddings.shape[1]
    if index.d != embedding_dim:
        print(f"Error: Dimension mismatch!")
        print(f"  FAISS index dimension: {index.d}")
        print(f"  Embeddings dimension: {embedding_dim}")
        print(f"  The embeddings and FAISS index must have the same dimension.")
        print(f"  Hint: Make sure you're using the correct index for your embedding type:")
        print(f"    - UCE embeddings (1280-dim) require a UCE index")
        print(f"    - SCimilarity embeddings (128-dim) require a SCimilarity index")
        sys.exit(1)

    # 6b. Validate FAISS index size matches reference annotations
    if index.ntotal != len(ref_df):
        print(f"Error: FAISS index has {index.ntotal} vectors but reference annotations has {len(ref_df)} rows.")
        print(f"  The index and reference annotations must have the same number of entries.")
        sys.exit(1)

    # 7. Query FAISS
    print(f"Querying FAISS index (k={args.k})...")
    squared_dists, neighbor_indices = execute_query(index, embeddings, k=args.k)  # euclidean (L2)
    # FAISS L2 indices return squared Euclidean distances; convert to true Euclidean
    dists = np.sqrt(np.maximum(squared_dists, 0))
    print(f"  Query complete for {n_cells} cells")

    run_mv = args.method in ('majority_voting', 'both')
    run_wt = args.method in ('distance_weighted_knn', 'both')

    # 8. Vote
    cols = {}
    if run_mv:
        print("Performing majority voting...")
        mv_preds, mv_scores = vote_neighbors(neighbor_indices, ref_df)
        cols['mv_cell_type_ontology_term_id'] = mv_preds
        cols['mv_cell_type'] = [cl_names.get(p, p) for p in mv_preds]
        cols['mv_score'] = mv_scores

    if run_wt:
        print("Performing distance-weighted KNN voting...")
        wt_preds, wt_scores = distance_weighted_knn_vote(neighbor_indices, dists, ref_df)
        cols['weighted_cell_type_ontology_term_id'] = wt_preds
        cols['weighted_cell_type'] = [cl_names.get(p, p) for p in wt_preds]
        cols['weighted_score'] = wt_scores

    # 9. Map each neighbor's CL term ID -> canonical readable name, sorted by distance
    ref_term_ids = ref_df['cell_type_ontology_term_id'].values
    neighbor_distances_list = []
    neighbor_names_list = []

    for i in range(n_cells):
        row_indices = neighbor_indices[i]
        row_dists = dists[i]

        pairs = [(d, idx) for d, idx in zip(row_dists, row_indices)
                 if idx >= 0 and np.isfinite(d)]
        pairs.sort(key=lambda x: x[0])

        sorted_dists = [p[0] for p in pairs]
        sorted_names = [str(cl_names.get(ref_term_ids[p[1]], ref_term_ids[p[1]])) for p in pairs]

        neighbor_distances_list.append(",".join(f"{d:.4f}" for d in sorted_dists))
        neighbor_names_list.append(",".join(sorted_names))

    # Compute mean and std of euclidean distance across all k neighbors
    mean_euclidean_distances = []
    std_euclidean_distances = []
    for row_dists in dists:
        valid_dists = row_dists[np.isfinite(row_dists)]
        if len(valid_dists) > 0:
            mean_euclidean_distances.append(float(np.mean(valid_dists)))
            std_euclidean_distances.append(float(np.std(valid_dists)))
        else:
            mean_euclidean_distances.append(float('nan'))
            std_euclidean_distances.append(float('nan'))

    cols['mean_euclidean_distance'] = mean_euclidean_distances
    cols['std_euclidean_distance'] = std_euclidean_distances
    cols['neighbor_distances'] = neighbor_distances_list
    cols['neighbor_cell_types'] = neighbor_names_list

    # 11. Build output DataFrame
    predictions_df = pd.DataFrame(cols)

    # If metadata was provided, prepend all metadata columns to output
    if metadata_df is not None:
        output_df = pd.concat([metadata_df.reset_index(drop=True),
                               predictions_df.reset_index(drop=True)],
                              axis=1)
    else:
        output_df = predictions_df

    # Ensure cell_id is the first column (generate if not present from metadata)
    if 'cell_id' not in output_df.columns:
        output_df.insert(0, 'cell_id', [f"row_{i}" for i in range(n_cells)])
    elif output_df.columns.tolist().index('cell_id') != 0:
        col = output_df.pop('cell_id')
        output_df.insert(0, 'cell_id', col)

    # Save with comment header describing the run
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f"# Reference annotations: {args.ref_annot} ({len(ref_df)} cells)\n")
        f.write(f"# Query embeddings: {args.embeddings} ({n_cells} cells, {embedding_dim}d)\n")
        f.write(f"# FAISS index: {args.index} ({index.ntotal} vectors, {index.d}d)\n")
        f.write(f"# OBO file: {args.obo} ({len(cl_names)} terms)\n")
        f.write(f"# k: {args.k}\n")
        f.write(f"# method: {args.method}\n")
        f.write(obsolete_comment)
        if args.metadata:
            f.write(f"# Query metadata: {args.metadata}\n")
        output_df.to_csv(f, sep='\t', index=False)
    print(f"\nSaved predictions for {n_cells} cells to {args.output}")

    # Summary — use whichever predictions were generated
    top_preds = wt_preds if run_wt else mv_preds
    n_empty = sum(1 for p in top_preds if p == '')
    if n_empty > 0:
        print(f"  Warning: {n_empty} cells had no valid neighbors for voting")
    print(f"  Unique predicted cell types: {len(set(p for p in top_preds if p))}")


if __name__ == "__main__":
    main()
