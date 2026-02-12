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
from prediction_module import execute_query, vote_neighbors
from obo_parser import parse_obo_names


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

    # 7. Query FAISS
    print(f"Querying FAISS index (k={args.k})...")
    dists, neighbor_indices = execute_query(index, embeddings, k=args.k)
    print(f"  Query complete for {n_cells} cells")

    # 8. Majority vote
    print("Performing majority voting...")
    predictions, vote_percentages = vote_neighbors(neighbor_indices, ref_df)

    # 9. Map predicted CL term IDs -> canonical readable names via OBO dict
    predicted_names = [cl_names.get(pred, pred) for pred in predictions]

    # 10. Map each neighbor's CL term ID -> canonical readable name, sorted by distance
    ref_term_ids = ref_df['cell_type_ontology_term_id'].values
    neighbor_distances_list = []
    neighbor_names_list = []

    for i in range(n_cells):
        row_indices = neighbor_indices[i]
        row_dists = dists[i]

        # Pair up (distance, neighbor_index), filter invalid
        pairs = [(d, idx) for d, idx in zip(row_dists, row_indices)
                 if idx >= 0 and np.isfinite(d)]

        # Sort by distance (closest first)
        pairs.sort(key=lambda x: x[0])

        sorted_dists = [p[0] for p in pairs]
        sorted_names = [str(cl_names.get(ref_term_ids[p[1]], ref_term_ids[p[1]])) for p in pairs]

        neighbor_distances_list.append(",".join(f"{d:.4f}" for d in sorted_dists))
        neighbor_names_list.append(",".join(sorted_names))

    # Compute mean euclidean distance across all k neighbors
    mean_euclidean_distances = []
    for row_dists in dists:
        valid_dists = row_dists[np.isfinite(row_dists)]
        if len(valid_dists) > 0:
            mean_euclidean_distances.append(float(np.mean(valid_dists)))
        else:
            mean_euclidean_distances.append(float('nan'))

    # 11. Build output DataFrame
    predictions_df = pd.DataFrame({
        'predicted_cell_type_ontology_term_id': predictions,
        'predicted_cell_type': predicted_names,
        'vote_percentage': vote_percentages,
        'mean_euclidean_distance': mean_euclidean_distances,
        'neighbor_distances': neighbor_distances_list,
        'neighbor_cell_types': neighbor_names_list,
    })

    # If metadata was provided, prepend all metadata columns to output
    if metadata_df is not None:
        # Concatenate metadata + predictions side-by-side (same row order)
        output_df = pd.concat([metadata_df.reset_index(drop=True),
                               predictions_df.reset_index(drop=True)],
                              axis=1)
    else:
        output_df = predictions_df

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(args.output, sep='\t', index=False)
    print(f"\nSaved predictions for {n_cells} cells to {args.output}")

    # Summary
    n_empty = sum(1 for p in predictions if p == '')
    if n_empty > 0:
        print(f"  Warning: {n_empty} cells had no valid neighbors for voting")
    unique_types = len(set(p for p in predictions if p != ''))
    print(f"  Unique predicted cell types: {unique_types}")


if __name__ == "__main__":
    main()
