#!/usr/bin/env python3
"""
Label cells using KNN voting on UCE embeddings.

Two voting methods are available:
  majority_voting          — each neighbor gets one vote
  distance_weighted_knn — neighbors weighted by Gaussian kernel of distance (default)
  both                   — run both and output all columns side by side

Usage:
  python3 predict.py \
    --index     path/to/index.faiss \
    --adata     path/to/dataset_uce_adata.h5ad \
    --ref_annot path/to/reference_metadata.tsv \
    --obo       path/to/cl.obo \
    --method    distance_weighted_knn \
    --k 30 \
    --output    labels.tsv

Embeddings are read from adata.obsm["X_uce"]. The h5ad file is opened
in backed ('r') mode to avoid loading the full file into memory.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import anndata
import faiss
from collections import Counter


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_index(path: str) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(path)
    ivf = faiss.extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = 20
    return index


def load_adata(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"h5ad file not found: {path}")
    adata = anndata.read_h5ad(path, backed='r')
    if 'X_uce' not in adata.obsm:
        raise ValueError("h5ad file does not contain adata.obsm['X_uce']")
    embeddings = np.array(adata.obsm['X_uce']).astype(np.float32)
    cell_ids = list(adata.obs_names)
    return embeddings, cell_ids


def load_ref_annot(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference annotations not found: {path}")
    df = pd.read_csv(path, sep='\t')
    validate_ref_columns(df)
    return df


def validate_ref_columns(ref_df: pd.DataFrame) -> None:
    has_cl_id  = 'cell_label_ontology_term_id' in ref_df.columns
    has_cl_lbl = 'cell_label' in ref_df.columns
    has_ct_id  = 'cell_type_ontology_term_id' in ref_df.columns
    has_ct_lbl = 'cell_type' in ref_df.columns

    if has_cl_id != has_cl_lbl:
        missing = 'cell_label' if has_cl_id else 'cell_label_ontology_term_id'
        raise ValueError(f"Incomplete column pair: '{missing}' is missing")
    if has_ct_id != has_ct_lbl:
        missing = 'cell_type' if has_ct_id else 'cell_type_ontology_term_id'
        raise ValueError(f"Incomplete column pair: '{missing}' is missing")
    if not (has_cl_id and has_cl_lbl) and not (has_ct_id and has_ct_lbl):
        raise ValueError(
            "Reference annotations must contain at least one complete column pair: "
            "('cell_label_ontology_term_id' + 'cell_label') or "
            "('cell_type_ontology_term_id' + 'cell_type')"
        )


def resolve_labels(ref_df: pd.DataFrame) -> np.ndarray:
    if 'cell_label_ontology_term_id' in ref_df.columns:
        id_col, lbl_col = ref_df['cell_label_ontology_term_id'], ref_df['cell_label']
    else:
        id_col, lbl_col = ref_df['cell_type_ontology_term_id'], ref_df['cell_type']

    def _clean(col):
        s = col.astype(str).str.strip()
        s = s.where(col.notna(), '')
        return s.where(s != 'nan', '')

    id_s  = _clean(id_col)
    lbl_s = _clean(lbl_col)
    result = id_s.where(id_s != '', lbl_s)
    return result.where(result != '', 'missing_label').values


def parse_obo_names(obo_path: str) -> dict:
    """Parse an OBO file and return {term_id: readable_name}."""
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"OBO file not found: {obo_path}")
    cl_map = {}
    current_id = None
    in_term = False
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                current_id = None
            elif line.startswith('[') and line.endswith(']'):
                in_term = False
                current_id = None
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('name: ') and current_id:
                    cl_map[current_id] = line[6:]
    return cl_map


def parse_obo_replacements(obo_path: str) -> dict:
    """Parse an OBO file and return {obsolete_id: replacement_id}."""
    replacements = {}
    current_id = None
    in_term = False
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                current_id = None
            elif line.startswith('[') and line.endswith(']'):
                in_term = False
                current_id = None
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('replaced_by: ') and current_id:
                    replacements[current_id] = line[13:]
    return replacements


# ---------------------------------------------------------------------------
# Core KNN + voting
# ---------------------------------------------------------------------------

def gaussian_kernel_weights(dists: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert distances to weights using a Gaussian kernel.
    sigma is set per-row to the median distance of that row's neighbors.

    w = exp(-d² / (2 * sigma²))
    """
    sigma = np.median(dists, axis=1, keepdims=True)
    sigma = np.clip(sigma, eps, None)
    return np.exp(-dists ** 2 / (2 * sigma ** 2))


def distance_weighted_knn_vote(weights: np.ndarray, neighbor_term_ids: np.ndarray):
    """
    Aggregate Gaussian kernel weights by cell type label, return top-2.

    Parameters:
        weights           — (N, k) weight array
        neighbor_term_ids — (N, k) string array of CL term IDs

    Returns:
        top1_ids, top1_scores, top2_ids, top2_scores — each length-N list
        Scores are normalized (sum of weights for winner / total weight).
    """
    top1_ids, top1_scores = [], []
    top2_ids, top2_scores = [], []

    for row_weights, row_terms in zip(weights, neighbor_term_ids):
        valid_mask = np.array([t != '' for t in row_terms])
        if not np.any(valid_mask):
            top1_ids.append('');   top1_scores.append(float('nan'))
            top2_ids.append('');   top2_scores.append(float('nan'))
            continue

        vw = row_weights[valid_mask]
        vt = row_terms[valid_mask]
        total = vw.sum()

        uniq, inv = np.unique(vt, return_inverse=True)
        scores = np.bincount(inv, weights=vw)
        order = np.argsort(scores)[::-1]

        top1_ids.append(uniq[order[0]])
        top1_scores.append(scores[order[0]] / total)
        if len(order) > 1:
            top2_ids.append(uniq[order[1]])
            top2_scores.append(scores[order[1]] / total)
        else:
            top2_ids.append('')
            top2_scores.append(float('nan'))

    return top1_ids, top1_scores, top2_ids, top2_scores


def knn_search(index: faiss.Index, embeddings: np.ndarray, k: int):
    squared_dists, indices = index.search(embeddings, k)
    dists = np.sqrt(np.maximum(squared_dists, 0))
    return dists, indices


def majority_voting(neighbor_indices: np.ndarray, neighbor_dists: np.ndarray, term_ids: np.ndarray):
    """
    Majority vote among k neighbors, return top-2 per cell.

    Parameters:
        neighbor_indices  — (N, k) FAISS indices
        neighbor_dists    — (N, k) Euclidean distances (sorted closest first)
        term_ids          — reference CL term ID array aligned to FAISS index

    Returns:
        top1_ids, top1_scores, top2_ids, top2_scores — each length-N list
        Scores are winner's share of valid votes (0.0-1.0).
    """
    top1_ids, top1_scores = [], []
    top2_ids, top2_scores = [], []

    for row_indices, row_dists in zip(neighbor_indices, neighbor_dists):
        pairs = [
            (d, idx) for d, idx in zip(row_dists, row_indices)
            if idx >= 0 and np.isfinite(d)
        ]
        pairs.sort(key=lambda x: x[0])

        valid_labels = [str(term_ids[idx]) for d, idx in pairs]

        if valid_labels:
            counts = Counter(valid_labels)
            top2 = counts.most_common(2)
            winner, count = top2[0]
            vote_pct = count / len(valid_labels)
            if len(top2) > 1:
                runner_up, count_2 = top2[1]
                vote_pct_2 = count_2 / len(valid_labels)
            else:
                runner_up = ''
                vote_pct_2 = float('nan')
        else:
            winner = runner_up = ''
            vote_pct = vote_pct_2 = float('nan')

        top1_ids.append(winner)
        top1_scores.append(vote_pct)
        top2_ids.append(runner_up)
        top2_scores.append(vote_pct_2)

    return top1_ids, top1_scores, top2_ids, top2_scores


def neighbor_distances_str(neighbor_indices: np.ndarray, neighbor_dists: np.ndarray) -> list:
    """Build comma-separated distance strings per cell, sorted closest to furthest."""
    result = []
    for row_dists, row_indices in zip(neighbor_dists, neighbor_indices):
        pairs = sorted(
            [(d, idx) for d, idx in zip(row_dists, row_indices)
             if idx >= 0 and np.isfinite(d)],
            key=lambda x: x[0]
        )
        result.append(','.join(f'{d:.4f}' for d, _ in pairs))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Label cells using KNN majority voting on UCE embeddings"
    )
    parser.add_argument('--index',      required=True, help='FAISS index file (.faiss)')
    parser.add_argument('--adata',      required=True, help='h5ad file with adata.obsm["X_uce"] embeddings')
    parser.add_argument('--ref_annot',  required=True, help='Reference metadata TSV (needs cell_label_ontology_term_id+cell_label, or cell_type_ontology_term_id+cell_type)')
    parser.add_argument('--obo',        required=True, help='Cell Ontology OBO file (cl.obo) for ID -> name translation')
    parser.add_argument('--method',     default='distance_weighted_knn',
                        choices=['majority_voting', 'distance_weighted_knn', 'both'],
                        help='Voting method (default: distance_weighted_knn)')
    parser.add_argument('--k',          type=int, default=30, help='Number of nearest neighbors (default: 30)')
    parser.add_argument('--output',     default='labels.tsv', help='Output TSV (default: labels.tsv)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading reference annotations from {args.ref_annot}...")
    ref_df = load_ref_annot(args.ref_annot)
    term_ids = resolve_labels(ref_df)
    print(f"  {len(ref_df)} reference cells")

    print(f"Parsing OBO file {args.obo}...")
    cl_names = parse_obo_names(args.obo)
    print(f"  {len(cl_names)} ontology terms")

    cl_replacements = parse_obo_replacements(args.obo)
    print(f"  {len(cl_replacements)} obsolete terms with replacements")
    term_col = ('cell_label_ontology_term_id' if 'cell_label_ontology_term_id' in ref_df.columns
                else 'cell_type_ontology_term_id')
    ref_terms = ref_df[term_col]
    n_replaced = int(ref_terms.isin(cl_replacements).sum())
    if n_replaced > 0:
        ref_df[term_col] = ref_terms.replace(cl_replacements)
        term_ids = resolve_labels(ref_df)
        print(f"  Resolved {n_replaced} obsolete reference terms")

    print(f"Loading embeddings from {args.adata}...")
    embeddings, cell_ids = load_adata(args.adata)
    n_cells, emb_dim = embeddings.shape
    print(f"  {n_cells} cells, {emb_dim}d")

    print(f"Loading FAISS index from {args.index}...")
    index = load_index(args.index)
    print(f"  {index.ntotal} vectors, {index.d}d")

    if index.d != emb_dim:
        print(f"Error: embedding dim ({emb_dim}) != index dim ({index.d})")
        sys.exit(1)
    if index.ntotal != len(ref_df):
        print(f"Error: index size ({index.ntotal}) != reference annotations ({len(ref_df)})")
        sys.exit(1)

    print(f"Searching {args.k} nearest neighbors...")
    dists, indices = knn_search(index, embeddings, k=args.k)

    run_mv = args.method in ('majority_voting', 'both')
    run_wt = args.method in ('distance_weighted_knn', 'both')

    valid_dists = np.where(indices >= 0, dists, np.nan)
    mean_distances = np.nanmean(valid_dists, axis=1).tolist()

    cols = {'cell_id': cell_ids}

    if run_mv:
        print("Majority voting (unweighted KNN)...")
        mv1_ids, mv1_scores, mv2_ids, mv2_scores = majority_voting(indices, dists, term_ids)
        cols.update({
            'mv_cell_type_ontology_term_id':   mv1_ids,
            'mv_cell_type':                    [cl_names.get(p, p) for p in mv1_ids],
            'mv_score':                        mv1_scores,
            'mv_cell_type_ontology_term_id_2': mv2_ids,
            'mv_cell_type_2':                  [cl_names.get(p, p) if p else '' for p in mv2_ids],
            'mv_score_2':                      mv2_scores,
        })

    if run_wt:
        print("Distance-weighted KNN voting...")
        neighbor_term_ids = np.vectorize(lambda i: term_ids[i] if i >= 0 else '')(indices)
        neighbor_term_ids = np.where(np.isfinite(dists) & (indices >= 0), neighbor_term_ids, '')
        weights = gaussian_kernel_weights(np.where(indices >= 0, dists, 0.0))
        w1_ids, w1_scores, w2_ids, w2_scores = distance_weighted_knn_vote(weights, neighbor_term_ids)
        cols.update({
            'weighted_cell_type_ontology_term_id':   w1_ids,
            'weighted_cell_type':                    [cl_names.get(p, p) for p in w1_ids],
            'weighted_score':                        w1_scores,
            'weighted_cell_type_ontology_term_id_2': w2_ids,
            'weighted_cell_type_2':                  [cl_names.get(p, p) if p else '' for p in w2_ids],
            'weighted_score_2':                      w2_scores,
        })

    cols.update({
        'mean_euclidean_distance': mean_distances,
    })

    output_df = pd.DataFrame(cols)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        f.write(f"# index:      {args.index} ({index.ntotal} vectors, {index.d}d)\n")
        f.write(f"# adata:      {args.adata} ({n_cells} cells, {emb_dim}d, obsm['X_uce'])\n")
        f.write(f"# ref_annot:  {args.ref_annot} ({len(ref_df)} cells)\n")
        f.write(f"# obo:        {args.obo} ({len(cl_names)} terms)\n")
        f.write(f"# k:          {args.k}\n")
        f.write(f"# method:     {args.method}\n")
        output_df.to_csv(f, sep='\t', index=False)

    print(f"\nSaved {n_cells} cell labels to {args.output}")
    top_preds = mv1_ids if run_mv else w1_ids
    n_empty = sum(1 for p in top_preds if p == '')
    if n_empty:
        print(f"  Warning: {n_empty} cells had no valid neighbors for voting")
    print(f"  Unique predicted types: {len(set(p for p in top_preds if p))}")


if __name__ == '__main__':
    main()
