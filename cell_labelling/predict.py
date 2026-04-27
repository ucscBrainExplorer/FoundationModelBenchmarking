#!/usr/bin/env python3
"""
Label cells using KNN voting on UCE embeddings.

Two voting methods are available:
  majority_voting       — each neighbor gets one vote
  distance_weighted_knn — neighbors weighted by Gaussian kernel of distance (default)

All label columns present in the reference TSV that match the allowed list are
predicted in a single run. Output columns are prefixed by the source column name.

Usage:
  python3 predict.py \
    --index     path/to/index.faiss \
    --adata     path/to/dataset_uce_adata.h5ad \
    --ref_annot path/to/reference_metadata.tsv \
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


LABEL_COLS = [
    'cell_type',
    'cell_label',
    'cell_type_original',
    'mapped_cell_type',
    'mapped_cell_label',
    'harmonized_cell_type',
    'harmonized_cell_label',
]


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
    found = [c for c in LABEL_COLS if c in ref_df.columns]
    if not found:
        raise ValueError(
            f"Reference TSV must contain at least one of: {LABEL_COLS}"
        )


def resolve_labels(ref_df: pd.DataFrame, col: str) -> np.ndarray:
    def _clean(series):
        s = series.astype(str).str.strip()
        s = s.where(series.notna(), '')
        return s.where(s != 'nan', '')

    cleaned = _clean(ref_df[col])
    return cleaned.where(cleaned != '', 'missing_label').values


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


def distance_weighted_knn_vote(weights: np.ndarray, neighbor_labels: np.ndarray):
    """
    Aggregate Gaussian kernel weights by cell label, return top-2.

    Parameters:
        weights          — (N, k) weight array
        neighbor_labels  — (N, k) string array of cell labels

    Returns:
        top1, top1_scores, top2, top2_scores — each length-N list
        Scores are normalized (sum of weights for winner / total weight).
    """
    top1, top1_scores = [], []
    top2, top2_scores = [], []

    for row_weights, row_labels in zip(weights, neighbor_labels):
        valid_mask = np.array([t != '' for t in row_labels])
        if not np.any(valid_mask):
            top1.append('');        top1_scores.append(float('nan'))
            top2.append('');        top2_scores.append(float('nan'))
            continue

        vw = row_weights[valid_mask]
        vt = row_labels[valid_mask]
        total = vw.sum()

        uniq, inv = np.unique(vt, return_inverse=True)
        scores = np.bincount(inv, weights=vw)
        order = np.argsort(scores)[::-1]

        top1.append(uniq[order[0]])
        top1_scores.append(scores[order[0]] / total)
        if len(order) > 1:
            top2.append(uniq[order[1]])
            top2_scores.append(scores[order[1]] / total)
        else:
            top2.append('')
            top2_scores.append(float('nan'))

    return top1, top1_scores, top2, top2_scores


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
        term_ids          — reference label array aligned to FAISS index

    Returns:
        top1, top1_scores, top2, top2_scores — each length-N list
        Scores are winner's share of valid votes (0.0-1.0).
    """
    top1, top1_scores = [], []
    top2, top2_scores = [], []

    for row_indices, row_dists in zip(neighbor_indices, neighbor_dists):
        pairs = [
            (d, idx) for d, idx in zip(row_dists, row_indices)
            if idx >= 0 and np.isfinite(d)
        ]
        pairs.sort(key=lambda x: x[0])

        valid_labels = [str(term_ids[idx]) for d, idx in pairs]

        if valid_labels:
            counts = Counter(valid_labels)
            top2_list = counts.most_common(2)
            winner, count = top2_list[0]
            vote_pct = count / len(valid_labels)
            if len(top2_list) > 1:
                runner_up, count_2 = top2_list[1]
                vote_pct_2 = count_2 / len(valid_labels)
            else:
                runner_up = ''
                vote_pct_2 = float('nan')
        else:
            winner = runner_up = ''
            vote_pct = vote_pct_2 = float('nan')

        top1.append(winner)
        top1_scores.append(vote_pct)
        top2.append(runner_up)
        top2_scores.append(vote_pct_2)

    return top1, top1_scores, top2, top2_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Label cells using KNN voting on UCE embeddings"
    )
    parser.add_argument('--index',     required=True, help='FAISS index file (.faiss)')
    parser.add_argument('--adata',     required=True, help='h5ad file with adata.obsm["X_uce"] embeddings')
    parser.add_argument('--ref_annot', required=True,
                        help='Reference metadata TSV — any of these columns found will be predicted: '
                             'cell_type, cell_label, cell_type_original, mapped_cell_type, '
                             'mapped_cell_label, harmonized_cell_type, harmonized_cell_label')
    parser.add_argument('--method',    default='distance_weighted_knn',
                        choices=['majority_voting', 'distance_weighted_knn'],
                        help='Voting method (default: distance_weighted_knn)')
    parser.add_argument('--k',         type=int, default=30, help='Number of nearest neighbors (default: 30)')
    parser.add_argument('--output',    default='labels.tsv', help='Output TSV (default: labels.tsv)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading reference annotations from {args.ref_annot}...")
    ref_df = load_ref_annot(args.ref_annot)
    active_cols = [c for c in LABEL_COLS if c in ref_df.columns]
    print(f"  {len(ref_df)} reference cells")
    print(f"  Label columns found: {active_cols}")

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

    valid_dists = np.where(indices >= 0, dists, np.nan)
    mean_distances = np.nanmean(valid_dists, axis=1).tolist()

    cols = {'cell_id': cell_ids}

    use_weighted = args.method == 'distance_weighted_knn'

    # Precompute weights once if using weighted method
    if use_weighted:
        neighbor_labels_cache = {}
        weights = gaussian_kernel_weights(np.where(indices >= 0, dists, 0.0))

    last_top1 = None
    for col in active_cols:
        print(f"\nVoting on '{col}' ({args.method})...")
        term_ids = resolve_labels(ref_df, col)

        if use_weighted:
            neighbor_labels = np.vectorize(lambda i: term_ids[i] if i >= 0 else '')(indices)
            neighbor_labels = np.where(np.isfinite(dists) & (indices >= 0), neighbor_labels, '')
            top1, top1_scores, top2, top2_scores = distance_weighted_knn_vote(weights, neighbor_labels)
            score_col  = f'{col}_weighted_score'
            score_col2 = f'{col}_weighted_score_2'
        else:
            top1, top1_scores, top2, top2_scores = majority_voting(indices, dists, term_ids)
            score_col  = f'{col}_score'
            score_col2 = f'{col}_score_2'

        cols[f'{col}_pred']  = top1
        cols[score_col]      = top1_scores
        cols[f'{col}_pred_2'] = top2
        cols[score_col2]     = top2_scores
        last_top1 = top1

    cols['mean_euclidean_distance'] = mean_distances

    output_df = pd.DataFrame(cols)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        f.write(f"# index:      {args.index} ({index.ntotal} vectors, {index.d}d)\n")
        f.write(f"# adata:      {args.adata} ({n_cells} cells, {emb_dim}d, obsm['X_uce'])\n")
        f.write(f"# ref_annot:  {args.ref_annot} ({len(ref_df)} cells)\n")
        f.write(f"# label_cols: {active_cols}\n")
        f.write(f"# k:          {args.k}\n")
        f.write(f"# method:     {args.method}\n")
        output_df.to_csv(f, sep='\t', index=False)

    print(f"\nSaved {n_cells} cell labels to {args.output}")
    if last_top1 is not None:
        n_empty = sum(1 for p in last_top1 if p == '')
        if n_empty:
            print(f"  Warning: {n_empty} cells had no valid neighbors for voting")
        print(f"  Unique predicted labels ({active_cols[-1]}): {len(set(p for p in last_top1 if p))}")


if __name__ == '__main__':
    main()
