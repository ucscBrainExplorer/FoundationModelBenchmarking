#!/usr/bin/env python3
"""
Distance distribution analysis for cell labelling quality assessment.

Compares three distributions to assess how well query cells map into the reference:
  - Null (negative control): random query → reference pairs
  - Query KNN: mean distance per query cell to its k nearest reference neighbors
  - Reference self-KNN (positive control): reference cells queried against themselves

Usage:
  python3 distance_analysis.py \
    --labels  labels.tsv \
    --adata   dataset_uce_adata.h5ad \
    --index   index.faiss \
    --output  distance_analysis.png
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import anndata
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_query_knn_distances(labels_path: str) -> np.ndarray:
    """Load per-cell mean KNN distances from labels.tsv."""
    df = pd.read_csv(labels_path, sep='\t', comment='#')
    if 'mean_euclidean_distance' in df.columns:
        dists = df['mean_euclidean_distance'].values
    else:
        # Fallback: compute from neighbor_distances column
        dists = df['neighbor_distances'].apply(
            lambda s: np.mean([float(x) for x in s.split(',')])
        ).values
    return dists.astype(np.float32)


def load_query_embeddings(adata_path: str) -> np.ndarray:
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"h5ad file not found: {adata_path}")
    adata = anndata.read_h5ad(adata_path, backed='r')
    if 'X_uce' not in adata.obsm:
        raise ValueError("h5ad does not contain adata.obsm['X_uce']")
    return np.array(adata.obsm['X_uce']).astype(np.float32)


def load_index(path: str) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(path)
    ivf = faiss.extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = 20
        ivf.make_direct_map()  # required for reconstruct() on IVF indices
    return index


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_null_distances(query_emb: np.ndarray, index: faiss.Index,
                          n_sample: int, seed: int = 42) -> np.ndarray:
    """
    Negative control: random query → reference distances.
    Pairs a random query cell with a random reference cell (not nearest neighbor).
    """
    rng = np.random.default_rng(seed)
    query_idx = rng.integers(0, query_emb.shape[0], size=n_sample)
    ref_idx   = rng.integers(0, index.ntotal,        size=n_sample)

    query_vecs = query_emb[query_idx]
    ref_vecs   = np.vstack([index.reconstruct(int(i)) for i in ref_idx])

    return np.linalg.norm(query_vecs - ref_vecs, axis=1).astype(np.float32)


def sample_ref_self_knn(index: faiss.Index, k: int,
                        n_sample: int, seed: int = 42) -> np.ndarray:
    """
    Positive control: reference self-KNN mean distances.
    Samples n_sample reference cells and queries them against the index,
    excluding self (rank 0) which is distance 0.
    """
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, index.ntotal, size=n_sample)
    ref_vecs   = np.vstack([index.reconstruct(int(i)) for i in sample_idx])

    sq_dists, _ = index.search(ref_vecs.astype(np.float32), k + 1)
    dists = np.sqrt(np.maximum(sq_dists[:, 1:], 0))  # exclude rank-0 self
    return dists.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(query_knn: np.ndarray, null_dists: np.ndarray,
                       ref_knn: np.ndarray, k: int, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals = np.concatenate([query_knn, null_dists, ref_knn])
    bins = np.linspace(np.percentile(all_vals, 0.5),
                       np.percentile(all_vals, 99.5), 80)

    ax.hist(null_dists, bins=bins, density=True, alpha=0.5, color='steelblue',
            label=f'Null — random query→reference (n={len(null_dists):,})')
    ax.hist(query_knn, bins=bins, density=True, alpha=0.5, color='coral',
            label=f'Query {k}-NN mean distance (n={len(query_knn):,})')
    ax.hist(ref_knn,   bins=bins, density=True, alpha=0.5, color='mediumseagreen',
            label=f'Reference self {k}-NN mean distance (n={len(ref_knn):,})')

    for vals, color, label in [
        (null_dists, 'navy',       f'Null mean: {null_dists.mean():.3f}'),
        (query_knn,  'darkred',    f'Query KNN mean: {query_knn.mean():.3f}'),
        (ref_knn,    'darkgreen',  f'Reference KNN mean: {ref_knn.mean():.3f}'),
    ]:
        ax.axvline(np.mean(vals), color=color, linestyle='--',
                   linewidth=1.2, label=label)

    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('Density')
    ax.set_title(f'KNN distance distributions (k={k}): query vs null vs reference self-KNN')
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Distance distribution analysis for cell labelling quality assessment"
    )
    parser.add_argument('--labels',   required=True, help='labels.tsv output from predict.py')
    parser.add_argument('--adata',    required=True, help='h5ad file with adata.obsm["X_uce"]')
    parser.add_argument('--index',    required=True, help='FAISS index file (.faiss)')
    parser.add_argument('--k',        type=int, default=30,    help='Neighbors used in predict.py (default: 30)')
    parser.add_argument('--n_sample', type=int, default=10000, help='Cells to sample for null and reference self-KNN (default: 10000)')
    parser.add_argument('--seed',     type=int, default=42,    help='Random seed (default: 42)')
    parser.add_argument('--output',   default='distance_analysis.png', help='Output PNG (default: distance_analysis.png)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading query KNN distances from {args.labels}...")
    query_knn = load_query_knn_distances(args.labels)
    print(f"  {len(query_knn):,} query cells, mean={query_knn.mean():.4f}")

    print(f"Loading query embeddings from {args.adata}...")
    query_emb = load_query_embeddings(args.adata)
    print(f"  {query_emb.shape[0]:,} cells, {query_emb.shape[1]}d")

    print(f"Loading FAISS index from {args.index}...")
    index = load_index(args.index)
    print(f"  {index.ntotal:,} reference vectors, {index.d}d")

    print(f"Sampling null distribution (n={args.n_sample:,}, random query→reference)...")
    null_dists = sample_null_distances(query_emb, index, args.n_sample, seed=args.seed)
    print(f"  Mean null distance: {null_dists.mean():.4f}")

    print(f"Sampling reference self-KNN (n={args.n_sample:,}, k={args.k})...")
    ref_knn = sample_ref_self_knn(index, args.k, args.n_sample, seed=args.seed)
    print(f"  Mean reference self-KNN distance: {ref_knn.mean():.4f}")

    print(f"\nSummary:")
    print(f"  Null mean:           {null_dists.mean():.4f}")
    print(f"  Query KNN mean:      {query_knn.mean():.4f}")
    print(f"  Reference KNN mean:  {ref_knn.mean():.4f}")
    ratio = (null_dists.mean() - query_knn.mean()) / (null_dists.mean() - ref_knn.mean())
    print(f"  Query KNN position between null and reference: {ratio:.1%}")

    plot_distributions(query_knn, null_dists, ref_knn, args.k, args.output)
    print(f"\nSaved plot to {args.output}")


if __name__ == '__main__':
    main()
