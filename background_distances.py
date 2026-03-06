#!/usr/bin/env python3
"""
Compute background pairwise Euclidean distances from random embedding pairs.

Samples random pairs of distinct points from an embeddings file and computes
Euclidean distances, providing a baseline distribution for interpreting KNN
neighbor distances from predict.py.

Can be used standalone or imported by predict.py for combined plotting.

Usage:
  # Background only:
  python3 background_distances.py \
    --embeddings reference.npy \
    --output background_distances.png

  # Combined plot with KNN distances:
  python3 background_distances.py \
    --embeddings reference.npy \
    --predictions predictions.tsv \
    --output background_vs_knn.png

  # Full three-distribution plot (null + query KNN + reference self-KNN):
  python3 background_distances.py \
    --embeddings reference.npy \
    --query      query.npy \
    --index      index.faiss \
    --predictions predictions.tsv \
    --output     three_distributions.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import faiss


def load_embeddings(path, h5_key=None):
    """Load embeddings from .npy or .h5/.hdf5 file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            if h5_key:
                if h5_key not in f:
                    print(f"Error: Key '{h5_key}' not found in {path}")
                    print(f"  Available keys: {list(f.keys())}")
                    sys.exit(1)
                key = h5_key
            else:
                key = None
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset) and f[k].ndim == 2:
                        key = k
                        break
                if key is None:
                    print(f"Error: No 2D dataset found in {path}")
                    print(f"  Available keys: {list(f.keys())}")
                    sys.exit(1)
            print(f"  Using h5 dataset key: '{key}'")
            return f[key][:]
    elif ext == ".npy":
        return np.load(path)
    else:
        print(f"Error: Unsupported file format '{ext}'. Use .npy or .h5/.hdf5")
        sys.exit(1)


def sample_background_distances(emb, n_pairs, seed=42):
    """Sample random pairs from embeddings and compute Euclidean distances.

    Args:
        emb: numpy array of shape (n_points, n_dims)
        n_pairs: number of random pairs to sample
        seed: random seed

    Returns:
        1D numpy array of Euclidean distances
    """
    n_points = emb.shape[0]
    max_pairs = n_points * (n_points - 1) // 2
    if n_pairs > max_pairs:
        print(f"  Warning: Requested {n_pairs} pairs but only {max_pairs} unique pairs exist. Using {max_pairs}.")
        n_pairs = max_pairs

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n_points, size=n_pairs)
    offsets = rng.integers(1, n_points, size=n_pairs)
    idx_b = (idx_a + offsets) % n_points

    return np.linalg.norm(emb[idx_a] - emb[idx_b], axis=1)


def summarize_distances(distances, label="BACKGROUND DISTANCE SUMMARY", extra_info=None):
    """Print summary statistics for a distance array."""
    print()
    print(label)
    print("=" * len(label))
    if extra_info:
        for line in extra_info:
            print(line)
    print(f"Count: {len(distances)}")
    print(f"Mean: {np.mean(distances):.2f}")
    print(f"Std: {np.std(distances):.2f}")
    print(f"Median: {np.median(distances):.2f}")
    print(f"Min: {np.min(distances):.2f}")
    print(f"Max: {np.max(distances):.2f}")
    print(f"25th percentile: {np.percentile(distances, 25):.2f}")
    print(f"75th percentile: {np.percentile(distances, 75):.2f}")
    print(f"90th percentile: {np.percentile(distances, 90):.2f}")
    print(f"95th percentile: {np.percentile(distances, 95):.2f}")


def plot_combined_histogram(background_distances, knn_distances, png_path, k=None):
    """Save a combined histogram of background and KNN mean distances.

    Args:
        background_distances: 1D array of random pair distances
        knn_distances: 1D array of mean KNN distances (one per query)
        png_path: output PNG file path
        k: number of neighbors (used in labels), or None if unknown
    """
    bg_mean = float(np.mean(background_distances))
    knn_mean = float(np.mean(knn_distances))

    k_label = f"{k}-nearest" if k else "k-nearest"

    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals = np.concatenate([background_distances, knn_distances])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), 81)

    ax.hist(background_distances, bins=bin_edges, alpha=0.5, edgecolor="black",
            linewidth=0.3, label=f"Background random pairs (n={len(background_distances)})",
            color="steelblue")
    ax.hist(knn_distances, bins=bin_edges, alpha=0.5, edgecolor="black",
            linewidth=0.3,
            label=f"{k_label} neighbors mean distance (n={len(knn_distances)})",
            color="coral")

    ax.axvline(bg_mean, color="navy", linestyle="--", linewidth=1.2,
               label=f"Background mean: {bg_mean:.2f}")
    ax.axvline(knn_mean, color="darkred", linestyle="--", linewidth=1.2,
               label=f"{k_label} neighbors mean: {knn_mean:.2f}")

    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Background vs {k_label} neighbors distances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_background_histogram(distances, png_path):
    """Save a background-only histogram."""
    mean = float(np.mean(distances))
    median = float(np.median(distances))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distances, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean:.2f}")
    ax.axvline(median, color="blue", linestyle="--", linewidth=1.2, label=f"Median: {median:.2f}")
    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Background pairwise distances ({len(distances)} pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def sample_null_query_ref_distances(query_emb: np.ndarray, ref_index: faiss.Index,
                                    n_sample: int, seed: int = 42) -> np.ndarray:
    """
    Negative control: random query → reference distances.
    Pairs a random query cell with a random reference cell (not nearest neighbor).
    """
    rng = np.random.default_rng(seed)
    query_idx = rng.integers(0, query_emb.shape[0], size=n_sample)
    ref_idx   = rng.integers(0, ref_index.ntotal,   size=n_sample)

    query_vecs = query_emb[query_idx]
    ref_vecs   = np.vstack([ref_index.reconstruct(int(i)) for i in ref_idx])

    return np.linalg.norm(query_vecs - ref_vecs, axis=1).astype(np.float32)


def sample_ref_self_knn(ref_index: faiss.Index, k: int,
                        n_sample: int, seed: int = 42) -> np.ndarray:
    """
    Positive control: reference self-KNN mean distances.
    Samples n_sample reference cells and queries them against the index,
    excluding self (rank 0, distance 0).
    """
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, ref_index.ntotal, size=n_sample)
    ref_vecs   = np.vstack([ref_index.reconstruct(int(i)) for i in sample_idx])

    sq_dists, _ = ref_index.search(ref_vecs.astype(np.float32), k + 1)
    dists = np.sqrt(np.maximum(sq_dists[:, 1:], 0))  # exclude rank-0 self
    return dists.mean(axis=1).astype(np.float32)


def plot_three_distributions(null_dists: np.ndarray, query_knn: np.ndarray,
                             ref_knn: np.ndarray, k: int, png_path: str):
    """Plot null, query KNN, and reference self-KNN as overlaid density histograms."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals = np.concatenate([null_dists, query_knn, ref_knn])
    bins = np.linspace(np.percentile(all_vals, 0.5),
                       np.percentile(all_vals, 99.5), 80)

    ax.hist(null_dists, bins=bins, density=True, alpha=0.5, color='steelblue',
            label=f'Null — random query→reference (n={len(null_dists):,})')
    ax.hist(query_knn,  bins=bins, density=True, alpha=0.5, color='coral',
            label=f'Query {k}-NN mean distance (n={len(query_knn):,})')
    ax.hist(ref_knn,    bins=bins, density=True, alpha=0.5, color='mediumseagreen',
            label=f'Reference self {k}-NN mean distance (n={len(ref_knn):,})')

    for vals, color, label in [
        (null_dists, 'navy',      f'Null mean: {null_dists.mean():.3f}'),
        (query_knn,  'darkred',   f'Query KNN mean: {query_knn.mean():.3f}'),
        (ref_knn,    'darkgreen', f'Reference KNN mean: {ref_knn.mean():.3f}'),
    ]:
        ax.axvline(vals.mean(), color=color, linestyle='--', linewidth=1.2, label=label)

    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('Density')
    ax.set_title(f'KNN distance distributions (k={k}): query vs null vs reference self-KNN')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI-only code below
# ---------------------------------------------------------------------------

def load_knn_distances(predictions_path):
    """Load mean KNN distances from a predict.py output TSV.

    Returns (dists, k) where dists is a 1D numpy array and k is the
    neighbor count parsed from the comment header (or None if not found).
    """
    import pandas as pd

    # Parse k from comment header (e.g. "# k: 30")
    k = None
    with open(predictions_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if line.startswith('# k:'):
                k = int(line.split(':')[1].strip())
    if k is not None:
        print(f"  Parsed k={k} from predictions header")

    pred_df = pd.read_csv(predictions_path, sep='\t', comment='#')

    for col in ("mean_euclidean_distance", "mean_distance"):
        if col in pred_df.columns:
            break
    else:
        # Fall back: compute from neighbor_distances column
        if 'neighbor_distances' in pred_df.columns:
            col = None
        else:
            print(f"Error: No distance column found in {predictions_path}")
            print(f"  Available columns: {list(pred_df.columns)}")
            sys.exit(1)

    if col:
        dists = pred_df[col].dropna().values.astype(float)
    else:
        dists = pred_df['neighbor_distances'].apply(
            lambda s: np.mean([float(x) for x in s.split(',')])
        ).values.astype(float)
    print(f"  Loaded {len(dists)} mean KNN distances")
    return dists, k


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute background pairwise Euclidean distances from random embedding pairs",
    )
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Reference embeddings file (.npy or .h5)")
    parser.add_argument("--dataset-key", type=str, default=None,
                        help="Dataset key for .h5 files (auto-detects first 2D dataset if omitted)")
    parser.add_argument("--query", type=str, default=None,
                        help="Query embeddings file (.npy or .h5); enables random query→reference null distribution")
    parser.add_argument("--index", type=str, default=None,
                        help="FAISS index file (.faiss); enables reference self-KNN positive control")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Predictions TSV from predict.py; reads mean_euclidean_distance column")
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="Number of random pairs/samples, default: 10000 "
                             "(auto-matches KNN count when --predictions is given)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed, default: 42")
    parser.add_argument("--output", type=str, default="background_distances.png",
                        help="Output histogram PNG path, default: background_distances.png")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Validate inputs
    if not os.path.isfile(args.embeddings):
        print(f"Error: Embeddings file not found: {args.embeddings}")
        sys.exit(1)

    if args.predictions and not os.path.isfile(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    # 2. Load KNN distances if predictions provided
    knn_distances = None
    k = None
    if args.predictions:
        print(f"Loading KNN mean distances from {args.predictions}...")
        knn_distances, k = load_knn_distances(args.predictions)

    # 3. Determine n_pairs
    if knn_distances is not None:
        n_pairs = len(knn_distances)
        print(f"  Auto-setting --n-pairs to {n_pairs} to match KNN distance count")
    else:
        n_pairs = args.n_pairs

    # 4. Load reference embeddings
    print(f"Loading reference embeddings from {args.embeddings}...")
    emb = load_embeddings(args.embeddings, args.dataset_key)
    n_points, n_dims = emb.shape
    print(f"  Loaded {n_points} x {n_dims} embeddings")

    if n_points < 2:
        print(f"Error: Need at least 2 points to form pairs, got {n_points}")
        sys.exit(1)

    # 5. Three-distribution mode: query + index provided
    if args.query and args.index and knn_distances is not None:
        print(f"Loading query embeddings from {args.query}...")
        query_emb = load_embeddings(args.query, args.dataset_key)
        print(f"  Loaded {query_emb.shape[0]} x {query_emb.shape[1]} query embeddings")

        print(f"Loading FAISS index from {args.index}...")
        ref_index = faiss.read_index(args.index)
        ivf = faiss.extract_index_ivf(ref_index)
        if ivf is not None:
            ivf.nprobe = 20
            ivf.make_direct_map()  # required for reconstruct() on IVF indices
        print(f"  {ref_index.ntotal} vectors, {ref_index.d}d")

        n_sample = min(args.n_pairs, len(knn_distances))
        print(f"Sampling null distribution (n={n_sample}, random query→reference)...")
        null_dists = sample_null_query_ref_distances(query_emb, ref_index, n_sample, seed=args.seed)

        print(f"Sampling reference self-KNN (n={n_sample}, k={k or 30})...")
        ref_knn = sample_ref_self_knn(ref_index, k or 30, n_sample, seed=args.seed)

        summarize_distances(null_dists,    "NULL DISTRIBUTION (random query→reference)")
        summarize_distances(knn_distances, "QUERY KNN DISTANCES")
        summarize_distances(ref_knn,       "REFERENCE SELF-KNN DISTANCES")

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plot_three_distributions(null_dists, knn_distances, ref_knn, k or 30, args.output)
        print(f"\nSaved histogram to {args.output}")
        return

    # 6. Original two-distribution or single mode
    print(f"Sampling {n_pairs} random pairs (seed={args.seed})...")
    distances = sample_background_distances(emb, n_pairs, seed=args.seed)
    n_pairs = len(distances)  # may have been capped

    mean = float(np.mean(distances))
    summarize_distances(distances, "BACKGROUND DISTANCE SUMMARY",
                        extra_info=[f"Embeddings: {args.embeddings} ({n_points} x {n_dims})",
                                    f"Pairs sampled: {n_pairs}"])

    knn_mean = None
    if knn_distances is not None:
        knn_mean = float(np.mean(knn_distances))
        summarize_distances(knn_distances, "KNN DISTANCE SUMMARY",
                            extra_info=[f"Predictions: {args.predictions}"])
        print(f"\nKNN mean / Background mean: {knn_mean / mean:.3f}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if knn_distances is not None:
        plot_combined_histogram(distances, knn_distances, args.output, k=k)
    else:
        plot_background_histogram(distances, args.output)

    print(f"\nSaved histogram to {args.output}")


if __name__ == "__main__":
    main()
