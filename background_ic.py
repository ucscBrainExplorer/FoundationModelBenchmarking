#!/usr/bin/env python3
"""
Compute background IC similarity from random pairs of reference cell types.

Samples random pairs of reference cells, computes Lin IC similarity between
their ontology terms, and plots the distribution. When --evaluation is provided,
overlays the prediction-vs-ground-truth IC similarity for comparison.

Usage:
  python3 background_ic.py \
    --ref-annot reference_data/prediction_obs.tsv \
    --obo cl-basic.obo \
    --output background_ic.png

  # Combined plot with evaluation IC scores:
  python3 background_ic.py \
    --ref-annot reference_data/prediction_obs.tsv \
    --obo cl-basic.obo \
    --evaluation evaluation_results/ic/per_cell_evaluation.tsv \
    --output background_vs_prediction_ic.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ontology_utils import load_ontology, precompute_ic, calculate_lin_similarity


def sample_background_ic(ref_terms, graph, ic_values, n_pairs, seed=42):
    """Sample random pairs of reference cell types and compute IC similarity.

    Args:
        ref_terms: array of CL term IDs from reference annotations
        graph: ontology graph
        ic_values: precomputed IC values
        n_pairs: number of random pairs to sample
        seed: random seed

    Returns:
        1D numpy array of IC similarity scores
    """
    n_points = len(ref_terms)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n_points, size=n_pairs)
    offsets = rng.integers(1, n_points, size=n_pairs)
    idx_b = (idx_a + offsets) % n_points

    scores = []
    for a, b in zip(idx_a, idx_b):
        sim = calculate_lin_similarity(graph, ref_terms[a], ref_terms[b], ic_values)
        if sim >= 0:
            scores.append(sim)
        else:
            scores.append(float('nan'))

    return np.array(scores)


def summarize_scores(scores, label, extra_info=None):
    """Print summary statistics for a score array."""
    valid = scores[~np.isnan(scores)]
    print()
    print(label)
    print("=" * len(label))
    if extra_info:
        for line in extra_info:
            print(line)
    print(f"Count: {len(valid)} ({len(scores) - len(valid)} skipped)")
    if len(valid) == 0:
        print("  No valid scores to summarize")
        return
    print(f"Mean: {np.mean(valid):.4f}")
    print(f"Std: {np.std(valid):.4f}")
    print(f"Median: {np.median(valid):.4f}")
    print(f"Min: {np.min(valid):.4f}")
    print(f"Max: {np.max(valid):.4f}")
    print(f"25th percentile: {np.percentile(valid, 25):.4f}")
    print(f"75th percentile: {np.percentile(valid, 75):.4f}")


def plot_combined_ic_histogram(background_scores, eval_scores, png_path):
    """Save a combined histogram of background and evaluation IC similarities."""
    bg_valid = background_scores[~np.isnan(background_scores)]
    eval_valid = eval_scores[~np.isnan(eval_scores)]

    bg_mean = float(np.mean(bg_valid))
    eval_mean = float(np.mean(eval_valid))

    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals = np.concatenate([bg_valid, eval_valid])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), 81)

    ax.hist(bg_valid, bins=bin_edges, alpha=0.5, edgecolor="black",
            linewidth=0.3, label=f"Background random pairs (n={len(bg_valid)})",
            color="steelblue")
    ax.hist(eval_valid, bins=bin_edges, alpha=0.5, edgecolor="black",
            linewidth=0.3,
            label=f"Prediction vs ground truth (n={len(eval_valid)})",
            color="coral")

    ax.axvline(bg_mean, color="navy", linestyle="--", linewidth=1.2,
               label=f"Background mean: {bg_mean:.4f}")
    ax.axvline(eval_mean, color="darkred", linestyle="--", linewidth=1.2,
               label=f"Prediction mean: {eval_mean:.4f}")

    ax.set_xlabel("IC similarity (Lin)")
    ax.set_ylabel("Count")
    ax.set_title("Background vs prediction IC similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_background_ic_histogram(scores, png_path):
    """Save a background-only IC histogram."""
    valid = scores[~np.isnan(scores)]
    mean = float(np.mean(valid))
    median = float(np.median(valid))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean:.4f}")
    ax.axvline(median, color="blue", linestyle="--", linewidth=1.2, label=f"Median: {median:.4f}")
    ax.set_xlabel("IC similarity (Lin)")
    ax.set_ylabel("Count")
    ax.set_title(f"Background pairwise IC similarity ({len(valid)} pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def load_eval_ic_scores(eval_path):
    """Load IC similarity scores from evaluate.py per_cell_evaluation.tsv.

    Returns a 1D numpy array of IC similarity scores.
    """
    import pandas as pd
    df = pd.read_csv(eval_path, sep='\t', comment='#')

    col = "ontology_IC_similarity"
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in {eval_path}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)
    scores = df[col].values.astype(float)
    n_valid = int(np.sum(~np.isnan(scores)))
    print(f"  Loaded {n_valid} IC similarity scores ({len(scores) - n_valid} NaN)")
    return scores


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute background IC similarity from random pairs of reference cell types",
    )
    parser.add_argument("--ref-annot", type=str, required=True,
                        help="Reference annotation TSV (needs cell_type_ontology_term_id column)")
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file")
    parser.add_argument("--evaluation", type=str, default=None,
                        help="per_cell_evaluation.tsv from evaluate.py; reads the "
                             "ontology_IC_similarity column and overlays on the histogram")
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="Number of random pairs to sample, default: 10000 "
                             "(auto-matches evaluation count when --evaluation is given)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed, default: 42")
    parser.add_argument("--output", type=str, default="background_ic.png",
                        help="Output histogram PNG path, default: background_ic.png")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Validate inputs
    if not os.path.isfile(args.ref_annot):
        print(f"Error: Reference annotation file not found: {args.ref_annot}")
        sys.exit(1)

    if not os.path.isfile(args.obo):
        print(f"Error: OBO file not found: {args.obo}")
        sys.exit(1)

    if args.evaluation and not os.path.isfile(args.evaluation):
        print(f"Error: Evaluation file not found: {args.evaluation}")
        sys.exit(1)

    # 2. Load evaluation IC scores if provided
    eval_scores = None
    if args.evaluation:
        print(f"Loading evaluation IC scores from {args.evaluation}...")
        eval_scores = load_eval_ic_scores(args.evaluation)

    # 3. Determine n_pairs
    if eval_scores is not None:
        n_pairs = len(eval_scores)
        print(f"  Auto-setting --n-pairs to {n_pairs} to match evaluation count")
    else:
        n_pairs = args.n_pairs

    # 4. Load ontology and precompute IC
    print(f"Loading ontology from {args.obo}...")
    graph = load_ontology(args.obo)
    print(f"  Loaded {len(graph.nodes)} terms")

    print("Precomputing IC values...")
    ic_values = precompute_ic(graph)

    # 5. Load reference annotations
    import pandas as pd
    print(f"Loading reference annotations from {args.ref_annot}...")
    ref_df = pd.read_csv(args.ref_annot, sep='\t', comment='#')
    term_col = 'cell_type_ontology_term_id'
    if term_col not in ref_df.columns:
        print(f"Error: Column '{term_col}' not found in {args.ref_annot}")
        sys.exit(1)
    ref_terms = ref_df[term_col].values
    print(f"  Loaded {len(ref_terms)} reference cells")

    # 6. Sample background IC similarities
    print(f"Sampling {n_pairs} random pairs (seed={args.seed})...")
    bg_scores = sample_background_ic(ref_terms, graph, ic_values, n_pairs, seed=args.seed)

    # 7. Print summaries
    summarize_scores(bg_scores, "BACKGROUND IC SIMILARITY SUMMARY",
                     extra_info=[f"Reference: {args.ref_annot} ({len(ref_terms)} cells)",
                                 f"Pairs sampled: {n_pairs}"])

    if eval_scores is not None:
        summarize_scores(eval_scores, "PREDICTION IC SIMILARITY SUMMARY",
                         extra_info=[f"Evaluation: {args.evaluation}"])
        bg_valid = bg_scores[~np.isnan(bg_scores)]
        eval_valid = eval_scores[~np.isnan(eval_scores)]
        if len(bg_valid) > 0 and len(eval_valid) > 0:
            print(f"\nPrediction mean / Background mean: "
                  f"{np.mean(eval_valid) / np.mean(bg_valid):.3f}")

    # 8. Save histogram
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if eval_scores is not None:
        plot_combined_ic_histogram(bg_scores, eval_scores, args.output)
    else:
        plot_background_ic_histogram(bg_scores, args.output)

    print(f"\nSaved histogram to {args.output}")


if __name__ == "__main__":
    main()
