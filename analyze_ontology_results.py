#!/usr/bin/env python3
"""
Analysis script for ontology tree distance results.

This script:
1. Reads per-cell prediction results
2. Calculates average ontology tree distance and its distribution
3. Analyzes the relationship between ontology distance and accuracy metrics
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path

def load_per_cell_results(results_dir: str) -> pd.DataFrame:
    """
    Load all per-cell results CSV files from a directory.
    
    Args:
        results_dir: Directory containing per_cell_results CSV files
        
    Returns:
        Combined DataFrame with all per-cell results
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    all_results = []
    for csv_file in results_dir.glob("*_per_cell_results.csv"):
        df = pd.read_csv(csv_file)
        all_results.append(df)
        print(f"Loaded {len(df)} cells from {csv_file.name}")
    
    if not all_results:
        raise ValueError(f"No per-cell results files found in {results_dir}")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal cells loaded: {len(combined_df)}")
    return combined_df

def _detect_ontology_column(df: pd.DataFrame, ontology_method: str = 'ic') -> str:
    """Return the ontology score column name present in *df*."""
    if ontology_method == 'ic' and 'ontology_IC_similarity' in df.columns:
        return 'ontology_IC_similarity'
    if ontology_method == 'shortest_path' and 'ontology_shortestpath_distance' in df.columns:
        return 'ontology_shortestpath_distance'
    # Legacy fallback
    if 'ontology_distance' in df.columns:
        return 'ontology_distance'
    # Try the other method's column as last resort
    for col in ('ontology_IC_similarity', 'ontology_shortestpath_distance'):
        if col in df.columns:
            return col
    raise KeyError("No ontology score column found in DataFrame")


def calculate_ontology_statistics(df: pd.DataFrame, ontology_method: str = 'ic') -> Dict:
    """
    Calculate average ontology score and distribution statistics.

    Args:
        df: DataFrame with ontology score column
        ontology_method: 'ic' or 'shortest_path'

    Returns:
        Dictionary with statistics (includes 'column' key with detected column name)
    """
    col = _detect_ontology_column(df, ontology_method)

    # Filter out NaN and invalid values (< 0)
    valid_distances = df[col].dropna()
    valid_distances = valid_distances[valid_distances >= 0]
    
    if len(valid_distances) == 0:
        return {
            'column': col,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'percentiles': {},
            'distribution': {}
        }

    stats = {
        'column': col,
        'mean': float(valid_distances.mean()),
        'median': float(valid_distances.median()),
        'std': float(valid_distances.std()),
        'min': float(valid_distances.min()),
        'max': float(valid_distances.max()),
        'percentiles': {
            '25th': float(valid_distances.quantile(0.25)),
            '75th': float(valid_distances.quantile(0.75)),
            '90th': float(valid_distances.quantile(0.90)),
            '95th': float(valid_distances.quantile(0.95)),
            '99th': float(valid_distances.quantile(0.99))
        },
        'distribution': valid_distances.value_counts().sort_index().to_dict()
    }
    
    return stats

def analyze_distance_metric_relationship(df: pd.DataFrame, output_dir: str, ontology_method: str = 'ic'):
    """
    Analyze the relationship between ontology score and exact match.

    Args:
        df: DataFrame with per-cell results
        output_dir: Directory to save analysis outputs
        ontology_method: 'ic' or 'shortest_path'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    col = _detect_ontology_column(df, ontology_method)
    is_similarity = (ontology_method == 'ic')

    # Filter valid values
    valid_df = df[df[col].notna() & (df[col] >= 0)].copy()

    if len(valid_df) == 0:
        print("No valid ontology scores found for relationship analysis.")
        return

    # Calculate per-cell exact match
    valid_df['is_correct'] = (valid_df['true_label'] == valid_df['prediction_label']).astype(int)

    # Group by score and calculate metrics
    distance_groups = valid_df.groupby(col).agg({
        'is_correct': ['count', 'sum', 'mean'],
    }).reset_index()
    distance_groups.columns = [col, 'cell_count', 'correct_count', 'accuracy']

    # Labels
    if is_similarity:
        x_label = 'Ontology Similarity (Lin/IC)'
        title_scatter = 'Exact Match vs Ontology IC Similarity'
        title_hist = 'Distribution of Ontology IC Similarities'
    else:
        x_label = 'Ontology Shortest Distance'
        title_scatter = 'Exact Match vs Ontology Distance'
        title_hist = 'Distribution of Ontology Distances'

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Plot 1: Exact match vs Score
    ax1 = plt.subplot(1, 2, 1)
    counts = distance_groups['cell_count']
    count_min, count_max = counts.min(), counts.max()
    min_size, max_size = 20, 500
    if count_max > count_min:
        sizes = min_size + (counts - count_min) / (count_max - count_min) * (max_size - min_size)
    else:
        sizes = (min_size + max_size) / 2
    scatter = ax1.scatter(distance_groups[col], distance_groups['accuracy'],
                          s=sizes, alpha=0.6)
    # Size legend
    legend_counts = [int(count_min), int((count_min + count_max) / 2), int(count_max)]
    legend_sizes = [min_size + (c - count_min) / max(count_max - count_min, 1) * (max_size - min_size) for c in legend_counts]
    for c, sz in zip(legend_counts, legend_sizes):
        ax1.scatter([], [], s=sz, alpha=0.6, color='C0', label=f'{c:,} cells')
    ax1.legend(title='Cell count', loc='best', framealpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel('Exact Match')
    plt.title(title_scatter)
    plt.grid(True, alpha=0.3)

    # Plot 2: Distribution
    plt.subplot(1, 2, 2)
    max_val = valid_df[col].max()
    #n_bins = int(min(50, max_val + 1)) if max_val > 0 else 50
    plt.hist(valid_df[col], edgecolor='black', alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel('Number of Cells')
    plt.title(title_hist)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = "ontology_similarity_analysis.png" if is_similarity else "ontology_distance_analysis.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.close()

def generate_summary_report(df: pd.DataFrame, stats: Dict, output_path: str, ontology_method: str = 'ic', comment_header: str = None):
    """
    Generate a comprehensive summary report.

    Args:
        df: DataFrame with per-cell results
        stats: Statistics dictionary from calculate_ontology_statistics
        output_path: Path to save the report
        ontology_method: 'ic' or 'shortest_path'
        comment_header: Optional metadata header to prepend to the report
    """
    output_path = Path(output_path)
    col = stats.get('column', _detect_ontology_column(df, ontology_method))
    is_similarity = (ontology_method == 'ic')

    # Terminology
    metric_noun = "similarity" if is_similarity else "distance"
    report_title = "ONTOLOGY SIMILARITY ANALYSIS REPORT" if is_similarity else "ONTOLOGY DISTANCE ANALYSIS REPORT"
    section_title = "ONTOLOGY SIMILARITY STATISTICS" if is_similarity else "ONTOLOGY DISTANCE STATISTICS"
    value_label = "Similarity" if is_similarity else "Distance"

    with open(output_path, 'w') as f:
        if comment_header:
            f.write(comment_header)
            f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"{report_title}\n")
        f.write("=" * 80 + "\n\n")

        # Evaluation Summary
        f.write("=" * 60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")

        # Calculate total cells and exact match rate
        total_cells = len(df)
        if 'true_label' in df.columns and 'prediction_label' in df.columns:
            exact_matches = (df['true_label'] == df['prediction_label']).sum()
            exact_match_rate = exact_matches / total_cells if total_cells > 0 else 0.0
            f.write(f"Cells evaluated: {total_cells}\n")
            f.write(f"Exact CL term match rate: {exact_match_rate:.4f}\n")
        else:
            f.write(f"Cells evaluated: {total_cells}\n")

        # Write ontology statistics
        f.write(f"Mean ontology {metric_noun}: {stats['mean']:.4f}\n")
        f.write(f"Median ontology {metric_noun}: {stats['median']:.4f}\n")
        f.write("=" * 60 + "\n\n")

        # Metric interpretation guide
        f.write("METRIC INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        if is_similarity:
            f.write("Method: Lin semantic similarity with Zhou Information Content (IC)\n")
            f.write("Range:  0.0 to 1.0\n")
            f.write("  1.0 = perfect match (predicted and true cell type are identical)\n")
            f.write("  High values (>0.8) = prediction is semantically very close to the true type\n")
            f.write("  Low values (<0.3)  = prediction is semantically distant from the true type\n")
            f.write("  0.0 = no common ancestor found in the ontology (unrelated or missing terms)\n")
            f.write("\nHigher values are better. A mean similarity near 1.0 indicates that even\n")
            f.write("when the model's prediction is not an exact match, it is typically a closely\n")
            f.write("related cell type in the ontology hierarchy.\n")
        else:
            f.write("Method: Shortest undirected path in the Cell Ontology graph\n")
            f.write("Range:  0 to N (number of edges between two terms)\n")
            f.write("  0 = perfect match (predicted and true cell type are identical)\n")
            f.write("  Low values (1-2)  = prediction is a close neighbor in the ontology tree\n")
            f.write("  High values (>5)  = prediction is far from the true type in the hierarchy\n")
            f.write("\nLower values are better. A mean distance near 0 indicates that even when\n")
            f.write("the model's prediction is not an exact match, it is typically a closely\n")
            f.write("related cell type in the ontology hierarchy.\n")
        f.write("\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cells analyzed: {len(df)}\n")
        valid_values = df[col].dropna()
        valid_values = valid_values[valid_values >= 0]
        f.write(f"Cells with valid ontology {metric_noun} values: {len(valid_values)}\n")
        f.write(f"Cells with missing/invalid values: {len(df) - len(valid_values)}\n\n")

        if len(valid_values) > 0:
            f.write(f"{section_title}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean {metric_noun}: {stats['mean']:.4f}\n")
            f.write(f"Median {metric_noun}: {stats['median']:.4f}\n")
            f.write(f"Standard deviation: {stats['std']:.4f}\n")
            f.write(f"Minimum {metric_noun}: {stats['min']:.4f}\n")
            f.write(f"Maximum {metric_noun}: {stats['max']:.4f}\n\n")

            f.write("PERCENTILES\n")
            f.write("-" * 80 + "\n")
            for percentile, value in stats['percentiles'].items():
                f.write(f"{percentile}: {value:.4f}\n")
            f.write("\n")

            # Distribution summary
            f.write("DISTRIBUTION SUMMARY\n")
            f.write("-" * 80 + "\n")
            if is_similarity:
                # Continuous values — bucket into ranges
                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f})" for i in range(len(bins)-2)] + [f"[0.9, 1.0]"]
                binned = pd.cut(valid_values, bins=bins, right=True, include_lowest=True)
                bin_counts = binned.value_counts().sort_index()
                exact_zero = int((valid_values == 0.0).sum())
                exact_one = int((valid_values == 1.0).sum())
                for interval, count in bin_counts.items():
                    count = int(count)
                    percentage = (count / len(valid_values)) * 100
                    f.write(f"  {value_label} {interval}: {count:>7} cells ({percentage:5.2f}%)\n")
                f.write(f"\n  Exact 0.0 (no common ancestor): {exact_zero:>7} cells ({exact_zero/len(valid_values)*100:5.2f}%)\n")
                f.write(f"  Exact 1.0 (perfect match):       {exact_one:>7} cells ({exact_one/len(valid_values)*100:5.2f}%)\n")
            else:
                # Discrete integer distances — show each value
                dist_dict = stats['distribution']
                for score in sorted(dist_dict.keys())[:20]:
                    count = dist_dict[score]
                    percentage = (count / len(valid_values)) * 100
                    f.write(f"  {value_label} {score}: {count:>7} cells ({percentage:5.2f}%)\n")
                if len(dist_dict) > 20:
                    f.write(f"  ... and {len(dist_dict) - 20} more {metric_noun} values\n")
            f.write("\n")

        # Relationship to exact match
        f.write("RELATIONSHIP TO EXACT MATCH\n")
        f.write("-" * 80 + "\n")
        valid_df = df[df[col].notna() & (df[col] >= 0)].copy()
        if len(valid_df) > 0:
            valid_df['is_correct'] = (valid_df['true_label'] == valid_df['prediction_label']).astype(int)

            # Overall exact match rate
            overall_accuracy = valid_df['is_correct'].mean()
            f.write(f"Overall exact match rate: {overall_accuracy:.4f}\n\n")

            # Exact match rate by score
            f.write(f"Exact Match Rate by Ontology {value_label}:\n")
            if is_similarity:
                # Bucket continuous similarities into ranges
                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f})" for i in range(len(bins)-2)] + ["[0.9, 1.0]"]
                valid_df['_bin'] = pd.cut(valid_df[col], bins=bins, right=True, include_lowest=True)
                bin_acc = valid_df.groupby('_bin', observed=True)['is_correct'].agg(['count', 'mean'])
                for interval in bin_acc.index:
                    count = int(bin_acc.loc[interval, 'count'])
                    acc = bin_acc.loc[interval, 'mean']
                    f.write(f"  {value_label} {interval}: {acc:.4f} exact match rate ({count} cells)\n")
                valid_df.drop(columns=['_bin'], inplace=True)
            else:
                score_acc = valid_df.groupby(col)['is_correct'].agg(['count', 'mean'])
                for val in sorted(score_acc.index)[:15]:
                    count = int(score_acc.loc[val, 'count'])
                    acc = score_acc.loc[val, 'mean']
                    f.write(f"  {value_label} {val}: {acc:.4f} exact match rate ({count} cells)\n")
            f.write("\n")

            # Correlation
            correlation = valid_df[col].corr(valid_df['is_correct'])
            f.write(f"Correlation between {metric_noun} and exact match: {correlation:.4f}\n")
            if is_similarity:
                f.write("(Positive correlation indicates higher similarity associated with exact matches)\n")
            else:
                f.write("(Negative correlation indicates smaller distances associated with exact matches)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    print(f"\nSaved summary report to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze ontology tree distance results")
    parser.add_argument("--results-dir", type=str,
                       default="per_cell_results",
                       help="Directory containing per-cell results CSV files")
    parser.add_argument("--output-dir", type=str,
                       default="ontology_analysis",
                       help="Directory to save analysis outputs")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Root directory for data (default: current directory or /data if exists)")
    parser.add_argument("--ontology-method", type=str, default="ic",
                       choices=["ic", "shortest_path"],
                       help="Ontology scoring method used when generating the per-cell results")

    args = parser.parse_args()
    ontology_method = args.ontology_method

    # Determine data root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path("/data") if Path("/data").exists() else Path(".")

    results_dir = data_root / args.results_dir
    output_dir = data_root / args.output_dir

    print("Loading per-cell results...")
    df = load_per_cell_results(results_dir)

    print("\nCalculating ontology statistics...")
    stats = calculate_ontology_statistics(df, ontology_method=ontology_method)
    col = stats['column']

    print("\nGenerating summary report...")
    report_path = output_dir / "ontology_analysis_report.txt"
    generate_summary_report(df, stats, report_path, ontology_method=ontology_method)

    print("\nAnalyzing ontology-metric relationships...")
    analyze_distance_metric_relationship(df, output_dir, ontology_method=ontology_method)

    # Print key statistics to console
    is_similarity = (ontology_method == 'ic')
    metric_noun = "similarity" if is_similarity else "distance"
    print("\n" + "=" * 80)
    print("KEY STATISTICS")
    print("=" * 80)
    valid_values = df[col].dropna()
    valid_values = valid_values[valid_values >= 0]
    if len(valid_values) > 0:
        print(f"Average ontology {metric_noun}: {stats['mean']:.4f}")
        print(f"Median ontology {metric_noun}: {stats['median']:.4f}")
        print(f"Standard deviation: {stats['std']:.4f}")
        print(f"Range: {stats['min']} to {stats['max']}")
    else:
        print(f"No valid ontology {metric_noun} values found.")
    print("=" * 80)

if __name__ == "__main__":
    main()
