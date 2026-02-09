#!/usr/bin/env python3
"""
Analyze per-cell prediction results from the benchmark.

This script reads per-cell result CSV files and provides:
1. Per-cell prediction results (cell, true label, prediction label, tree distance)
2. Average ontology tree distance and its distribution
3. Relationship between tree distance and accuracy/F1-weighted/Top-K accuracy
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_per_cell_results(results_dir):
    """Load all per-cell result CSV files from the directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return None
    
    csv_files = list(results_dir.glob("*_per_cell_results.csv"))
    
    if not csv_files:
        print(f"No per-cell result files found in {results_dir}")
        return None
    
    print(f"Found {len(csv_files)} per-cell result file(s)")
    
    dfs = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        dfs.append(pd.read_csv(csv_file))
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal cells: {len(combined_df)}")
    
    return combined_df

def analyze_tree_distances(df):
    """Analyze ontology tree distance statistics."""
    # Filter out invalid distances (-1 indicates error/missing, NaN means ontology wasn't loaded)
    valid_distances = df[df['ontology_distance'].notna() & (df['ontology_distance'] >= 0)]['ontology_distance']
    
    if len(valid_distances) == 0:
        print("\n⚠️  Warning: No valid ontology distances found.")
        print("   This likely means the ontology file (cl.obo) failed to load.")
        print("   Ontology distances are NaN in the results.")
        return None
    
    stats = {
        'mean': float(valid_distances.mean()),
        'median': float(valid_distances.median()),
        'std': float(valid_distances.std()),
        'min': int(valid_distances.min()),
        'max': int(valid_distances.max()),
        'q25': float(valid_distances.quantile(0.25)),
        'q75': float(valid_distances.quantile(0.75)),
        'total_cells': len(df),
        'valid_cells': len(valid_distances),
        'invalid_cells': len(df) - len(valid_distances)
    }
    
    return stats

def print_per_cell_results(df, output_file=None, max_rows=100):
    """Print per-cell results to console and optionally save to file."""
    display_cols = ['cell_id', 'true_label', 'prediction_label', 'ontology_distance']
    if 'true_label_readable' in df.columns:
        display_cols.insert(2, 'true_label_readable')
    if 'dataset' in df.columns:
        display_cols.append('dataset')
    
    display_df = df[display_cols].copy()
    
    # Replace -1 and NaN with "N/A" for readability
    display_df['ontology_distance'] = display_df['ontology_distance'].replace(-1, 'N/A')
    display_df['ontology_distance'] = display_df['ontology_distance'].fillna('N/A (ontology not loaded)')
    
    print("\n" + "="*80)
    print("PER-CELL PREDICTION RESULTS")
    print("="*80)
    
    if max_rows and len(display_df) > max_rows:
        print(f"Showing first {max_rows} of {len(display_df)} cells:")
        print(display_df.head(max_rows).to_string(index=False))
        print(f"\n... ({len(display_df) - max_rows} more rows)")
    else:
        print(display_df.to_string(index=False))
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        display_df.to_csv(output_file, index=False)
        print(f"\nFull results saved to: {output_file}")
        print(f"Total rows: {len(display_df)}")

def correlate_with_metrics(df, benchmark_results_path):
    """Correlate tree distances with accuracy, F1-weighted, and Top-K accuracy."""
    if not os.path.exists(benchmark_results_path):
        print(f"\nWarning: Benchmark results file not found: {benchmark_results_path}")
        return None
    
    benchmark_df = pd.read_csv(benchmark_results_path)
    
    print("\n" + "="*80)
    print("RELATIONSHIP BETWEEN TREE DISTANCE AND METRICS")
    print("="*80)
    
    for (dataset, index_type, metric), group_df in df.groupby(['dataset', 'index_type', 'metric']):
        summary = benchmark_df[
            (benchmark_df['Dataset'] == dataset) &
            (benchmark_df['Index'] == index_type) &
            (benchmark_df['Metric'] == metric)
        ]
        
        if len(summary) == 0:
            continue
        
        summary_row = summary.iloc[0]
        
        # Check if ontology distances are available
        valid_distances = group_df[group_df['ontology_distance'].notna() & (group_df['ontology_distance'] >= 0)]['ontology_distance']
        
        if len(valid_distances) == 0:
            print(f"\nDataset: {dataset}, Index: {index_type}, Metric: {metric}")
            print(f"  ⚠️  Ontology distances are not available (NaN)")
            print(f"  This is because the ontology file (cl.obo) failed to load.")
            print(f"  Summary Metrics:")
            print(f"    Accuracy: {summary_row['accuracy']:.4f}")
            print(f"    F1-weighted: {summary_row['f1_weighted']:.4f}")
            print(f"    Top-K Accuracy: {summary_row['top_k_accuracy']:.4f}")
            continue
        
        print(f"\nDataset: {dataset}, Index: {index_type}, Metric: {metric}")
        print(f"  Summary Metrics:")
        print(f"    Accuracy: {summary_row['accuracy']:.4f}")
        print(f"    F1-weighted: {summary_row['f1_weighted']:.4f}")
        print(f"    Top-K Accuracy: {summary_row['top_k_accuracy']:.4f}")
        print(f"    Mean Ontology Distance: {summary_row['mean_ontology_dist']:.2f}")
        print(f"\n  Per-Cell Distance Analysis:")
        print(f"    Mean Distance: {valid_distances.mean():.2f}")
        print(f"    Median Distance: {valid_distances.median():.2f}")
        print(f"    Cells with distance = 0 (exact match): {(valid_distances == 0).sum()} ({(valid_distances == 0).sum() / len(valid_distances) * 100:.1f}%)")
        print(f"    Cells with distance <= 1: {(valid_distances <= 1).sum()} ({(valid_distances <= 1).sum() / len(valid_distances) * 100:.1f}%)")
        print(f"    Cells with distance <= 2: {(valid_distances <= 2).sum()} ({(valid_distances <= 2).sum() / len(valid_distances) * 100:.1f}%)")
        print(f"    Cells with distance <= 3: {(valid_distances <= 3).sum()} ({(valid_distances <= 3).sum() / len(valid_distances) * 100:.1f}%)")
        
        exact_match_rate = (valid_distances == 0).sum() / len(valid_distances)
        print(f"\n  Relationship Analysis:")
        print(f"    Exact match rate (distance=0): {exact_match_rate:.4f}")
        print(f"    Summary accuracy: {summary_row['accuracy']:.4f}")
        print(f"    Difference: {abs(exact_match_rate - summary_row['accuracy']):.4f}")
        
        print(f"\n  Distance Distribution:")
        distance_counts = valid_distances.value_counts().sort_index()
        for dist, count in distance_counts.head(10).items():
            pct = (count / len(valid_distances)) * 100
            print(f"    Distance {int(dist)}: {count} cells ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze per-cell prediction results")
    parser.add_argument("--results-dir", type=str, default="per_cell_results",
                       help="Directory containing per-cell result CSV files")
    parser.add_argument("--benchmark-results", type=str, default="benchmark_results.csv",
                       help="Path to benchmark_results.csv")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Optional path to save per-cell results summary")
    parser.add_argument("--max-rows", type=int, default=100,
                       help="Maximum number of rows to display")
    
    args = parser.parse_args()
    
    data_root = "/data" if os.path.exists("/data") else "."
    results_dir = os.path.join(data_root, args.results_dir) if data_root != "." else args.results_dir
    benchmark_results_path = os.path.join(data_root, args.benchmark_results) if data_root != "." else args.benchmark_results
    
    df = load_per_cell_results(results_dir)
    if df is None:
        return
    
    output_file = args.output_file
    if output_file and data_root != ".":
        output_file = os.path.join(data_root, output_file)
    print_per_cell_results(df, output_file=output_file, max_rows=args.max_rows)
    
    print("\n" + "="*80)
    print("ONTOLOGY TREE DISTANCE STATISTICS")
    print("="*80)
    stats = analyze_tree_distances(df)
    if stats:
        print(f"\nAverage (Mean) Tree Distance: {stats['mean']:.4f}")
        print(f"Median Tree Distance: {stats['median']:.4f}")
        print(f"Standard Deviation: {stats['std']:.4f}")
        print(f"Minimum Distance: {stats['min']}")
        print(f"Maximum Distance: {stats['max']}")
        print(f"25th Percentile: {stats['q25']:.2f}")
        print(f"75th Percentile: {stats['q75']:.2f}")
        print(f"\nTotal Cells: {stats['total_cells']}")
        print(f"Valid Distances: {stats['valid_cells']}")
        print(f"Invalid/Missing Distances: {stats['invalid_cells']}")
    
    correlate_with_metrics(df, benchmark_results_path)

if __name__ == "__main__":
    main()
