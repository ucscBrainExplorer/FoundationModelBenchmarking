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

def calculate_ontology_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate average ontology distance and distribution statistics.
    
    Args:
        df: DataFrame with 'ontology_distance' column
        
    Returns:
        Dictionary with statistics
    """
    # Filter out NaN and invalid distances (< 0)
    valid_distances = df['ontology_distance'].dropna()
    valid_distances = valid_distances[valid_distances >= 0]
    
    if len(valid_distances) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'percentiles': {},
            'distribution': {}
        }
    
    stats = {
        'mean': float(valid_distances.mean()),
        'median': float(valid_distances.median()),
        'std': float(valid_distances.std()),
        'min': int(valid_distances.min()),
        'max': int(valid_distances.max()),
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

def analyze_distance_metric_relationship(df: pd.DataFrame, output_dir: str):
    """
    Analyze the relationship between ontology distance and accuracy metrics.
    
    Args:
        df: DataFrame with per-cell results
        output_dir: Directory to save analysis outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid distances
    valid_df = df[df['ontology_distance'].notna() & (df['ontology_distance'] >= 0)].copy()
    
    if len(valid_df) == 0:
        print("No valid ontology distances found for relationship analysis.")
        return
    
    # Calculate per-cell accuracy (exact match)
    valid_df['is_correct'] = (valid_df['true_label'] == valid_df['prediction_label']).astype(int)
    
    # Group by distance and calculate metrics
    distance_groups = valid_df.groupby('ontology_distance').agg({
        'is_correct': ['count', 'sum', 'mean'],
    }).reset_index()
    distance_groups.columns = ['ontology_distance', 'cell_count', 'correct_count', 'accuracy']
    
    # Save distance-accuracy relationship
    relationship_path = output_dir / "ontology_distance_accuracy_relationship.csv"
    distance_groups.to_csv(relationship_path, index=False)
    print(f"\nSaved distance-accuracy relationship to {relationship_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Accuracy vs Distance
    plt.subplot(1, 2, 1)
    plt.scatter(distance_groups['ontology_distance'], distance_groups['accuracy'], 
                s=distance_groups['cell_count']*2, alpha=0.6)
    plt.xlabel('Ontology Tree Distance')
    plt.ylabel('Accuracy (Exact Match)')
    plt.title('Accuracy vs Ontology Distance')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of distances
    plt.subplot(1, 2, 2)
    plt.hist(valid_df['ontology_distance'], bins=min(50, valid_df['ontology_distance'].max() + 1), 
             edgecolor='black', alpha=0.7)
    plt.xlabel('Ontology Tree Distance')
    plt.ylabel('Number of Cells')
    plt.title('Distribution of Ontology Distances')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "ontology_distance_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.close()

def generate_summary_report(df: pd.DataFrame, stats: Dict, output_path: str):
    """
    Generate a comprehensive summary report.
    
    Args:
        df: DataFrame with per-cell results
        stats: Statistics dictionary from calculate_ontology_statistics
        output_path: Path to save the report
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ONTOLOGY TREE DISTANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cells analyzed: {len(df)}\n")
        valid_distances = df['ontology_distance'].dropna()
        valid_distances = valid_distances[valid_distances >= 0]
        f.write(f"Cells with valid ontology distances: {len(valid_distances)}\n")
        f.write(f"Cells with missing/invalid distances: {len(df) - len(valid_distances)}\n\n")
        
        if len(valid_distances) > 0:
            f.write("ONTOLOGY DISTANCE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean distance: {stats['mean']:.4f}\n")
            f.write(f"Median distance: {stats['median']:.4f}\n")
            f.write(f"Standard deviation: {stats['std']:.4f}\n")
            f.write(f"Minimum distance: {stats['min']}\n")
            f.write(f"Maximum distance: {stats['max']}\n\n")
            
            f.write("PERCENTILES\n")
            f.write("-" * 80 + "\n")
            for percentile, value in stats['percentiles'].items():
                f.write(f"{percentile}: {value:.4f}\n")
            f.write("\n")
            
            # Distribution summary
            f.write("DISTRIBUTION SUMMARY\n")
            f.write("-" * 80 + "\n")
            dist_dict = stats['distribution']
            for distance in sorted(dist_dict.keys())[:20]:  # Show first 20 distances
                count = dist_dict[distance]
                percentage = (count / len(valid_distances)) * 100
                f.write(f"Distance {distance}: {count} cells ({percentage:.2f}%)\n")
            if len(dist_dict) > 20:
                f.write(f"... and {len(dist_dict) - 20} more distance values\n")
            f.write("\n")
        
        # Relationship to accuracy
        f.write("RELATIONSHIP TO ACCURACY METRICS\n")
        f.write("-" * 80 + "\n")
        valid_df = df[df['ontology_distance'].notna() & (df['ontology_distance'] >= 0)].copy()
        if len(valid_df) > 0:
            valid_df['is_correct'] = (valid_df['true_label'] == valid_df['prediction_label']).astype(int)
            
            # Overall accuracy
            overall_accuracy = valid_df['is_correct'].mean()
            f.write(f"Overall accuracy (exact match): {overall_accuracy:.4f}\n\n")
            
            # Accuracy by distance
            f.write("Accuracy by Ontology Distance:\n")
            distance_acc = valid_df.groupby('ontology_distance')['is_correct'].agg(['count', 'mean'])
            for dist in sorted(distance_acc.index)[:15]:  # Show first 15 distances
                count = distance_acc.loc[dist, 'count']
                acc = distance_acc.loc[dist, 'mean']
                f.write(f"  Distance {dist}: {acc:.4f} ({count} cells)\n")
            f.write("\n")
            
            # Correlation
            correlation = valid_df['ontology_distance'].corr(valid_df['is_correct'])
            f.write(f"Correlation between distance and correctness: {correlation:.4f}\n")
            f.write("(Negative correlation indicates that smaller distances are associated with correct predictions)\n")
        
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
    
    args = parser.parse_args()
    
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
    stats = calculate_ontology_statistics(df)
    
    print("\nGenerating summary report...")
    report_path = output_dir / "ontology_analysis_report.txt"
    generate_summary_report(df, stats, report_path)
    
    print("\nAnalyzing distance-metric relationships...")
    analyze_distance_metric_relationship(df, output_dir)
    
    # Print key statistics to console
    print("\n" + "=" * 80)
    print("KEY STATISTICS")
    print("=" * 80)
    valid_distances = df['ontology_distance'].dropna()
    valid_distances = valid_distances[valid_distances >= 0]
    if len(valid_distances) > 0:
        print(f"Average ontology tree distance: {stats['mean']:.4f}")
        print(f"Median ontology tree distance: {stats['median']:.4f}")
        print(f"Standard deviation: {stats['std']:.4f}")
        print(f"Range: {stats['min']} to {stats['max']}")
    else:
        print("No valid ontology distances found.")
    print("=" * 80)

if __name__ == "__main__":
    main()
