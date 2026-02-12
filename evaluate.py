#!/usr/bin/env python3
"""
Evaluate cell type predictions against ground truth using ontology-based metrics.

Accepts predictions from predict.py or any external tool -- only requires two
columns: cell_id + predicted CL term ID. The primary metrics are IC similarity
or shortest-path distance; exact CL term match rate is supplementary.

Usage:
  python3 evaluate.py \
    --predictions predictions/output.tsv \
    --ground_truth test_data/dataset_prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --ontology-method ic \
    --output evaluation_results/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from ontology_utils import load_ontology, precompute_ic, calculate_per_cell_distances
from analyze_ontology_results import (
    calculate_ontology_statistics,
    generate_summary_report,
    analyze_distance_metric_relationship,
)
from obo_parser import parse_obo_names


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate cell type predictions against ground truth using ontology metrics"
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions TSV (min columns: cell_id + CL term ID)")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Ground truth TSV (min columns: cell_id + CL term ID)")
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file")
    parser.add_argument("--ontology-method", type=str, default="ic",
                        choices=["ic", "shortest_path"],
                        help="Ontology scoring method (default: ic)")
    parser.add_argument("--pred_id_col", type=str,
                        default="predicted_cell_type_ontology_term_id",
                        help="CL term ID column in predictions file")
    parser.add_argument("--pred_cell_id_col", type=str, default="cell_id",
                        help="Cell ID column in predictions file")
    parser.add_argument("--truth_id_col", type=str,
                        default="cell_type_ontology_term_id",
                        help="CL term ID column in ground truth file")
    parser.add_argument("--truth_cell_id_col", type=str, default="cell_id",
                        help="Cell ID column in ground truth file")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory (default: evaluation_results/)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read predictions
    print(f"Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions, sep='\t')
    if args.pred_cell_id_col not in pred_df.columns:
        print(f"Error: column '{args.pred_cell_id_col}' not found in predictions file.")
        print(f"  Available columns: {list(pred_df.columns)}")
        sys.exit(1)
    if args.pred_id_col not in pred_df.columns:
        print(f"Error: column '{args.pred_id_col}' not found in predictions file.")
        print(f"  Available columns: {list(pred_df.columns)}")
        sys.exit(1)
    print(f"  Loaded {len(pred_df)} predictions")

    # 2. Read ground truth
    print(f"Loading ground truth from {args.ground_truth}...")
    truth_df = pd.read_csv(args.ground_truth, sep='\t')
    if args.truth_cell_id_col not in truth_df.columns:
        print(f"Error: column '{args.truth_cell_id_col}' not found in ground truth file.")
        print(f"  Available columns: {list(truth_df.columns)}")
        sys.exit(1)
    if args.truth_id_col not in truth_df.columns:
        print(f"Error: column '{args.truth_id_col}' not found in ground truth file.")
        print(f"  Available columns: {list(truth_df.columns)}")
        sys.exit(1)
    print(f"  Loaded {len(truth_df)} ground truth entries")

    # 3. Inner join on cell_id
    pred_subset = pred_df[[args.pred_cell_id_col, args.pred_id_col]].copy()
    pred_subset.columns = ['cell_id', 'predicted_cl_term_id']

    truth_subset = truth_df[[args.truth_cell_id_col, args.truth_id_col]].copy()
    truth_subset.columns = ['cell_id', 'truth_cl_term_id']

    merged = pred_subset.merge(truth_subset, on='cell_id', how='inner')
    print(f"  Matched {len(merged)} cells (inner join on cell_id)")

    if len(merged) == 0:
        print("Error: no matching cell IDs between predictions and ground truth.")
        print(f"  Prediction cell IDs (first 5): {pred_subset['cell_id'].head().tolist()}")
        print(f"  Ground truth cell IDs (first 5): {truth_subset['cell_id'].head().tolist()}")
        sys.exit(1)

    n_pred_only = len(pred_subset) - len(merged)
    n_truth_only = len(truth_subset) - len(merged)
    if n_pred_only > 0:
        print(f"  Note: {n_pred_only} prediction cells not in ground truth (excluded)")
    if n_truth_only > 0:
        print(f"  Note: {n_truth_only} ground truth cells not in predictions (excluded)")

    predictions = merged['predicted_cl_term_id'].tolist()
    ground_truth = merged['truth_cl_term_id'].tolist()
    cell_ids = merged['cell_id'].tolist()

    # 4. Load ontology and precompute IC
    print(f"Loading ontology from {args.obo}...")
    ontology_graph = load_ontology(args.obo)
    print(f"  Ontology loaded ({ontology_graph.number_of_nodes()} terms)")

    ic_values = None
    if args.ontology_method == 'ic':
        print("Precomputing Information Content...")
        ic_values = precompute_ic(ontology_graph, k=0.5)
        print(f"  IC computed for {len(ic_values)} terms")

    # 5. Compute per-cell ontology scores
    print(f"Computing per-cell ontology scores (method: {args.ontology_method})...")
    per_cell_scores = calculate_per_cell_distances(
        ontology_graph, predictions, ground_truth,
        method=args.ontology_method, ic_values=ic_values
    )

    # 6. Parse OBO for readable names
    cl_names = parse_obo_names(args.obo)

    # 7. Compute exact match rate
    is_exact_match = [1 if p == t else 0 for p, t in zip(predictions, ground_truth)]
    exact_match_rate = sum(is_exact_match) / len(is_exact_match)

    # Determine ontology score column name (matching what analyze_ontology_results expects)
    if args.ontology_method == 'ic':
        onto_col = 'ontology_IC_similarity'
    else:
        onto_col = 'ontology_shortestpath_distance'

    # 8. Build per-cell results DataFrame
    # Use column names that analyze_ontology_results expects (true_label, prediction_label)
    per_cell_df = pd.DataFrame({
        'cell_id': cell_ids,
        'predicted_cl_term_id': predictions,
        'truth_cl_term_id': ground_truth,
        'predicted_cell_type': [cl_names.get(p, p) for p in predictions],
        'truth_cell_type': [cl_names.get(t, t) for t in ground_truth],
        onto_col: per_cell_scores,
        'is_exact_match': is_exact_match,
        # These aliases are needed by generate_summary_report / analyze_distance_metric_relationship
        'true_label': ground_truth,
        'prediction_label': predictions,
    })

    # Save per-cell results (without the internal alias columns)
    per_cell_output = per_cell_df[['cell_id', 'predicted_cl_term_id', 'truth_cl_term_id',
                                    'predicted_cell_type', 'truth_cell_type',
                                    onto_col, 'is_exact_match']].copy()
    per_cell_output.rename(columns={onto_col: 'ontology_score'}, inplace=True)
    per_cell_path = os.path.join(output_dir, "per_cell_evaluation.tsv")
    per_cell_output.to_csv(per_cell_path, sep='\t', index=False)
    print(f"  Saved per-cell evaluation to {per_cell_path}")

    # 9. Compute aggregate statistics using analyze_ontology_results
    stats = calculate_ontology_statistics(per_cell_df, ontology_method=args.ontology_method)

    # Build evaluation summary
    metric_name = "IC_similarity" if args.ontology_method == 'ic' else "shortest_path_distance"
    summary_data = {
        'metric': [
            f'mean_ontology_{metric_name}',
            f'median_ontology_{metric_name}',
            f'std_ontology_{metric_name}',
            'exact_match_rate',
            'total_cells_evaluated',
        ],
        'value': [
            stats['mean'],
            stats['median'],
            stats['std'],
            exact_match_rate,
            len(merged),
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "evaluation_summary.tsv")
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"  Saved evaluation summary to {summary_path}")

    # 10. Generate report and plots
    report_path = os.path.join(output_dir, "ontology_analysis_report.txt")
    generate_summary_report(per_cell_df, stats, report_path,
                            ontology_method=args.ontology_method)

    analyze_distance_metric_relationship(per_cell_df, output_dir,
                                         ontology_method=args.ontology_method)

    # Print summary to console
    is_similarity = (args.ontology_method == 'ic')
    metric_noun = "similarity" if is_similarity else "distance"
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Cells evaluated: {len(merged)}")
    print(f"Exact CL term match rate: {exact_match_rate:.4f}")
    valid_scores = [s for s in per_cell_scores if not (s != s)]  # filter NaN
    if valid_scores:
        print(f"Mean ontology {metric_noun}: {stats['mean']:.4f}")
        print(f"Median ontology {metric_noun}: {stats['median']:.4f}")
    else:
        print(f"No valid ontology {metric_noun} scores computed")
    print(f"{'=' * 60}")
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
