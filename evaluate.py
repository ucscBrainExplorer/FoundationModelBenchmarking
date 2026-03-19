#!/usr/bin/env python3
"""
Evaluate cell type predictions against ground truth using ontology-based metrics.

Accepts predictions from predict.py or any external tool. Matches predictions to
ground truth by row index (row-by-row). Files must have the same number of rows
in the same order. The primary metrics are IC similarity or shortest-path distance;
exact CL term match rate is supplementary.

Usage:
  python3 evaluate.py \
    --predictions predictions/output.tsv \
    --ground_truth test_data/dataset_prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --ontology-method ic \
    --output-dir evaluation_results/
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
from obo_parser import parse_obo_names, parse_obo_replacements
from annotate_cl_terms import parse_obo_synonyms, fuzzy_normalize


import re


def _is_cl_id(value):
    """Check if a string looks like a CL term ID (e.g. CL:0000540)."""
    return bool(re.match(r'^CL:\d+$', str(value).strip()))


def _score_column_cl(col_values):
    """Score how likely a column contains CL term IDs (0.0 to 1.0)."""
    sample = [v for v in col_values if pd.notna(v) and str(v).strip()][:200]
    if not sample:
        return 0.0
    return sum(1 for v in sample if _is_cl_id(v)) / len(sample)


def _score_column_names(col_values, name_to_id):
    """Score how likely a column contains resolvable cell type names (0.0 to 1.0)."""
    sample = [v for v in col_values if pd.notna(v) and str(v).strip()][:200]
    if not sample:
        return 0.0
    return sum(1 for v in sample if str(v).lower().strip() in name_to_id) / len(sample)


def auto_detect_column(df, obo_path, role="cell type", user_col=None):
    """Auto-detect which column contains CL term IDs or resolvable cell type names.

    Strategy:
      1. If user specified a column and it exists, use it.
      2. Scan all columns: prefer one with CL IDs (>90% match).
      3. Fall back to column with highest resolvable name fraction (>50%).
      4. If nothing found, error out with available columns.

    Args:
        df: DataFrame to search.
        obo_path: Path to OBO file (for name resolution lookups).
        role: Human-readable role string for messages (e.g. "predictions").
        user_col: User-specified column name, or None for auto-detect.

    Returns:
        Tuple of (column_name, detection_report).
    """
    # If user specified and it exists, use it
    if user_col and user_col in df.columns:
        return user_col, f"  Using column '{user_col}' (user-specified)"

    # Build name lookup for scoring
    cl_names = parse_obo_names(obo_path)
    name_to_id = {name.lower(): cl_id for cl_id, name in cl_names.items()}

    # Score every column
    best_cl_col = None
    best_cl_score = 0.0
    best_name_col = None
    best_name_score = 0.0

    for col in df.columns:
        vals = df[col].tolist()

        cl_score = _score_column_cl(vals)
        if cl_score > best_cl_score:
            best_cl_score = cl_score
            best_cl_col = col

        name_score = _score_column_names(vals, name_to_id)
        if name_score > best_name_score:
            best_name_score = name_score
            best_name_col = col

    # Prefer CL ID columns
    if best_cl_score > 0.9:
        return best_cl_col, (
            f"  Auto-detected column '{best_cl_col}' for {role}"
            f" ({best_cl_score:.0%} CL term IDs)")

    # Fall back to name columns
    if best_name_score > 0.5:
        return best_name_col, (
            f"  Auto-detected column '{best_name_col}' for {role}"
            f" ({best_name_score:.0%} resolvable cell type names)")

    # Nothing found
    if user_col:
        msg = f"  Column '{user_col}' not found."
    else:
        msg = f"  No column with CL term IDs or resolvable cell type names found."
    msg += f"\n  Available columns: {list(df.columns)}"
    return None, msg


def build_label_mapping(obo_path, cl_names=None):
    """Build lookup tables for resolving cell type names to CL term IDs.

    Constructs exact, synonym, and fuzzy-normalized lookup dicts from an OBO
    file. These tables power the name-to-CL-ID resolution cascade used by
    both evaluate.py and generate_remap.py.

    Args:
        obo_path: Path to OBO file.
        cl_names: Optional pre-parsed {CL_id: name} dict. Parsed from OBO if None.

    Returns:
        Tuple of (cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id).
    """
    if cl_names is None:
        cl_names = parse_obo_names(obo_path)
    name_to_id = {name.lower(): cl_id for cl_id, name in cl_names.items()}
    synonym_to_id = parse_obo_synonyms(obo_path)
    fuzzy_name_to_id = {fuzzy_normalize(name): cl_id
                        for cl_id, name in cl_names.items()}
    fuzzy_synonym_to_id = {fuzzy_normalize(syn): cl_id
                           for syn, cl_id in synonym_to_id.items()}
    return cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id


def resolve_name(name, name_to_id, synonym_to_id, fuzzy_name_to_id,
                 fuzzy_synonym_to_id, cl_names):
    """Resolve a single cell type name to a CL term ID.

    Tries exact name match, synonym match, then fuzzy match.

    Args:
        name: Cell type name string.
        name_to_id: {lowercase_name: CL_id} dict.
        synonym_to_id: {lowercase_synonym: CL_id} dict.
        fuzzy_name_to_id: {fuzzy_name: CL_id} dict.
        fuzzy_synonym_to_id: {fuzzy_synonym: CL_id} dict.
        cl_names: {CL_id: canonical_name} dict.

    Returns:
        Tuple of (cl_id_or_None, canonical_name, method) where method is one of
        'exact', 'synonym', 'fuzzy', 'fuzzy_synonym', or None if unresolved.
    """
    name_lower = name.lower().strip()

    if name_lower in name_to_id:
        cl_id = name_to_id[name_lower]
        return cl_id, cl_names.get(cl_id, ''), 'exact'

    if name_lower in synonym_to_id:
        cl_id = synonym_to_id[name_lower]
        return cl_id, cl_names.get(cl_id, ''), 'synonym'

    fuzzy = fuzzy_normalize(name)
    if fuzzy in fuzzy_name_to_id:
        cl_id = fuzzy_name_to_id[fuzzy]
        return cl_id, cl_names.get(cl_id, ''), 'fuzzy'
    if fuzzy in fuzzy_synonym_to_id:
        cl_id = fuzzy_synonym_to_id[fuzzy]
        return cl_id, cl_names.get(cl_id, ''), 'fuzzy_synonym'

    return None, '', None


def resolve_to_cl_ids(values, obo_path, cl_names=None):
    """Auto-resolve a list of values to CL term IDs.

    If values already look like CL IDs, return them unchanged.
    Otherwise, resolve readable names to CL IDs using exact name match,
    synonym match, and fuzzy match from the OBO file.

    Args:
        values: List of strings (CL IDs or readable cell type names).
        obo_path: Path to OBO file.
        cl_names: Optional pre-parsed {CL_id: name} dict.

    Returns:
        Tuple of (resolved_list, was_resolved, resolution_report, unresolved_names):
            resolved_list: List of CL IDs (or original value if unresolved).
            was_resolved: True if name-to-ID resolution was performed.
            resolution_report: String describing what happened.
            unresolved_names: List of names that could not be resolved.
    """
    # Check if values are already CL IDs by sampling non-null values
    sample = [v for v in values if pd.notna(v) and str(v).strip()][:100]
    cl_id_frac = sum(1 for v in sample if _is_cl_id(v)) / max(len(sample), 1)

    if cl_id_frac > 0.9:
        return values, False, "Values are CL term IDs — no resolution needed.", []

    # Build lookup tables
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(obo_path, cl_names)

    # Resolve unique names first, then apply to full list
    unique_names = set(str(v).strip() for v in values if pd.notna(v))
    mapping = {}  # {original_name: CL_id or original_name}
    resolved_as = {}  # {original_name: (CL_id, canonical_name, method)}
    method_counts = {'exact': 0, 'synonym': 0, 'fuzzy': 0, 'already_cl': 0,
                     'unresolved': 0}

    for name in unique_names:
        if _is_cl_id(name):
            mapping[name] = name
            method_counts['already_cl'] += 1
            continue

        cl_id, canon_name, method = resolve_name(
            name, name_to_id, synonym_to_id,
            fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)

        if cl_id is not None:
            mapping[name] = cl_id
            resolved_as[name] = (cl_id, canon_name, method)
            if method in ('fuzzy', 'fuzzy_synonym'):
                method_counts['fuzzy'] += 1
            else:
                method_counts[method] += 1
        else:
            mapping[name] = name  # keep original
            method_counts['unresolved'] += 1

    resolved = [mapping.get(str(v).strip(), v) for v in values]

    # Count how many cells are affected by unresolved names
    unresolved_names = sorted(n for n, m_id in mapping.items()
                              if not _is_cl_id(m_id))
    unresolved_cell_count = sum(
        1 for v in values
        if str(v).strip() in unresolved_names
    )

    lines = [f"Auto-resolving readable names to CL term IDs..."]
    lines.append(f"    {len(unique_names)} unique names")
    lines.append(f"    Exact name match: {method_counts['exact']}")
    if method_counts['synonym']:
        lines.append(f"    Synonym match:    {method_counts['synonym']}")
    if method_counts['fuzzy']:
        lines.append(f"    Fuzzy match:      {method_counts['fuzzy']}")
    if method_counts['already_cl']:
        lines.append(f"    Already CL IDs:   {method_counts['already_cl']}")
    if method_counts['unresolved']:
        lines.append(f"    WARNING — UNRESOLVED: {method_counts['unresolved']} names"
                     f" ({unresolved_cell_count} cells will get NaN scores)")
        for n in unresolved_names:
            lines.append(f"      - \"{n}\"")
    report = '\n'.join(lines)

    return resolved, True, report, unresolved_names



def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate cell type predictions against ground truth using ontology metrics"
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions TSV. Must have same row count and order as ground_truth.")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Ground truth TSV. Must have same row count and order as predictions.")
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file")
    parser.add_argument("--ontology-method", type=str, default="ic",
                        choices=["ic", "shortest_path"],
                        help="Ontology scoring method (default: ic)")
    parser.add_argument("--pred_id_col", type=str,
                        default="weighted_cell_type_ontology_term_id",
                        help="CL term ID column in predictions file (default: weighted_cell_type_ontology_term_id)")
    parser.add_argument("--truth_id_col", type=str,
                        default="mapped_cell_label_ontology_term_id",
                        help="CL term ID column in ground truth file (default: mapped_cell_label_ontology_term_id)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory (default: evaluation_results/)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read predictions (and extract any comment header for provenance)
    print(f"Loading predictions from {args.predictions}...")
    prediction_provenance = []
    with open(args.predictions, 'r') as f:
        for line in f:
            if line.startswith('#'):
                prediction_provenance.append(line.rstrip('\n'))
            else:
                break
    pred_df = pd.read_csv(args.predictions, sep='\t', comment='#')
    print(f"  Loaded {len(pred_df)} predictions")

    # Auto-detect prediction column
    pred_col, pred_col_report = auto_detect_column(
        pred_df, args.obo, role="predictions", user_col=args.pred_id_col)
    if pred_col is None:
        print(f"Error: {pred_col_report}")
        sys.exit(1)
    print(pred_col_report)

    # 2. Read ground truth
    print(f"Loading ground truth from {args.ground_truth}...")
    truth_df = pd.read_csv(args.ground_truth, sep='\t')
    print(f"  Loaded {len(truth_df)} ground truth entries")

    # Auto-detect ground truth column
    truth_col, truth_col_report = auto_detect_column(
        truth_df, args.obo, role="ground truth", user_col=args.truth_id_col)
    if truth_col is None:
        print(f"Error: {truth_col_report}")
        sys.exit(1)
    print(truth_col_report)

    # 3. Match by row index - validate same row count
    print(f"\nMatching predictions to ground truth by row index...")
    print(f"  Predictions: {len(pred_df)} rows")
    print(f"  Ground truth: {len(truth_df)} rows")

    if len(pred_df) != len(truth_df):
        print(f"\nError: Row count mismatch!")
        print(f"  Predictions file has {len(pred_df)} rows")
        print(f"  Ground truth file has {len(truth_df)} rows")
        print(f"  Files must have the same number of rows in the same order.")
        sys.exit(1)

    print(f"  ✓ Row counts match ({len(pred_df)} cells)")
    print(f"  Matching row 0 → row 0, row 1 → row 1, etc.\n")

    # Extract columns by row index (already aligned)
    predictions = pred_df[pred_col].tolist()
    ground_truth = truth_df[truth_col].tolist()

    # 3a. Auto-resolve readable names to CL term IDs if needed
    predictions, pred_resolved, pred_report, pred_unresolved = resolve_to_cl_ids(
        predictions, args.obo)
    print(f"  Predictions: {pred_report}")

    ground_truth, truth_resolved, truth_report, truth_unresolved = resolve_to_cl_ids(
        ground_truth, args.obo)
    print(f"  Ground truth: {truth_report}")

    # 3b. Resolve obsolete CL terms in predictions and ground truth
    cl_replacements = parse_obo_replacements(args.obo)
    print(f"  Found {len(cl_replacements)} obsolete terms with replacements")
    cl_names = parse_obo_names(args.obo)
    n_pred_replaced = 0
    n_truth_replaced = 0
    for i, p in enumerate(predictions):
        if p in cl_replacements:
            predictions[i] = cl_replacements[p]
            n_pred_replaced += 1
    for i, t in enumerate(ground_truth):
        if t in cl_replacements:
            ground_truth[i] = cl_replacements[t]
            n_truth_replaced += 1
    # Count obsolete terms with no replacement
    all_terms = set(predictions) | set(ground_truth)
    n_unresolvable = len(all_terms - set(cl_names) - set(cl_replacements))
    if n_pred_replaced > 0 or n_truth_replaced > 0:
        print(f"  Resolved {n_pred_replaced} obsolete prediction terms and {n_truth_replaced} obsolete ground truth terms to current terms")
    if n_unresolvable > 0:
        print(f"  {n_unresolvable} obsolete terms have no replacement and will remain as-is")
    obsolete_comment = f"# Obsolete terms resolved: {n_pred_replaced} predictions replaced, {n_truth_replaced} ground truth replaced, {n_unresolvable} unresolvable"

    # Generate cell indices for output (since no cell_id column)
    cell_ids = [f"row_{i}" for i in range(len(predictions))]

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

    # 6. Compute exact match rate (cl_names already parsed in step 3a)
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

    # Build comment header for output files
    comment_lines = []
    comment_lines.append(f"# Ontology method: {args.ontology_method}")
    comment_lines.append(f"# Predictions file: {args.predictions}")
    comment_lines.append(f"# Ground truth file: {args.ground_truth}")
    comment_lines.append(f"# OBO file: {args.obo}")
    if pred_resolved:
        comment_lines.append(f"# Predictions: readable names auto-resolved to CL term IDs (column: {pred_col})")
    if truth_resolved:
        comment_lines.append(f"# Ground truth: readable names auto-resolved to CL term IDs (column: {truth_col})")
    comment_lines.append(obsolete_comment)
    if prediction_provenance:
        comment_lines.append("# --- Prediction provenance (from predictions file) ---")
        comment_lines.extend(prediction_provenance)
    comment_header = '\n'.join(comment_lines) + '\n'

    # Save per-cell results (without the internal alias columns)
    per_cell_output = per_cell_df[['cell_id', 'predicted_cl_term_id', 'truth_cl_term_id',
                                    'predicted_cell_type', 'truth_cell_type',
                                    onto_col, 'is_exact_match']].copy()
    per_cell_path = os.path.join(output_dir, "per_cell_evaluation.tsv")
    with open(per_cell_path, 'w') as f:
        f.write(comment_header)
        per_cell_output.to_csv(f, sep='\t', index=False, na_rep='NaN')
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
            len(predictions),
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "evaluation_summary.tsv")
    with open(summary_path, 'w') as f:
        f.write(comment_header)
        summary_df.to_csv(f, sep='\t', index=False, na_rep='NaN')
    print(f"  Saved evaluation summary to {summary_path}")

    # 10. Generate report and plots
    report_path = os.path.join(output_dir, "ontology_analysis_report.txt")
    generate_summary_report(per_cell_df, stats, report_path,
                            ontology_method=args.ontology_method,
                            comment_header=comment_header)

    analyze_distance_metric_relationship(per_cell_df, output_dir,
                                         ontology_method=args.ontology_method)

    # Save unresolved names if any
    all_unresolved = sorted(set(pred_unresolved + truth_unresolved))
    if all_unresolved:
        unresolved_path = os.path.join(output_dir, "unresolved_names.txt")
        with open(unresolved_path, 'w') as f:
            f.write("# Cell type names that could not be resolved to CL term IDs.\n")
            f.write("# These cells will have NaN ontology scores.\n")
            f.write(f"# OBO file: {args.obo}\n")
            f.write("#\n")
            if pred_unresolved:
                f.write("# From predictions:\n")
                for n in sorted(pred_unresolved):
                    f.write(f"{n}\n")
            if truth_unresolved:
                f.write("# From ground truth:\n")
                for n in sorted(truth_unresolved):
                    f.write(f"{n}\n")
        print(f"\n  WARNING: {len(all_unresolved)} names could not be resolved to CL terms.")
        print(f"  Saved to {unresolved_path}")

    # Print summary to console
    is_similarity = (args.ontology_method == 'ic')
    metric_noun = "similarity" if is_similarity else "distance"
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Cells evaluated: {len(predictions)}")
    print(f"Exact CL term match rate: {exact_match_rate:.2f}")
    valid_scores = [s for s in per_cell_scores if not (s != s)]  # filter NaN
    if valid_scores:
        print(f"Mean ontology {metric_noun}: {stats['mean']:.2f}")
        print(f"Median ontology {metric_noun}: {stats['median']:.2f}")
    else:
        print(f"No valid ontology {metric_noun} scores computed")
    print(f"{'=' * 60}")
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
