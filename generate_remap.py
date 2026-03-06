#!/usr/bin/env python3
"""
Auto-detect annotation columns in a ground truth file and generate a remap TSV
that maps author labels to Cell Ontology (CL) term IDs.

Scans each column for cell-type-like labels, scores by OBO resolvability, picks
the best column (or uses --column), and writes a mapping table.  The output can
be fed to ``evaluate.py --remap-file`` to replace lossy cellxgene CL labels with
more specific author annotations.

Usage:
  python3 generate_remap.py \
    --input ground_truth.tsv \
    --obo cl-basic.obo \
    --output remap.tsv \
    [--column annot_level_3_rev2]
"""

import argparse
import os
import re
import sys

import pandas as pd

from evaluate import build_label_mapping, resolve_name, _is_cl_id
from obo_parser import parse_obo_names
from annotate_cl_terms import fuzzy_normalize, query_llm_mapping


def build_parser():
    parser = argparse.ArgumentParser(
        description="Auto-detect annotation columns and generate a remap TSV"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Ground truth TSV file with annotation columns")
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output remap TSV path")
    parser.add_argument("--column", type=str, default=None,
                        help="Annotation column to use (auto-detect if omitted)")
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="Minimum resolvability fraction to consider a column (default: 0.10)")
    parser.add_argument("--use-llm", action="store_true", default=False,
                        help="Use LLM to resolve unmatched labels. "
                             "Requires ANTHROPIC_API_KEY or OPENAI_API_KEY env var.")
    return parser


def _is_metadata_column(col_name, values):
    """Heuristic: return True if the column looks like metadata, not annotations."""
    non_null = [v for v in values if pd.notna(v)]
    if not non_null:
        return True

    # Numeric columns (ages, coordinates, counts)
    numeric_count = 0
    for v in non_null[:200]:
        try:
            float(v)
            numeric_count += 1
        except (ValueError, TypeError):
            pass
    if numeric_count / max(len(non_null[:200]), 1) > 0.8:
        return True

    # Count unique values across the full column (cheap set operation)
    n_unique = len(set(str(v) for v in non_null))

    # High-cardinality columns (barcodes, IDs)
    if n_unique > 1000:
        return True

    # Very low cardinality (< 3 unique values) — likely a constant/flag field
    if n_unique < 3:
        return True

    # Columns that are already CL term IDs (we want to IMPROVE upon those)
    cl_count = sum(1 for v in non_null[:200] if _is_cl_id(str(v).strip()))
    if cl_count / max(len(non_null[:200]), 1) > 0.9:
        return True

    return False


def score_column(values, name_to_id, synonym_to_id, fuzzy_name_to_id,
                 fuzzy_synonym_to_id, cl_names):
    """Score a column by the fraction of unique labels that resolve to CL terms.

    Returns:
        Tuple of (score, n_unique, n_resolved).
    """
    unique_labels = set(str(v).strip() for v in values
                        if pd.notna(v) and str(v).strip())
    if not unique_labels:
        return 0.0, 0, 0

    n_resolved = 0
    for label in unique_labels:
        if _is_cl_id(label):
            n_resolved += 1
            continue
        cl_id, _, method = resolve_name(
            label, name_to_id, synonym_to_id,
            fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
        if cl_id is not None:
            n_resolved += 1

    score = n_resolved / len(unique_labels)
    return score, len(unique_labels), n_resolved


def detect_best_column(df, obo_path, min_score=0.10):
    """Auto-detect the best annotation column for remapping.

    Scans all columns, skips metadata, scores by OBO resolvability,
    and returns the best column above the threshold.

    Args:
        df: DataFrame to scan.
        obo_path: Path to OBO file.
        min_score: Minimum resolvability fraction to accept.

    Returns:
        Tuple of (column_name_or_None, scores_report_string).
    """
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(obo_path)

    # Detect cellxgene paired columns: if "{col}_ontology_term_id" exists
    # and contains CL IDs, then "{col}" is a cellxgene label column — skip both.
    cellxgene_cols = set()
    for col in df.columns:
        paired = col + '_ontology_term_id'
        if paired in df.columns:
            sample = [v for v in df[paired] if pd.notna(v) and str(v).strip()][:200]
            cl_frac = sum(1 for v in sample if _is_cl_id(str(v).strip())) / max(len(sample), 1)
            if cl_frac > 0.9:
                cellxgene_cols.add(col)
    if cellxgene_cols:
        print(f"  Skipping cellxgene label columns: {sorted(cellxgene_cols)}")

    results = []  # (col_name, score, n_unique, n_resolved)

    for col in df.columns:
        if col in cellxgene_cols:
            continue
        vals = df[col].tolist()
        if _is_metadata_column(col, vals):
            continue

        score, n_unique, n_resolved = score_column(
            vals, name_to_id, synonym_to_id,
            fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
        results.append((col, score, n_unique, n_resolved))

    if not results:
        return None, "No candidate annotation columns found."

    # Rank by: most resolved labels first, then most unique labels (granularity)
    results.sort(key=lambda x: (x[3], x[2]), reverse=True)

    lines = ["Column resolvability scores:"]
    for col, score, n_unique, n_resolved in results:
        marker = " <-- best" if col == results[0][0] and results[0][1] >= min_score else ""
        lines.append(f"  {col}: {score:.0%} ({n_resolved}/{n_unique} unique labels){marker}")

    best_col, best_score, _, _ = results[0]
    if best_score < min_score:
        lines.append(f"\nNo column meets the minimum threshold ({min_score:.0%}).")
        return None, '\n'.join(lines)

    return best_col, '\n'.join(lines)


def _filter_candidate_terms(label, cl_names, max_candidates=200):
    """Select a focused subset of CL terms relevant to a label.

    Extracts keywords from the label and filters CL terms that share
    at least one keyword. Falls back to the full dict if no keywords match.
    """
    # Extract keywords (2+ chars, lowercased, skip common stop words)
    stop = {'of', 'the', 'and', 'or', 'in', 'to', 'a', 'an', 'by', 'is',
            'at', 'on', 'for', 'from', 'with', 'derived', 'type'}
    words = set(w.lower() for w in re.split(r'[\s\-_/]+', label)
                if len(w) >= 2 and w.lower() not in stop)

    if not words:
        return cl_names

    # Score each CL term by how many keywords it shares with the label
    scored = {}
    for cl_id, name in cl_names.items():
        name_words = set(w.lower() for w in re.split(r'[\s\-_/]+', name))
        overlap = len(words & name_words)
        if overlap > 0:
            scored[cl_id] = name

    if len(scored) < 5:
        # Too few matches — broaden: include partial substring matches
        for cl_id, name in cl_names.items():
            if cl_id in scored:
                continue
            name_lower = name.lower()
            if any(w in name_lower for w in words):
                scored[cl_id] = name
            if len(scored) >= max_candidates:
                break

    return scored if scored else cl_names


def _resolve_with_llm(unresolved_labels, cl_names):
    """Attempt LLM resolution for a list of unresolved labels.

    Tries Claude first (ANTHROPIC_API_KEY), then OpenAI (OPENAI_API_KEY).
    Returns dict {label: (cl_id, cl_name, method)} for successfully resolved labels.
    """
    has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
    has_openai = bool(os.environ.get('OPENAI_API_KEY'))

    if not has_anthropic and not has_openai:
        print("\n  Error: --use-llm requires ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("  Set one of these environment variables:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    export OPENAI_API_KEY=sk-...")
        return {}

    apis = []
    if has_anthropic:
        apis.append('claude')
    if has_openai:
        apis.append('openai')
    print(f"\n  LLM resolution using: {', '.join(apis)}")

    resolved = {}
    for label in unresolved_labels:
        candidates = _filter_candidate_terms(label, cl_names)
        print(f"    \"{label}\" ({len(candidates)} candidate terms)...", end="", flush=True)

        results = {}
        for api in apis:
            cl_id = query_llm_mapping(label, candidates, api=api)
            if cl_id:
                results[api] = cl_id

        if len(results) == 2 and results.get('claude') == results.get('openai'):
            # Both agree — auto-accept
            cl_id = results['claude']
            cl_name = cl_names.get(cl_id, '')
            resolved[label] = (cl_id, cl_name, 'llm_consensus')
            print(f" {cl_id} ({cl_name}) [consensus]")
        elif results:
            # Take whatever we got (prefer Claude if only one)
            api_used = 'claude' if 'claude' in results else 'openai'
            cl_id = results[api_used]
            cl_name = cl_names.get(cl_id, '')
            resolved[label] = (cl_id, cl_name, f'llm_{api_used}')
            print(f" {cl_id} ({cl_name}) [{api_used}]")
        else:
            print(" NONE")

    return resolved


def generate_remap(df, column, obo_path, use_llm=False):
    """Generate a remap table from a column of labels.

    Args:
        df: DataFrame with the annotation column.
        column: Column name to remap.
        obo_path: Path to OBO file.
        use_llm: If True, use LLM to resolve unmatched labels.

    Returns:
        Tuple of (remap_df, summary_report).
    """
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(obo_path)

    unique_labels = sorted(set(
        str(v).strip() for v in df[column]
        if pd.notna(v) and str(v).strip()
    ))

    rows = []
    unresolved_labels = []
    method_counts = {'exact': 0, 'synonym': 0, 'fuzzy': 0, 'fuzzy_synonym': 0,
                     'already_cl': 0, 'llm_consensus': 0, 'llm_claude': 0,
                     'llm_openai': 0, 'unresolved': 0}

    for label in unique_labels:
        if _is_cl_id(label):
            rows.append({
                'original_label': label,
                'cl_term_id': label,
                'cl_term_name': cl_names.get(label, ''),
                'match_method': 'already_cl',
            })
            method_counts['already_cl'] += 1
            continue

        cl_id, canon_name, method = resolve_name(
            label, name_to_id, synonym_to_id,
            fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)

        if cl_id is not None:
            rows.append({
                'original_label': label,
                'cl_term_id': cl_id,
                'cl_term_name': canon_name,
                'match_method': method,
            })
            method_counts[method] += 1
        else:
            unresolved_labels.append(label)

    # LLM pass for unresolved labels
    llm_resolved = {}
    if use_llm and unresolved_labels:
        print(f"\n  {len(unresolved_labels)} labels unresolved — trying LLM...")
        llm_resolved = _resolve_with_llm(unresolved_labels, cl_names)

    for label in unresolved_labels:
        if label in llm_resolved:
            cl_id, cl_name, method = llm_resolved[label]
            rows.append({
                'original_label': label,
                'cl_term_id': cl_id,
                'cl_term_name': cl_name,
                'match_method': method,
            })
            method_counts[method] += 1
        else:
            rows.append({
                'original_label': label,
                'cl_term_id': '',
                'cl_term_name': '',
                'match_method': 'unresolved',
            })
            method_counts['unresolved'] += 1

    remap_df = pd.DataFrame(rows)

    total = len(unique_labels)
    resolved = total - method_counts['unresolved']
    lines = [
        f"Remap summary for column '{column}':",
        f"  Total unique labels: {total}",
        f"  Resolved:            {resolved} ({resolved/max(total,1):.0%})",
    ]
    for m in ('exact', 'synonym', 'fuzzy', 'fuzzy_synonym', 'already_cl',
              'llm_consensus', 'llm_claude', 'llm_openai'):
        if method_counts[m]:
            lines.append(f"    {m}: {method_counts[m]}")
    if method_counts['unresolved']:
        lines.append(f"  Unresolved:          {method_counts['unresolved']}")
        still_unresolved = [r['original_label'] for r in rows if r['match_method'] == 'unresolved']
        for name in still_unresolved:
            lines.append(f"    - \"{name}\"")

    return remap_df, '\n'.join(lines)


def detect_parent_columns(df, child_col):
    """Find columns that are valid hierarchical parents of child_col.

    A column P is a parent of C if every unique value in C maps to exactly
    one value in P (functional dependency C -> P), and P has fewer unique
    values than C (coarser granularity).

    Returns:
        List of (parent_col, n_unique) sorted from coarsest to finest.
    """
    child_values = df[[child_col]].copy()
    child_values['_child'] = df[child_col].astype(str).str.strip()
    child_nunique = child_values['_child'].nunique()

    parents = []
    for col in df.columns:
        if col == child_col:
            continue
        # Skip numeric / high-cardinality / low-cardinality columns
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        col_nunique = vals.nunique()
        # Parent must be coarser (fewer unique values)
        if col_nunique >= child_nunique or col_nunique < 2:
            continue

        # Check functional dependency: each child value maps to exactly one parent
        pairs = df[[child_col, col]].dropna()
        grouped = pairs.groupby(child_col)[col].nunique()
        if (grouped == 1).all():
            parents.append((col, col_nunique))

    # Sort coarsest first
    parents.sort(key=lambda x: x[1])
    return parents


def check_hierarchy_consistency(df, child_col, remap_dict, parent_cols, obo_path):
    """Check that CL term mappings are consistent within parent groups.

    For each parent group, finds the Most Informative Common Ancestor (MICA)
    of all CL terms in that group. Reports whether the MICA is biologically
    sensible.

    Returns:
        Consistency report string.
    """
    from ontology_utils import load_ontology, precompute_ic
    import networkx as nx

    cl_names = parse_obo_names(obo_path)
    G = load_ontology(obo_path)
    ic_values = precompute_ic(G, k=0.5)

    lines = ["\n=== Hierarchy consistency check ==="]

    for parent_col, _ in parent_cols:
        lines.append(f"\nParent column: '{parent_col}'")

        # Build: parent_value -> [child_labels] -> [CL terms]
        pairs = df[[child_col, parent_col]].drop_duplicates()
        parent_groups = pairs.groupby(parent_col)[child_col].apply(list).to_dict()

        for parent_val in sorted(parent_groups.keys()):
            child_labels = sorted(set(str(v).strip() for v in parent_groups[parent_val]))
            cl_ids = []
            label_to_cl = {}
            for label in child_labels:
                cl_id = remap_dict.get(label)
                if cl_id and _is_cl_id(cl_id):
                    cl_ids.append(cl_id)
                    label_to_cl[label] = cl_id

            if len(cl_ids) < 2:
                if len(cl_ids) == 1:
                    cl_name = cl_names.get(cl_ids[0], '?')
                    lines.append(f"  {parent_val}: 1 CL term — {cl_ids[0]} ({cl_name})")
                else:
                    lines.append(f"  {parent_val}: no resolved CL terms")
                continue

            # Find common ancestors of all CL terms in this group
            unique_cl = list(set(cl_ids))
            # Start with ancestors of the first term, intersect with the rest
            try:
                common_anc = set(nx.descendants(G, unique_cl[0])) | {unique_cl[0]}
                for cl_id in unique_cl[1:]:
                    anc_i = set(nx.descendants(G, cl_id)) | {cl_id}
                    common_anc &= anc_i
            except nx.NodeNotFound:
                lines.append(f"  {parent_val}: ontology lookup failed")
                continue

            if not common_anc:
                lines.append(f"  {parent_val} ({len(unique_cl)} CL terms): "
                             f"WARNING — no common ancestor found!")
                for label, cl_id in sorted(label_to_cl.items()):
                    lines.append(f"    {label} -> {cl_id} ({cl_names.get(cl_id, '?')})")
                continue

            # Find MICA (highest IC among common ancestors)
            mica_id = max(common_anc, key=lambda a: ic_values.get(a, 0))
            mica_name = cl_names.get(mica_id, '?')
            mica_ic = ic_values.get(mica_id, 0)

            # Assess: is the MICA reasonable?
            if mica_ic > 0.1:
                status = "consistent"
            elif mica_id == 'CL:0000000':
                status = "WARNING — only common ancestor is root 'cell'"
            else:
                status = "weak — common ancestor is very generic"

            n_cl = len(unique_cl)
            lines.append(f"  {parent_val} ({n_cl} CL term{'s' if n_cl != 1 else ''}): "
                         f"MICA = {mica_id} ({mica_name}, IC={mica_ic:.3f}) — {status}")
            for label, cl_id in sorted(label_to_cl.items()):
                marker = " *" if cl_id == mica_id else ""
                lines.append(f"    {label} -> {cl_id} ({cl_names.get(cl_id, '?')}){marker}")

    return '\n'.join(lines)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load input
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, sep='\t')
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Detect or validate column
    if args.column:
        if args.column not in df.columns:
            print(f"Error: column '{args.column}' not found.")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)
        selected_col = args.column
        print(f"  Using column '{selected_col}' (user-specified)")
    else:
        print(f"\nAuto-detecting best annotation column (min resolvability: {args.min_score:.0%})...")
        selected_col, detect_report = detect_best_column(df, args.obo, args.min_score)
        print(detect_report)
        if selected_col is None:
            print("\nError: No suitable annotation column found.")
            print(f"  Available columns: {list(df.columns)}")
            print("  Use --column to specify one explicitly.")
            sys.exit(1)
        print(f"\n  Selected column: '{selected_col}'")

    # Generate remap
    print(f"\nResolving labels from '{selected_col}' to CL term IDs...")
    remap_df, summary = generate_remap(df, selected_col, args.obo, use_llm=args.use_llm)
    print(summary)

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    comment_lines = [
        f"# Generated by generate_remap.py",
        f"# Input file: {args.input}",
        f"# Source column: {selected_col}",
        f"# OBO file: {args.obo}",
    ]
    with open(args.output, 'w') as f:
        f.write('\n'.join(comment_lines) + '\n')
        remap_df.to_csv(f, sep='\t', index=False)

    print(f"\nSaved remap to {args.output}")
    print(f"  Use with: evaluate.py --remap-file {args.output}")

    # Hierarchy consistency check
    parent_cols = detect_parent_columns(df, selected_col)
    if parent_cols:
        print(f"\nDetected hierarchical parent columns: "
              f"{[col for col, _ in parent_cols]}")
        remap_dict = dict(zip(remap_df['original_label'], remap_df['cl_term_id']))
        report = check_hierarchy_consistency(
            df, selected_col, remap_dict, parent_cols, args.obo)
        print(report)
    else:
        print("\nNo hierarchical parent columns detected — skipping consistency check.")


if __name__ == "__main__":
    main()
