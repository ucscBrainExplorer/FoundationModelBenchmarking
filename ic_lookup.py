#!/usr/bin/env python3
"""
Look up IC similarity between two cell types in the Cell Ontology.

Usage:
  python3 ic_lookup.py "neural stem cell" "forebrain radial glial cell"
  python3 ic_lookup.py "neural stem cell" "forebrain radial glial cell" --obo /path/to/cl.obo
  python3 ic_lookup.py CL:0000047 CL:0013000
"""

import argparse
import sys

from obo_parser import parse_obo_names
from ontology_utils import (
    load_ontology, precompute_ic, calculate_lin_similarity,
    calculate_graph_distance, _get_all_ancestors,
)

# Cache so repeated lookups in an interactive session are fast
_CACHE = {}


def _load(obo_path):
    if obo_path in _CACHE:
        return _CACHE[obo_path]

    names = parse_obo_names(obo_path)
    name_to_id = {v.lower(): k for k, v in names.items()}
    graph = load_ontology(obo_path, include_relationships=True)
    ic_values = precompute_ic(graph, k=0.5)

    _CACHE[obo_path] = (names, name_to_id, graph, ic_values)
    return names, name_to_id, graph, ic_values


def resolve_term(term, names, name_to_id):
    """Resolve a term string to (cl_id, name). Accepts CL IDs or readable names."""
    if term.startswith('CL:'):
        if term in names:
            return term, names[term]
        return None, None
    cl_id = name_to_id.get(term.lower())
    if cl_id:
        return cl_id, names[cl_id]
    return None, None


def ic_lookup(term_a, term_b, obo_path):
    names, name_to_id, graph, ic_values = _load(obo_path)

    id_a, name_a = resolve_term(term_a, names, name_to_id)
    id_b, name_b = resolve_term(term_b, names, name_to_id)

    if not id_a:
        print(f"Error: '{term_a}' not found in ontology")
        return
    if not id_b:
        print(f"Error: '{term_b}' not found in ontology")
        return

    ic_a = ic_values.get(id_a, 0.0)
    ic_b = ic_values.get(id_b, 0.0)
    sim = calculate_lin_similarity(graph, id_a, id_b, ic_values)
    path_dist = calculate_graph_distance(graph, id_a, id_b)

    # Find MICA and common ancestors
    ancestors_a = _get_all_ancestors(id_a, graph)
    ancestors_b = _get_all_ancestors(id_b, graph)
    common = ancestors_a & ancestors_b

    print(f"\n{'=' * 64}")
    print(f"  Term A:  {name_a} ({id_a})")
    print(f"  Term B:  {name_b} ({id_b})")
    print(f"{'=' * 64}")
    print(f"  IC(A)              = {ic_a:.6f}")
    print(f"  IC(B)              = {ic_b:.6f}")

    if common:
        mica_id = max(common, key=lambda a: ic_values.get(a, 0.0))
        mica_ic = ic_values.get(mica_id, 0.0)
        mica_name = names.get(mica_id, mica_id)
        print(f"  IC(MICA)           = {mica_ic:.6f}  [{mica_name} ({mica_id})]")
    else:
        print(f"  MICA               = (no common ancestor)")

    print(f"  Lin IC similarity  = {sim:.6f}")
    print(f"  Shortest path dist = {path_dist}")
    print(f"  Exact match        = {'Yes' if id_a == id_b else 'No'}")

    if common:
        print(f"\n  Common ancestors ({len(common)}):")
        ranked = sorted(common, key=lambda a: ic_values.get(a, 0.0), reverse=True)
        for a in ranked[:10]:
            tag = " <-- MICA" if a == mica_id else ""
            print(f"    {ic_values.get(a, 0.0):.6f}  {names.get(a, a)} ({a}){tag}")
        if len(common) > 10:
            print(f"    ... and {len(common) - 10} more")

    # Show immediate parents for context
    parents_a = [p for p in graph.successors(id_a)]
    parents_b = [p for p in graph.successors(id_b)]
    print(f"\n  Parents of A: {', '.join(names.get(p, p) + ' (' + p + ')' for p in parents_a)}")
    print(f"  Parents of B: {', '.join(names.get(p, p) + ' (' + p + ')' for p in parents_b)}")

    print(f"\n  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
          f" (is_a + relationships)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Look up IC similarity between two cell types"
    )
    parser.add_argument("term_a", help="Cell type name or CL term ID")
    parser.add_argument("term_b", help="Cell type name or CL term ID")
    parser.add_argument("--obo", required=True,
                        help="Path to Cell Ontology OBO file (e.g. cl.obo)")
    args = parser.parse_args()

    print("Loading ontology and computing IC...", file=sys.stderr)
    ic_lookup(args.term_a, args.term_b, args.obo)


if __name__ == "__main__":
    main()
