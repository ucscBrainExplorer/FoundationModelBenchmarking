#!/usr/bin/env python3
"""
Normalize cell type labels using the Cell Ontology (OBO) file.

Reads a prediction_obs TSV with columns:
  - cell_type_ontology_term_id  (CL:XXXXXXX)
  - cell_type                   (free-text label from source dataset)

Produces a new TSV with columns:
  - cell_type_ontology_term_id
  - raw_cell_type               (original cell_type, renamed)
  - normalized_cell_type        (official name from the OBO file)

Usage:
  python3 normalize_cell_types.py --obo cl.obo --input prediction_obs.tsv --output prediction_obs_normalized.tsv
"""

import argparse
import sys
import os
import pandas as pd

# Allow importing obo_parser from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obo_parser import parse_obo_names


def normalize(obo_path: str, input_path: str, output_path: str):
    """Read input TSV, add normalized_cell_type column, write output TSV."""
    # Parse ontology
    cl_map = parse_obo_names(obo_path)
    print(f"Parsed {len(cl_map)} terms from {obo_path}")

    # Read input
    df = pd.read_csv(input_path, sep='\t')
    print(f"Read {len(df)} rows from {input_path}")

    # Rename and add column
    df.rename(columns={'cell_type': 'raw_cell_type'}, inplace=True)
    df['normalized_cell_type'] = df['cell_type_ontology_term_id'].map(cl_map).fillna('')

    # Reorder
    df = df[['cell_type_ontology_term_id', 'raw_cell_type', 'normalized_cell_type']]

    # Save
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    # Summary
    unique_ids = df['cell_type_ontology_term_id'].nunique()
    mapped = df[df['normalized_cell_type'] != '']['cell_type_ontology_term_id'].nunique()
    unmapped = df[df['normalized_cell_type'] == '']['cell_type_ontology_term_id'].unique()
    print(f"\nUnique CL IDs: {unique_ids}")
    print(f"Mapped to OBO name: {mapped}")
    print(f"Not found in OBO: {len(unmapped)}")
    for uid in sorted(unmapped):
        count = (df['cell_type_ontology_term_id'] == uid).sum()
        print(f"  {uid} ({count} cells)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize cell type labels using a Cell Ontology OBO file")
    parser.add_argument('--obo', type=str, required=True,
                        help="Path to OBO ontology file (e.g. cl.obo)")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to input prediction_obs.tsv")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output normalized TSV")
    args = parser.parse_args()

    normalize(args.obo, args.input, args.output)
