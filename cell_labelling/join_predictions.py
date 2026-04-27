#!/usr/bin/env python3
"""
Join predicted_labels.tsv with the query h5ad obs table.

Usage:
  python3 join_predictions.py \
    --labels  predicted_labels.tsv \
    --adata   query_uce_adata.h5ad \
    --output  predictions_with_obs.tsv
"""

import argparse
import anndata
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True, metavar='FILE', help='predicted_labels.tsv from predict.py')
    parser.add_argument('--adata',  required=True, metavar='FILE', help='Query h5ad file')
    parser.add_argument('--output', default='predictions_with_obs.tsv', metavar='FILE', help='Output TSV (default: predictions_with_obs.tsv)')
    return parser


def main():
    args = build_parser().parse_args()

    print(f"Reading predictions from {args.labels}...")
    labels = pd.read_csv(args.labels, sep='\t', comment='#')
    print(f"  {len(labels)} cells, {len(labels.columns)} columns")

    print(f"Reading obs from {args.adata}...")
    adata = anndata.read_h5ad(args.adata, backed='r')
    obs = adata.obs.copy()
    obs.index.name = 'cell_id'
    obs = obs.reset_index()
    print(f"  {len(obs)} cells, {len(obs.columns)} columns")

    result = obs.merge(labels, on='cell_id', how='left')
    print(f"  Merged: {len(result)} cells, {len(result.columns)} columns")

    result.to_csv(args.output, sep='\t', index=False)
    print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
