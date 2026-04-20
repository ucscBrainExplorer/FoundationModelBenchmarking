#!/usr/bin/env python3
"""
Summarize the reference neighborhood for each query cell.

For each query cell, finds k nearest neighbors in the reference and reports
a Gaussian-weighted distribution over multiple metadata fields (developmental
stage, tissue, disease, etc.) plus cell type with OBO-resolved names.

Each metadata field reports the top-N labels and their scores (normalized
weights summing to 1.0).  Cell type is handled specially: votes are cast on
the CL ontology term ID (resolved via OBO), with the canonical name reported
alongside.

Usage:
  python3 neighbor_profile.py \
    --index     path/to/index.faiss \
    --embeddings path/to/embeddings.npy \
    --ref_annot path/to/reference_metadata.tsv \
    --obo       path/to/cl.obo \
    --k 30 \
    --top_n 3 \
    --output    neighbor_profile.tsv

Embeddings can be a .npy file or a .h5ad file (extracts obsm['X_uce']).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import faiss


# ---------------------------------------------------------------------------
# Configuration — edit these to add, remove, or reorder fields
# ---------------------------------------------------------------------------

# Metadata fields to profile. Each entry is the column name in the reference.
# These are reported as top-N distributions (label + score).
DEFAULT_PROFILE_FIELDS = [
    'development_stage',
    'tissue',
    'tissue_type',
    'disease',
    'assay',
    'suspension_type',
    'sex',
    'collection_doi_label',
]

# Cell type column pairs in priority order.
# The first pair found in the reference will be used for voting.
CELL_TYPE_COLUMN_PAIRS = [
    ('harmonized_cell_type_ontology_term_id', 'harmonized_cell_type'),
    ('mapped_cell_label_ontology_term_id',    'mapped_cell_label'),
    ('cell_label_ontology_term_id',           'cell_label'),
    ('cell_type_ontology_term_id',            'cell_type'),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_index(path: str) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(path)
    ivf = faiss.extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = 20
    return index


def load_embeddings(path: str):
    """Load embeddings from .npy or .h5ad. Returns (embeddings, cell_ids)."""
    if path.endswith('.h5ad'):
        import anndata
        adata = anndata.read_h5ad(path, backed='r')
        if 'X_uce' not in adata.obsm:
            raise ValueError("h5ad file does not contain adata.obsm['X_uce']")
        embeddings = np.array(adata.obsm['X_uce']).astype(np.float32)
        cell_ids = list(adata.obs_names)
    else:
        embeddings = np.load(path).astype(np.float32)
        cell_ids = [f"row_{i}" for i in range(len(embeddings))]
    return embeddings, cell_ids


def resolve_cell_type_columns(ref_df: pd.DataFrame):
    """Return (id_col, lbl_col) using the first available pair in CELL_TYPE_COLUMN_PAIRS."""
    for id_col, lbl_col in CELL_TYPE_COLUMN_PAIRS:
        if id_col in ref_df.columns and lbl_col in ref_df.columns:
            return id_col, lbl_col
    raise ValueError(
        "No recognized cell type column pair found in reference annotations.\n"
        "Expected one of: " + ", ".join(f"({a}+{b})" for a, b in CELL_TYPE_COLUMN_PAIRS)
    )


def parse_obo_names(obo_path: str) -> dict:
    """Parse an OBO file and return {term_id: canonical_name}."""
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"OBO file not found: {obo_path}")
    cl_map = {}
    current_id = None
    in_term = False
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                current_id = None
            elif line.startswith('[') and line.endswith(']'):
                in_term = False
                current_id = None
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('name: ') and current_id:
                    cl_map[current_id] = line[6:]
    return cl_map


def parse_obo_replacements(obo_path: str) -> dict:
    """Parse an OBO file and return {obsolete_id: replacement_id}."""
    replacements = {}
    current_id = None
    in_term = False
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                current_id = None
            elif line.startswith('[') and line.endswith(']'):
                in_term = False
                current_id = None
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('replaced_by: ') and current_id:
                    replacements[current_id] = line[13:]
    return replacements


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def gaussian_kernel_weights(dists: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-row Gaussian kernel weights.  sigma = median distance of that row's neighbors.
    w = exp(-d^2 / (2 * sigma^2))
    """
    sigma = np.median(dists, axis=1, keepdims=True)
    sigma = np.clip(sigma, eps, None)
    return np.exp(-dists ** 2 / (2 * sigma ** 2))


def top_n_weighted(weights: np.ndarray, neighbor_labels: np.ndarray, n: int = 3):
    """
    Gaussian-weighted distribution over labels, return top-N per cell.

    Parameters:
        weights         — (N_cells, k) weight array
        neighbor_labels — (N_cells, k) string label array
        n               — number of top labels to return

    Returns:
        top_labels — list of n lists, each length N_cells
        top_scores — list of n lists, each length N_cells (normalized, sum to 1)
    """
    top_labels = [[] for _ in range(n)]
    top_scores = [[] for _ in range(n)]

    for row_weights, row_labels in zip(weights, neighbor_labels):
        valid = row_labels != ''
        if not valid.any():
            for i in range(n):
                top_labels[i].append('')
                top_scores[i].append(float('nan'))
            continue

        vw = row_weights[valid]
        vl = row_labels[valid]
        total = vw.sum()

        uniq, inv = np.unique(vl, return_inverse=True)
        scores = np.bincount(inv, weights=vw) / total
        order = np.argsort(scores)[::-1]

        for i in range(n):
            if i < len(order):
                top_labels[i].append(uniq[order[i]])
                top_scores[i].append(float(scores[order[i]]))
            else:
                top_labels[i].append('')
                top_scores[i].append(float('nan'))

    return top_labels, top_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize reference neighborhood for each query cell across multiple metadata fields"
    )
    parser.add_argument('--index',      required=True,
                        help='FAISS index file (.faiss)')
    parser.add_argument('--embeddings', required=True,
                        help='Query embeddings (.npy or .h5ad with obsm["X_uce"])')
    parser.add_argument('--ref_annot',  required=True,
                        help='Reference metadata TSV')
    parser.add_argument('--obo',        required=True,
                        help='Cell Ontology OBO file for CL ID -> canonical name resolution')
    parser.add_argument('--k',          type=int, default=30,
                        help='Number of nearest neighbors (default: 30)')
    parser.add_argument('--top_n',      type=int, default=3,
                        help='Number of top labels to report per field (default: 3)')
    parser.add_argument('--fields',     nargs='+', default=DEFAULT_PROFILE_FIELDS,
                        help='Metadata fields to profile (default: see DEFAULT_PROFILE_FIELDS)')
    parser.add_argument('--output',     default='neighbor_profile.tsv',
                        help='Output TSV (default: neighbor_profile.tsv)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # -- Reference annotations -----------------------------------------------
    print(f"Loading reference annotations from {args.ref_annot}...")
    # Identify cell type columns before loading so we can read only needed cols
    probe_df = pd.read_csv(args.ref_annot, sep='\t', nrows=0)
    ct_id_col, ct_lbl_col = resolve_cell_type_columns(probe_df)
    print(f"  Cell type columns: '{ct_id_col}' + '{ct_lbl_col}'")

    needed_cols = set(args.fields) | {ct_id_col, ct_lbl_col}
    available_cols = [c for c in needed_cols if c in probe_df.columns]
    missing_fields = [f for f in args.fields if f not in probe_df.columns]
    if missing_fields:
        print(f"  Warning: fields not found in reference and will be skipped: {missing_fields}")
    profile_fields = [f for f in args.fields if f in probe_df.columns]

    ref_df = pd.read_csv(args.ref_annot, sep='\t', usecols=available_cols, low_memory=False)
    n_ref = len(ref_df)
    print(f"  {n_ref} reference cells")

    # -- OBO -----------------------------------------------------------------
    print(f"Parsing OBO file {args.obo}...")
    cl_names = parse_obo_names(args.obo)
    cl_replacements = parse_obo_replacements(args.obo)
    print(f"  {len(cl_names)} terms, {len(cl_replacements)} obsolete replacements")

    # Resolve obsolete CL terms in reference
    ref_ct_ids = ref_df[ct_id_col].fillna('').astype(str)
    n_replaced = int(ref_ct_ids.isin(cl_replacements).sum())
    if n_replaced > 0:
        ref_df[ct_id_col] = ref_ct_ids.replace(cl_replacements)
        print(f"  Resolved {n_replaced} obsolete terms in reference")

    # Build cell type label array: prefer ID, fall back to label
    ct_id_arr  = ref_df[ct_id_col].fillna('').astype(str).str.strip().values
    ct_lbl_arr = ref_df[ct_lbl_col].fillna('').astype(str).str.strip().values
    ct_arr = np.where(ct_id_arr != '', ct_id_arr, ct_lbl_arr)
    ct_arr = np.where(ct_arr != 'nan', ct_arr, '')

    # Build label arrays for each profile field
    field_arrays = {}
    for field in profile_fields:
        field_arrays[field] = ref_df[field].fillna('').astype(str).values

    # -- Embeddings ----------------------------------------------------------
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings, cell_ids = load_embeddings(args.embeddings)
    n_cells, emb_dim = embeddings.shape
    print(f"  {n_cells} cells, {emb_dim}d")

    # -- FAISS index ---------------------------------------------------------
    print(f"Loading FAISS index from {args.index}...")
    index = load_index(args.index)
    print(f"  {index.ntotal} vectors, {index.d}d")

    if index.d != emb_dim:
        print(f"Error: embedding dim ({emb_dim}) != index dim ({index.d})")
        sys.exit(1)
    if index.ntotal != n_ref:
        print(f"Error: index size ({index.ntotal}) != reference annotations ({n_ref})")
        sys.exit(1)

    # -- KNN search ----------------------------------------------------------
    print(f"Searching {args.k} nearest neighbors...")
    squared_dists, indices = index.search(embeddings, args.k)
    dists = np.sqrt(np.maximum(squared_dists, 0))

    valid_mask = indices >= 0
    mean_distances = np.where(valid_mask, dists, np.nan).mean(axis=1).tolist()
    n_valid = valid_mask.sum(axis=1).tolist()

    # Gaussian weights (shared across all fields — one KNN pass)
    weights = gaussian_kernel_weights(np.where(valid_mask, dists, 0.0))

    # -- Cell type voting ----------------------------------------------------
    print("Computing cell type neighborhood distribution...")
    neighbor_ct = np.where(
        valid_mask,
        np.vectorize(lambda i: ct_arr[i] if i >= 0 else '')(indices),
        ''
    )
    ct_top_labels, ct_top_scores = top_n_weighted(weights, neighbor_ct, n=args.top_n)

    # -- Profile fields voting -----------------------------------------------
    field_results = {}
    for field in profile_fields:
        print(f"Computing {field} neighborhood distribution...")
        arr = field_arrays[field]
        neighbor_labels = np.where(
            valid_mask,
            np.vectorize(lambda i: arr[i] if i >= 0 else '')(indices),
            ''
        )
        top_labels, top_scores = top_n_weighted(weights, neighbor_labels, n=args.top_n)
        field_results[field] = (top_labels, top_scores)

    # -- Assemble output -----------------------------------------------------
    cols = {'cell_id': cell_ids}

    # Cell type columns: ID + resolved name side by side
    for i in range(args.top_n):
        id_vals = ct_top_labels[i]
        cols[f'cell_type_ontology_term_id_{i+1}'] = id_vals
        cols[f'cell_type_{i+1}'] = [cl_names.get(v, v) if v else '' for v in id_vals]
        cols[f'cell_type_score_{i+1}'] = ct_top_scores[i]

    # Profile field columns
    for field in profile_fields:
        top_labels, top_scores = field_results[field]
        for i in range(args.top_n):
            cols[f'{field}_{i+1}']       = top_labels[i]
            cols[f'{field}_score_{i+1}'] = top_scores[i]

    # Quality indicators
    cols['mean_euclidean_distance'] = mean_distances
    cols['n_neighbors']             = n_valid

    output_df = pd.DataFrame(cols)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        f.write(f"# index:      {args.index} ({index.ntotal} vectors, {index.d}d)\n")
        f.write(f"# embeddings: {args.embeddings} ({n_cells} cells, {emb_dim}d)\n")
        f.write(f"# ref_annot:  {args.ref_annot} ({n_ref} cells)\n")
        f.write(f"# obo:        {args.obo} ({len(cl_names)} terms)\n")
        f.write(f"# k:          {args.k}\n")
        f.write(f"# top_n:      {args.top_n}\n")
        f.write(f"# cell_type_columns: {ct_id_col} + {ct_lbl_col}\n")
        f.write(f"# profile_fields: {', '.join(profile_fields)}\n")
        output_df.to_csv(f, sep='\t', index=False)

    print(f"\nSaved neighborhood profile for {n_cells} cells to {args.output}")
    print(f"  Columns: cell_type (top-{args.top_n}) + "
          f"{len(profile_fields)} profile fields (top-{args.top_n} each) + quality indicators")


if __name__ == '__main__':
    main()
