import numpy as np
import pandas as pd
import faiss
from collections import Counter
from typing import List, Tuple, Union


def validate_ref_columns(ref_df: pd.DataFrame) -> None:
    """
    Validate that ref_df contains at least one complete label pair:
      - ('cell_label_ontology_term_id' + 'cell_label'), or
      - ('cell_type_ontology_term_id'  + 'cell_type')

    Raises ValueError if a pair is only partially present or if neither pair exists.
    """
    has_cl_id  = 'cell_label_ontology_term_id' in ref_df.columns
    has_cl_lbl = 'cell_label' in ref_df.columns
    has_ct_id  = 'cell_type_ontology_term_id' in ref_df.columns
    has_ct_lbl = 'cell_type' in ref_df.columns

    if has_cl_id != has_cl_lbl:
        missing = 'cell_label' if has_cl_id else 'cell_label_ontology_term_id'
        raise ValueError(f"Incomplete column pair: '{missing}' is missing")
    if has_ct_id != has_ct_lbl:
        missing = 'cell_type' if has_ct_id else 'cell_type_ontology_term_id'
        raise ValueError(f"Incomplete column pair: '{missing}' is missing")
    if not (has_cl_id and has_cl_lbl) and not (has_ct_id and has_ct_lbl):
        raise ValueError(
            "Reference annotations must contain at least one complete column pair: "
            "('cell_label_ontology_term_id' + 'cell_label') or "
            "('cell_type_ontology_term_id' + 'cell_type')"
        )


def resolve_labels(ref_df: pd.DataFrame) -> np.ndarray:
    """
    Resolve the effective voting label for each reference cell.

    Source detection (cell_label pair takes priority over cell_type pair):
      - If 'cell_label_ontology_term_id' is present: use it, fall back to 'cell_label'
        for rows where it is missing/invalid.
      - Otherwise: use 'cell_type_ontology_term_id', fall back to 'cell_type'.

    Assumes ref_df has been validated by validate_ref_columns().
    Returns a string ndarray of length len(ref_df).
    """
    if 'cell_label_ontology_term_id' in ref_df.columns:
        id_col, lbl_col = ref_df['cell_label_ontology_term_id'], ref_df['cell_label']
    else:
        id_col, lbl_col = ref_df['cell_type_ontology_term_id'], ref_df['cell_type']

    def _clean(col):
        s = col.astype(str).str.strip()
        s = s.where(col.notna(), '')
        return s.where(s != 'nan', '')

    id_s  = _clean(id_col)
    lbl_s = _clean(lbl_col)
    result = id_s.where(id_s != '', lbl_s)
    return result.where(result != '', 'missing_label').values


def gaussian_kernel_weights(dists: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert distances to Gaussian kernel weights.
    sigma is set per-row to the median distance of that row's neighbors.

    w = exp(-d² / (2 * sigma²))
    """
    sigma = np.median(dists, axis=1, keepdims=True)
    sigma = np.clip(sigma, eps, None)
    return np.exp(-dists ** 2 / (2 * sigma ** 2))


def distance_weighted_knn_vote(neighbor_indices: np.ndarray, neighbor_dists: np.ndarray,
                               reference_annotations: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Distance-weighted KNN voting using Gaussian kernel weights.
    Closer neighbors contribute more weight to their cell type label.

    Args:
        neighbor_indices: (N, k) FAISS indices
        neighbor_dists:   (N, k) Euclidean distances (already converted from squared)
        reference_annotations: DataFrame with 'cell_type_ontology_term_id'

    Returns:
        Tuple of (predictions, vote_percentages) — top-1 CL term IDs and normalized
        weight fractions (0.0–1.0), matching the signature of vote_neighbors.
    """
    if neighbor_indices.size == 0:
        return [], []

    term_ids = resolve_labels(reference_annotations)

    # Zero out distances for invalid FAISS sentinels before computing weights
    valid_mask = neighbor_indices >= 0
    safe_dists = np.where(valid_mask, neighbor_dists, 0.0)
    weights = gaussian_kernel_weights(safe_dists.astype(np.float32))

    predictions = []
    vote_percentages = []

    for row_indices, row_weights, row_valid in zip(neighbor_indices, weights, valid_mask):
        valid_idx = row_indices[row_valid]
        valid_w = row_weights[row_valid]

        if len(valid_idx) == 0:
            predictions.append('')
            vote_percentages.append(float('nan'))
            continue

        valid_terms = [str(t) for t in term_ids[valid_idx]]
        filtered_w = list(valid_w)

        total = sum(filtered_w)
        label_weights: dict = {}
        for term, w in zip(valid_terms, filtered_w):
            label_weights[term] = label_weights.get(term, 0.0) + w

        winner = max(label_weights, key=label_weights.get)
        predictions.append(winner)
        vote_percentages.append(label_weights[winner] / total)

    return predictions, vote_percentages

def execute_query(index: faiss.Index, query_embeddings: np.ndarray, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query the FAISS index for the top k nearest neighbors.

    Args:
        index (faiss.Index): Loaded FAISS index.
        query_embeddings (np.ndarray): Query vectors (n_queries, dim).
        k (int): Number of neighbors to retrieve.

    Returns:
        dists (np.ndarray): Distances to neighbors.
        indices (np.ndarray): Indices of neighbors.
    """
    queries = query_embeddings.astype(np.float32)
    dists, indices = index.search(queries, k)
    return dists, indices

def vote_neighbors(neighbor_indices: np.ndarray, reference_annotations: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Map neighbor indices to cell types and perform majority voting.

    Args:
        neighbor_indices (np.ndarray): Indices of neighbors (n_queries, k).
        reference_annotations (pd.DataFrame): DataFrame with 'cell_type_ontology_term_id'.
                                              Assumes index of DataFrame aligns with FAISS index IDs.

    Returns:
        Tuple[List[str], List[float]]:
            - Predicted label for each query (empty string only if all FAISS neighbors are -1)
            - Vote percentage (0.0-1.0) for the winning label, or NaN if no valid FAISS neighbors

    Note:
        All valid FAISS neighbors vote, including those with missing source labels (which
        vote as 'missing_label'). Only FAISS sentinel (-1) neighbors are excluded.
    """
    predictions = []
    vote_percentages = []

    # Handle empty array
    if neighbor_indices.size == 0:
        return predictions, vote_percentages

    # We assume the dataframe index corresponds to the faiss ids (0 to N-1)
    # Check if max index is within bounds
    max_idx = neighbor_indices.max()
    if max_idx >= len(reference_annotations):
        raise IndexError(f"Neighbor index {max_idx} out of bounds for reference annotations with size {len(reference_annotations)}")
    
    term_ids = resolve_labels(reference_annotations)
    
    for row_indices in neighbor_indices:
        # Filter out -1 indices (FAISS returns -1 for unfound neighbors)
        valid_mask = row_indices >= 0
        valid_indices = row_indices[valid_mask]
        if len(valid_indices) == 0:
            predictions.append('')
            vote_percentages.append(float('nan'))
            continue
        valid_terms = [str(t) for t in term_ids[valid_indices]]
        
        # Majority vote among valid labels only
        # In case of ties, Counter.most_common returns the first one encountered which is arbitrary but standard
        vote_counts = Counter(valid_terms)
        most_common_pair = vote_counts.most_common(1)[0]
        most_common = most_common_pair[0]
        vote_count = most_common_pair[1]
        
        # Calculate percentage: votes for winner / total valid votes
        vote_percentage = vote_count / len(valid_terms)
        
        predictions.append(most_common)
        vote_percentages.append(vote_percentage)
        
    return predictions, vote_percentages
