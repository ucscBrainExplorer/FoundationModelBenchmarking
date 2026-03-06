import numpy as np
import pandas as pd
import faiss
from collections import Counter
from typing import List, Tuple, Union


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

    term_ids = reference_annotations['cell_type_ontology_term_id'].values

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

        # Filter out empty/NaN labels
        valid_terms = []
        filtered_w = []
        for t, w in zip(term_ids[valid_idx], valid_w):
            if pd.notna(t) and str(t).strip() not in ('', 'nan'):
                valid_terms.append(str(t).strip())
                filtered_w.append(w)

        if not valid_terms:
            predictions.append('')
            vote_percentages.append(float('nan'))
            continue

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
            - Predicted cell_type_terminology_id for each query (empty string if all neighbors missing)
            - Vote percentage (0.0-1.0) for the winning prediction, or NaN if no valid votes

    Note:
        Invalid labels (NaN, empty strings, None) are filtered out before voting.
        Only neighbors with valid cell_type_ontology_term_id values participate in the vote.
        If all neighbors have missing labels, an empty string is returned with NaN percentage.
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
    
    # Extract the relevant column once to speed up lookups
    # Assuming the column name is 'cell_type_ontology_term_id' based on plan
    term_ids = reference_annotations['cell_type_ontology_term_id'].values
    
    for row_indices in neighbor_indices:
        # Filter out -1 indices (FAISS returns -1 for unfound neighbors)
        valid_mask = row_indices >= 0
        valid_indices = row_indices[valid_mask]
        if len(valid_indices) == 0:
            predictions.append('')
            vote_percentages.append(float('nan'))
            continue
        # Retrieve the terms for the valid neighbors
        neighbor_terms = term_ids[valid_indices]
        
        # Filter out invalid labels (NaN, empty strings, None)
        # Convert to list to handle numpy array indexing properly
        valid_terms = []
        for term in neighbor_terms:
            # Check if term is valid: not NaN, not None, and not empty string
            if pd.notna(term) and term is not None:
                term_str = str(term).strip()
                if term_str != '' and term_str.lower() != 'nan':
                    valid_terms.append(term_str)
        
        # If no valid labels found, return empty string with NaN percentage
        if len(valid_terms) == 0:
            predictions.append('')
            vote_percentages.append(float('nan'))
            continue
        
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
