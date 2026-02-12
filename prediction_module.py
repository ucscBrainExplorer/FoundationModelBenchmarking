import numpy as np
import pandas as pd
import faiss
from collections import Counter
from typing import List, Tuple, Union

def execute_query(index: faiss.Index, query_embeddings: np.ndarray, k: int = 30, metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Query the FAISS index for the top k nearest neighbors.

    Args:
        index (faiss.Index): Loaded FAISS index.
        query_embeddings (np.ndarray): Query vectors (n_queries, dim).
        k (int): Number of neighbors to retrieve.
        metric (str): Distance metric. Only 'euclidean' is supported for L2-built indices.

    Returns:
        dists (np.ndarray): Distances to neighbors.
        indices (np.ndarray): Indices of neighbors.

    Raises:
        ValueError: If metric='cosine' is used with an L2-built FAISS index.
    """
    # Ensure query_embeddings is float32 as Faiss expects
    queries = query_embeddings.astype(np.float32)

    if metric.lower() == 'cosine':
        # Normalizing only query vectors does NOT produce cosine similarity
        # when the FAISS index was built with L2 metric. It computes:
        #   ||q_normalized - v_unnormalized||²
        # which is neither euclidean nor cosine — it's a meaningless hybrid.
        # True cosine requires METRIC_INNER_PRODUCT index with all vectors normalized.
        raise ValueError(
            "Cosine metric is not supported with an L2-built FAISS index. "
            "Normalizing only query vectors produces neither cosine similarity "
            "nor euclidean distance. Use 'euclidean' metric instead. "
            "If embeddings are already normalized, euclidean and cosine rankings "
            "are mathematically identical: ||q-v||² = 2(1 - q·v)."
        )

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
