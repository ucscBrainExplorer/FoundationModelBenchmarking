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
        metric (str): Distance metric to use ('euclidean' or 'cosine').
                      If 'cosine', query vectors will be normalized.
                      
    Returns:
        dists (np.ndarray): Distances to neighbors.
        indices (np.ndarray): Indices of neighbors.
    """
    # Ensure query_embeddings is float32 as Faiss expects
    queries = query_embeddings.astype(np.float32)
    
    if metric.lower() == 'cosine':
        # Normalize vectors for cosine similarity (assuming index is compatible or IP)
        faiss.normalize_L2(queries)
        
    dists, indices = index.search(queries, k)
    return dists, indices

def vote_neighbors(neighbor_indices: np.ndarray, reference_annotations: pd.DataFrame) -> List[str]:
    """
    Map neighbor indices to cell types and perform majority voting.
    
    Args:
        neighbor_indices (np.ndarray): Indices of neighbors (n_queries, k).
        reference_annotations (pd.DataFrame): DataFrame with 'cell_type_ontology_term_id'.
                                              Assumes index of DataFrame aligns with FAISS index IDs.
                                              
    Returns:
        List[str]: Predicted cell_type_terminology_id for each query.
    """
    predictions = []
    
    # We assume the dataframe index corresponds to the faiss ids (0 to N-1)
    # Check if max index is within bounds
    max_idx = neighbor_indices.max()
    if max_idx >= len(reference_annotations):
        raise IndexError(f"Neighbor index {max_idx} out of bounds for reference annotations with size {len(reference_annotations)}")
    
    # Extract the relevant column once to speed up lookups
    # Assuming the column name is 'cell_type_ontology_term_id' based on plan
    term_ids = reference_annotations['cell_type_ontology_term_id'].values
    
    for row_indices in neighbor_indices:
        # Retrieve the terms for the neighbors
        neighbor_terms = term_ids[row_indices]
        
        # Majority vote
        # In case of ties, Counter.most_common returns the first one encountered which is arbitrary but standard
        vote_counts = Counter(neighbor_terms)
        most_common = vote_counts.most_common(1)[0][0]
        predictions.append(most_common)
        
    return predictions
