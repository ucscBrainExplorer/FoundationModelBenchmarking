from typing import List, Dict, Optional, Union
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def calculate_accuracy(
    predictions: List[str], 
    ground_truth: List[str], 
    neighbor_labels: Optional[List[List[str]]] = None
) -> Dict[str, float]:
    """
    Compute Overall Accuracy, Top-k Accuracy (if neighbor_labels provided), and F1 Score.
    
    Args:
        predictions (List[str]): Final predicted labels (voting result).
        ground_truth (List[str]): Ground truth labels.
        neighbor_labels (List[List[str]], optional): List of labels for the k neighbors for each query.
                                                     Used for Top-k accuracy (is truth in neighbors?).
                                                     
    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    metrics = {}
    
    # 1. Overall Accuracy
    acc = accuracy_score(ground_truth, predictions)
    metrics['accuracy'] = acc
    
    # 2. F1 Score (Macro and Weighted)
    # Use zero_division=0 to handle cases where classes are missing
    metrics['f1_macro'] = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    
    # 3. Top-k Accuracy
    # Definition: Is the ground truth label present in the retrieved neighbors?
    if neighbor_labels:
        hits = 0
        for truth, neighbors in zip(ground_truth, neighbor_labels):
            if truth in neighbors:
                hits += 1
        metrics['top_k_accuracy'] = hits / len(ground_truth) if ground_truth else 0.0
        
    return metrics
