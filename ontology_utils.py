import logging
from typing import Dict, Optional, Tuple, List
import statistics

# Try checking for networkx and pronto, but don't fail at module level 
# to allow inspection of code even if deps are missing.
try:
    import networkx as nx
    import pronto
except ImportError:
    nx = None
    pronto = None

def load_ontology(obo_path: str) -> 'nx.DiGraph':
    """
    Load the Cell Ontology (CL) OBO file into a NetworkX DiGraph.
    
    Args:
        obo_path (str): Path to the .obo file.
        
    Returns:
        nx.DiGraph: Directed graph representing the ontology (edges point to parents/superclasses).
    """
    if pronto is None or nx is None:
        raise ImportError("pronto and networkx are required for ontology_utils.")
        
    ont = pronto.Ontology(obo_path)
    g = nx.DiGraph()
    
    for term in ont.terms():
        g.add_node(term.id, name=term.name)
        for superclass in term.superclasses(distance=1, with_self=False):
            # In OBO, 'is_a' points to parent. 
            # We want graph distance. Edge direction depends on how we calculate LCA/Distance.
            # Usually child->parent is natural for 'is_a'.
            g.add_edge(term.id, superclass.id)
            
    return g

def calculate_graph_distance(graph: 'nx.DiGraph', predicted_id: str, truth_id: str) -> int:
    """
    Calculate the distance between predicted node and ground truth node.
    Distance = shortest path in the undirected version of the graph.
    The plan says: "via their Lowest Common Ancestor (LCA)."
    Distance = dist(pred, LCA) + dist(truth, LCA).

    For ontology graphs, we use the shortest path in the undirected graph,
    which represents the semantic distance through the ontology hierarchy.

    Args:
        graph (nx.DiGraph): Ontology graph (edges: child -> parent).
        predicted_id (str): Predicted CL ID.
        truth_id (str): Ground truth CL ID.

    Returns:
        int: Distance (number of edges). Returns -1 if not reachable.
    """
    if predicted_id == truth_id:
        return 0

    if predicted_id not in graph or truth_id not in graph:
        # If term not found, return -1 to indicate error/missing.
        return -1

    # Convert to undirected graph for distance calculation
    # This allows us to traverse both up and down the hierarchy
    undirected_graph = graph.to_undirected()

    try:
        # Calculate shortest path in undirected graph
        distance = nx.shortest_path_length(undirected_graph, source=predicted_id, target=truth_id)
        return distance
    except nx.NetworkXNoPath:
        # If no path exists (disconnected components), return -1
        return -1
    except Exception:
        # Catch any other exceptions
        return -1

def score_batch(graph: 'nx.DiGraph', predictions: List[str], ground_truth: List[str]) -> Tuple[float, float]:
    """
    Report Mean and Median ontology distance for the dataset.

    Args:
        graph (nx.DiGraph): Ontology graph.
        predictions (List[str]): List of predicted IDs.
        ground_truth (List[str]): List of ground truth IDs.

    Returns:
        Tuple[float, float]: (Mean Distance, Median Distance)
    """
    distances = []

    for p, g in zip(predictions, ground_truth):
        dist = calculate_graph_distance(graph, p, g)
        if dist >= 0:
            distances.append(dist)
        # Else: ignore or penalize? Ignoring for now to avoid skewing with errors.

    if not distances:
        return 0.0, 0.0

    # Ensure we return floats
    return float(statistics.mean(distances)), float(statistics.median(distances))
