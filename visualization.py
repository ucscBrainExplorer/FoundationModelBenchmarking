"""
Visualization module for UCE Benchmarking.
Generates UMAP plots, confusion matrices, and summary tables.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP plots will use t-SNE instead.")
    print("Install with: pip install umap-learn")

def generate_umap_embedding(embeddings: np.ndarray, n_components: int = 2, random_state: int = 42, max_samples: int = 50000) -> np.ndarray:
    """
    Generate UMAP (or t-SNE fallback) embedding for visualization.
    
    Args:
        embeddings: High-dimensional embeddings (n_samples, n_features)
        n_components: Number of dimensions for embedding (2 for visualization)
        random_state: Random seed for reproducibility
        max_samples: Maximum number of samples to use for UMAP (for large datasets)
        
    Returns:
        2D embedding array (n_samples, 2)
    """
    n_samples = embeddings.shape[0]
    
    # For very large datasets, sample for faster computation
    if n_samples > max_samples:
        print(f"  Sampling {max_samples} from {n_samples} samples for faster UMAP computation...")
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, max_samples, replace=False)
        sample_embeddings = embeddings[indices]
        use_indices = True
    else:
        sample_embeddings = embeddings
        indices = None
        use_indices = False
    
    if UMAP_AVAILABLE:
        # Optimize UMAP parameters for large datasets
        n_neighbors = min(15, sample_embeddings.shape[0] // 10)  # Adaptive neighbors
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='euclidean',
            random_state=random_state,
            verbose=True,  # Enable progress output
            n_jobs=1  # Use single thread to avoid memory issues
        )
        print(f"  Computing UMAP on {sample_embeddings.shape[0]} samples...")
        embedding_2d = reducer.fit_transform(sample_embeddings)
        
        # If we sampled, interpolate to full dataset
        if use_indices:
            print(f"  Interpolating UMAP to full dataset ({n_samples} samples)...")
            reducer_full = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='euclidean',
                random_state=random_state,
                verbose=False
            )
            reducer_full.fit(sample_embeddings)
            embedding = reducer_full.transform(embeddings)
        else:
            embedding = embedding_2d
    else:
        # Fallback to t-SNE
        print(f"  Computing t-SNE on {sample_embeddings.shape[0]} samples...")
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
        embedding_2d = reducer.fit_transform(sample_embeddings)
        
        if use_indices:
            # For t-SNE, we can't easily interpolate, so just return sampled version
            embedding = np.zeros((n_samples, n_components))
            embedding[indices] = embedding_2d
            # Fill remaining with nearest neighbor
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(sample_embeddings)
            _, nearest = nn.kneighbors(embeddings[~np.isin(np.arange(n_samples), indices)])
            embedding[~np.isin(np.arange(n_samples), indices)] = embedding_2d[nearest.flatten()]
        else:
            embedding = embedding_2d
    
    return embedding

def plot_umap_by_labels(
    embedding_2d: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
    label_map: Optional[Dict[str, str]] = None
):
    """
    Generate UMAP plot colored by labels (ground truth or predictions).
    
    Args:
        embedding_2d: 2D embedding coordinates (n_samples, 2)
        labels: List of label strings (e.g., cell type ontology IDs)
        title: Plot title
        output_path: Path to save the plot
        label_map: Optional mapping from ontology IDs to readable names
    """
    plt.figure(figsize=(12, 8))
    
    # Filter out NaN/None values and convert to strings for consistent sorting
    import math
    import numpy as np
    filtered_labels = []
    for label in labels:
        if label is None:
            filtered_labels.append('Unknown')
        elif isinstance(label, float) and (math.isnan(label) or np.isnan(label)):
            filtered_labels.append('Unknown')
        else:
            # Convert to string to ensure consistent type for sorting
            filtered_labels.append(str(label))
    
    # Get unique labels and assign colors (all are strings now, so sorting will work)
    unique_labels = sorted(list(set(filtered_labels)))
    n_labels = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_labels, 20)))
    if n_labels > 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        # Cycle colors for more than 20 labels
        colors = np.tile(colors, (n_labels // 20 + 1, 1))[:n_labels]
    
    # Plot each label group
    for i, label in enumerate(unique_labels):
        mask = np.array(filtered_labels) == label
        display_label = label_map.get(label, label) if label_map else label
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i % len(colors)]],
            label=display_label,
            alpha=0.6,
            s=10
        )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    # Only show legend if reasonable number of labels
    if n_labels <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        plt.text(0.02, 0.98, f'{n_labels} unique labels', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_umap_by_error_magnitude(
    embedding_2d: np.ndarray,
    predictions: List[str],
    ground_truth: List[str],
    ontology_graph: Optional[object],
    title: str,
    output_path: str
):
    """
    Generate UMAP plot colored by ontology error magnitude.
    
    Args:
        embedding_2d: 2D embedding coordinates (n_samples, 2)
        predictions: List of predicted ontology IDs
        ground_truth: List of ground truth ontology IDs
        ontology_graph: NetworkX graph for distance calculation (optional)
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate error magnitudes
    error_magnitudes = []
    for pred, truth in zip(predictions, ground_truth):
        if pred == truth:
            error_magnitudes.append(0)
        elif ontology_graph:
            try:
                from ontology_utils import calculate_graph_distance
                dist = calculate_graph_distance(ontology_graph, pred, truth)
                error_magnitudes.append(dist if dist >= 0 else 10)  # Cap at 10 for visualization
            except:
                error_magnitudes.append(10 if pred != truth else 0)
        else:
            error_magnitudes.append(10 if pred != truth else 0)
    
    error_magnitudes = np.array(error_magnitudes)
    
    # Create scatter plot colored by error magnitude
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=error_magnitudes,
        cmap='RdYlGn_r',  # Red (high error) to Green (low error)
        s=10,
        alpha=0.6
    )
    
    plt.colorbar(scatter, label='Ontology Distance (edges)', shrink=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    # Add statistics text
    correct = np.sum(error_magnitudes == 0)
    total = len(error_magnitudes)
    accuracy = correct / total if total > 0 else 0
    mean_error = np.mean(error_magnitudes[error_magnitudes > 0]) if np.any(error_magnitudes > 0) else 0
    
    stats_text = f'Accuracy: {accuracy:.2%}\nMean Error: {mean_error:.2f} edges'
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    title: str,
    output_path: str,
    label_map: Optional[Dict[str, str]] = None,
    max_labels: int = 20
):
    """
    Generate confusion matrix heatmap.
    
    Args:
        predictions: List of predicted ontology IDs
        ground_truth: List of ground truth ontology IDs
        title: Plot title
        output_path: Path to save the plot
        label_map: Optional mapping from ontology IDs to readable names
        max_labels: Maximum number of labels to show (top N by frequency)
    """
    # Get top N most frequent labels
    all_labels = ground_truth + predictions
    label_counts = pd.Series(all_labels).value_counts()
    top_labels = label_counts.head(max_labels).index.tolist()
    
    # Filter to only top labels
    filtered_pred = [p if p in top_labels else 'Other' for p in predictions]
    filtered_truth = [t if t in top_labels else 'Other' for t in ground_truth]
    
    # Generate confusion matrix
    labels_for_cm = top_labels + ['Other'] if 'Other' in filtered_pred or 'Other' in filtered_truth else top_labels
    cm = confusion_matrix(filtered_truth, filtered_pred, labels=labels_for_cm)
    
    # Convert to DataFrame for better labeling
    if label_map:
        display_labels = [label_map.get(l, l) for l in labels_for_cm]
    else:
        display_labels = labels_for_cm
    
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    
    # Plot heatmap
    plt.figure(figsize=(max(12, len(labels_for_cm) * 0.8), max(10, len(labels_for_cm) * 0.8)))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_summary_table(results_df: pd.DataFrame, output_path: str):
    """
    Generate a comprehensive summary table comparing all permutations.
    
    Args:
        results_df: DataFrame with benchmark results
        output_path: Path to save the summary table (CSV)
    """
    # Create summary statistics
    summary_data = []
    
    # Group by Index and Metric
    for (index_type, metric), group in results_df.groupby(['Index', 'Metric']):
        summary_data.append({
            'Index': index_type,
            'Metric': metric,
            'Mean_Accuracy': group['accuracy'].mean(),
            'Std_Accuracy': group['accuracy'].std(),
            'Mean_F1_Macro': group['f1_macro'].mean(),
            'Mean_F1_Weighted': group['f1_weighted'].mean(),
            'Mean_TopK_Accuracy': group['top_k_accuracy'].mean(),
            'Mean_Ontology_Dist': group['mean_ontology_dist'].mean() if 'mean_ontology_dist' in group.columns else np.nan,
            'Mean_Query_Time_ms': group['Avg_Query_Time_ms'].mean(),
            'Num_Datasets': len(group)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by accuracy (descending)
    summary_df = summary_df.sort_values('Mean_Accuracy', ascending=False)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary table: {output_path}")
    
    # Also print formatted table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Performance by Index Type and Metric")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    
    return summary_df

def generate_visualizations(
    results_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    predictions_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]],
    ontology_graph: Optional[object],
    output_dir: str,
    label_map: Optional[Dict[str, str]] = None
):
    """
    Generate all visualizations for a benchmark run.
    
    Args:
        results_df: DataFrame with benchmark results
        embeddings_dict: Dict mapping dataset_id to embeddings array
        predictions_dict: Dict mapping dataset_id to predictions list
        ground_truth_dict: Dict mapping dataset_id to ground truth list
        ontology_graph: NetworkX graph for ontology distance (optional)
        output_dir: Directory to save all visualizations
        label_map: Optional mapping from ontology IDs to readable names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate UMAP plots for each dataset
    for dataset_id in embeddings_dict.keys():
        if dataset_id not in predictions_dict or dataset_id not in ground_truth_dict:
            continue
        
        embeddings = embeddings_dict[dataset_id]
        predictions = predictions_dict[dataset_id]
        ground_truth = ground_truth_dict[dataset_id]
        
        # Skip if no embeddings
        if len(embeddings) == 0:
            continue
        
        print(f"\nGenerating visualizations for dataset: {dataset_id}")
        
        # Generate 2D embedding
        print("  Computing UMAP embedding...")
        embedding_2d = generate_umap_embedding(embeddings)
        
        # Plot by ground truth
        print("  Plotting UMAP by Ground Truth...")
        plot_umap_by_labels(
            embedding_2d,
            ground_truth,
            f'UMAP: {dataset_id} (Ground Truth)',
            os.path.join(output_dir, f'{dataset_id}_umap_ground_truth.png'),
            label_map
        )
        
        # Plot by predictions
        print("  Plotting UMAP by Predictions...")
        plot_umap_by_labels(
            embedding_2d,
            predictions,
            f'UMAP: {dataset_id} (Predictions)',
            os.path.join(output_dir, f'{dataset_id}_umap_predictions.png'),
            label_map
        )
        
        # Plot by error magnitude
        print("  Plotting UMAP by Error Magnitude...")
        plot_umap_by_error_magnitude(
            embedding_2d,
            predictions,
            ground_truth,
            ontology_graph,
            f'UMAP: {dataset_id} (Ontology Error Magnitude)',
            os.path.join(output_dir, f'{dataset_id}_umap_errors.png')
        )
        
        # Generate confusion matrix
        print("  Generating confusion matrix...")
        plot_confusion_matrix(
            predictions,
            ground_truth,
            f'Confusion Matrix: {dataset_id}',
            os.path.join(output_dir, f'{dataset_id}_confusion_matrix.png'),
            label_map
        )
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary_path = os.path.join(output_dir, 'summary_table.csv')
    generate_summary_table(results_df, summary_path)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
