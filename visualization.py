"""Confusion matrix visualization for UCE Benchmarking."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    title: str,
    output_path: str,
    label_map: Optional[Dict[str, str]] = None,
    max_labels: int = 20
):
    """
    Generate a confusion matrix heatmap.

    Args:
        predictions: List of predicted CL term IDs.
        ground_truth: List of ground truth CL term IDs.
        title: Plot title.
        output_path: Path to save the PNG.
        label_map: Optional {CL_id: readable_name} for axis labels.
        max_labels: Maximum number of labels to show (top N by frequency).
                    Less frequent labels are collapsed into 'Other'.
    """
    # Determine top N labels by combined frequency
    all_labels = ground_truth + predictions
    top_labels = pd.Series(all_labels).value_counts().head(max_labels).index.tolist()

    filtered_pred  = [p if p in top_labels else 'Other' for p in predictions]
    filtered_truth = [t if t in top_labels else 'Other' for t in ground_truth]

    has_other = 'Other' in filtered_pred or 'Other' in filtered_truth
    labels_for_cm = top_labels + (['Other'] if has_other else [])

    cm = confusion_matrix(filtered_truth, filtered_pred, labels=labels_for_cm)

    if label_map:
        display_labels = [label_map.get(l, l) for l in labels_for_cm]
    else:
        display_labels = labels_for_cm

    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    n = len(labels_for_cm)
    fig_size = max(10, n * 0.8)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
