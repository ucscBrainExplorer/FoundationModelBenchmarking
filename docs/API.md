# API Reference

Programmatic usage of the core modules.

---

## obo_parser

### `parse_obo_names(obo_path) -> dict`

Parse an OBO file and return `{term_id: canonical_name}`.

```python
from obo_parser import parse_obo_names

cl_names = parse_obo_names("cl.obo")
print(cl_names["CL:0000540"])  # "neuron"
```

### `parse_obo_replacements(obo_path) -> dict`

Return `{obsolete_id: replacement_id}` for terms with `replaced_by` in the OBO file.

```python
from obo_parser import parse_obo_replacements

replacements = parse_obo_replacements("cl.obo")
```

---

## data_loader

### `load_faiss_index(path, index_type='ivfFlat') -> faiss.Index`

Load a FAISS index from disk. Automatically sets `nprobe=20` for IVF indices.

Raises `FileNotFoundError` if the file doesn't exist, `RuntimeError` if loading fails.

```python
from data_loader import load_faiss_index

index = load_faiss_index("indices/index_ivfflat.faiss")
print(f"dim={index.d}, vectors={index.ntotal}")
```

### `load_reference_annotations(path) -> pd.DataFrame`

Load a reference annotation TSV. Validates that `cell_type_ontology_term_id` is present.

Raises `FileNotFoundError` or `ValueError` (missing column).

```python
from data_loader import load_reference_annotations

ref_df = load_reference_annotations("reference_data/prediction_obs.tsv")
```

**Important:** The DataFrame's integer index (0 to N-1) must align with FAISS index vector IDs.

### `load_test_batch(test_dir) -> List[Dict]`

Discover paired `.npy` / `_prediction_obs.tsv` files in a directory. Returns a list of dicts with keys `id`, `embedding_path`, `metadata_path`.

```python
from data_loader import load_test_batch

for ds in load_test_batch("test_data/"):
    print(ds["id"], ds["embedding_path"])
```

---

## prediction_module

### `execute_query(index, query_embeddings, k=30) -> (np.ndarray, np.ndarray)`

Query a FAISS index for k nearest neighbors. Returns `(squared_dists, indices)`, both shape `(n_queries, k)`. Convert squared distances to Euclidean with `np.sqrt(np.maximum(sq_dists, 0))`.

```python
import numpy as np
from data_loader import load_faiss_index
from prediction_module import execute_query

index = load_faiss_index("index.faiss")
embeddings = np.load("test.npy")

sq_dists, neighbor_indices = execute_query(index, embeddings, k=30)
dists = np.sqrt(np.maximum(sq_dists, 0))
```

### `distance_weighted_knn_vote(neighbor_indices, neighbor_dists, reference_annotations) -> (List[str], List[float])`

Distance-weighted KNN voting using Gaussian kernel weights (sigma = per-row median distance). Closer neighbors contribute more weight.

Returns `(predictions, vote_fractions)` — predicted CL term IDs and winner weight fraction (0–1).

```python
from prediction_module import distance_weighted_knn_vote

predictions, scores = distance_weighted_knn_vote(neighbor_indices, dists, ref_df)
```

### `vote_neighbors(neighbor_indices, reference_annotations) -> (List[str], List[float])`

Simple majority voting. Returns `(predictions, vote_percentages)`. Empty string and `NaN` for cells where all neighbors have invalid labels.

```python
from prediction_module import vote_neighbors

predictions, scores = vote_neighbors(neighbor_indices, ref_df)
```

### `gaussian_kernel_weights(dists, eps=1e-8) -> np.ndarray`

Convert Euclidean distances to Gaussian kernel weights. Sigma is set per-row to the median distance of that row's neighbors.

---

## ontology_utils

### `load_ontology(obo_path, include_relationships=True) -> nx.DiGraph`

Load the Cell Ontology into a NetworkX directed graph. Edges point child → parent. Tries `pronto` first, falls back to a manual OBO parser for the full `cl.obo`.

```python
from ontology_utils import load_ontology

graph = load_ontology("cl.obo")
print(graph.number_of_nodes(), graph.number_of_edges())
```

### `precompute_ic(graph, k=0.5) -> Dict[str, float]`

Precompute Zhou (2008) intrinsic Information Content for all terms:

```
IC(t) = k * Seco_IC(t) + (1 - k) * depth_component(t)
```

Returns `{term_id: IC_value}`. With k=0.5, descendant count and structural depth contribute equally.

```python
from ontology_utils import load_ontology, precompute_ic

graph = load_ontology("cl.obo")
ic_values = precompute_ic(graph, k=0.5)
print(ic_values["CL:0000540"])  # IC of neuron
```

### `calculate_lin_similarity(graph, predicted_id, truth_id, ic_values) -> float`

Lin (1998) semantic similarity between two terms: `2 * IC(MICA) / (IC(A) + IC(B))`. Returns a value in [0, 1], or -1.0 if either term is missing from the graph.

### `calculate_graph_distance(graph, predicted_id, truth_id) -> int`

Shortest undirected path length between two terms. Returns -1 if not reachable.

### `calculate_per_cell_distances(graph, predictions, ground_truth, method='shortest_path', ic_values=None) -> List[float]`

Compute an ontology score for each cell. Returns `NaN` for cells where the score cannot be computed (missing terms, no common ancestor).

```python
from ontology_utils import load_ontology, precompute_ic, calculate_per_cell_distances

graph = load_ontology("cl.obo")
ic_values = precompute_ic(graph)

scores = calculate_per_cell_distances(
    graph,
    predictions=["CL:0000540", "CL:0000128"],
    ground_truth=["CL:0000540", "CL:0000127"],
    method='ic',
    ic_values=ic_values,
)
# [1.0, 0.87]  — first is exact match, second is similar
```

### `score_batch(graph, predictions, ground_truth, method='shortest_path', ic_values=None) -> (float, float)`

Returns `(mean, median)` ontology score over the full dataset, skipping cells where the score is undefined.

---

## analyze_ontology_results

### `calculate_ontology_statistics(df, ontology_method='ic') -> Dict`

Compute descriptive statistics on the ontology score column of a per-cell results DataFrame.

Returns a dict with keys: `column`, `mean`, `median`, `std`, `min`, `max`, `percentiles`, `distribution`.

```python
import pandas as pd
from analyze_ontology_results import calculate_ontology_statistics

df = pd.read_csv("evaluation_results/per_cell_evaluation.tsv", sep='\t', comment='#')
stats = calculate_ontology_statistics(df, ontology_method='ic')
print(f"Mean IC similarity: {stats['mean']:.3f}")
```

### `generate_summary_report(df, stats, output_path, ontology_method='ic', comment_header=None)`

Write a comprehensive text report to `output_path`. The DataFrame must have columns `true_label`, `prediction_label`, and an ontology score column.

### `analyze_distance_metric_relationship(df, output_dir, ontology_method='ic')`

Save a two-panel PNG (accuracy vs. score scatter + score histogram) to `output_dir`.

---

## End-to-End Example

```python
import numpy as np
import pandas as pd

from data_loader import load_faiss_index, load_reference_annotations
from prediction_module import execute_query, distance_weighted_knn_vote
from ontology_utils import load_ontology, precompute_ic, calculate_per_cell_distances
from analyze_ontology_results import calculate_ontology_statistics, generate_summary_report
from obo_parser import parse_obo_names

# Load resources
index   = load_faiss_index("indices/index.faiss")
ref_df  = load_reference_annotations("reference_data/ref.tsv")
cl_names = parse_obo_names("reference_data/cl.obo")
graph   = load_ontology("reference_data/cl.obo")
ic_values = precompute_ic(graph, k=0.5)

# Predict
embeddings = np.load("test_data/test.npy")
sq_dists, neighbor_indices = execute_query(index, embeddings, k=30)
dists = np.sqrt(np.maximum(sq_dists, 0))
predictions, vote_scores = distance_weighted_knn_vote(neighbor_indices, dists, ref_df)

# Evaluate against ground truth
truth_df = pd.read_csv("test_data/ground_truth.tsv", sep='\t')
ground_truth = truth_df['cell_type_ontology_term_id'].tolist()

scores = calculate_per_cell_distances(
    graph, predictions, ground_truth, method='ic', ic_values=ic_values)

# Build results DataFrame
results_df = pd.DataFrame({
    'predicted_cl_term_id': predictions,
    'truth_cl_term_id':     ground_truth,
    'predicted_cell_type':  [cl_names.get(p, p) for p in predictions],
    'truth_cell_type':      [cl_names.get(t, t) for t in ground_truth],
    'ontology_IC_similarity': scores,
    'is_exact_match':       [1 if p == t else 0 for p, t in zip(predictions, ground_truth)],
    'true_label':           ground_truth,
    'prediction_label':     predictions,
})

# Report
stats = calculate_ontology_statistics(results_df, ontology_method='ic')
generate_summary_report(results_df, stats, "report.txt", ontology_method='ic')

print(f"Mean IC similarity:  {stats['mean']:.3f}")
print(f"Exact match rate:    {results_df['is_exact_match'].mean():.3f}")
```
