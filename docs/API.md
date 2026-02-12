# API Reference

Programmatic usage of the benchmarking modules.

## Table of Contents

1. [obo_parser](#obo_parser)
2. [data_loader](#data_loader)
3. [prediction_module](#prediction_module)
4. [ontology_utils](#ontology_utils)
5. [evaluate (analyze_ontology_results)](#analyze_ontology_results)

---

## obo_parser

Parse Cell Ontology OBO files.

### `parse_obo_names(obo_path: str) -> dict`

Parse an OBO file and return a mapping of term IDs to canonical names.

**Args:**
- `obo_path` (str): Path to OBO file (e.g., `cl.obo`)

**Returns:**
- `dict`: Mapping `{term_id: name}`, e.g., `{"CL:0000540": "neuron", ...}`

**Example:**
```python
from obo_parser import parse_obo_names

cl_names = parse_obo_names("reference_data/cl.obo")
print(cl_names["CL:0000540"])  # "neuron"
```

---

## data_loader

Load FAISS indices, reference annotations, and test data.

### `load_faiss_index(path: str, index_type: str = 'ivfFlat') -> faiss.Index`

Load a FAISS index from disk.

**Args:**
- `path` (str): Path to `.faiss` file
- `index_type` (str): Index type (currently unused, for future validation)

**Returns:**
- `faiss.Index`: Loaded FAISS index

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `RuntimeError`: If loading fails

**Example:**
```python
from data_loader import load_faiss_index

index = load_faiss_index("indices/index_ivfflat.faiss")
print(f"Dimension: {index.d}, Vectors: {index.ntotal}")
```

### `load_reference_annotations(path: str) -> pd.DataFrame`

Load reference annotations from a TSV file.

**Args:**
- `path` (str): Path to TSV file

**Returns:**
- `pd.DataFrame`: DataFrame with required column `cell_type_ontology_term_id`

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If required column is missing

**Example:**
```python
from data_loader import load_reference_annotations

ref_df = load_reference_annotations("reference_data/prediction_obs.tsv")
print(ref_df.head())
```

**Important:** The DataFrame's integer index (0 to N-1) must align with FAISS index vector IDs.

### `load_test_batch(test_dir: str) -> List[Dict[str, str]]`

Discover test dataset pairs in a directory.

**Args:**
- `test_dir` (str): Directory containing test data

**Returns:**
- `List[Dict]`: List of dicts with keys `'id'`, `'embedding_path'`, `'metadata_path'`

**Example:**
```python
from data_loader import load_test_batch

datasets = load_test_batch("test_data")
for ds in datasets:
    print(f"Dataset {ds['id']}: {ds['embedding_path']}")
```

---

## prediction_module

Execute FAISS queries and perform majority voting.

### `execute_query(index: faiss.Index, query_embeddings: np.ndarray, k: int = 30, metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]`

Query a FAISS index for k-nearest neighbors.

**Args:**
- `index` (faiss.Index): FAISS index
- `query_embeddings` (np.ndarray): Query vectors, shape `(n_queries, dim)`
- `k` (int): Number of nearest neighbors (default: 30)
- `metric` (str): Distance metric (`'euclidean'` only, default)

**Returns:**
- `tuple`: `(dists, indices)`
  - `dists` (np.ndarray): Shape `(n_queries, k)`, L2 distances
  - `indices` (np.ndarray): Shape `(n_queries, k)`, neighbor vector IDs

**Raises:**
- `ValueError`: If `metric != 'euclidean'`

**Example:**
```python
import numpy as np
from data_loader import load_faiss_index
from prediction_module import execute_query

index = load_faiss_index("index.faiss")
embeddings = np.load("test.npy")

dists, neighbor_indices = execute_query(index, embeddings, k=30)
print(f"Queried {len(embeddings)} cells, got {neighbor_indices.shape[1]} neighbors each")
```

### `vote_neighbors(neighbor_indices: np.ndarray, reference_annotations: pd.DataFrame) -> Tuple[List[str], List[float]]`

Perform majority voting on nearest neighbors.

**Args:**
- `neighbor_indices` (np.ndarray): Shape `(n_queries, k)`, neighbor indices from FAISS
- `reference_annotations` (pd.DataFrame): Reference annotations with `cell_type_ontology_term_id` column

**Returns:**
- `tuple`: `(predictions, vote_percentages)`
  - `predictions` (List[str]): Predicted CL term IDs (empty string if no valid vote)
  - `vote_percentages` (List[float]): Confidence in [0.0, 1.0] (NaN if no valid neighbors)

**Example:**
```python
from data_loader import load_reference_annotations
from prediction_module import vote_neighbors

ref_df = load_reference_annotations("ref.tsv")
predictions, confidences = vote_neighbors(neighbor_indices, ref_df)

for pred, conf in zip(predictions[:5], confidences[:5]):
    print(f"Predicted: {pred}, Confidence: {conf:.2f}")
```

---

## ontology_utils

Ontology graph construction and semantic similarity metrics.

### `load_ontology(obo_path: str) -> nx.DiGraph`

Load Cell Ontology from OBO file into a NetworkX directed graph.

**Args:**
- `obo_path` (str): Path to OBO file

**Returns:**
- `nx.DiGraph`: Directed graph where nodes are CL term IDs with `name` attribute, edges are child→parent

**Raises:**
- `ImportError`: If `pronto` or `networkx` not installed

**Example:**
```python
from ontology_utils import load_ontology

graph = load_ontology("reference_data/cl.obo")
print(f"Loaded {graph.number_of_nodes()} terms, {graph.number_of_edges()} edges")
```

### `precompute_ic(graph: nx.DiGraph, k: float = 0.5) -> Dict[str, float]`

Precompute Zhou (2008) intrinsic Information Content for all terms.

**Args:**
- `graph` (nx.DiGraph): Ontology graph from `load_ontology()`
- `k` (float): Blend parameter (0.5 = equal weight to Seco IC and depth component)

**Returns:**
- `Dict[str, float]`: Mapping `{term_id: IC_value}`

**Example:**
```python
from ontology_utils import load_ontology, precompute_ic

graph = load_ontology("cl.obo")
ic_values = precompute_ic(graph, k=0.5)

print(f"IC(neuron) = {ic_values['CL:0000540']:.4f}")
```

### `calculate_per_cell_distances(graph: nx.DiGraph, predictions: List[str], ground_truth: List[str], method: str = 'shortest_path', ic_values: Dict[str, float] = None) -> List[float]`

Calculate ontology-based score for each cell.

**Args:**
- `graph` (nx.DiGraph): Ontology graph
- `predictions` (List[str]): Predicted CL term IDs
- `ground_truth` (List[str]): True CL term IDs
- `method` (str): `'shortest_path'` or `'ic'`
- `ic_values` (Dict): IC values (required if `method='ic'`)

**Returns:**
- `List[float]`: Ontology scores (one per cell). `NaN` if score cannot be computed.

**Example:**
```python
from ontology_utils import load_ontology, precompute_ic, calculate_per_cell_distances

graph = load_ontology("cl.obo")
ic_values = precompute_ic(graph)

predictions = ["CL:0000540", "CL:0000128"]
ground_truth = ["CL:0000540", "CL:0000127"]

scores = calculate_per_cell_distances(graph, predictions, ground_truth, method='ic', ic_values=ic_values)
print(scores)  # [1.0, 0.85]  (first is exact match, second is similar)
```

---

## analyze_ontology_results

Statistical analysis and reporting on ontology-based evaluation results.

### `calculate_ontology_statistics(df: pd.DataFrame, ontology_method: str = 'ic') -> Dict`

Calculate descriptive statistics on ontology scores.

**Args:**
- `df` (pd.DataFrame): Per-cell results with ontology score column
- `ontology_method` (str): `'ic'` or `'shortest_path'`

**Returns:**
- `Dict`: Statistics with keys `'column'`, `'mean'`, `'median'`, `'std'`, `'min'`, `'max'`, `'percentiles'`, `'distribution'`

**Example:**
```python
import pandas as pd
from analyze_ontology_results import calculate_ontology_statistics

df = pd.read_csv("per_cell_evaluation.tsv", sep='\t')
stats = calculate_ontology_statistics(df, ontology_method='ic')

print(f"Mean IC similarity: {stats['mean']:.4f}")
print(f"Median: {stats['median']:.4f}")
```

### `generate_summary_report(df: pd.DataFrame, stats: Dict, output_path: str, ontology_method: str = 'ic')`

Generate a comprehensive text report.

**Args:**
- `df` (pd.DataFrame): Per-cell results with columns `true_label`, `prediction_label`, and ontology score
- `stats` (Dict): Statistics from `calculate_ontology_statistics()`
- `output_path` (str): Path to save report
- `ontology_method` (str): `'ic'` or `'shortest_path'`

**Returns:**
- None (writes to file)

### `analyze_distance_metric_relationship(df: pd.DataFrame, output_dir: str, ontology_method: str = 'ic')`

Generate visualization and CSV analyzing ontology score vs. accuracy relationship.

**Args:**
- `df` (pd.DataFrame): Per-cell results
- `output_dir` (str): Directory to save outputs
- `ontology_method` (str): `'ic'` or `'shortest_path'`

**Returns:**
- None (saves PNG and CSV)

---

## Evaluation Matching Behavior

**Important:** `evaluate.py` uses **row-by-row matching** to align predictions with ground truth. This means:

- Predictions and ground truth files must have the **same number of rows**
- Rows are matched by **position/index**, not by cell_id or any join key
- The i-th row in predictions is compared to the i-th row in ground truth
- Both files must be in the **same order**

This design ensures consistent evaluation when cell identifiers may vary or be missing.

**Example:**
```python
# predictions.tsv (row 0, 1, 2, ...)
predicted_cell_type_ontology_term_id
CL:0000540
CL:0000128
CL:0000127

# ground_truth.tsv (row 0, 1, 2, ...)
cell_type_ontology_term_id
CL:0000540    # matched with row 0 of predictions
CL:0000128    # matched with row 1 of predictions
CL:0000540    # matched with row 2 of predictions
```

---

## Complete Example: Programmatic Workflow

```python
import numpy as np
import pandas as pd
from data_loader import load_faiss_index, load_reference_annotations
from prediction_module import execute_query, vote_neighbors
from ontology_utils import load_ontology, precompute_ic, calculate_per_cell_distances
from analyze_ontology_results import calculate_ontology_statistics, generate_summary_report
from obo_parser import parse_obo_names

# 1. Load resources
index = load_faiss_index("indices/index.faiss")
ref_df = load_reference_annotations("reference_data/ref.tsv")
cl_names = parse_obo_names("reference_data/cl.obo")
graph = load_ontology("reference_data/cl.obo")
ic_values = precompute_ic(graph, k=0.5)

# 2. Predict
embeddings = np.load("test_data/test.npy")
dists, neighbor_indices = execute_query(index, embeddings, k=30)
predictions, vote_percentages = vote_neighbors(neighbor_indices, ref_df)

# 3. Evaluate (requires ground truth)
ground_truth_df = pd.read_csv("test_data/ground_truth.tsv", sep='\t')
ground_truth = ground_truth_df['cell_type_ontology_term_id'].tolist()

scores = calculate_per_cell_distances(
    graph, predictions, ground_truth,
    method='ic', ic_values=ic_values
)

# 4. Build results DataFrame (row-by-row alignment)
results_df = pd.DataFrame({
    'predicted_cl_term_id': predictions,
    'truth_cl_term_id': ground_truth,
    'predicted_cell_type': [cl_names.get(p, p) for p in predictions],
    'truth_cell_type': [cl_names.get(t, t) for t in ground_truth],
    'ontology_IC_similarity': scores,
    'is_exact_match': [1 if p == t else 0 for p, t in zip(predictions, ground_truth)],
    'true_label': ground_truth,
    'prediction_label': predictions,
})

# 5. Generate report
stats = calculate_ontology_statistics(results_df, ontology_method='ic')
generate_summary_report(results_df, stats, "evaluation_report.txt", ontology_method='ic')

print(f"Mean IC similarity: {stats['mean']:.4f}")
print(f"Exact match rate: {results_df['is_exact_match'].mean():.4f}")
```
