# Detailed Usage Guide

## Table of Contents

1. [predict.py — Prediction](#predictpy--prediction)
2. [evaluate.py — Evaluation](#evaluatepy--evaluation)
3. [annotate_cl_terms.py — Annotation](#annotate_cl_termspy--annotation)
4. [File Formats](#file-formats)
5. [Troubleshooting](#troubleshooting)

---

## predict.py — Prediction

### Purpose

Predict cell types using FAISS k-nearest-neighbor voting. **No ground truth labels required.**
Two voting methods are available via `--method`: `distance_weighted_knn` (default) and `majority_voting`.

### Example: Basic Usage

```bash
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings test_data/dataset.npy \
  --output predictions.tsv
```

### Example: With Metadata

If you have a metadata file with additional information (all columns will be preserved in output):

```bash
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings test_data/dataset.npy \
  --metadata test_data/dataset_metadata.tsv \
  --output predictions.tsv
```

### Example: Custom k Value

Use 50 nearest neighbors instead of default 30:

```bash
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings test_data/dataset.npy \
  --k 50 \
  --output predictions.tsv
```

### Output Format

The output TSV contains one row per cell. Prediction columns depend on `--method`:

**`distance_weighted_knn` (default):**

| Column | Type | Description |
|---|---|---|
| `cell_id` | string | Cell identifier from metadata, if provided |
| `weighted_cell_type_ontology_term_id` | string | Predicted CL term ID (e.g., "CL:0000540") |
| `weighted_cell_type` | string | Canonical name from OBO (e.g., "neuron") |
| `weighted_score` | float | Normalized weight fraction for winner (0–1) |
| `mean_euclidean_distance` | float | Mean distance to all k neighbors |
| `std_euclidean_distance` | float | Std of distance to all k neighbors |
| `neighbor_distances` | string | Comma-separated distances (sorted closest→farthest) |
| `neighbor_cell_types` | string | Comma-separated canonical names (same order) |

**`majority_voting`:** same but with `mv_cell_type_ontology_term_id`, `mv_cell_type`, `mv_score`.

**`both`:** all columns from both methods.

**Row order:** Always matches the input `.npy` row order. No cells are dropped.

### Important Notes

- **Reference annotations:** Used only for the `cell_type_ontology_term_id` column during voting. The FAISS index vector IDs must align with the reference DataFrame row indices.
- **Empty predictions:** Cells where all k neighbors had invalid labels will have an empty CL term ID and `NaN` score.

---

## evaluate.py — Evaluation

### Purpose

Evaluate cell type predictions against ground truth using ontology-based semantic metrics. Accepts predictions from `predict.py` or any external tool.

### Example: Basic Usage

```bash
python3 evaluate.py \
  --predictions predictions.tsv \
  --ground_truth test_data/ground_truth.tsv \
  --obo reference_data/cl.obo \
  --output-dir evaluation_results/
```

### Example: Shortest Path Distance

Use shortest-path ontology distance instead of IC similarity:

```bash
python3 evaluate.py \
  --predictions predictions.tsv \
  --ground_truth test_data/ground_truth.tsv \
  --obo reference_data/cl.obo \
  --ontology-method shortest_path \
  --output-dir evaluation_results/
```

### Example: External Predictions with Custom Columns

Your predictions file has different column names:

```bash
python3 evaluate.py \
  --predictions external_preds.tsv \
  --pred_id_col predicted_cl_id \
  --ground_truth truth.tsv \
  --truth_id_col true_cl_id \
  --obo reference_data/cl.obo \
  --output-dir evaluation_results/
```

### Output Files

The output directory contains:

#### 1. `evaluation_summary.tsv`

Aggregate metrics in vertical format:

```
metric                              value
mean_ontology_IC_similarity         0.8723
median_ontology_IC_similarity       0.9234
std_ontology_IC_similarity          0.1456
exact_match_rate                    0.6500
total_cells_evaluated               1000
```

#### 2. `per_cell_evaluation.tsv`

Per-cell results with columns:
- `cell_id`, `predicted_cl_term_id`, `truth_cl_term_id`
- `predicted_cell_type`, `truth_cell_type` (canonical names)
- `ontology_score` (IC similarity or shortest-path distance)
- `is_exact_match` (1 if predicted == truth, 0 otherwise)

#### 3. `ontology_analysis_report.txt`

Comprehensive text report including:
- Metric interpretation guide
- Overall statistics (mean, median, std, percentiles)
- Distribution summary (bucketed for IC, per-value for shortest path)
- Relationship to accuracy (correlation, accuracy by score bucket)

#### 4. `ontology_distance_analysis.png`

Two-panel visualization:
- Left: Scatter plot of accuracy vs. ontology score
- Right: Histogram of ontology score distribution

#### 5. `ontology_distance_accuracy_relationship.csv`

CSV with columns: ontology score, cell count, correct count, accuracy (grouped by score).

### Evaluation Logic

1. Read predictions and ground truth TSVs
2. **Row-by-row matching** (predictions and ground truth must have same row count and order)
3. Load ontology, precompute IC if needed
4. Compute per-cell ontology scores
5. Compute exact match rate as supplementary statistic
6. Generate aggregate statistics, report, and visualizations

**Important:** The predictions and ground truth files must have the same number of rows in the same order. Evaluation is performed by matching rows at the same index.

---

## annotate_cl_terms.py — Annotation

### Purpose

Map readable cell type names to CL ontology term IDs using a multi-step matching pipeline with LLM fallback.

### Example: Basic Usage

```bash
python3 annotate_cl_terms.py \
  --obo reference_data/cl.obo \
  --input biologist_predictions.tsv \
  --output annotated.tsv
```

Assumes the input file has a column named `cell_type` with readable names.

### Example: Custom Column Name

Your input file has a column named `predicted_type`:

```bash
python3 annotate_cl_terms.py \
  --obo reference_data/cl.obo \
  --input predictions.tsv \
  --name_col predicted_type \
  --output annotated.tsv
```

### Matching Logic

For each unique cell type name in the input:

#### Step 1: Exact Match (case-insensitive)
- `"neuron"` → `CL:0000540` ✓
- `"Neuron"` → `CL:0000540` ✓

#### Step 2: Synonym Match
- `"nerve cell"` → `CL:0000540` (synonym in OBO) ✓

#### Step 3: Fuzzy Normalization
Strips hyphens, underscores, and trailing 's' (plurals):
- `"T-cell"` → `CL:0000084` ✓
- `"oligodendrocytes"` → `CL:0000128` ✓
- `"micro_glial"` → `CL:0000129` ✓

#### Step 4: LLM-Assisted Fallback

For names that don't match any of the above:

1. Query both Claude and OpenAI APIs with a prompt listing all CL terms
2. **If both LLMs agree** → auto-accept the mapping
3. **If LLMs disagree** → interactive terminal prompt:
   ```
   LLMs disagree for "mystery cell":
     [c] Claude: CL:0000540 (neuron)
     [o] OpenAI: CL:0000084 (T cell)
     [s] Skip (leave unmapped)
     [m] Manual (type CL ID)
   Your choice:
   ```
4. **If neither LLM has a suggestion** → mark as unresolved (empty string)

### Environment Variables

For LLM fallback, set these environment variables:

```bash
export ANTHROPIC_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

If neither key is set, unmatched names are left as empty strings (no LLM fallback).

### Output Format

The output TSV contains all original columns plus a new column:
- `cell_type_ontology_term_id` — the mapped CL term ID (or empty string if unresolved)

### Console Summary

After processing, prints a summary:

```
ANNOTATION SUMMARY
============================================================
Total unique names:     45
  Exact match:          30
  Synonym match:        5
  Fuzzy match:          7
  LLM consensus:        2
  User resolved:        1
  Unresolved:           0
  Total mapped:         45/45
============================================================
```

---

## File Formats

### Embeddings (.npy)

NumPy array with shape `(n_cells, embedding_dim)`, dtype `float32` or `float64`.

```python
import numpy as np
embeddings = np.load("dataset.npy")
print(embeddings.shape)  # (1000, 128)
```

### Reference Annotations (TSV)

Tab-separated file with required column:
- `cell_type_ontology_term_id` — CL term IDs (e.g., "CL:0000540")

Optional columns:
- `cell_type` — readable names (e.g., "neuron")

**Critical:** The DataFrame row index (0 to N-1) must align with FAISS index vector IDs.

### Ground Truth / Metadata (TSV)

Tab-separated file with required column:
- `cell_type_ontology_term_id` — CL term IDs

Optional columns:
- `cell_id` — unique cell identifiers (preserved in output if provided)
- `cell_type` — readable names

### OBO File

Cell Ontology in OBO format. Download from:
- Full: http://purl.obolibrary.org/obo/cl.obo
- Basic (subset): http://purl.obolibrary.org/obo/cl/cl-basic.obo

---

## Troubleshooting

### Issue: "Row count mismatch between predictions and ground truth"

**Cause:** The predictions and ground truth files have different numbers of rows.

**Solution:** Ensure both files have the same number of rows in the same order. Print row counts:
```bash
wc -l predictions.tsv
wc -l ground_truth.tsv
```

### Issue: "Column 'cell_type_ontology_term_id' not found"

**Cause:** Column name mismatch.

**Solution:** Use `--pred_id_col` / `--truth_id_col` to specify custom column names:
```bash
python3 evaluate.py ... --pred_id_col my_column_name
```

### Issue: Empty predictions (score = NaN)

**Cause:** All k neighbors had invalid labels (NaN, empty, or "nan").

**Solution:** Check reference annotations file for missing or invalid labels. Filter:
```python
import pandas as pd
df = pd.read_csv("reference_data/prediction_obs.tsv", sep='\t')
print(df[df['cell_type_ontology_term_id'].isna()])
```

### Issue: LLM fallback not working

**Cause:** API keys not set or incorrect.

**Solution:** Verify environment variables:
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

### Issue: FAISS index dimension mismatch

**Cause:** Embeddings dimension doesn't match FAISS index.

**Solution:** Check dimensions:
```python
import faiss
import numpy as np

index = faiss.read_index("index.faiss")
embeddings = np.load("embeddings.npy")

print(f"Index dimension: {index.d}")
print(f"Embeddings dimension: {embeddings.shape[1]}")
```

They must match.
