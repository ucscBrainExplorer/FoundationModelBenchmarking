# README Update — Three Standalone Programs

**Add this section after "Quick Start" in the main README.md**

---

## Three Standalone Programs (New)

The benchmarking suite now provides three composable programs for flexible workflows:

### 1. **`predict.py`** — Predict cell types
Predict cell types using FAISS k-NN + majority voting. **No ground truth required.**

```bash
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings test_data/dataset.npy \
  --output predictions.tsv
```

**Output:** `predictions.tsv` with predicted CL term IDs, readable names, vote confidence, and neighbor information.

### 2. **`evaluate.py`** — Evaluate predictions
Evaluate predictions against ground truth using ontology-based semantic metrics.

```bash
python3 evaluate.py \
  --predictions predictions.tsv \
  --ground_truth test_data/ground_truth.tsv \
  --obo reference_data/cl.obo \
  --ontology-method ic \
  --output-dir evaluation_results/
```

**Output:** Aggregate metrics, per-cell scores, detailed report, and visualizations.

**Works with any predictions**, not just from `predict.py` — accepts external tools or biologist annotations.

### 3. **`annotate_cl_terms.py`** — Map names → CL IDs
Map readable cell type names to CL ontology term IDs for external predictions.

```bash
python3 annotate_cl_terms.py \
  --obo reference_data/cl.obo \
  --input biologist_predictions.tsv \
  --name_col cell_type \
  --output annotated.tsv
```

Uses exact match, synonym match, fuzzy normalization, and LLM-assisted fallback (Claude + OpenAI).

**Output:** Original file + added `cell_type_ontology_term_id` column.

---

### Workflows

#### Workflow A: Full Benchmark
```bash
# Predict
python3 predict.py --index idx.faiss --ref_annot ref.tsv --obo cl.obo \
  --embeddings test.npy --output preds.tsv

# Evaluate
python3 evaluate.py --predictions preds.tsv --ground_truth truth.tsv \
  --obo cl.obo --output-dir eval_results/
```

#### Workflow B: External Predictions
```bash
# Map readable names → CL IDs
python3 annotate_cl_terms.py --obo cl.obo \
  --input external_preds.tsv --output annotated.tsv

# Evaluate
python3 evaluate.py --predictions annotated.tsv --ground_truth truth.tsv \
  --obo cl.obo --output-dir eval_results/
```

#### Workflow C: Predict Only (no ground truth)
```bash
python3 predict.py --index idx.faiss --ref_annot ref.tsv --obo cl.obo \
  --embeddings unlabeled.npy --output preds.tsv
```

---

### Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** — Get started quickly
- **[Detailed Usage](docs/USAGE.md)** — All options, examples, troubleshooting
- **[API Reference](docs/API.md)** — Programmatic usage

---
