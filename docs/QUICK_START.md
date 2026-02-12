# Quick Start Guide

This guide covers the three standalone programs for cell type prediction and evaluation.

## Overview

The benchmarking suite has been split into three composable programs:

1. **`predict.py`** — Predict cell types using FAISS k-NN + majority voting
2. **`evaluate.py`** — Evaluate predictions using ontology-based semantic metrics
3. **`annotate_cl_terms.py`** — Map readable names to CL ontology term IDs

These can be used independently or chained together.

---

## Workflow A: Full Benchmark (Predict + Evaluate)

### Step 1: Predict cell types

```bash
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings test_data/dataset.npy \
  --metadata test_data/dataset_prediction_obs.tsv \
  --k 30 \
  --output predictions.tsv
```

**Output:** `predictions.tsv` with columns:
- `predicted_cell_type_ontology_term_id`, `predicted_cell_type`
- `vote_percentage`, `mean_euclidean_distance`
- `neighbor_distances`, `neighbor_cell_types`
- Plus all metadata columns if `--metadata` provided

### Step 2: Evaluate predictions

```bash
python3 evaluate.py \
  --predictions predictions.tsv \
  --ground_truth test_data/dataset_prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --ontology-method ic \
  --output-dir evaluation_results/
```

**Output:** `evaluation_results/` directory containing:
- `evaluation_summary.tsv` — Aggregate metrics
- `per_cell_evaluation.tsv` — Per-cell scores
- `ontology_analysis_report.txt` — Full text report
- `ontology_distance_analysis.png` — Visualization

---

## Workflow B: Evaluate External Predictions

You have predictions from a custom tool or biologist annotations with readable names.

### Step 1: Map readable names → CL IDs

```bash
python3 annotate_cl_terms.py \
  --obo reference_data/cl.obo \
  --input biologist_predictions.tsv \
  --name_col predicted_type \
  --output annotated.tsv
```

**Input:** TSV with a column of readable cell type names (e.g., "oligodendrocyte", "T-cell").

**Matching logic:**
1. Exact match (case-insensitive)
2. Synonym match from OBO file
3. Fuzzy normalization (strips hyphens/underscores/plurals)
4. LLM-assisted fallback (Claude + OpenAI consensus, with interactive prompt if they disagree)

**Output:** Original TSV + added `cell_type_ontology_term_id` column.

### Step 2: Evaluate

```bash
python3 evaluate.py \
  --predictions annotated.tsv \
  --pred_id_col cell_type_ontology_term_id \
  --ground_truth ground_truth.tsv \
  --obo reference_data/cl.obo \
  --output-dir evaluation_results/
```

**Note:** Use `--pred_id_col` and `--truth_id_col` to specify custom column names if your files don't use the defaults.

---

## Workflow C: Predict Only (No Ground Truth)

If you don't have ground truth labels (e.g., predicting on unlabeled data):

```bash
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings unlabeled_data.npy \
  --output predictions.tsv
```

**No evaluation step needed.** The predictions file contains confidence scores (`vote_percentage`) and neighbor distances.

---

## Understanding Ontology Metrics

### IC Similarity (default, `--ontology-method ic`)
- **Range:** 0.0 to 1.0 (higher is better)
- **1.0** = perfect match (predicted == ground truth)
- **High (>0.8)** = semantically very close (e.g., predicted "motor neuron" when truth is "neuron")
- **Low (<0.3)** = semantically distant (e.g., predicted "hepatocyte" when truth is "neuron")
- **0.0** = no common ancestor in ontology

### Shortest Path Distance (`--ontology-method shortest_path`)
- **Range:** 0 to N edges (lower is better)
- **0** = perfect match
- **Low (1-2)** = close neighbor in ontology tree
- **High (>5)** = far from true type in hierarchy

**Why use ontology metrics instead of exact match?**

Exact CL term match rate treats all mistakes equally. If your model predicts "motor neuron" (CL:0000100) instead of "neuron" (CL:0000540), exact match scores this as 0% correct — even though motor neuron *is a* neuron and they're highly related in the Cell Ontology hierarchy.

Ontology-based metrics capture semantic closeness, giving partial credit for "nearby" predictions.

---

## Common Arguments

### `predict.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--index` | yes | — | FAISS index file (.faiss) |
| `--ref_annot` | yes | — | Reference annotation TSV |
| `--obo` | yes | — | Cell Ontology OBO file |
| `--embeddings` | yes | — | Test embeddings (.npy) |
| `--metadata` | no | none | Test metadata TSV. All columns will be included in output. |
| `--k` | no | 30 | Number of nearest neighbors |
| `--output` | no | `predictions.tsv` | Output TSV path |

### `evaluate.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--predictions` | yes | — | Predictions TSV |
| `--ground_truth` | yes | — | Ground truth TSV |
| `--obo` | yes | — | Cell Ontology OBO file |
| `--ontology-method` | no | `ic` | `ic` or `shortest_path` |
| `--pred_id_col` | no | `predicted_cell_type_ontology_term_id` | CL term ID column in predictions |
| `--truth_id_col` | no | `cell_type_ontology_term_id` | CL term ID column in ground truth |
| `--output-dir` | no | `evaluation_results/` | Output directory |

### `annotate_cl_terms.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--obo` | yes | — | Cell Ontology OBO file |
| `--input` | yes | — | Input TSV with readable names |
| `--output` | yes | — | Output TSV (input + added CL ID column) |
| `--name_col` | no | `cell_type` | Column with readable cell type names |

**LLM fallback:** Requires `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` environment variables for unmatched names.

---

## Next Steps

- See [USAGE.md](USAGE.md) for detailed examples
- See [API.md](API.md) for programmatic usage
- See [../tests/README.md](../tests/README.md) for running tests
