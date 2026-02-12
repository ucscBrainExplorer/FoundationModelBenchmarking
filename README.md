# Universal Cell Embedding (UCE) Benchmarking

A modular Python framework for benchmarking K-Nearest Neighbors (KNN) model accuracy in cell type prediction using foundation models like Universal Cell Embedding (UCE) and SCimilarity.

## Overview

This project benchmarks cell type prediction accuracy by comparing:
- **Foundation models**: UCE, SCimilarity, etc.
- **FAISS index types**: IVF Flat, IVF PQ
- **Distance metrics**: Euclidean
- **Test datasets**: Organoids, post-mortem adult brain, etc.

The evaluation uses **ontology-aware metrics** based on Cell Ontology (CL) graph structures to measure how semantically "close" predictions are to ground truth in the biological hierarchy.

---

## Three Standalone Programs

The benchmarking suite provides three composable programs for flexible workflows:

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

- **[Quick Start Guide](docs/QUICK_START.md)** — Get started quickly with all three programs
- **[Detailed Usage](docs/USAGE.md)** — All options, examples, troubleshooting
- **[API Reference](docs/API.md)** — Programmatic usage

---

## Ontology-Aware Metrics

Two methods are available, selectable via `--ontology-method`:

### IC-based Lin Similarity (default, `--ontology-method ic`)

Uses Information Content (IC) to measure semantic similarity between predicted and ground truth cell types.

| Score | Meaning |
|-------|---------|
| **1.0** | Identical terms (perfect prediction) |
| **> 0.8** | Closely related (e.g., neuron subtypes) |
| **0.4 - 0.7** | Moderately related (e.g., neuron vs. astrocyte) |
| **~0.0** | Unrelated (e.g., near the ontology root) |

**Higher similarity = better prediction.** A prediction of "Purkinje neuron" when the truth is "cerebellar granule cell" will score higher than a prediction of "erythrocyte", reflecting the biological relatedness of the cell types.

IC values are precomputed using the Zhou (2008) weighted intrinsic IC formula, which blends descendant count with structural depth. Similarity between two terms is computed using Lin (1998): `Sim(A,B) = 2 * IC(MICA) / (IC(A) + IC(B))`, where MICA is the Most Informative Common Ancestor.

### Shortest-Path Distance (`--ontology-method shortest_path`)

Computes the shortest undirected path between two terms in the ontology graph.

**Lower distance = better prediction.** A distance of 0 means an exact match; larger values mean the terms are farther apart in the hierarchy.

Note: on DAGs with multiple inheritance (the Cell Ontology has 33.5% multi-parent terms), shortest undirected path can shortcut across separate branches, potentially underestimating true semantic distance. The IC method is recommended for this reason. See `IC_FORMULA_ANALYSIS.md` for detailed evaluation.

---

## Project Structure

```
FoundationModelBenchmarking/
├── predict.py                     # NEW: Standalone prediction
├── evaluate.py                    # NEW: Standalone evaluation
├── annotate_cl_terms.py           # NEW: Standalone annotation
├── obo_parser.py                  # NEW: OBO file parsing
│
├── data_loader.py                 # Data ingestion, S3 download/upload
├── prediction_module.py           # KNN search and voting logic
├── ontology_utils.py              # Cell Ontology: IC similarity + path distance
├── analyze_ontology_results.py    # Statistical analysis and reporting
├── visualization.py               # UMAP plots and confusion matrices
│
├── utility/
│   └── normalize_cell_types.py    # Cell type name normalization
│
├── tests/                         # Modern pytest suite
│   ├── conftest.py
│   ├── test_obo_parser.py
│   ├── test_predict.py
│   ├── test_evaluate.py
│   └── test_annotate_cl_terms.py
│
├── unit-tests/                    # Legacy test suite
│   ├── test_ontology_utils.py     # 32 tests (IC similarity, path distance, scoring)
│   ├── test_prediction_module.py  # 10 tests (requires faiss)
│   └── test_data_loader.py        # 8 tests (requires faiss)
│
├── docs/                          # Documentation
│   ├── QUICK_START.md             # Quick start guide
│   ├── USAGE.md                   # Detailed usage with examples
│   └── API.md                     # Programmatic API reference
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container definition
├── BUGFIX_NOTES.md                # Documents 6 bug fixes with lessons
├── IC_FORMULA_ANALYSIS.md         # IC formula comparison and analysis
├── CHANGELOG.md                   # Commit-level change log
├── IMPLEMENTATION_SUMMARY.md      # Refactoring summary
└── README.md                      # This file
```

## Input File Formats

| File | Format | Key Details |
|------|--------|-------------|
| FAISS index (`*.faiss`) | Binary FAISS index | IVF Flat, IVF PQ, or any FAISS type |
| Reference annotations (`prediction_obs.tsv`) | TSV | Required: `cell_type_ontology_term_id`. Optional: `cell_type`. Row order must match FAISS index. |
| Test embeddings (`{dataset_id}_*.npy`) | NumPy binary | Shape: `(n_cells, embedding_dim)`, float32/64. Matched to metadata by shared `{dataset_id}` prefix. |
| Test metadata (`{dataset_id}_prediction_obs.tsv`) | TSV | Required: `cell_type_ontology_term_id`. Optional: `cell_type`, `cell_id`. The `{dataset_id}` is everything before `_prediction_obs.tsv`. |
| Cell Ontology (`cl.obo`) | OBO | Source: [OBO Foundry](http://obofoundry.org/ontology/cl.html) |

## Local Development

```bash
git clone <repository-url>
cd FoundationModelBenchmarking
pip install -r requirements.txt    # or: conda install -c conda-forge faiss-cpu

# Run new programs
python3 predict.py --help
python3 evaluate.py --help
python3 annotate_cl_terms.py --help

# Run tests
pip install pytest pytest-cov
pytest tests/ -v
pytest unit-tests/ -v  # Legacy test suite
```

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Cell Ontology](http://obofoundry.org/ontology/cl.html)
- [Universal Cell Embeddings](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)
- [CELLxGENE](https://cellxgene.cziscience.com/)
- Lin, D. (1998). An Information-Theoretic Definition of Similarity. In *Proceedings of the 15th International Conference on Machine Learning (ICML 1998)*, Vol. 98, pp. 296-304.
- Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content for Semantic Similarity in WordNet. In *2008 Second International Conference on Future Generation Communication and Networking Symposia*, Hainan, China, pp. 85-89. doi: 10.1109/FGCNS.2008.16.
