# UCE Benchmarking

A modular Python framework for benchmarking KNN cell type prediction using foundation models (UCE, SCimilarity). Evaluation uses **ontology-aware metrics** based on the Cell Ontology (CL) graph — measuring how semantically close predictions are to ground truth, not just whether they match exactly.

---

## Installation

```bash
git clone <repository-url>
cd FoundationModelBenchmarking
pip install -r requirements.txt
```

**Dependencies:** `faiss-cpu`, `numpy`, `pandas`, `networkx`, `pronto`, `scikit-learn`, `matplotlib`, `seaborn`, `umap-learn`

---

## Programs

The suite has four composable CLI programs:

| Program | Purpose |
|---|---|
| `predict.py` | Predict cell types via FAISS KNN (no ground truth needed) |
| `evaluate.py` | Score predictions against ground truth using ontology metrics |
| `annotate_cl_terms.py` | Map readable cell type names to CL ontology term IDs |

---

## Workflows

### Workflow A: Predict + Evaluate

```bash
# Step 1: Predict
python3 predict.py \
  --index indices/index_ivfflat.faiss \
  --ref_annot reference_data/prediction_obs.tsv \
  --obo reference_data/cl.obo \
  --embeddings test_data/dataset.npy \
  --metadata test_data/dataset_metadata.tsv \
  --output predictions.tsv

# Step 2: Evaluate
python3 evaluate.py \
  --predictions predictions.tsv \
  --ground_truth test_data/ground_truth.tsv \
  --obo reference_data/cl.obo \
  --ontology-method ic \
  --output-dir evaluation_results/
```

### Workflow B: Evaluate External Predictions

When predictions come from a third-party tool or biologist annotations with readable names:

```bash
# Map readable names → CL IDs
python3 annotate_cl_terms.py \
  --obo reference_data/cl.obo \
  --input external_predictions.tsv \
  --name_col cell_type \
  --output annotated.tsv

# Evaluate
python3 evaluate.py \
  --predictions annotated.tsv \
  --ground_truth ground_truth.tsv \
  --obo reference_data/cl.obo \
  --output-dir evaluation_results/
```

---

## Program Reference

### `predict.py`

Predicts cell types by querying a FAISS index and voting among the k nearest neighbors.

```
--index         FAISS index file (.faiss)                                        [required]
--ref_annot     Reference annotation TSV (cell_type_ontology_term_id required)   [required]
--obo           Cell Ontology OBO file                                           [required]
--embeddings    Test embeddings (.npy), shape (n_cells, dim)                     [required]
--metadata      Optional metadata TSV; all columns carried through to output
--method        majority_voting | distance_weighted_knn | both  [default: distance_weighted_knn]
--k             Number of nearest neighbors                      [default: 30]
--output        Output TSV path                                  [default: predictions.tsv]
```

**Output columns** (`distance_weighted_knn`):

| Column | Description |
|---|---|
| `cell_id` | From metadata if provided, otherwise `row_N` |
| `weighted_cell_type_ontology_term_id` | Predicted CL term ID |
| `weighted_cell_type` | Canonical name from OBO |
| `weighted_score` | Weight fraction for the winner (0–1) |
| `mean_euclidean_distance` | Mean distance to all k neighbors |
| `std_euclidean_distance` | Std of distances to all k neighbors |
| `neighbor_distances` | Comma-separated distances, closest first |
| `neighbor_cell_types` | Comma-separated neighbor names, same order |

With `--method majority_voting`, columns are prefixed `mv_` instead of `weighted_`. With `--method both`, all columns appear.

---

### `evaluate.py`

Evaluates predictions against ground truth row-by-row. Both files must have the same number of rows in the same order.

```
--predictions       Predictions TSV                                [required]
--ground_truth      Ground truth TSV                               [required]
--obo               Cell Ontology OBO file                         [required]
--ontology-method   ic | shortest_path                             [default: ic]
--pred_id_col       CL term ID column in predictions file          [default: predicted_cell_type_ontology_term_id]
--truth_id_col      CL term ID column in ground truth file         [default: cell_type_ontology_term_id]
--remap-file        Remap TSV                                        [optional]
--output-dir        Output directory                               [default: evaluation_results/]
```

**Auto-detection:** If the specified column is not found, `evaluate.py` scans all columns and picks the one with the highest fraction of CL term IDs (>90%) or resolvable cell type names (>50%). Readable names are auto-resolved to CL IDs via exact match, synonym match, and fuzzy normalization before scoring.

**Output files:**

| File | Contents |
|---|---|
| `evaluation_summary.tsv` | Mean, median, std ontology score; exact match rate; total cells |
| `per_cell_evaluation.tsv` | Per-cell predicted/truth CL IDs, names, ontology score, exact match flag |
| `ontology_analysis_report.txt` | Full text report: metric guide, distribution, correlation with exact match |
| `ontology_similarity_analysis.png` | Scatter (accuracy vs. score) + histogram of score distribution |

---

### `annotate_cl_terms.py`

Maps readable cell type names to CL ontology term IDs using a cascading match strategy.

```
--obo        Cell Ontology OBO file        [required]
--input      Input TSV with readable names [required]
--output     Output TSV path               [required]
--name_col   Column with cell type names   [default: cell_type]
```

**Matching cascade:**
1. Exact match (case-insensitive)
2. OBO synonym match
3. Fuzzy normalization (strips hyphens, underscores, trailing 's')
4. LLM fallback via Claude and/or OpenAI — auto-accepts if both agree; prompts interactively if they disagree

Set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` for LLM fallback. Without API keys, unmatched names are left empty.

**Output:** Original TSV with an added `cell_type_ontology_term_id` column.

---

---

## Ontology Metrics

### IC Similarity (`--ontology-method ic`, default)

Uses Lin (1998) semantic similarity with Zhou (2008) intrinsic Information Content:

```
Sim(A, B) = 2 * IC(MICA) / (IC(A) + IC(B))
```

where MICA is the Most Informative Common Ancestor. IC values blend descendant count (Seco) with structural depth at k=0.5.

| Score | Meaning |
|---|---|
| 1.0 | Identical terms (exact match) |
| > 0.8 | Closely related (e.g., neuron subtypes) |
| 0.4 – 0.7 | Moderately related (e.g., neuron vs. astrocyte) |
| ~ 0.0 | Unrelated or no common ancestor |

Higher is better.

### Shortest-Path Distance (`--ontology-method shortest_path`)

Shortest undirected path (in edges) between two terms in the ontology graph. Lower is better; 0 = exact match.

**Why not just use exact match?** Exact match treats all errors equally. A prediction of "motor neuron" when the truth is "neuron" gets the same zero score as a prediction of "erythrocyte". Ontology metrics give partial credit for biologically related predictions.

**Why IC over shortest path?** The Cell Ontology has 33.5% multi-parent terms (DAG, not tree). Shortest undirected path can shortcut across unrelated branches, underestimating true semantic distance. IC-based similarity handles DAGs correctly via the MICA. See `IC_FORMULA_ANALYSIS.md` for a detailed comparison.

---

## Supporting Tools

### `ic_lookup.py` — Interactive IC lookup

Look up IC similarity and shortest-path distance between any two cell types:

```bash
python3 ic_lookup.py "neural stem cell" "forebrain radial glial cell" --obo cl.obo
python3 ic_lookup.py CL:0000047 CL:0013000 --obo cl.obo
```

Prints IC values, Lin similarity, path distance, MICA, and common ancestors.

### `background_distances.py` — Euclidean distance baseline

Compute a background distribution of random-pair Euclidean distances to contextualize KNN neighbor distances:

```bash
python3 background_distances.py \
  --embeddings reference.npy \
  --predictions predictions.tsv \
  --output background_vs_knn.png
```

Also supports a three-distribution plot (null / query KNN / reference self-KNN) when `--query` and `--index` are provided.

### `background_ic.py` — IC similarity baseline

Compute a background distribution of random-pair IC similarities to contextualize evaluation scores:

```bash
python3 background_ic.py \
  --ref-annot reference_data/prediction_obs.tsv \
  --obo cl.obo \
  --evaluation evaluation_results/per_cell_evaluation.tsv \
  --output background_vs_prediction_ic.png
```

### `cell_labelling/` — Standalone cell labelling tool

A self-contained tool for labelling cells against a UCE reference index without requiring ground truth. See [`cell_labelling/README.md`](cell_labelling/README.md).

---

## Input File Formats

### Embeddings (`.npy`)

NumPy array, shape `(n_cells, embedding_dim)`, dtype `float32` or `float64`.

- UCE embeddings: 1280-dimensional
- SCimilarity embeddings: 128-dimensional

The FAISS index and embeddings must have the same dimension.

### Reference Annotations (TSV)

Required column: `cell_type_ontology_term_id` (CL term IDs, e.g. `CL:0000540`).

Row order must match the FAISS index (row 0 = vector 0, etc.).

### OBO File

Cell Ontology in OBO format. Download:
- Full: http://purl.obolibrary.org/obo/cl.obo
- Basic subset: http://purl.obolibrary.org/obo/cl/cl-basic.obo

---

## Project Structure

```
FoundationModelBenchmarking/
├── predict.py                     # KNN prediction
├── evaluate.py                    # Ontology-based evaluation
├── annotate_cl_terms.py           # Name → CL ID annotation
│
├── data_loader.py                 # FAISS + annotation loading
├── prediction_module.py           # KNN voting logic
├── ontology_utils.py              # IC similarity + path distance
├── obo_parser.py                  # OBO file parsing
├── analyze_ontology_results.py    # Statistics and reporting
├── visualization.py               # UMAP plots and confusion matrices
│
├── ic_lookup.py                   # Interactive IC/distance lookup
├── background_distances.py        # Euclidean distance baseline tool
├── background_ic.py               # IC similarity baseline tool
│
├── cell_labelling/                # Standalone cell labelling tool
│   ├── predict.py
│   ├── distance_analysis.py
│   └── README.md
│
├── utility/
│   └── normalize_cell_types.py
│
├── tests/                         # pytest suite
├── unit-tests/                    # Legacy unittest suite
│
├── docs/
│   └── API.md
│
├── requirements.txt
├── IC_FORMULA_ANALYSIS.md         # IC formula evaluation and rationale
├── BUGFIX_NOTES.md
└── CHANGELOG.md
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v          # main test suite
pytest unit-tests/ -v     # legacy test suite
```

---

## References

**Ontology similarity — foundational methods**

- Resnik, P. (1995). Using Information Content to Evaluate Semantic Similarity in a Taxonomy. *IJCAI 1995*, pp. 448–453. — Introduced Information Content as a measure of term specificity in ontologies; the root of the IC-based similarity family.
- Lin, D. (1998). An Information-Theoretic Definition of Similarity. *ICML 1998*, pp. 296–304. — Defined the Lin similarity formula `2·IC(MICA) / (IC(A) + IC(B))` used here to score predictions against ground truth.
- Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content for Semantic Similarity in WordNet. *FGCNS 2008*, pp. 85–89. doi:10.1109/FGCNS.2008.16. — Source of the blended IC formula (descendant count + depth) used to precompute IC values on the Cell Ontology.

**IC similarity applied to biological ontologies**

- Lord, P.W., Stevens, R.D., Brass, A., & Goble, C.A. (2003). Investigating semantic similarity measures across the Gene Ontology: the relationship between sequence and annotation. *Bioinformatics*, 19(10), 1275–1283. — First demonstration that IC-based semantic similarity of ontology annotations correlates with biological similarity (sequence identity), establishing that this approach is meaningful for evaluating cell type predictions.
- Pesquita, C., Faria, D., Falcão, A.O., Lord, P., & Couto, F.M. (2009). Semantic Similarity in Biomedical Ontologies. *PLOS Computational Biology*, 5(7), e1000443. — Comprehensive review of IC and graph-based semantic similarity across biomedical ontologies including GO; the primary reference for using these metrics to evaluate biological annotations.
- Schlicker, A., Domingues, F.S., Rahnenführer, J., & Lengauer, T. (2006). A new measure for functional similarity of gene products based on Gene Ontology. *BMC Bioinformatics*, 7, 302. — Applied Lin similarity to GO to compare gene function; the direct precedent for using the same metric to compare cell type annotations.

**Data and tools**

- [Cell Ontology (OBO Foundry)](http://obofoundry.org/ontology/cl.html)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Universal Cell Embeddings (UCE)](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)
- [CELLxGENE](https://cellxgene.cziscience.com/)
