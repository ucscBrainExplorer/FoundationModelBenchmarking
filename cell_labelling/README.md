# cell_labelling

Standalone tools for assigning cell type labels to query cells using KNN search
against a UCE reference atlas.

## Programs

| Script | Purpose |
|---|---|
| `predict.py` | Assign cell type labels to query cells |
| `join_predictions.py` | Join predicted labels with query h5ad obs table |
| `distance_analysis.py` | Assess mapping quality via distance distributions |

## Inputs

| File | Description |
|---|---|
| `*.faiss` | FAISS index of reference UCE embeddings |
| `*_uce_adata.h5ad` | Query dataset; embeddings in `adata.obsm["X_uce"]` (or use `--npy`/`--obs`) |
| `ref_annot.tsv(.gz)` | Reference annotation TSV aligned to the FAISS index; all columns predicted |

Demo data for all inputs is in `demodata/`.

---

## predict.py

Assigns cell type labels by querying a FAISS index and voting among the k nearest
reference neighbors. Outputs top-1 and top-2 predictions with confidence scores per cell.

All columns in the reference TSV are predicted except those ending with `_term_id`.
Column order in the output follows the ref TSV column order.

### Usage

```bash
# with h5ad input
python3 predict.py \
  --index     demodata/index_ivfflat.faiss \
  --adata     demodata/query_uce_adata.h5ad \
  --ref_annot demodata/ref_obs.tsv.gz \
  --method    enrichment_weighted_knn \
  --output    labels.tsv

# with npy + obs input
python3 predict.py \
  --index     demodata/index_ivfflat.faiss \
  --npy       query_uce.npy \
  --obs       query_obs.tsv \
  --ref_annot ref_annot.tsv \
  --method    enrichment_weighted_knn \
  --output    labels.tsv
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--index` | yes | — | FAISS index file (`.faiss`) |
| `--adata` | one of | — | Query h5ad file with `adata.obsm["X_uce"]` |
| `--npy` | one of | — | Query UCE embeddings `.npy` file (requires `--obs`) |
| `--obs` | with `--npy` | — | Query obs metadata TSV (first column = cell index) |
| `--ref_annot` | yes | — | Reference annotation TSV aligned to the FAISS index |
| `--method` | no | `distance_weighted_knn` | Voting method (see below) |
| `--k` | no | `30` | Number of nearest neighbors |
| `--output` | no | `labels.tsv` | Output TSV path |

### Voting methods

Three methods are available:

#### `enrichment_weighted_knn` (recommended for cross mapping)

Weights each neighbor label by how enriched it is among the k neighbors relative
to its abundance in the reference. Corrects for reference composition bias — abundant
cell types no longer dominate predictions just because they are common in the reference.

Use this method whenever the query and reference are expected to be significantly
different: human-mouse mapping, scRNAseq-MERFISH mapping, developing tissue-organoid
mapping, developing brain-adult brain mapping, cross-assay mapping.

**Algorithm** for each query cell:
1. Count observed label frequency among k neighbors
2. Compute enrichment per label:
   - `observed_frac = count / k`
   - `expected_frac = label_count_in_ref / total_ref`
   - `enrichment = observed_frac / expected_frac`
3. Score per label: `score = count × log2(enrichment)`, floored at 0
4. Winner = label with highest score; normalized score = `score[winner] / sum(all scores)`

**Design decisions:**
- **log2 compression:** Raw enrichment can reach thousands-fold for rare types.
  `log2` compresses this so a single hit of a 5000x-enriched type doesn't dominate
  10 hits of a 50x-enriched type. Score balances frequency (count) and enrichment (log2).
- **Floor at 0:** Labels with enrichment < 1 (depleted relative to reference) are
  ignored rather than penalized. Without this floor, negative scores would cause
  normalized scores to exceed 1.0. A depleted label contributes nothing to the vote.
- **When to use:** Cross mapping (see method header). Validated on human BG → mouse
  Zhuang ABCA-1 where Oligo (17% of reference) dominated distance-weighted predictions
  for primate-specific cell types with no mouse equivalent; enrichment weighting
  correctly surfaced biologically meaningful hindbrain/GABA analogs instead.

#### `distance_weighted_knn`

Neighbors weighted by a Gaussian kernel of Euclidean distance. Closer neighbors
count more. sigma is set per-row to the median neighbor distance.

`w = exp(-d² / (2σ²))`

**When to use:** Query and reference from the same species/tissue with similar
cell type composition. Fast and reliable when reference coverage is good.

#### `majority_voting`

Each neighbor gets one equal vote regardless of distance. Winner = most common label.

**When to use:** Quick baseline. Less accurate than distance-weighted in most settings.

### Output columns

For each column in the reference annotation TSV, four prediction columns are written:

| Column | Description |
|---|---|
| `prediction_by_{col}_top1` | Top-1 predicted label |
| `prediction_by_{col}_top1_score` | Normalized confidence score for top-1 (0–1) |
| `prediction_by_{col}_top2` | Top-2 predicted label |
| `prediction_by_{col}_top2_score` | Normalized confidence score for top-2 (0–1) |

Additional columns always present:

| Column | Description |
|---|---|
| `cell_id` | Cell identifier from obs index |
| `prediction_mean_euclidean_distance` | Mean Euclidean distance to all k neighbors |

The output file includes a `#`-prefixed provenance header tracking inputs and parameters.

### Choosing a reference annotation file

The `--ref_annot` TSV must be row-aligned to the FAISS index (row i = index vector i).
Use a file with only the biological columns you want to predict — avoid coordinate
columns, color columns, or other metadata that isn't meaningful as a cell label.

Example: `Zhuang-ABCA-1_shuffled/prediction.tsv` contains only
`class, subclass, supertype, cluster, division, structure, substructure` (7 columns,
2.49M rows aligned to the index).

### Interpreting scores

- **High score + low distance:** confident, well-matched prediction
- **High score + high distance:** label consistently dominates neighbors but cell is
  far from all reference cells — may indicate missing reference coverage
- **Low score:** neighbors are scattered across many labels — ambiguous prediction
- **enrichment_weighted_knn score vs distance_weighted_knn score:** not directly
  comparable; enrichment scores reflect how much a label stands out above chance,
  not proximity

### Known limitations

- **Primate-specific cell types** (e.g. CN ONECUT1 GABA, CN GABA-Glut from CCN20250428)
  have no mouse equivalent. Even enrichment weighting cannot produce a correct label
  against a mouse reference — it will surface the closest mouse transcriptional analog
  instead, which is biologically interpretable but not a true match.
- **Small query populations** (< ~20 cells): enrichment weighting may still be
  dominated by abundant types due to insufficient neighbor sampling.

---

## join_predictions.py

Joins the `labels.tsv` output from `predict.py` with the query h5ad obs table,
producing a single TSV with all metadata and predictions together.

### Usage

```bash
python3 join_predictions.py \
  --labels  labels.tsv \
  --adata   query_uce_adata.h5ad \
  --output  predictions_with_obs.tsv
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--labels` | yes | — | `labels.tsv` output from `predict.py` |
| `--adata` | yes | — | Query h5ad file (obs table joined on `cell_id`) |
| `--output` | no | `predictions_with_obs.tsv` | Output TSV path |

---

## distance_analysis.py

Assesses how well query cells map into the reference by comparing three distance
distributions and generating a density plot.

| Distribution | Role | Description |
|---|---|---|
| Random query → reference | Negative control (null) | Distances if neighbors were picked randomly |
| Query KNN | Measurement | Actual mean distances from `predict.py` |
| Reference self-KNN | Positive control | How close reference cells are to their own neighbors |

A well-mapped query dataset should sit clearly between null and reference self-KNN.

### Usage

```bash
python3 distance_analysis.py \
  --labels    labels.tsv \
  --adata     demodata/query_uce_adata.h5ad \
  --index     demodata/index_ivfflat.faiss
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--labels` | yes | — | `labels.tsv` output from `predict.py` |
| `--adata` | yes | — | Query h5ad file with `adata.obsm["X_uce"]` |
| `--index` | yes | — | FAISS index file (`.faiss`) |
| `--k` | no | `30` | Neighbors used in `predict.py` |
| `--n_sample` | no | `10000` | Cells sampled for null and reference self-KNN |
| `--seed` | no | `42` | Random seed |
| `--output` | no | `distance_analysis.png` | Output PNG path |

---

## Quick start with demo data

```bash
python3 predict.py \
  --index     demodata/index_ivfflat.faiss \
  --adata     demodata/query_uce_adata.h5ad \
  --ref_annot demodata/ref_obs.tsv.gz

python3 distance_analysis.py \
  --labels  labels.tsv \
  --adata   demodata/query_uce_adata.h5ad \
  --index   demodata/index_ivfflat.faiss
```

Output: `labels.tsv` and `distance_analysis.png` in the current directory.

---

## Typical workflow

```bash
# Step 1 — label cells
python3 predict.py \
  --index     demodata/index_ivfflat.faiss \
  --npy       query_uce.npy \
  --obs       query_obs.tsv \
  --ref_annot ref_annot.tsv \
  --method    enrichment_weighted_knn \
  --output    labels.tsv

# Step 2 — join predictions with query obs
python3 join_predictions.py \
  --labels  labels.tsv \
  --adata   demodata/query_uce_adata.h5ad \
  --output  predictions_with_obs.tsv

# Step 3 — assess mapping quality
python3 distance_analysis.py \
  --labels  labels.tsv \
  --adata   demodata/query_uce_adata.h5ad \
  --index   demodata/index_ivfflat.faiss \
  --output  distance_analysis.png
```

## Dependencies

```
numpy
pandas
anndata
faiss-cpu
matplotlib
```
