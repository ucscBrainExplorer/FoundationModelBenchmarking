# cell_labelling

Standalone tools for assigning cell type labels to query cells using KNN search
against a UCE reference atlas.

## Programs

| Script | Purpose |
|---|---|
| `predict.py` | Assign cell type labels to query cells |
| `distance_analysis.py` | Assess mapping quality via distance distributions |

## Inputs

| File | Description |
|---|---|
| `*.faiss` | FAISS index of reference UCE embeddings |
| `*_uce_adata.h5ad` | Query dataset; embeddings in `adata.obsm["X_uce"]` |
| `ref_obs.tsv(.gz)` | Reference metadata TSV; must contain at least one recognized label column (see below) |

Demo data for all inputs is in `demodata/`.

---

## predict.py

Assigns cell type labels by querying a FAISS index and voting among the k nearest
reference neighbors. Outputs top-1 and top-2 predictions per cell.

All recognized label columns present in the reference TSV are predicted in a single
run. Output columns are prefixed by the source column name.

### Recognized label columns (predicted in this order)

```
harmonized_cell_label
harmonized_cell_type
mapped_cell_label
mapped_cell_type
cell_type_original
cell_label
cell_type
```

### Usage

```bash
python3 predict.py \
  --index     demodata/index_ivfflat.faiss \
  --adata     demodata/query_uce_adata.h5ad \
  --ref_annot demodata/ref_obs.tsv.gz \
  --method    distance_weighted_knn \
  --output    labels.tsv
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--index` | yes | — | FAISS index file (`.faiss`) |
| `--adata` | yes | — | Query h5ad file with `adata.obsm["X_uce"]` |
| `--ref_annot` | yes | — | Reference metadata TSV with at least one recognized label column |
| `--method` | no | `distance_weighted_knn` | Voting method (see below) |
| `--k` | no | `30` | Number of nearest neighbors |
| `--output` | no | `labels.tsv` | Output TSV path |

### Voting methods

| Method | Description |
|---|---|
| `distance_weighted_knn` | Neighbors weighted by Gaussian kernel of distance; closer neighbors count more |
| `majority_voting` | Each neighbor gets one equal vote |

`distance_weighted_knn` is the default and recommended method.

### Output columns

For each recognized label column found in the reference TSV, four columns are added
to the output (example for `harmonized_cell_label` with `distance_weighted_knn`):

| Column | Description |
|---|---|
| `harmonized_cell_label_pred` | Top-1 predicted label |
| `harmonized_cell_label_weighted_score` | Normalized weight fraction for top-1 (0–1) |
| `harmonized_cell_label_pred_2` | Top-2 predicted label |
| `harmonized_cell_label_weighted_score_2` | Normalized weight fraction for top-2 |

For `majority_voting`, score columns are named `{col}_score` / `{col}_score_2`
(no `weighted` prefix).

Additional columns always present:

| Column | Description |
|---|---|
| `cell_id` | Cell barcode from `adata.obs_names` |
| `mean_euclidean_distance` | Mean distance to all k neighbors |

The output file includes a `#`-prefixed provenance header tracking inputs and parameters.

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
  --adata     demodata/query_uce_adata.h5ad \
  --ref_annot demodata/ref_obs.tsv.gz \
  --output    labels.tsv

# Step 2 — assess mapping quality
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
