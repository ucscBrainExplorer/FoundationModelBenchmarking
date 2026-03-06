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
| `ref_obs.tsv(.gz)` | Reference metadata TSV; must contain `cell_type_ontology_term_id` |
| `cl.obo` | Cell Ontology OBO file for translating CL IDs to readable names |

Demo data for all inputs is in `demodata/`.

---

## predict.py

Assigns cell type labels by querying a FAISS index and voting among the k nearest
reference neighbors. Outputs top-1 and top-2 predictions per cell.

### Usage

```bash
python3 predict.py \
  --index     demodata/index_ivfflat.faiss \
  --adata     demodata/query_uce_adata.h5ad \
  --ref_annot demodata/ref_obs.tsv.gz \
  --obo       demodata/cl-basic.obo \
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--index` | yes | — | FAISS index file (`.faiss`) |
| `--adata` | yes | — | Query h5ad file with `adata.obsm["X_uce"]` |
| `--ref_annot` | yes | — | Reference metadata TSV (needs `cell_type_ontology_term_id`) |
| `--obo` | yes | — | Cell Ontology OBO file (`cl.obo`) |
| `--method` | no | `distance_weighted_knn` | Voting method (see below) |
| `--k` | no | `30` | Number of nearest neighbors |
| `--output` | no | `labels.tsv` | Output TSV path |

### Voting methods

| Method | Description |
|---|---|
| `distance_weighted_knn` | Neighbors weighted by Gaussian kernel of distance; closer neighbors count more |
| `majority_voting` | Each neighbor gets one equal vote |
| `both` | Run both methods and output all columns side by side |

`distance_weighted_knn` is the default and recommended method.

### Output columns

**Both methods** always produce:

| Column | Description |
|---|---|
| `cell_id` | Cell barcode from `adata.obs_names` |
| `mean_euclidean_distance` | Mean distance to all k neighbors |

**`distance_weighted_knn`** columns (prefix `weighted_`):

| Column | Description |
|---|---|
| `weighted_cell_type_ontology_term_id` | Top-1 predicted CL term ID |
| `weighted_cell_type` | Top-1 predicted cell type (human-readable) |
| `weighted_score` | Normalized weight fraction for top-1 (0–1) |
| `weighted_cell_type_ontology_term_id_2` | Top-2 predicted CL term ID |
| `weighted_cell_type_2` | Top-2 predicted cell type |
| `weighted_score_2` | Normalized weight fraction for top-2 |

**`majority_voting`** columns (prefix `mv_`):

| Column | Description |
|---|---|
| `mv_cell_type_ontology_term_id` | Top-1 predicted CL term ID |
| `mv_cell_type` | Top-1 predicted cell type (human-readable) |
| `mv_score` | Vote fraction for top-1 (0–1) |
| `mv_cell_type_ontology_term_id_2` | Top-2 predicted CL term ID |
| `mv_cell_type_2` | Top-2 predicted cell type |
| `mv_score_2` | Vote fraction for top-2 |

The output file includes a `#`-prefixed provenance header tracking all inputs and
parameters used.

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
  --index     demodata/index_ivfflat.faiss \
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
  --ref_annot demodata/ref_obs.tsv.gz \
  --obo       demodata/cl-basic.obo

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
  --obo       demodata/cl-basic.obo \
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
