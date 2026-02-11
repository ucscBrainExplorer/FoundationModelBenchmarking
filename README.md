# Universal Cell Embedding (UCE) Benchmarking

A modular Python framework for benchmarking K-Nearest Neighbors (KNN) model accuracy in cell type prediction using foundation models like Universal Cell Embedding (UCE) and SCimilarity.

## Overview

This project benchmarks cell type prediction accuracy by comparing:
- **Foundation models**: UCE, SCimilarity, etc.
- **FAISS index types**: IVF Flat, IVF PQ
- **Distance metrics**: Euclidean
- **Test datasets**: Organoids, post-mortem adult brain, etc.

The evaluation uses both standard metrics (accuracy, F1 score) and **ontology-aware metrics** based on Cell Ontology (CL) graph structures to measure how semantically "close" predictions are to ground truth in the biological hierarchy.

## Quick Start

```bash
# Deploy to Kubernetes
kubectl delete job fm-benchmark-job -n braingeneers 2>/dev/null
kubectl apply -f benchmarking-job.yaml

# Monitor
kubectl logs -n braingeneers -l job-name=fm-benchmark-job -f

# Get results from S3
aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/
aws s3 cp --recursive s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/ ./benchmark_results/{timestamp}/
```

For detailed step-by-step instructions, troubleshooting, and result retrieval options, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

## Generated Results

Each run creates a timestamped directory so previous results are never overwritten.
On the cluster, results are uploaded to S3. Locally (`--no-s3`), results are saved to `./benchmark_results/{timestamp}/`.

```
benchmark_results/{timestamp}/
├── benchmark_results.csv              Summary metrics per index/metric/dataset
├── benchmark.log                      Run log (warnings, filtered cells, timing)
├── per_cell_results/
│   └── {dataset}_{index}_{metric}_per_cell_results.csv
├── ontology_analysis/
│   └── ontology_analysis_report.txt
└── visualizations/                    (if --generate-plots)
```

---

## Project Structure

```
FoundationModelBenchmarking/
├── main_benchmark.py              # Main orchestration script
├── data_loader.py                 # Data ingestion, S3 download/upload
├── prediction_module.py           # KNN search and voting logic
├── ontology_utils.py              # Cell Ontology: IC similarity + path distance
├── evaluation_metrics.py          # Accuracy, F1, Top-k metrics
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container definition
├── benchmarking-job.yaml          # Kubernetes job configuration
├── unit-tests/                    # Test suite
│   ├── test_ontology_utils.py     # 32 tests (IC similarity, path distance, scoring)
│   ├── test_evaluation_metrics.py # 12 tests
│   ├── test_prediction_module.py  # 10 tests (requires faiss)
│   ├── test_data_loader.py        # 8 tests (requires faiss)
│   └── test_main_benchmark.py     # 11 tests (requires faiss)
├── BUGFIX_NOTES.md                # Documents 6 bug fixes with lessons
├── IC_FORMULA_ANALYSIS.md         # IC formula comparison and analysis
├── CHANGELOG.md                   # Commit-level change log
├── HOW_TO_RUN.md                  # Step-by-step operational guide
└── README.md                      # This file
```

## Architecture

### Module Responsibilities

| Module | Core Responsibility | Key Dependencies |
|--------|---------------------|------------------|
| `data_loader.py` | FAISS index loading, TSV annotations, NumPy embeddings, S3 download/upload | faiss, numpy, pandas, boto3 |
| `prediction_module.py` | KNN search (Euclidean L2), majority voting for cell type prediction | numpy, collections, faiss |
| `ontology_utils.py` | OBO file parsing, DAG construction, IC-based Lin similarity (default) and shortest-path distance | pronto, networkx |
| `evaluation_metrics.py` | Accuracy, F1 (macro & weighted), Top-k accuracy | sklearn.metrics |
| `main_benchmark.py` | Orchestrates benchmarking loops across Index Type x Metric x Dataset | All of the above |

### Pipeline

1. **Data Ingestion** — Load FAISS index, reference annotations, test embeddings (from local or S3)
2. **Prediction** — KNN search with Euclidean distance, majority voting across k neighbors
3. **Evaluation** — Standard metrics (accuracy, F1) + ontology-aware similarity/distance
4. **Output** — Per-cell CSVs, summary CSV, ontology analysis report (uploaded to S3 or saved locally)

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

Result columns: `mean_ontology_similarity`, `median_ontology_similarity`

### Shortest-Path Distance (`--ontology-method shortest_path`)

Computes the shortest undirected path between two terms in the ontology graph.

**Lower distance = better prediction.** A distance of 0 means an exact match; larger values mean the terms are farther apart in the hierarchy.

Note: on DAGs with multiple inheritance (the Cell Ontology has 33.5% multi-parent terms), shortest undirected path can shortcut across separate branches, potentially underestimating true semantic distance. The IC method is recommended for this reason. See `IC_FORMULA_ANALYSIS.md` for detailed evaluation.

Result columns: `mean_ontology_dist`, `median_ontology_dist`

## CLI Usage

```bash
# Basic local run (skip S3, results saved to ./benchmark_results/{timestamp}/)
python3 -m main_benchmark --no-s3

# Custom paths
python3 -m main_benchmark \
    --index indices/index_ivfflat.faiss \
    --test_dir test_data/ \
    --ref_annot reference_data/prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --no-s3

# With S3 (download data, run benchmark, upload results)
python3 -m main_benchmark \
    --s3_bucket latentbrain \
    --s3_prefix combined_UCE_5neuro/

# Choose ontology method
python3 -m main_benchmark --ontology-method ic              # default
python3 -m main_benchmark --ontology-method shortest_path   # original method
```

## Input File Formats

| File | Format | Key Details |
|------|--------|-------------|
| FAISS index (`*.faiss`) | Binary FAISS index | IVF Flat, IVF PQ, or any FAISS type |
| Reference annotations (`prediction_obs.tsv`) | TSV | Columns: `cell_type_ontology_term_id`, `cell_type`. Row order must match FAISS index. |
| Test embeddings (`{dataset_id}_*.npy`) | NumPy binary | Shape: `(n_cells, embedding_dim)`, float32/64. Matched to metadata by shared `{dataset_id}` prefix. |
| Test metadata (`{dataset_id}_prediction_obs.tsv`) | TSV | Columns: `cell_type_ontology_term_id`, `cell_type`. The `{dataset_id}` is everything before `_prediction_obs.tsv`. |
| Cell Ontology (`cl.obo`) | OBO | Source: [OBO Foundry](http://obofoundry.org/ontology/cl.html) |

## Output Columns

### Summary CSV (`benchmark_results.csv`)

| Column | Description |
|--------|-------------|
| Index | Index type (e.g., "ivfFlat") |
| Metric | Distance metric (e.g., "euclidean") |
| Dataset | Test dataset ID |
| accuracy | Overall prediction accuracy (0-1) |
| f1_macro | Macro-averaged F1 score |
| f1_weighted | Weighted F1 score |
| top_k_accuracy | Proportion where truth in top-k neighbors |
| mean_ontology_similarity | Mean Lin similarity (0-1, **higher = more similar**). Present with `--ontology-method ic`. |
| median_ontology_similarity | Median Lin similarity |
| mean_ontology_dist | Mean shortest-path distance (**lower = more similar**). Present with `--ontology-method shortest_path`. |
| median_ontology_dist | Median shortest-path distance |
| Avg_Query_Time_ms | Average query time per cell (ms) |

### Per-Cell CSV

| Column | Description |
|--------|-------------|
| cell_id | Cell identifier |
| true_label / true_label_readable | Ground truth (ontology ID and name) |
| prediction_label / prediction_label_readable | Prediction (ontology ID and name) |
| vote_percentage | Fraction of k neighbors that voted for the prediction |
| mean_euclidean_distance | Mean FAISS distance to all k neighbors |
| nearest_neighbor_euclidean_distance | Distance to closest neighbor |
| ontology_distance | Ontology score between prediction and truth |
| avg_neighbor_ontology_distance | Mean ontology score across all neighbors vs truth |

## Local Development

```bash
git clone <repository-url>
cd FoundationModelBenchmarking
pip install -r requirements.txt    # or: conda install -c conda-forge faiss-cpu
python3 verify_setup.py            # check dependencies
python3 -m main_benchmark --no-s3  # run locally
```

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Cell Ontology](http://obofoundry.org/ontology/cl.html)
- [Universal Cell Embeddings](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)
- [CELLxGENE](https://cellxgene.cziscience.com/)
- Lin, D. (1998). An Information-Theoretic Definition of Similarity. In *Proceedings of the 15th International Conference on Machine Learning (ICML 1998)*, Vol. 98, pp. 296-304.
- Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content for Semantic Similarity in WordNet. In *2008 Second International Conference on Future Generation Communication and Networking Symposia*, Hainan, China, pp. 85-89. doi: 10.1109/FGCNS.2008.16.
