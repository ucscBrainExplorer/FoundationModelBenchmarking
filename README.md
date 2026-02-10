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

### Prerequisites

- **Docker** installed and running
- **kubectl** configured for your Kubernetes cluster
- **Access to Kubernetes namespace**: `braingeneers` (or update namespace in YAML files)
- **Docker Hub account** (for pushing images, or use a different registry)

### Running the Benchmark

1. **Build the Docker image**:
   ```bash
   cd FoundationModelBenchmarking
   docker build --no-cache -t suhaso123/uce-benchmark:v1 .
   ```

2. **Push to Docker Hub** (or your registry):
   ```bash
   docker login
   docker push suhaso123/uce-benchmark:v1
   ```
   
   **Note**: Update the image name (`suhaso123/uce-benchmark:v1`) in `benchmarking-job.yaml` if using a different registry.

3. **Ensure PVC exists** (for storing results):
   ```bash
   kubectl get pvc benchmark-data-pvc -n braingeneers
   # If missing, create it (see pvc.yaml or create manually)
   ```

4. **Deploy the benchmark job**:
   ```bash
   kubectl delete job uce-benchmark-job -n braingeneers 2>/dev/null  # Delete old job if exists
   kubectl apply -f benchmarking-job.yaml
   ```

5. **Monitor the job**:
   ```bash
   # Check job status
   kubectl get job uce-benchmark-job -n braingeneers
   
   # Watch logs in real-time
   kubectl logs -n braingeneers -l job-name=uce-benchmark-job -f
   ```

6. **Access results** (after job completes):
   ```bash
   # Create temporary pod to access PVC
   kubectl apply -f pvc-access-pod.yaml
   kubectl wait --for=condition=Ready pod/pvc-access-pod -n braingeneers --timeout=60s

   # List available runs
   kubectl exec -n braingeneers pvc-access-pod -- ls /mnt/data/benchmark_results/

   # View a run's summary (replace {timestamp} with actual value)
   kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results/{timestamp}/benchmark_results.csv

   # Or download from S3 (results are uploaded automatically)
   aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/
   aws s3 cp --recursive s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/ ./results/

   # Clean up access pod when done
   kubectl delete pod pvc-access-pod -n braingeneers
   ```

### Expected Runtime

- **Docker Build**: 5-10 minutes (first time), 1-2 minutes (cached)
- **Docker Push**: 2-5 minutes (depends on network)
- **Benchmark Execution**: 10-20 minutes (depends on dataset size)
- **Total**: ~20-35 minutes end-to-end

### Generated Results

Each run creates a timestamped directory so previous results are never overwritten.
The same structure is used locally and on S3.

**Local (cluster at `/data`, or `.` for local development):**
```
{DATA_ROOT}/benchmark_results/{timestamp}/
├── benchmark_results.csv              Summary metrics per index/metric/dataset
├── benchmark.log                      Run log (warnings, filtered cells, timing)
├── per_cell_results/
│   └── {dataset}_{index}_{metric}_per_cell_results.csv
├── ontology_analysis/
│   └── ontology_analysis_report.txt
└── visualizations/                    (if --generate-plots)
```

**S3 (uploaded automatically unless `--no-upload` or `--no-s3`):**
```
s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/
├── benchmark_results.csv
├── benchmark.log
├── per_cell_results/
│   └── {dataset}_{index}_{metric}_per_cell_results.csv
├── ontology_analysis/
│   └── ontology_analysis_report.txt
└── visualizations/
```

**Per-cell results CSV columns:**
- Cell ID, true label, prediction label (both ontology ID and readable name)
- Vote percentage (how many neighbors voted for the prediction)
- Euclidean distances (mean and nearest neighbor)
- Ontology score (prediction vs truth, and average across neighbors)

**Retrieving results from the cluster:**
```bash
# List available runs
kubectl exec -n braingeneers pvc-access-pod -- ls /mnt/data/benchmark_results/

# View a specific run's summary
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results/{timestamp}/benchmark_results.csv

# Or download from S3
aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/
aws s3 cp --recursive s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/ ./results/
```

### Detailed Instructions

For complete step-by-step instructions, troubleshooting, and advanced usage, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

---

## Project Structure

```
FoundationModelBenchmarking/
├── __init__.py                    # Package initialization
├── data_loader.py                 # Data ingestion and S3 download
├── prediction_module.py           # KNN search and voting logic
├── ontology_utils.py              # Cell Ontology processing
├── evaluation_metrics.py          # Statistical metrics calculation
├── main_benchmark.py              # Main orchestration script
├── requirements.txt               # Python dependencies
├── verify_setup.py                # Dependency verification script
├── Dockerfile                     # Container definition
├── benchmarking-job.yaml          # Kubernetes job configuration
└── README.md                      # This file
```

## Architecture

### Module Responsibilities

| Module | Core Responsibility | Key Dependencies |
|--------|---------------------|------------------|
| `data_loader.py` | IO operations for FAISS indices, TSV annotations, and NumPy embeddings. Handles S3 data downloads. | faiss, numpy, pandas, boto3, os |
| `prediction_module.py` | KNN search execution with metric switching (Euclidean/Cosine). Implements majority voting for cell type prediction. | numpy, collections, faiss |
| `ontology_utils.py` | OBO file parsing, DAG construction, IC-based Lin similarity (default) and shortest-path distance. | pronto, networkx |
| `evaluation_metrics.py` | Statistical calculations: Accuracy, F1 (Macro & Weighted), Top-k Accuracy | sklearn.metrics |
| `main_benchmark.py` | Orchestrates benchmarking loops across Index Type × Metric × Dataset permutations | All of the above |

## How It Works

### 1. Data Ingestion (Phase 1)

#### Reference Data
- **FAISS Index**: Pre-computed graph structure for reference cells (e.g., `index_ivfflat.faiss`)
- **Annotations**: `prediction_obs.tsv` containing `cell_type_ontology_term_id` and `cell_type` columns
  - **Critical**: Must align perfectly with FAISS index row order

#### Test Data
Located in `test_data/` directory with paired files:
- **Embeddings**: `{dataset_id}_{embedding}.npy` (NumPy binary array)
- **Metadata**: `{dataset_id}_prediction_obs.tsv` (ground truth labels)

The `load_test_batch()` function automatically discovers and pairs these files.

#### S3 Data Source
Data can be automatically downloaded from:
```
s3://latentbrain/combined_UCE_5neuro/
```

### 2. Prediction Pipeline (Phase 2)

#### KNN Search
```python
execute_query(index, query_embeddings, k=30, metric='euclidean')
```
- Queries FAISS index for top k nearest neighbors
- Uses Euclidean (L2) distance

#### Majority Voting
```python
vote_neighbors(neighbor_indices, reference_annotations)
```
- Maps neighbor indices → reference annotation rows → cell type ontology term IDs
- Performs majority voting to determine final prediction
- Returns most common cell type among k neighbors

### 3. Evaluation (Phase 3)

#### Standard Metrics
- **Overall Accuracy**: Exact match between prediction and ground truth
- **Top-k Accuracy**: Ground truth appears in k neighbors
- **F1 Score**: Macro and weighted averages across cell types

#### Ontology-Aware Metrics

Two methods are available, selectable via `--ontology-method`:

**Method 1: IC-based Lin Similarity (default, `--ontology-method ic`)**

Uses Information Content (IC) to measure semantic similarity between predicted and ground truth cell types.

| Score | Meaning |
|-------|---------|
| **1.0** | Identical terms (perfect prediction) |
| **> 0.8** | Closely related (e.g., neuron subtypes) |
| **0.4 – 0.7** | Moderately related (e.g., neuron vs. astrocyte) |
| **~0.0** | Unrelated (e.g., near the ontology root) |

**Higher similarity = better prediction.** A prediction of "Purkinje neuron" when the truth is "cerebellar granule cell" will score higher than a prediction of "erythrocyte", reflecting the biological relatedness of the cell types.

IC values are precomputed using the Zhou (2008) weighted intrinsic IC formula, which blends descendant count with structural depth. Similarity between two terms is computed using Lin (1998): `Sim(A,B) = 2 * IC(MICA) / (IC(A) + IC(B))`, where MICA is the Most Informative Common Ancestor.

Result columns: `mean_ontology_similarity`, `median_ontology_similarity`

**Method 2: Shortest-Path Distance (`--ontology-method shortest_path`)**

Computes the shortest undirected path between two terms in the ontology graph.

**Lower distance = better prediction.** A distance of 0 means an exact match; larger values mean the terms are farther apart in the hierarchy.

Note: on DAGs with multiple inheritance (the Cell Ontology has 33.5% multi-parent terms), shortest undirected path can shortcut across separate branches, potentially underestimating true semantic distance. The IC method is recommended for this reason.

Result columns: `mean_ontology_dist`, `median_ontology_dist`

### 4. Benchmarking Orchestration

The `run_benchmark()` function orchestrates nested loops:

```python
for index_type in ['ivfFlat', 'ivfPQ']:
    for metric in ['euclidean']:
        for dataset in test_datasets:
            # Execute query
            # Calculate metrics (standard + ontology)
            # Record results
```

Results are saved to `benchmark_results.csv` with columns:
- Index, Metric, Dataset
- accuracy, f1_macro, f1_weighted, top_k_accuracy
- mean_ontology_dist, median_ontology_dist
- Avg_Query_Time_ms

## Installation

### Recommended: Kubernetes Deployment (Production)

**See [Quick Start](#quick-start) section above for complete instructions.**

The benchmark is designed to run as a Kubernetes job with:
- Data volume (PVC) mounted at `/data` for persistent storage
- AWS credentials for S3 access (via Kubernetes secrets or mounted credentials)
- Automatic data download from S3 if needed

### Local Development Setup

For local development or testing:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FoundationModelBenchmarking
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - faiss-cpu==1.7.4
   - numpy<2.0
   - pandas
   - scikit-learn
   - pronto
   - networkx
   - boto3
   - matplotlib
   - seaborn
   - umap-learn

3. **Verify installation**:
   ```bash
   python3 verify_setup.py
   ```

4. **Run locally** (requires data files):
   ```bash
   python3 -m main_benchmark \
       --test_dir test_data/ \
       --ref_annot reference_data/prediction_obs.tsv \
       --obo reference_data/cl.obo
   ```

### Docker Setup (Alternative)

Build and run locally with Docker:
```bash
# Build
docker build -t uce-benchmark:v1 .

# Run (mount data directory)
docker run -v /path/to/data:/data uce-benchmark:v1
```

**Note**: For production use, we recommend Kubernetes deployment (see Quick Start above).

## Usage

### Basic Usage

```bash
python3 -m main_benchmark \
    --test_dir test_data/ \
    --ref_annot reference_data/prediction_obs.tsv \
    --obo reference_data/cl.obo
```

### With S3 Download

```bash
python3 -m main_benchmark \
    --test_dir test_data/ \
    --ref_annot reference_data/prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --s3_bucket latentbrain \
    --s3_prefix combined_UCE_5neuro/
```

### Choose Ontology Scoring Method

```bash
# IC-based Lin similarity (default, recommended)
python3 -m main_benchmark --ontology-method ic

# Shortest-path distance (original method)
python3 -m main_benchmark --ontology-method shortest_path
```

### Skip S3 Download

```bash
python3 -m main_benchmark --no-s3
```

### Configure AWS Profile

Set environment variable or configure in `~/.aws/config`:
```bash
export AWS_PROFILE=braingeneers
```

## Configuration

### Kubernetes Job Configuration

Before deploying, check/update `benchmarking-job.yaml`:

1. **Image name**: Update if using different Docker registry:
   ```yaml
   image: suhaso123/uce-benchmark:v1  # Change to your image
   ```

2. **Namespace**: Default is `braingeneers`, update if needed:
   ```yaml
   namespace: braingeneers
   ```

3. **PVC name**: Ensure PVC exists or update name:
   ```yaml
   persistentVolumeClaim:
     claimName: benchmark-data-pvc  # Must exist in namespace
   ```

4. **AWS credentials**: Configure via Kubernetes secrets or environment variables:
   ```yaml
   env:
     - name: AWS_ACCESS_KEY_ID
       valueFrom:
         secretKeyRef:
           name: aws-credentials
           key: access-key-id
   ```

### Index Configuration

The benchmark automatically discovers FAISS indices in `/data/indices/` directory. Expected files:
- `index_ivfflat.faiss` (or similar naming)

To use different indices, edit `main_benchmark.py` or provide via command-line:
```bash
--indices_config indices/config.txt
```

### Default Paths (in container)

When running in Kubernetes, input data paths are:
- `/data/indices/` - FAISS index files
- `/data/reference_data/` - Reference annotations and ontology files
- `/data/test_data/` - Test datasets (or downloaded from S3)

Output is written to `/data/benchmark_results/{timestamp}/` (see [Generated Results](#generated-results)).

## Expected Input File Formats

### FAISS Index
- Format: Binary FAISS index file
- Types supported: IVF Flat, IVF PQ, or any FAISS index type
- File extension: `.faiss`

### Reference Annotations (`prediction_obs.tsv`)
- Format: Tab-separated values (TSV)
- Required columns:
  - `cell_type_ontology_term_id`: CL term ID (e.g., "CL:0000000")
  - `cell_type`: Human-readable cell type name
- **Critical**: Row order must exactly match FAISS index order

### Test Embeddings (`{dataset_id}_{embedding}.npy`)
- Format: NumPy binary array (`.npy`)
- Shape: `(n_cells, embedding_dim)`
- Data type: float32 or float64

### Test Metadata (`{dataset_id}_prediction_obs.tsv`)
- Format: Tab-separated values (TSV)
- Required columns:
  - `cell_type_ontology_term_id`: Ground truth CL term ID
  - `cell_type`: Human-readable cell type name

### Cell Ontology (`cl.obo`)
- Format: OBO ontology file
- Source: [Cell Ontology](http://obofoundry.org/ontology/cl.html)

## Output

### Results CSV
`benchmark_results.csv` contains:

| Column | Description |
|--------|-------------|
| Index | Index type used (e.g., "ivfFlat") |
| Metric | Distance metric (e.g., "euclidean") |
| Dataset | Test dataset ID |
| accuracy | Overall prediction accuracy (0-1) |
| f1_macro | Macro-averaged F1 score |
| f1_weighted | Weighted F1 score |
| top_k_accuracy | Proportion where truth in top-k neighbors |
| mean_ontology_similarity | Mean Lin similarity (0-1, **higher = more similar**). Present when `--ontology-method ic` (default). |
| median_ontology_similarity | Median Lin similarity. |
| mean_ontology_dist | Mean shortest-path distance (**lower = more similar**). Present when `--ontology-method shortest_path`. |
| median_ontology_dist | Median shortest-path distance. |
| Avg_Query_Time_ms | Average query time per cell (milliseconds) |

## Implementation Status

### Completed Features ✅

**Phase 1: Infrastructure & Data Ingestion**
- ✅ `load_faiss_index()` - Loads different FAISS index types
- ✅ `load_reference_annotations()` - Validates required columns
- ✅ `load_test_batch()` - Auto-discovers test dataset pairs
- ✅ `download_data_from_s3()` - Downloads data from S3

**Phase 2: Benchmarking Logic**
- ✅ `execute_query()` - KNN search with metric switching
- ✅ `vote_neighbors()` - Majority voting implementation
- ✅ `run_benchmark()` - Orchestrates all permutations

**Phase 3: Evaluation & Ontology**
- ✅ `calculate_accuracy()` - Standard metrics (Accuracy, F1, Top-k)
- ✅ `load_ontology()` - OBO file parsing to NetworkX DAG
- ✅ `calculate_graph_distance()` - LCA-based distance calculation
- ✅ `score_batch()` - Mean/median ontology distance

**Infrastructure**
- ✅ Docker containerization
- ✅ Kubernetes job configuration
- ✅ AWS S3 integration

### Known Gaps ⚠️

#### High Priority

1. **Phase 4: Visualization & Reporting** ❌
   - Missing UMAP plots colored by:
     - Ground Truth
     - Prediction
     - Ontology Error Magnitude
   - Missing confusion matrices
   - Missing comprehensive summary table (beyond CSV)

2. **Data Validation**
   - `load_reference_annotations()` doesn't verify row count alignment with FAISS index
   - No explicit validation that positive control file exists in test batch
   - Missing file format validation

3. **Package Structure Issue**
   - `main_benchmark.py` uses relative imports (e.g., `from .data_loader`)
   - Fails when run directly as a script
   - Should be run as module: `python3 -m main_benchmark`
   - Or refactor to absolute imports

#### Medium Priority

4. **Index Configuration**
   - Hardcoded index paths in `main_benchmark.py`
   - `--indices_config` argument exists but not fully implemented
   - Only ivfFlat configured by default, ivfPQ commented out

5. **Foundation Model Comparison**
   - No explicit support for comparing UCE vs SCimilarity
   - Currently assumes different embeddings come as different test datasets
   - Could benefit from explicit model tracking/labeling

6. **Error Handling**
   - Limited error handling for malformed data files
   - No graceful degradation if ontology file missing
   - Silent failures in some edge cases

#### Low Priority

7. **Testing**
   - No unit tests
   - No integration tests
   - No example/mock data for testing

8. **Performance Optimization**
   - No batch processing optimization
   - Could parallelize dataset processing
   - No caching of ontology graph computations

9. **Logging**
   - Basic print statements instead of proper logging framework
   - No log levels (DEBUG, INFO, WARNING, ERROR)
   - Difficult to troubleshoot in production

10. **Documentation**
    - Limited inline code documentation
    - No examples of expected data formats
    - No troubleshooting guide

## Recommended Next Steps

### Critical Fixes
1. **Fix package structure** - Refactor imports or add `setup.py` for proper package installation
2. **Add data validation** - Verify FAISS index and annotation alignment before processing
3. **Implement visualizations** - UMAP plots and confusion matrices as specified in plan

### Enhancements
4. **Add unit tests** - Test each module independently
5. **Improve error handling** - Better error messages and recovery
6. **Add logging framework** - Replace print statements with proper logging
7. **Create example data** - Small test dataset for validation

### Nice to Have
8. **Parallelization** - Speed up multi-dataset benchmarking
9. **Web dashboard** - Interactive visualization of results
10. **Automated reporting** - Generate PDF reports with plots and tables

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError` when running `main_benchmark.py`

**Solution**: Run as a module:
```bash
python3 -m main_benchmark
```

Or add current directory to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 main_benchmark.py
```

### S3 Access Issues

**Problem**: `NoCredentialsError` or access denied

**Solutions**:
- Configure AWS credentials: `aws configure --profile braingeneers`
- Check IAM permissions for S3 bucket access
- Use `--no-s3` flag to skip S3 download

### Missing Dependencies

**Problem**: Import errors for `pronto` or `networkx`

**Solution**:
```bash
pip install -r requirements.txt
```

Verify with:
```bash
python3 verify_setup.py
```

### FAISS Index Mismatch

**Problem**: `IndexError: Neighbor index out of bounds`

**Cause**: Reference annotation file doesn't match FAISS index size

**Solution**: Ensure `prediction_obs.tsv` has exactly the same number of rows as vectors in FAISS index

### Ontology Distance Returns -1

**Problem**: `mean_ontology_dist` shows -1 or very large negative values

**Causes**:
- Cell type ID not found in ontology graph
- Disconnected components in ontology (no common ancestor)
- Malformed CL term IDs

**Solution**: Check that CL term IDs in data match format in `cl.obo` file (e.g., "CL:0000001")

## Performance Considerations

### Query Time
- IVF indices require training but offer faster search
- Euclidean distance is slightly faster than cosine
- Query time scales with k (number of neighbors)

### Memory Usage
- FAISS indices are memory-mapped (efficient)
- Ontology graph loaded once and reused
- Large test datasets processed sequentially to limit memory

### Optimization Tips
- Use IVF PQ for memory-constrained environments
- Reduce k if speed is critical (typical range: 10-50)
- Process large batches with `faiss.IndexShards` for parallelization

## Citation

If you use this benchmarking framework, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please open an issue on GitHub or contact [contact information].

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Cell Ontology](http://obofoundry.org/ontology/cl.html)
- [Universal Cell Embeddings](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)
- [CELLxGENE](https://cellxgene.cziscience.com/)
- Lin, D. (1998). An Information-Theoretic Definition of Similarity. In *Proceedings of the 15th International Conference on Machine Learning (ICML 1998)*, Vol. 98, pp. 296-304.
- Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content for Semantic Similarity in WordNet. In *2008 Second International Conference on Future Generation Communication and Networking Symposia*, Hainan, China, pp. 85-89. doi: 10.1109/FGCNS.2008.16.
