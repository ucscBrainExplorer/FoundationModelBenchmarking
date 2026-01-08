# Universal Cell Embedding (UCE) Benchmarking

A modular Python framework for benchmarking K-Nearest Neighbors (KNN) model accuracy in cell type prediction using foundation models like Universal Cell Embedding (UCE) and SCimilarity.

## Overview

This project benchmarks cell type prediction accuracy by comparing:
- **Foundation models**: UCE, SCimilarity, etc.
- **FAISS index types**: IVF Flat, IVF PQ
- **Distance metrics**: Euclidean, Cosine
- **Test datasets**: Organoids, post-mortem adult brain, etc.

The evaluation uses both standard metrics (accuracy, F1 score) and **biological distance metrics** based on Cell Ontology (CL) graph structures to measure how "close" predictions are to ground truth in the ontology hierarchy.

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
| `ontology_utils.py` | OBO file parsing, DAG construction, and graph distance calculations using Lowest Common Ancestor (LCA). | pronto, networkx |
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
- Supports metric switching:
  - `'euclidean'`: L2 distance
  - `'cosine'`: Cosine similarity (vectors are normalized)

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
Uses Cell Ontology (CL) graph structure to calculate biological distance:

```python
calculate_graph_distance(graph, predicted_id, truth_id)
```

**Distance Calculation**:
1. Find Lowest Common Ancestor (LCA) of predicted and truth nodes
2. Calculate: `distance = path_length(predicted → LCA) + path_length(truth → LCA)`
3. Returns edge count (0 = exact match, higher = more distant)

**Batch Scoring**:
- Mean ontology distance
- Median ontology distance

This accounts for cases where predictions are "close" in the ontology hierarchy even if not exact matches.

### 4. Benchmarking Orchestration

The `run_benchmark()` function orchestrates nested loops:

```python
for index_type in ['ivfFlat', 'ivfPQ']:
    for metric in ['euclidean', 'cosine']:
        for dataset in test_datasets:
            # Execute query
            # Calculate metrics
            # Record results
```

Results are saved to `benchmark_results.csv` with columns:
- Index, Metric, Dataset
- accuracy, f1_macro, f1_weighted, top_k_accuracy
- mean_ontology_dist, median_ontology_dist
- Avg_Query_Time_ms

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FoundationModelBenchmarking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- faiss-cpu
- numpy
- pandas
- scikit-learn
- pronto
- networkx
- boto3

3. Verify installation:
```bash
python3 verify_setup.py
```

### Docker Setup

Build the container:
```bash
docker build -t uce-benchmark:v1 .
```

Run the container:
```bash
docker run -v /path/to/data:/data uce-benchmark:v1
```

### Kubernetes Deployment

Deploy as a Kubernetes job:
```bash
kubectl apply -f benchmarking-job.yaml
```

The job expects:
- Data volume mounted at `/data`
- AWS credentials for S3 access mounted at `/root/.aws`

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

### Index Configuration

Edit `main_benchmark.py` to specify index paths:

```python
index_paths = {
    "ivfFlat": "indices/index_ivfflat.faiss",
    "ivfPQ": "indices/index_ivfpq.faiss",
}
```

Or provide via command-line argument:
```bash
--indices_config indices/config.txt
```

### Default Paths

Configured in `main_benchmark.py`:
- `DEFAULT_TEST_DIR = "test_data/"`
- `DEFAULT_INDEX_DIR = "indices/"`
- `DEFAULT_REF_ANNOTATION = "reference_data/prediction_obs.tsv"`
- `DEFAULT_OBO_PATH = "reference_data/cl.obo"`

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
| mean_ontology_dist | Mean graph distance in CL ontology |
| median_ontology_dist | Median graph distance in CL ontology |
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
