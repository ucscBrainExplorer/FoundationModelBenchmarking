# Unit Tests for UCE Benchmarking

This directory contains comprehensive unit tests for all modules in the UCE benchmarking project.

## Directory Structure

```
unit-tests/
├── README.md                       # This file
├── create_mock_data.py             # Script to generate mock test data
├── run_all_tests.py                # Script to run all tests
├── test_data_loader.py             # Tests for data_loader.py
├── test_prediction_module.py       # Tests for prediction_module.py
├── test_evaluation_metrics.py      # Tests for evaluation_metrics.py
├── test_ontology_utils.py          # Tests for ontology_utils.py
├── test_main_benchmark.py          # Tests for main_benchmark.py
└── mock_data/                      # Mock data for testing
    ├── indices/                    # Mock FAISS indices
    ├── reference_data/             # Mock reference annotations and ontology
    └── test_data/                  # Mock test datasets
```

## Running Tests

### Run All Tests

```bash
# From the project root directory
python3 unit-tests/run_all_tests.py

# Or using unittest directly
python3 -m unittest discover -s unit-tests -p "test_*.py" -v
```

### Run Individual Test Files

```bash
python3 -m unittest unit-tests.test_data_loader
python3 -m unittest unit-tests.test_prediction_module
python3 -m unittest unit-tests.test_evaluation_metrics
python3 -m unittest unit-tests.test_ontology_utils
python3 -m unittest unit-tests.test_main_benchmark
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
python3 -m unittest unit-tests.test_data_loader.TestLoadFaissIndex

# Run a specific test method
python3 -m unittest unit-tests.test_data_loader.TestLoadFaissIndex.test_load_valid_index
```

## Test Coverage

### test_data_loader.py
- `load_faiss_index()`: Loading indices, handling missing/invalid files
- `load_reference_annotations()`: Loading TSV files, validating required columns
- `load_test_batch()`: Discovering test datasets, pairing embeddings with metadata
- `download_data_from_s3()`: S3 download error handling

### test_prediction_module.py
- `execute_query()`: FAISS querying with euclidean/cosine metrics, type conversion
- `vote_neighbors()`: Majority voting, tie-breaking, bounds checking

### test_evaluation_metrics.py
- `calculate_accuracy()`: Overall accuracy, F1 scores (macro/weighted), top-k accuracy

### test_ontology_utils.py
- `load_ontology()`: Loading OBO files, graph construction
- `calculate_graph_distance()`: LCA-based distance calculation, handling invalid terms
- `score_batch()`: Mean/median distance computation

### test_main_benchmark.py
- `run_benchmark()`: Full integration tests, multiple indices/metrics/datasets, error handling

## Mock Data

Mock data is automatically created by `create_mock_data.py` and includes:

- **FAISS Indices**: Flat and IVFFlat indices with 100 vectors (64 dimensions)
- **Reference Annotations**: 100 cells with 5 different cell types
- **Test Datasets**: 3 datasets (organoid_test, brain_test, positive_control)
- **Cell Ontology**: Minimal OBO file with hierarchical cell type relationships

## Requirements

All tests require the same dependencies as the main project:
- faiss-cpu or faiss-gpu
- numpy
- pandas
- scikit-learn
- networkx
- pronto
- boto3 (for S3 tests)

## Notes

- Tests use mock data to avoid dependencies on external data sources
- S3 download tests expect failures without valid AWS credentials (this is intentional)
- All tests are self-contained and can run in any order
- Tests automatically clean up generated files (e.g., benchmark_results.csv)
