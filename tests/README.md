# Test Suite

Modern pytest-based test suite for the Foundation Model Benchmarking project.

## Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── test_obo_parser.py             # Tests for obo_parser.py
├── test_predict.py                # Tests for predict.py
├── test_evaluate.py               # Tests for evaluate.py
├── test_annotate_cl_terms.py      # Tests for annotate_cl_terms.py
├── test_data_loader.py            # Tests for data_loader.py
├── test_prediction_module.py      # Tests for prediction_module.py
├── test_ontology_utils.py         # Tests for ontology_utils.py
├── test_evaluation_metrics.py     # Tests for evaluation_metrics.py
└── fixtures/                      # Test data files
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_obo_parser.py -v
```

### Run specific test function
```bash
pytest tests/test_obo_parser.py::test_parse_obo_names_basic -v
```

### Run with coverage report
```bash
pytest tests/ --cov=. --cov-report=html
```

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `temp_dir` — Temporary directory that auto-cleans
- `sample_obo_file` — Minimal OBO file with test terms
- `sample_embeddings` — Random embeddings numpy file
- `sample_reference_annotations` — Reference annotation TSV
- `sample_predictions` — Sample predictions TSV
- `sample_ground_truth` — Sample ground truth TSV

## Legacy Tests

The old `unit-tests/` directory contains the original test suite. It will be migrated incrementally to the new `tests/` structure.

## Requirements

```bash
pip install pytest pytest-cov
```
