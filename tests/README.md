# Test Suite

pytest-based test suite for the Foundation Model Benchmarking project.

## Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── test_obo_parser.py             # Tests for obo_parser.py
├── test_predict.py                # Tests for predict.py
├── test_evaluate.py               # Tests for evaluate.py
├── test_annotate_cl_terms.py      # Tests for annotate_cl_terms.py
├── test_generate_remap.py         # Tests for generate_remap.py
└── fixtures/                      # Test data files
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_predict.py -v
```

### Run with coverage report
```bash
pytest tests/ --cov=. --cov-report=html
```

## Requirements

```bash
pip install pytest pytest-cov
```

## Legacy Tests

The `unit-tests/` directory contains an older unittest-based suite covering
`data_loader.py`, `prediction_module.py`, and `ontology_utils.py`.
