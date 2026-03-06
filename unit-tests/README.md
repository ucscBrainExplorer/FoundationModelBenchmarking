# Unit Tests for UCE Benchmarking

Legacy unittest-based test suite covering core modules.

## Directory Structure

```
unit-tests/
├── README.md                       # This file
├── create_mock_data.py             # Script to generate mock test data
├── run_all_tests.py                # Script to run all tests
├── test_data_loader.py             # Tests for data_loader.py
├── test_prediction_module.py       # Tests for prediction_module.py
├── test_ontology_utils.py          # Tests for ontology_utils.py
└── mock_data/                      # Mock data for testing
    ├── indices/                    # Mock FAISS indices
    ├── reference_data/             # Mock reference annotations and ontology
    └── test_data/                  # Mock test datasets
```

## Running Tests

### Run All Tests

```bash
python3 -m unittest discover -s unit-tests -p "test_*.py" -v
```

### Run Individual Test Files

```bash
python3 -m unittest unit-tests.test_data_loader
python3 -m unittest unit-tests.test_prediction_module
python3 -m unittest unit-tests.test_ontology_utils
```

## Test Coverage

### test_data_loader.py
- `load_faiss_index()`: Loading indices, handling missing/invalid files
- `load_reference_annotations()`: Loading TSV files, validating required columns
- `load_test_batch()`: Discovering test datasets, pairing embeddings with metadata

### test_prediction_module.py
- `execute_query()`: FAISS querying, type conversion
- `vote_neighbors()`: Majority voting, tie-breaking, bounds checking

### test_ontology_utils.py
- `load_ontology()`: Loading OBO files, graph construction
- `calculate_graph_distance()`: LCA-based distance calculation, handling invalid terms
- `score_batch()`: Mean/median distance computation

## Requirements

- faiss-cpu or faiss-gpu
- numpy
- pandas
- networkx
- pronto
