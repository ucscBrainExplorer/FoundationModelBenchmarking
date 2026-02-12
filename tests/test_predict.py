"""Tests for predict.py

Note: Full integration test requires FAISS index.
These tests validate the argument parsing and data flow.
"""

import os
import sys
import pytest
import pandas as pd

# Import the build_parser from predict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import build_parser


def test_predict_parser_required_args():
    """Test that parser enforces required arguments."""
    parser = build_parser()

    # Missing all required args should fail
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_predict_parser_with_all_required():
    """Test parser with all required arguments."""
    parser = build_parser()

    args = parser.parse_args([
        '--index', 'test.faiss',
        '--ref_annot', 'ref.tsv',
        '--obo', 'cl.obo',
        '--embeddings', 'test.npy',
    ])

    assert args.index == 'test.faiss'
    assert args.ref_annot == 'ref.tsv'
    assert args.obo == 'cl.obo'
    assert args.embeddings == 'test.npy'
    assert args.k == 30  # default
    assert args.output == 'predictions.tsv'  # default


def test_predict_parser_with_optional_args():
    """Test parser with optional arguments."""
    parser = build_parser()

    args = parser.parse_args([
        '--index', 'test.faiss',
        '--ref_annot', 'ref.tsv',
        '--obo', 'cl.obo',
        '--embeddings', 'test.npy',
        '--metadata', 'meta.tsv',
        '--k', '50',
        '--output', 'my_predictions.tsv',
    ])

    assert args.metadata == 'meta.tsv'
    assert args.k == 50
    assert args.output == 'my_predictions.tsv'


def test_predict_output_format(sample_predictions):
    """Test that predict.py output has expected columns."""
    path, df = sample_predictions

    required_cols = [
        'predicted_cell_type_ontology_term_id',
        'predicted_cell_type',
        'vote_percentage',
        'mean_euclidean_distance',
        'neighbor_distances',
        'neighbor_cell_types',
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
