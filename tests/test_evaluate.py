"""Tests for evaluate.py"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import build_parser


def test_evaluate_parser_required_args():
    """Test that parser enforces required arguments."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_evaluate_parser_with_all_required():
    """Test parser with all required arguments."""
    parser = build_parser()

    args = parser.parse_args([
        '--predictions', 'preds.tsv',
        '--ground_truth', 'truth.tsv',
        '--obo', 'cl.obo',
    ])

    assert args.predictions == 'preds.tsv'
    assert args.ground_truth == 'truth.tsv'
    assert args.obo == 'cl.obo'
    assert args.ontology_method == 'ic'  # default
    assert args.output == 'evaluation_results'  # default


def test_evaluate_parser_column_overrides():
    """Test parser with custom column name overrides."""
    parser = build_parser()

    args = parser.parse_args([
        '--predictions', 'preds.tsv',
        '--ground_truth', 'truth.tsv',
        '--obo', 'cl.obo',
        '--pred_id_col', 'my_pred_col',
        '--truth_id_col', 'my_truth_col',
        '--pred_cell_id_col', 'my_cell_id',
        '--truth_cell_id_col', 'my_cell_id',
    ])

    assert args.pred_id_col == 'my_pred_col'
    assert args.truth_id_col == 'my_truth_col'
    assert args.pred_cell_id_col == 'my_cell_id'
    assert args.truth_cell_id_col == 'my_cell_id'


def test_evaluate_parser_ontology_methods():
    """Test parser accepts both ontology methods."""
    parser = build_parser()

    args_ic = parser.parse_args([
        '--predictions', 'p.tsv',
        '--ground_truth', 't.tsv',
        '--obo', 'cl.obo',
        '--ontology-method', 'ic',
    ])
    assert args_ic.ontology_method == 'ic'

    args_sp = parser.parse_args([
        '--predictions', 'p.tsv',
        '--ground_truth', 't.tsv',
        '--obo', 'cl.obo',
        '--ontology-method', 'shortest_path',
    ])
    assert args_sp.ontology_method == 'shortest_path'


def test_evaluate_expected_output_files():
    """Test that evaluate.py is expected to create specific output files."""
    # This is a documentation test - evaluate.py should create:
    expected_files = [
        'evaluation_summary.tsv',
        'per_cell_evaluation.tsv',
        'ontology_analysis_report.txt',
        'ontology_distance_analysis.png',
    ]

    # Just verify the list is documented
    assert len(expected_files) == 4
