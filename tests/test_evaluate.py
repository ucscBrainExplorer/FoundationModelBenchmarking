"""Tests for evaluate.py"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import (
    build_parser,
    build_label_mapping,
    resolve_name,
    resolve_to_cl_ids,
    load_remap_file,
    apply_remap,
    _is_cl_id,
)


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
    assert args.output_dir == 'evaluation_results'  # default


def test_evaluate_parser_column_overrides():
    """Test parser with custom column name overrides."""
    parser = build_parser()

    args = parser.parse_args([
        '--predictions', 'preds.tsv',
        '--ground_truth', 'truth.tsv',
        '--obo', 'cl.obo',
        '--pred_id_col', 'my_pred_col',
        '--truth_id_col', 'my_truth_col',
    ])

    assert args.pred_id_col == 'my_pred_col'
    assert args.truth_id_col == 'my_truth_col'


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


def test_evaluate_parser_remap_file():
    """Test parser accepts --remap-file argument."""
    parser = build_parser()

    args = parser.parse_args([
        '--predictions', 'p.tsv',
        '--ground_truth', 't.tsv',
        '--obo', 'cl.obo',
        '--remap-file', 'remap.tsv',
    ])
    assert args.remap_file == 'remap.tsv'

    # Default is None
    args_no_remap = parser.parse_args([
        '--predictions', 'p.tsv',
        '--ground_truth', 't.tsv',
        '--obo', 'cl.obo',
    ])
    assert args_no_remap.remap_file is None


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


def test_build_label_mapping(sample_obo_file):
    """Test that build_label_mapping returns correct lookup dicts."""
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(sample_obo_file)

    # cl_names: {CL_id: name}
    assert cl_names['CL:0000540'] == 'neuron'
    assert cl_names['CL:0000127'] == 'astrocyte'
    assert cl_names['CL:0000128'] == 'oligodendrocyte'

    # name_to_id: {lowercase_name: CL_id}
    assert name_to_id['neuron'] == 'CL:0000540'
    assert name_to_id['astrocyte'] == 'CL:0000127'

    # synonym_to_id: {lowercase_synonym: CL_id}
    assert synonym_to_id['nerve cell'] == 'CL:0000540'

    # fuzzy lookups should exist
    assert len(fuzzy_name_to_id) > 0
    assert len(fuzzy_synonym_to_id) > 0


def test_resolve_name(sample_obo_file):
    """Test resolve_name with different match types."""
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(sample_obo_file)

    # Exact match
    cl_id, name, method = resolve_name(
        'neuron', name_to_id, synonym_to_id,
        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
    assert cl_id == 'CL:0000540'
    assert method == 'exact'

    # Synonym match
    cl_id, name, method = resolve_name(
        'nerve cell', name_to_id, synonym_to_id,
        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
    assert cl_id == 'CL:0000540'
    assert method == 'synonym'

    # Unresolved
    cl_id, name, method = resolve_name(
        'zygomorphic blastocell', name_to_id, synonym_to_id,
        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
    assert cl_id is None
    assert method is None


def test_resolve_to_cl_ids_already_cl(sample_obo_file):
    """Test resolve_to_cl_ids with values that are already CL IDs."""
    values = ['CL:0000540', 'CL:0000127', 'CL:0000128']
    resolved, was_resolved, report, unresolved = resolve_to_cl_ids(values, sample_obo_file)

    assert not was_resolved
    assert resolved == values
    assert len(unresolved) == 0


def test_resolve_to_cl_ids_names(sample_obo_file):
    """Test resolve_to_cl_ids with readable names."""
    values = ['neuron', 'astrocyte', 'nerve cell', 'totally fake']
    resolved, was_resolved, report, unresolved = resolve_to_cl_ids(values, sample_obo_file)

    assert was_resolved
    assert resolved[0] == 'CL:0000540'  # neuron -> exact
    assert resolved[1] == 'CL:0000127'  # astrocyte -> exact
    assert resolved[2] == 'CL:0000540'  # nerve cell -> synonym
    assert resolved[3] == 'totally fake'  # unresolved
    assert 'totally fake' in unresolved


def test_load_remap_file(sample_remap_file):
    """Test loading a remap TSV file."""
    remap = load_remap_file(sample_remap_file)

    assert remap['neuron'] == 'CL:0000540'
    assert remap['nerve cell'] == 'CL:0000540'
    assert remap['astrocyte'] == 'CL:0000127'
    assert remap['oligodendrocyte'] == 'CL:0000128'
    # Unresolved entries should be skipped
    assert 'zygomorphic blastocell' not in remap


def test_apply_remap():
    """Test applying a remap dict to ground truth values."""
    values = ['neuron', 'astrocyte', 'unknown_type', 'neuron']
    remap_dict = {
        'neuron': 'CL:0000540',
        'astrocyte': 'CL:0000127',
    }

    remapped, n_remapped, report = apply_remap(values, remap_dict)

    assert remapped == ['CL:0000540', 'CL:0000127', 'unknown_type', 'CL:0000540']
    assert n_remapped == 3
    assert 'neuron -> CL:0000540' in report
    assert 'astrocyte -> CL:0000127' in report


def test_is_cl_id():
    """Test CL ID detection."""
    assert _is_cl_id('CL:0000540')
    assert _is_cl_id('CL:0000000')
    assert not _is_cl_id('neuron')
    assert not _is_cl_id('CL:abc')
    assert not _is_cl_id('')
    assert not _is_cl_id('GO:0000540')
