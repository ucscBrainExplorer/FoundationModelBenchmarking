"""Tests for obo_parser.py"""

import pytest
from obo_parser import parse_obo_names


def test_parse_obo_names_basic(sample_obo_file):
    """Test basic OBO parsing."""
    cl_map = parse_obo_names(sample_obo_file)

    assert 'CL:0000000' in cl_map
    assert cl_map['CL:0000000'] == 'cell'

    assert 'CL:0000540' in cl_map
    assert cl_map['CL:0000540'] == 'neuron'

    assert 'CL:0000128' in cl_map
    assert cl_map['CL:0000128'] == 'oligodendrocyte'


def test_parse_obo_names_ignores_typedef(sample_obo_file):
    """Test that [Typedef] sections are ignored."""
    cl_map = parse_obo_names(sample_obo_file)

    # part_of is a typedef, not a term, should not be in the map
    assert 'part_of' not in cl_map


def test_parse_obo_names_count(sample_obo_file):
    """Test correct number of terms parsed."""
    cl_map = parse_obo_names(sample_obo_file)

    # Should have 4 terms: cell, neuron, oligodendrocyte, astrocyte
    assert len(cl_map) == 4
