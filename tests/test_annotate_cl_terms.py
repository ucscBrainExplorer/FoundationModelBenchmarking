"""Tests for annotate_cl_terms.py"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotate_cl_terms import (
    build_parser,
    parse_obo_synonyms,
    fuzzy_normalize,
)


def test_annotate_parser_required_args():
    """Test that parser enforces required arguments."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_annotate_parser_with_all_required():
    """Test parser with all required arguments."""
    parser = build_parser()

    args = parser.parse_args([
        '--obo', 'cl.obo',
        '--input', 'input.tsv',
        '--output', 'output.tsv',
    ])

    assert args.obo == 'cl.obo'
    assert args.input == 'input.tsv'
    assert args.output == 'output.tsv'
    assert args.name_col == 'cell_type'  # default


def test_parse_obo_synonyms(sample_obo_file):
    """Test OBO synonym parsing."""
    synonyms = parse_obo_synonyms(sample_obo_file)

    # "nerve cell" is a synonym for neuron (CL:0000540)
    assert 'nerve cell' in synonyms
    assert synonyms['nerve cell'] == 'CL:0000540'


def test_fuzzy_normalize():
    """Test fuzzy normalization logic."""
    # Test hyphen removal
    assert fuzzy_normalize('T-cell') == 'tcell'

    # Test underscore removal
    assert fuzzy_normalize('micro_glial') == 'microglial'

    # Test plural handling (trailing 's')
    assert fuzzy_normalize('neurons') == 'neuron'
    assert fuzzy_normalize('oligodendrocytes') == 'oligodendrocyte'

    # Test case normalization
    assert fuzzy_normalize('NEURON') == 'neuron'

    # Test combined
    assert fuzzy_normalize('T-Cells') == 'tcell'

    # Test that very short names don't get plural stripping
    assert fuzzy_normalize('as') == 'as'  # too short, no stripping


def test_fuzzy_normalize_preserves_short_words():
    """Test that fuzzy normalize doesn't strip 's' from very short words."""
    # Words 3 chars or less should not have trailing 's' stripped
    assert fuzzy_normalize('as') == 'as'
    assert fuzzy_normalize('is') == 'is'
