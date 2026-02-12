"""Pytest fixtures and configuration for the test suite."""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_obo_file(temp_dir):
    """Create a minimal OBO file for testing."""
    obo_path = os.path.join(temp_dir, "test.obo")
    content = """format-version: 1.2
ontology: cl

[Term]
id: CL:0000000
name: cell

[Term]
id: CL:0000540
name: neuron
synonym: "nerve cell" EXACT []

[Term]
id: CL:0000128
name: oligodendrocyte
is_a: CL:0000540

[Term]
id: CL:0000127
name: astrocyte
is_a: CL:0000540

[Typedef]
id: part_of
name: part of
"""
    with open(obo_path, 'w') as f:
        f.write(content)
    return obo_path


@pytest.fixture
def sample_embeddings(temp_dir):
    """Create sample embeddings numpy file."""
    embeddings = np.random.rand(10, 128).astype(np.float32)
    path = os.path.join(temp_dir, "embeddings.npy")
    np.save(path, embeddings)
    return path, embeddings


@pytest.fixture
def sample_reference_annotations(temp_dir):
    """Create sample reference annotations TSV."""
    df = pd.DataFrame({
        'cell_type_ontology_term_id': ['CL:0000540'] * 5 + ['CL:0000128'] * 5,
        'cell_type': ['neuron'] * 5 + ['oligodendrocyte'] * 5,
    })
    path = os.path.join(temp_dir, "ref_annot.tsv")
    df.to_csv(path, sep='\t', index=False)
    return path, df


@pytest.fixture
def sample_predictions(temp_dir):
    """Create sample predictions TSV."""
    df = pd.DataFrame({
        'predicted_cell_type_ontology_term_id': ['CL:0000540'] * 6 + ['CL:0000128'] * 4,
        'predicted_cell_type': ['neuron'] * 6 + ['oligodendrocyte'] * 4,
        'vote_percentage': [0.8] * 10,
        'mean_euclidean_distance': np.random.rand(10).tolist(),
        'neighbor_distances': [','.join(['0.1'] * 30) for _ in range(10)],
        'neighbor_cell_types': [','.join(['neuron'] * 30) for _ in range(10)],
    })
    path = os.path.join(temp_dir, "predictions.tsv")
    df.to_csv(path, sep='\t', index=False)
    return path, df


@pytest.fixture
def sample_ground_truth(temp_dir):
    """Create sample ground truth TSV."""
    df = pd.DataFrame({
        'cell_type_ontology_term_id': ['CL:0000540'] * 5 + ['CL:0000128'] * 5,
        'cell_type': ['neuron'] * 5 + ['oligodendrocyte'] * 5,
    })
    path = os.path.join(temp_dir, "ground_truth.tsv")
    df.to_csv(path, sep='\t', index=False)
    return path, df
