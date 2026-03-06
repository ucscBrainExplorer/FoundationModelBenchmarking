"""
Tests for cell_labelling/predict.py

Run from cell_labelling/ directory:
  pytest tests/test_predict.py -v

All tests use demodata/ so no network access or large files are needed.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import (
    majority_voting,
    distance_weighted_knn_vote,
    gaussian_kernel_weights,
    neighbor_distances_str,
    parse_obo_names,
    load_ref_annot,
)

DEMODATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demodata')
INDEX    = os.path.join(DEMODATA, 'index_ivfflat.faiss')
ADATA    = os.path.join(DEMODATA, 'query_uce_adata.h5ad')
REF      = os.path.join(DEMODATA, 'ref_obs.tsv.gz')
OBO      = os.path.join(DEMODATA, 'cl-basic.obo')
PREDICT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predict.py')


# ---------------------------------------------------------------------------
# Unit tests — majority_voting
# ---------------------------------------------------------------------------

class TestMajorityVoting:

    def _make_inputs(self, label_rows, dist_rows):
        """Build minimal numpy inputs for majority_voting."""
        n, k = len(label_rows), len(label_rows[0])
        term_ids = np.array([t for row in label_rows for t in row])
        # Map each unique label to a sequential index
        uniq = list(dict.fromkeys(term_ids))
        label_to_idx = {l: i for i, l in enumerate(uniq)}
        indices = np.array([[label_to_idx[t] for t in row] for row in label_rows])
        dists   = np.array(dist_rows, dtype=np.float32)
        # Build term_ids array where index i → uniq[i]
        ref_term_ids = np.array(uniq)
        return indices, dists, ref_term_ids

    def test_clear_winner(self):
        labels = [['CL:A', 'CL:A', 'CL:A', 'CL:B', 'CL:B']]
        dists  = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        idx, d, terms = self._make_inputs(labels, dists)
        top1, s1, top2, s2 = majority_voting(idx, d, terms)
        assert top1[0] == 'CL:A'
        assert pytest.approx(s1[0]) == 3 / 5
        assert top2[0] == 'CL:B'
        assert pytest.approx(s2[0]) == 2 / 5

    def test_tie_returns_one_winner(self):
        labels = [['CL:A', 'CL:A', 'CL:B', 'CL:B']]
        dists  = [[0.1, 0.2, 0.3, 0.4]]
        idx, d, terms = self._make_inputs(labels, dists)
        top1, s1, top2, s2 = majority_voting(idx, d, terms)
        assert top1[0] in ('CL:A', 'CL:B')
        assert pytest.approx(s1[0]) == 0.5

    def test_single_label(self):
        labels = [['CL:A', 'CL:A', 'CL:A']]
        dists  = [[0.1, 0.2, 0.3]]
        idx, d, terms = self._make_inputs(labels, dists)
        top1, s1, top2, s2 = majority_voting(idx, d, terms)
        assert top1[0] == 'CL:A'
        assert pytest.approx(s1[0]) == 1.0
        assert top2[0] == ''
        assert np.isnan(s2[0])

    def test_multiple_cells(self):
        labels = [['CL:A', 'CL:A', 'CL:B'],
                  ['CL:C', 'CL:C', 'CL:C']]
        dists  = [[0.1, 0.2, 0.3],
                  [0.1, 0.2, 0.3]]
        idx, d, terms = self._make_inputs(labels, dists)
        top1, s1, top2, s2 = majority_voting(idx, d, terms)
        assert top1[0] == 'CL:A'
        assert top1[1] == 'CL:C'
        assert pytest.approx(s1[1]) == 1.0


# ---------------------------------------------------------------------------
# Unit tests — distance_weighted_knn_vote
# ---------------------------------------------------------------------------

class TestDistanceWeightedKNN:

    def test_closer_neighbor_wins(self):
        # CL:A is 1 neighbor but much closer; CL:B has 2 neighbors but further
        # With Gaussian weighting, CL:A should win
        weights = np.array([[0.9, 0.1, 0.1]])  # CL:A gets 0.9, CL:B gets 0.1+0.1
        terms   = np.array([['CL:A', 'CL:B', 'CL:B']])
        top1, s1, top2, s2 = distance_weighted_knn_vote(weights, terms)
        assert top1[0] == 'CL:A'

    def test_empty_labels(self):
        weights = np.array([[0.5, 0.5]])
        terms   = np.array([['', '']])
        top1, s1, top2, s2 = distance_weighted_knn_vote(weights, terms)
        assert top1[0] == ''
        assert np.isnan(s1[0])

    def test_scores_sum_to_one(self):
        weights = np.array([[0.5, 0.3, 0.2]])
        terms   = np.array([['CL:A', 'CL:B', 'CL:A']])
        top1, s1, top2, s2 = distance_weighted_knn_vote(weights, terms)
        assert pytest.approx(s1[0] + s2[0], abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# Unit tests — gaussian_kernel_weights
# ---------------------------------------------------------------------------

class TestGaussianKernelWeights:

    def test_closer_gets_higher_weight(self):
        dists = np.array([[0.1, 0.5, 1.0]])
        w = gaussian_kernel_weights(dists)
        assert w[0, 0] > w[0, 1] > w[0, 2]

    def test_output_shape(self):
        dists = np.random.rand(10, 30).astype(np.float32)
        w = gaussian_kernel_weights(dists)
        assert w.shape == (10, 30)

    def test_weights_positive(self):
        dists = np.random.rand(5, 10).astype(np.float32)
        w = gaussian_kernel_weights(dists)
        assert np.all(w > 0)


# ---------------------------------------------------------------------------
# Unit tests — neighbor_distances_str
# ---------------------------------------------------------------------------

class TestNeighborDistancesStr:

    def test_sorted_closest_first(self):
        indices = np.array([[2, 0, 1]])
        dists   = np.array([[0.3, 0.1, 0.2]])
        result  = neighbor_distances_str(indices, dists)
        vals = [float(x) for x in result[0].split(',')]
        assert vals == sorted(vals)

    def test_faiss_sentinels_excluded(self):
        indices = np.array([[0, -1, 1]])
        dists   = np.array([[0.2, 99.9, 0.4]])
        result  = neighbor_distances_str(indices, dists)
        assert '-1' not in result[0]
        assert len(result[0].split(',')) == 2


# ---------------------------------------------------------------------------
# Unit tests — parse_obo_names
# ---------------------------------------------------------------------------

class TestParseOboNames:

    def test_parses_demo_obo(self):
        cl_names = parse_obo_names(OBO)
        assert isinstance(cl_names, dict)
        assert len(cl_names) > 0
        # All values should be non-empty strings
        assert all(isinstance(v, str) and v for v in cl_names.values())

    def test_file_not_found(self):
        from predict import load_ref_annot
        with pytest.raises(FileNotFoundError):
            parse_obo_names('nonexistent.obo')


# ---------------------------------------------------------------------------
# Integration tests — CLI via subprocess
# ---------------------------------------------------------------------------

class TestCLI:

    def test_distance_weighted_knn(self, tmp_path, small_adata_path):
        out = tmp_path / 'labels.tsv'
        result = subprocess.run(
            ['python3', PREDICT,
             '--index',     INDEX,
             '--adata',     small_adata_path,
             '--ref_annot', REF,
             '--obo',       OBO,
             '--method',    'distance_weighted_knn',
             '--output',    str(out)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        df = pd.read_csv(out, sep='\t', comment='#')
        assert 'cell_id' in df.columns
        assert 'weighted_cell_type_ontology_term_id' in df.columns
        assert 'weighted_cell_type' in df.columns
        assert 'weighted_score' in df.columns
        assert 'mean_euclidean_distance' in df.columns
        assert 'neighbor_distances' in df.columns
        assert len(df) > 0
        assert (df['weighted_score'] >= 0).all()
        assert (df['weighted_score'] <= 1 + 1e-5).all()  # float32 sum tolerance

    def test_majority_voting(self, tmp_path, small_adata_path):
        out = tmp_path / 'labels_mv.tsv'
        result = subprocess.run(
            ['python3', PREDICT,
             '--index',     INDEX,
             '--adata',     small_adata_path,
             '--ref_annot', REF,
             '--obo',       OBO,
             '--method',    'majority_voting',
             '--output',    str(out)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        df = pd.read_csv(out, sep='\t', comment='#')
        assert 'mv_cell_type_ontology_term_id' in df.columns
        assert 'mv_cell_type' in df.columns
        assert 'mv_score' in df.columns
        assert df['mv_score'].between(0, 1, inclusive='both').all()

    def test_both_methods(self, tmp_path, small_adata_path):
        out = tmp_path / 'labels_both.tsv'
        result = subprocess.run(
            ['python3', PREDICT,
             '--index',     INDEX,
             '--adata',     small_adata_path,
             '--ref_annot', REF,
             '--obo',       OBO,
             '--method',    'both',
             '--output',    str(out)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        df = pd.read_csv(out, sep='\t', comment='#')
        for col in ['mv_cell_type', 'mv_score',
                    'weighted_cell_type', 'weighted_score']:
            assert col in df.columns

    def test_provenance_header(self, tmp_path, small_adata_path):
        out = tmp_path / 'labels.tsv'
        subprocess.run(
            ['python3', PREDICT,
             '--index',     INDEX,
             '--adata',     small_adata_path,
             '--ref_annot', REF,
             '--obo',       OBO,
             '--output',    str(out)],
            capture_output=True
        )
        with open(out) as f:
            header = [l for l in f if l.startswith('#')]
        assert any('index' in l for l in header)
        assert any('method' in l for l in header)
        assert any('k:' in l for l in header)

    def test_invalid_index_path(self, tmp_path, small_adata_path):
        result = subprocess.run(
            ['python3', PREDICT,
             '--index',     'nonexistent.faiss',
             '--adata',     small_adata_path,
             '--ref_annot', REF,
             '--obo',       OBO],
            capture_output=True, text=True
        )
        assert result.returncode != 0
