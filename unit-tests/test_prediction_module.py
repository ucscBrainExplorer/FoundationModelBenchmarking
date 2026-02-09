"""
Unit tests for prediction_module.py module.
"""
import unittest
import os
import sys
import numpy as np
import pandas as pd
import faiss

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prediction_module import execute_query, vote_neighbors
from data_loader import load_faiss_index, load_reference_annotations


class TestExecuteQuery(unittest.TestCase):
    """Test execute_query function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.index_path = os.path.join(self.test_dir, "indices/index_flat.faiss")
        self.index = load_faiss_index(self.index_path)

        # Create query embeddings (same dimension as index)
        np.random.seed(42)
        self.query_embeddings = np.random.randn(10, 64).astype('float32')

    def test_execute_query_euclidean(self):
        """Test query execution with Euclidean metric."""
        k = 5
        dists, indices = execute_query(
            self.index,
            self.query_embeddings,
            k=k,
            metric='euclidean'
        )

        # Check output shapes
        self.assertEqual(dists.shape, (10, k))
        self.assertEqual(indices.shape, (10, k))

        # Check that indices are valid (within index size)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < self.index.ntotal))

        # Check that distances are non-negative
        self.assertTrue(np.all(dists >= 0))

    def test_execute_query_cosine(self):
        """Test query execution with cosine metric."""
        k = 5
        dists, indices = execute_query(
            self.index,
            self.query_embeddings,
            k=k,
            metric='cosine'
        )

        # Check output shapes
        self.assertEqual(dists.shape, (10, k))
        self.assertEqual(indices.shape, (10, k))

        # Check that indices are valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < self.index.ntotal))

    def test_default_k_value(self):
        """Test that default k=30 works correctly."""
        dists, indices = execute_query(
            self.index,
            self.query_embeddings,
            k=30,
            metric='euclidean'
        )

        self.assertEqual(dists.shape[1], 30)
        self.assertEqual(indices.shape[1], 30)

    def test_query_type_conversion(self):
        """Test that query embeddings are converted to float32."""
        # Create float64 queries
        queries_f64 = np.random.randn(5, 64).astype('float64')

        # Should work without error (function converts to float32)
        dists, indices = execute_query(
            self.index,
            queries_f64,
            k=5,
            metric='euclidean'
        )

        self.assertEqual(dists.shape, (5, 5))

    def test_k_larger_than_index(self):
        """Test behavior when k is larger than index size."""
        # Our index has 100 vectors, try to get 200 neighbors
        # FAISS will return all available vectors
        dists, indices = execute_query(
            self.index,
            self.query_embeddings[:1],
            k=200,
            metric='euclidean'
        )

        # Should return all 100 vectors in the index
        self.assertEqual(dists.shape[1], 200)


class TestVoteNeighbors(unittest.TestCase):
    """Test vote_neighbors function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.ref_path = os.path.join(self.test_dir, "reference_data/prediction_obs.tsv")
        self.ref_df = load_reference_annotations(self.ref_path)

    def test_vote_neighbors_basic(self):
        """Test basic majority voting."""
        # Create simple neighbor indices
        neighbor_indices = np.array([
            [0, 1, 2, 3, 4],  # First query
            [5, 6, 7, 8, 9],  # Second query
        ])

        predictions, vote_percentages = vote_neighbors(neighbor_indices, self.ref_df)

        # Check output
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(vote_percentages), 2)

        # Each prediction should be a valid cell type ontology term ID
        for pred in predictions:
            self.assertIsInstance(pred, str)
            self.assertTrue(pred.startswith('CL:'))
        
        # Vote percentages should be between 0 and 1
        for pct in vote_percentages:
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 1.0)

    def test_vote_neighbors_unanimous(self):
        """Test voting when all neighbors have the same label."""
        # Find indices that have the same label
        first_label = self.ref_df['cell_type_ontology_term_id'].iloc[0]
        same_label_indices = self.ref_df[
            self.ref_df['cell_type_ontology_term_id'] == first_label
        ].index.tolist()

        if len(same_label_indices) >= 5:
            # All neighbors have the same label
            neighbor_indices = np.array([same_label_indices[:5]])

            predictions, vote_percentages = vote_neighbors(neighbor_indices, self.ref_df)

            # Should return the unanimous label
            self.assertEqual(predictions[0], first_label)
            # Unanimous vote should be 100% (1.0)
            self.assertEqual(vote_percentages[0], 1.0)

    def test_vote_neighbors_tie_breaking(self):
        """Test that tie-breaking works (returns first encountered in Counter)."""
        # Create a scenario with ties if possible
        # Counter.most_common returns first encountered in case of tie
        neighbor_indices = np.array([[0, 1, 2, 3, 4]])

        predictions, vote_percentages = vote_neighbors(neighbor_indices, self.ref_df)

        # Should return some valid prediction (tie-breaking is deterministic)
        self.assertIsInstance(predictions[0], str)
        self.assertTrue(predictions[0].startswith('CL:'))
        # Vote percentage should be valid
        self.assertGreater(vote_percentages[0], 0.0)
        self.assertLessEqual(vote_percentages[0], 1.0)

    def test_vote_neighbors_out_of_bounds(self):
        """Test that IndexError is raised for out-of-bounds indices."""
        # Create neighbor indices that exceed reference size
        neighbor_indices = np.array([[0, 1, 2, 1000, 2000]])

        with self.assertRaises(IndexError):
            vote_neighbors(neighbor_indices, self.ref_df)

    def test_vote_neighbors_integration(self):
        """Test integration with execute_query."""
        # Load index and execute a query
        index_path = os.path.join(self.test_dir, "indices/index_flat.faiss")
        index = load_faiss_index(index_path)

        np.random.seed(42)
        queries = np.random.randn(5, 64).astype('float32')

        dists, neighbor_indices = execute_query(index, queries, k=10, metric='euclidean')

        # Vote on neighbors
        predictions, vote_percentages = vote_neighbors(neighbor_indices, self.ref_df)

        # Check predictions
        self.assertEqual(len(predictions), 5)
        self.assertEqual(len(vote_percentages), 5)
        for pred in predictions:
            self.assertIn(pred, self.ref_df['cell_type_ontology_term_id'].values)
        for pct in vote_percentages:
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 1.0)

    def test_empty_neighbor_indices(self):
        """Test behavior with empty neighbor indices."""
        neighbor_indices = np.array([]).reshape(0, 0)

        predictions, vote_percentages = vote_neighbors(neighbor_indices, self.ref_df)

        self.assertEqual(len(predictions), 0)
        self.assertEqual(len(vote_percentages), 0)

    def test_single_neighbor(self):
        """Test voting with k=1 (single neighbor)."""
        neighbor_indices = np.array([[5], [10], [15]])

        predictions, vote_percentages = vote_neighbors(neighbor_indices, self.ref_df)

        # Should return the label of each single neighbor
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(vote_percentages), 3)
        self.assertEqual(predictions[0], self.ref_df['cell_type_ontology_term_id'].iloc[5])
        self.assertEqual(predictions[1], self.ref_df['cell_type_ontology_term_id'].iloc[10])
        self.assertEqual(predictions[2], self.ref_df['cell_type_ontology_term_id'].iloc[15])
        # Single neighbor = 100% vote
        self.assertEqual(vote_percentages[0], 1.0)
        self.assertEqual(vote_percentages[1], 1.0)
        self.assertEqual(vote_percentages[2], 1.0)
    
    def test_vote_neighbors_filters_missing_labels(self):
        """Test that missing/empty/NaN labels are filtered out before voting."""
        import pandas as pd
        
        # Create a test DataFrame with some missing labels
        test_df = pd.DataFrame({
            'cell_type_ontology_term_id': ['CL:0000001', 'CL:0000002', '', np.nan, 'CL:0000001', 'CL:0000002'],
            'cell_type': ['type1', 'type2', '', 'missing', 'type1', 'type2']
        })
        
        # Neighbors: [0, 1, 2, 3, 4, 5]
        # Valid labels: CL:0000001 (indices 0, 4), CL:0000002 (indices 1, 5)
        # Missing labels: '' (index 2), NaN (index 3)
        # Should vote among valid labels only
        neighbor_indices = np.array([[0, 1, 2, 3, 4, 5]])
        
        predictions, vote_percentages = vote_neighbors(neighbor_indices, test_df)
        
        # Should return a valid label (CL:0000001 or CL:0000002), not empty/NaN
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(vote_percentages), 1)
        self.assertIn(predictions[0], ['CL:0000001', 'CL:0000002'])
        self.assertNotEqual(predictions[0], '')
        self.assertNotEqual(predictions[0], np.nan)
        # Vote percentage should be valid
        self.assertGreater(vote_percentages[0], 0.0)
        self.assertLessEqual(vote_percentages[0], 1.0)
    
    def test_vote_neighbors_all_missing_labels(self):
        """Test that empty string is returned when all neighbors have missing labels."""
        import pandas as pd
        
        # Create a test DataFrame with all missing labels
        test_df = pd.DataFrame({
            'cell_type_ontology_term_id': ['', np.nan, '', None],
            'cell_type': ['', 'missing', '', 'missing']
        })
        
        # All neighbors have missing labels
        neighbor_indices = np.array([[0, 1, 2, 3]])
        
        predictions, vote_percentages = vote_neighbors(neighbor_indices, test_df)
        
        # Should return empty string when all labels are missing
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(vote_percentages), 1)
        self.assertEqual(predictions[0], '')
        # Vote percentage should be 0.0 when all labels missing
        self.assertEqual(vote_percentages[0], 0.0)


if __name__ == '__main__':
    unittest.main()
