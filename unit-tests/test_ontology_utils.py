"""
Unit tests for ontology_utils.py module.
"""
import unittest
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ontology_utils import load_ontology, calculate_graph_distance, score_batch


class TestLoadOntology(unittest.TestCase):
    """Test load_ontology function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.obo_path = os.path.join(self.test_dir, "reference_data/cl.obo")

    def test_load_valid_ontology(self):
        """Test loading a valid OBO file."""
        graph = load_ontology(self.obo_path)

        # Check that graph was created
        self.assertIsNotNone(graph)

        # Check that nodes were added
        self.assertGreater(graph.number_of_nodes(), 0)

        # Check specific nodes from our mock OBO
        expected_nodes = ['CL:0000000', 'CL:0000001', 'CL:0000002',
                          'CL:0000003', 'CL:0000004', 'CL:0000005']
        for node in expected_nodes:
            self.assertIn(node, graph.nodes)

    def test_ontology_hierarchy(self):
        """Test that ontology hierarchy is correctly represented."""
        graph = load_ontology(self.obo_path)

        # All specific cell types should have an edge to CL:0000000 (cell)
        # Either directly or indirectly
        # CL:0000001 (neuron) should have edge to CL:0000000 (cell)
        # This means there should be a path from neuron to cell
        self.assertTrue(graph.has_edge('CL:0000001', 'CL:0000000'))

    def test_node_attributes(self):
        """Test that node attributes (like name) are preserved."""
        graph = load_ontology(self.obo_path)

        # Check that nodes have 'name' attribute
        node_data = graph.nodes['CL:0000001']
        self.assertIn('name', node_data)
        self.assertEqual(node_data['name'], 'neuron')

    def test_file_not_found(self):
        """Test that error is raised for missing OBO file."""
        with self.assertRaises(Exception):  # Could be FileNotFoundError or pronto exception
            load_ontology("nonexistent.obo")


class TestCalculateGraphDistance(unittest.TestCase):
    """Test calculate_graph_distance function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.obo_path = os.path.join(self.test_dir, "reference_data/cl.obo")
        self.graph = load_ontology(self.obo_path)

    def test_identical_terms(self):
        """Test distance between identical terms is 0."""
        dist = calculate_graph_distance(self.graph, 'CL:0000001', 'CL:0000001')
        self.assertEqual(dist, 0)

    def test_parent_child_distance(self):
        """Test distance between parent and child."""
        # CL:0000001 (neuron) is child of CL:0000000 (cell)
        dist = calculate_graph_distance(self.graph, 'CL:0000001', 'CL:0000000')

        # Distance should be 1 (direct parent-child)
        self.assertEqual(dist, 1)

    def test_sibling_distance(self):
        """Test distance between sibling terms."""
        # CL:0000001 (neuron) and CL:0000002 (astrocyte) are both children of CL:0000000
        dist = calculate_graph_distance(self.graph, 'CL:0000001', 'CL:0000002')

        # Distance should be 2 (up 1 to parent, down 1 to sibling)
        self.assertEqual(dist, 2)

    def test_subtype_distance(self):
        """Test distance between specific subtypes."""
        # CL:0000006 (GABAergic neuron) is child of CL:0000001 (neuron)
        # CL:0000007 (glutamatergic neuron) is also child of CL:0000001 (neuron)
        dist = calculate_graph_distance(self.graph, 'CL:0000006', 'CL:0000007')

        # Distance should be 2 (up 1 to neuron, down 1 to sibling)
        self.assertEqual(dist, 2)

    def test_ancestor_descendant_distance(self):
        """Test distance between ancestor and descendant."""
        # CL:0000006 (GABAergic neuron) -> CL:0000001 (neuron) -> CL:0000000 (cell)
        dist = calculate_graph_distance(self.graph, 'CL:0000006', 'CL:0000000')

        # Distance should be 2 (two hops to ancestor)
        self.assertEqual(dist, 2)

    def test_nonexistent_term(self):
        """Test that -1 is returned for nonexistent terms."""
        dist = calculate_graph_distance(self.graph, 'CL:9999999', 'CL:0000001')
        self.assertEqual(dist, -1)

        dist = calculate_graph_distance(self.graph, 'CL:0000001', 'CL:9999999')
        self.assertEqual(dist, -1)

    def test_both_nonexistent(self):
        """Test that -1 is returned when both terms don't exist."""
        dist = calculate_graph_distance(self.graph, 'CL:9999999', 'CL:8888888')
        self.assertEqual(dist, -1)


class TestScoreBatch(unittest.TestCase):
    """Test score_batch function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.obo_path = os.path.join(self.test_dir, "reference_data/cl.obo")
        self.graph = load_ontology(self.obo_path)

    def test_perfect_predictions(self):
        """Test batch scoring with perfect predictions."""
        predictions = ['CL:0000001', 'CL:0000002', 'CL:0000003']
        ground_truth = ['CL:0000001', 'CL:0000002', 'CL:0000003']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # All distances should be 0
        self.assertEqual(mean_dist, 0.0)
        self.assertEqual(median_dist, 0.0)

    def test_parent_child_predictions(self):
        """Test batch scoring with parent-child predictions."""
        # All predictions are off by one level (parent instead of child)
        predictions = ['CL:0000000', 'CL:0000000', 'CL:0000000']
        ground_truth = ['CL:0000001', 'CL:0000002', 'CL:0000003']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # All distances should be 1
        self.assertEqual(mean_dist, 1.0)
        self.assertEqual(median_dist, 1.0)

    def test_mixed_distances(self):
        """Test batch scoring with varying distances."""
        predictions = ['CL:0000001', 'CL:0000002', 'CL:0000001']
        ground_truth = ['CL:0000001', 'CL:0000001', 'CL:0000002']
        # Distances: 0, 2, 2

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # Mean should be (0 + 2 + 2) / 3 = 1.333...
        self.assertAlmostEqual(mean_dist, 4/3, places=5)
        # Median should be 2
        self.assertEqual(median_dist, 2.0)

    def test_with_invalid_terms(self):
        """Test that invalid terms are ignored in scoring."""
        predictions = ['CL:0000001', 'CL:9999999', 'CL:0000002']
        ground_truth = ['CL:0000001', 'CL:0000002', 'CL:0000002']
        # Valid distances: 0 (first), invalid (second), 0 (third)

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # Should only count valid distances (0, 0)
        self.assertEqual(mean_dist, 0.0)
        self.assertEqual(median_dist, 0.0)

    def test_empty_batch(self):
        """Test batch scoring with empty lists."""
        predictions = []
        ground_truth = []

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # Should return 0.0 for empty batch
        self.assertEqual(mean_dist, 0.0)
        self.assertEqual(median_dist, 0.0)

    def test_all_invalid_terms(self):
        """Test batch scoring when all terms are invalid."""
        predictions = ['CL:9999999', 'CL:8888888']
        ground_truth = ['CL:7777777', 'CL:6666666']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # Should return 0.0 when no valid distances
        self.assertEqual(mean_dist, 0.0)
        self.assertEqual(median_dist, 0.0)

    def test_single_prediction(self):
        """Test batch scoring with single prediction."""
        predictions = ['CL:0000001']
        ground_truth = ['CL:0000002']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # Mean and median should be the same for single value
        self.assertEqual(mean_dist, median_dist)
        # Distance between siblings should be 2
        self.assertEqual(mean_dist, 2.0)

    def test_large_batch(self):
        """Test batch scoring with larger batch."""
        # Create a batch of 100 predictions
        predictions = ['CL:0000001'] * 50 + ['CL:0000002'] * 50
        ground_truth = ['CL:0000001'] * 25 + ['CL:0000002'] * 25 + \
                       ['CL:0000001'] * 25 + ['CL:0000002'] * 25

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        # Should handle large batches without error
        self.assertIsInstance(mean_dist, float)
        self.assertIsInstance(median_dist, float)
        self.assertGreaterEqual(mean_dist, 0.0)
        self.assertGreaterEqual(median_dist, 0.0)


if __name__ == '__main__':
    unittest.main()
