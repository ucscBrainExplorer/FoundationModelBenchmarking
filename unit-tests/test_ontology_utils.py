"""
Unit tests for ontology_utils.py module.

Uses real Cell Ontology (CL) IDs from cl.obo:
    CL:0000000 = cell (root)
    CL:0000540 = neuron (is_a: electrically responsive cell, electrically signaling cell, neural cell)
    CL:0000127 = astrocyte (is_a: macroglial cell)
    CL:0000128 = oligodendrocyte (is_a: macroglial cell, myelinating glial cell)
    CL:0000617 = GABAergic neuron (is_a: secretory cell, neuron)
    CL:0000679 = glutamatergic neuron (is_a: secretory cell, neuron)
    CL:0000125 = glial cell
    CL:0000126 = macroglial cell (is_a: glial cell)
    CL:0002319 = neural cell
"""
import unittest
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ontology_utils import load_ontology, calculate_graph_distance, score_batch, precompute_ic, calculate_lin_similarity


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

        # Check specific nodes exist
        expected_nodes = ['CL:0000000', 'CL:0000540', 'CL:0000127',
                          'CL:0000617', 'CL:0000679']
        for node in expected_nodes:
            self.assertIn(node, graph.nodes)

    def test_ontology_hierarchy(self):
        """Test that ontology hierarchy is correctly represented."""
        graph = load_ontology(self.obo_path)

        # neuron (CL:0000540) should have a direct is_a edge to neural cell (CL:0002319)
        self.assertTrue(graph.has_edge('CL:0000540', 'CL:0002319'))

        # astrocyte (CL:0000127) should have a direct is_a edge to macroglial cell (CL:0000126)
        self.assertTrue(graph.has_edge('CL:0000127', 'CL:0000126'))

    def test_node_attributes(self):
        """Test that node attributes (like name) are preserved."""
        graph = load_ontology(self.obo_path)

        # Check that nodes have 'name' attribute with correct values
        self.assertEqual(graph.nodes['CL:0000000']['name'], 'cell')
        self.assertEqual(graph.nodes['CL:0000540']['name'], 'neuron')
        self.assertEqual(graph.nodes['CL:0000127']['name'], 'astrocyte')
        self.assertEqual(graph.nodes['CL:0000617']['name'], 'GABAergic neuron')

    def test_multiple_parents(self):
        """Test that DAG structure (multiple parents) is preserved."""
        graph = load_ontology(self.obo_path)

        # neuron has 3 parents: electrically responsive cell, electrically signaling cell, neural cell
        neuron_parents = list(graph.successors('CL:0000540'))
        self.assertGreaterEqual(len(neuron_parents), 3)
        self.assertIn('CL:0002319', neuron_parents)  # neural cell

        # glutamatergic neuron has 2 parents: secretory cell and neuron
        glut_parents = list(graph.successors('CL:0000679'))
        self.assertIn('CL:0000540', glut_parents)  # neuron
        self.assertIn('CL:0000151', glut_parents)  # secretory cell

    def test_file_not_found(self):
        """Test that error is raised for missing OBO file."""
        with self.assertRaises(Exception):
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
        dist = calculate_graph_distance(self.graph, 'CL:0000540', 'CL:0000540')
        self.assertEqual(dist, 0)

    def test_parent_child_distance(self):
        """Test distance between direct parent and child."""
        # neuron (CL:0000540) is_a neural cell (CL:0002319) — 1 hop
        dist = calculate_graph_distance(self.graph, 'CL:0000540', 'CL:0002319')
        self.assertEqual(dist, 1)

        # astrocyte (CL:0000127) is_a macroglial cell (CL:0000126) — 1 hop
        dist = calculate_graph_distance(self.graph, 'CL:0000127', 'CL:0000126')
        self.assertEqual(dist, 1)

    def test_sibling_distance(self):
        """Test distance between sibling terms (share a parent)."""
        # astrocyte and oligodendrocyte are both is_a macroglial cell
        # distance = 2 (up 1 to macroglial cell, down 1 to sibling)
        dist = calculate_graph_distance(self.graph, 'CL:0000127', 'CL:0000128')
        self.assertEqual(dist, 2)

    def test_subtype_distance(self):
        """Test distance between neurotransmitter subtypes."""
        # GABAergic neuron (CL:0000617) and glutamatergic neuron (CL:0000679)
        # both is_a neuron and is_a secretory cell
        # distance = 2 (up 1 to shared parent, down 1)
        dist = calculate_graph_distance(self.graph, 'CL:0000617', 'CL:0000679')
        self.assertEqual(dist, 2)

    def test_ancestor_descendant_distance(self):
        """Test distance between ancestor and deeper descendant."""
        # GABAergic neuron (CL:0000617) to cell (CL:0000000)
        # GABAergic -> secretory cell -> cell = 2 hops (shortest path)
        dist = calculate_graph_distance(self.graph, 'CL:0000617', 'CL:0000000')
        self.assertEqual(dist, 2)

        # neuron (CL:0000540) to cell (CL:0000000) = 3 hops
        dist = calculate_graph_distance(self.graph, 'CL:0000540', 'CL:0000000')
        self.assertEqual(dist, 3)

    def test_cross_lineage_distance(self):
        """Test distance between cells in different lineages."""
        # neuron vs astrocyte — different branches under neural cell
        dist = calculate_graph_distance(self.graph, 'CL:0000540', 'CL:0000127')
        self.assertEqual(dist, 4)

    def test_nonexistent_term(self):
        """Test that -1 is returned for nonexistent terms."""
        dist = calculate_graph_distance(self.graph, 'CL:9999999', 'CL:0000540')
        self.assertEqual(dist, -1)

        dist = calculate_graph_distance(self.graph, 'CL:0000540', 'CL:9999999')
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
        predictions = ['CL:0000540', 'CL:0000127', 'CL:0000617']
        ground_truth = ['CL:0000540', 'CL:0000127', 'CL:0000617']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertEqual(mean_dist, 0.0)
        self.assertEqual(median_dist, 0.0)

    def test_parent_child_predictions(self):
        """Test batch scoring with parent-child predictions."""
        # Predict neural cell instead of neuron (off by 1 hop)
        # Predict macroglial cell instead of astrocyte (off by 1 hop)
        predictions = ['CL:0002319', 'CL:0000126']
        ground_truth = ['CL:0000540', 'CL:0000127']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertEqual(mean_dist, 1.0)
        self.assertEqual(median_dist, 1.0)

    def test_mixed_distances(self):
        """Test batch scoring with varying distances."""
        predictions = ['CL:0000540', 'CL:0000127', 'CL:0000617']
        ground_truth = ['CL:0000540', 'CL:0000128', 'CL:0000679']
        # Distances: 0 (exact), 2 (astrocyte vs oligo), 2 (GABAergic vs glutamatergic)

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertAlmostEqual(mean_dist, 4/3, places=5)
        self.assertEqual(median_dist, 2.0)

    def test_with_invalid_terms(self):
        """Test that invalid terms are ignored in scoring."""
        predictions = ['CL:0000540', 'CL:9999999', 'CL:0000127']
        ground_truth = ['CL:0000540', 'CL:0000127', 'CL:0000127']
        # Valid distances: 0 (first), invalid (second, skipped), 0 (third)

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertEqual(mean_dist, 0.0)
        self.assertEqual(median_dist, 0.0)

    def test_empty_batch(self):
        """Test batch scoring with empty lists."""
        import math
        predictions = []
        ground_truth = []

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertTrue(math.isnan(mean_dist))
        self.assertTrue(math.isnan(median_dist))

    def test_all_invalid_terms(self):
        """Test batch scoring when all terms are invalid."""
        import math
        predictions = ['CL:9999999', 'CL:8888888']
        ground_truth = ['CL:7777777', 'CL:6666666']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertTrue(math.isnan(mean_dist))
        self.assertTrue(math.isnan(median_dist))

    def test_single_prediction(self):
        """Test batch scoring with single prediction."""
        # astrocyte vs oligodendrocyte = distance 2 (siblings under macroglial cell)
        predictions = ['CL:0000127']
        ground_truth = ['CL:0000128']

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertEqual(mean_dist, median_dist)
        self.assertEqual(mean_dist, 2.0)

    def test_large_batch(self):
        """Test batch scoring with larger batch."""
        # 100 predictions: half exact, half off by 2 (astrocyte vs oligo)
        predictions = ['CL:0000540'] * 50 + ['CL:0000127'] * 50
        ground_truth = ['CL:0000540'] * 50 + ['CL:0000128'] * 50
        # 50 distances of 0, 50 distances of 2

        mean_dist, median_dist = score_batch(self.graph, predictions, ground_truth)

        self.assertIsInstance(mean_dist, float)
        self.assertIsInstance(median_dist, float)
        self.assertAlmostEqual(mean_dist, 1.0, places=5)  # (50*0 + 50*2) / 100
        self.assertAlmostEqual(median_dist, 1.0, places=5)  # median of [0,0,...,2,2,...]


class TestLinSimilarity(unittest.TestCase):
    """Test IC-based Lin similarity (Zhou k=0.5 IC).

    Expected values verified against ic_formula_comparison.py output.
    See IC_FORMULA_ANALYSIS.md and ic_formula_comparison_results.txt.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.obo_path = os.path.join(self.test_dir, "reference_data/cl.obo")
        self.graph = load_ontology(self.obo_path)
        self.ic_values = precompute_ic(self.graph, k=0.5)

    def test_identical_terms(self):
        """Identical terms should have similarity 1.0."""
        sim = calculate_lin_similarity(self.graph, 'CL:0000540', 'CL:0000540', self.ic_values)
        self.assertEqual(sim, 1.0)

    def test_nonexistent_term(self):
        """Missing term should return -1.0."""
        sim = calculate_lin_similarity(self.graph, 'CL:9999999', 'CL:0000540', self.ic_values)
        self.assertEqual(sim, -1.0)

    def test_neuron_subtypes_high_similarity(self):
        """GABAergic and glutamatergic neurons should be very similar (both neuron subtypes)."""
        sim = calculate_lin_similarity(self.graph, 'CL:0000617', 'CL:0000679', self.ic_values)
        self.assertGreater(sim, 0.8)  # verified: ~0.870

    def test_sibling_glia_high_similarity(self):
        """Astrocyte and oligodendrocyte should be very similar (both macroglial cells)."""
        sim = calculate_lin_similarity(self.graph, 'CL:0000127', 'CL:0000128', self.ic_values)
        self.assertGreater(sim, 0.8)  # verified: ~0.880

    def test_neuron_vs_astrocyte_moderate(self):
        """Neuron and astrocyte should have moderate similarity (different cell classes, both neural)."""
        sim = calculate_lin_similarity(self.graph, 'CL:0000540', 'CL:0000127', self.ic_values)
        self.assertGreater(sim, 0.4)
        self.assertLess(sim, 0.7)  # verified: ~0.582

    def test_similarity_ordering(self):
        """Biologically closer pairs should have higher similarity."""
        # neuron subtypes > neuron vs astrocyte
        sim_subtypes = calculate_lin_similarity(self.graph, 'CL:0000617', 'CL:0000679', self.ic_values)
        sim_cross = calculate_lin_similarity(self.graph, 'CL:0000540', 'CL:0000127', self.ic_values)
        self.assertGreater(sim_subtypes, sim_cross)

    def test_symmetry(self):
        """Lin similarity should be symmetric: sim(A,B) = sim(B,A)."""
        sim_ab = calculate_lin_similarity(self.graph, 'CL:0000540', 'CL:0000127', self.ic_values)
        sim_ba = calculate_lin_similarity(self.graph, 'CL:0000127', 'CL:0000540', self.ic_values)
        self.assertAlmostEqual(sim_ab, sim_ba, places=10)

    def test_root_has_lowest_ic(self):
        """Root node (cell) should have the lowest IC of all terms."""
        root_ic = self.ic_values['CL:0000000']
        for term_id, ic in self.ic_values.items():
            if term_id != 'CL:0000000':
                self.assertLessEqual(root_ic, ic)


class TestScoreBatchIC(unittest.TestCase):
    """Test score_batch with method='ic'."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.obo_path = os.path.join(self.test_dir, "reference_data/cl.obo")
        self.graph = load_ontology(self.obo_path)
        self.ic_values = precompute_ic(self.graph, k=0.5)

    def test_perfect_predictions(self):
        """Perfect predictions should give mean similarity = 1.0."""
        predictions = ['CL:0000540', 'CL:0000127', 'CL:0000617']
        ground_truth = ['CL:0000540', 'CL:0000127', 'CL:0000617']

        mean_sim, median_sim = score_batch(self.graph, predictions, ground_truth,
                                           method='ic', ic_values=self.ic_values)
        self.assertEqual(mean_sim, 1.0)
        self.assertEqual(median_sim, 1.0)

    def test_all_invalid_returns_nan(self):
        """All invalid terms should return NaN."""
        import math
        predictions = ['CL:9999999', 'CL:8888888']
        ground_truth = ['CL:7777777', 'CL:6666666']

        mean_sim, median_sim = score_batch(self.graph, predictions, ground_truth,
                                           method='ic', ic_values=self.ic_values)
        self.assertTrue(math.isnan(mean_sim))
        self.assertTrue(math.isnan(median_sim))

    def test_similar_pairs_high_score(self):
        """Closely related pairs should produce high mean similarity."""
        # astrocyte vs oligodendrocyte (~0.880), GABAergic vs glutamatergic (~0.870)
        predictions = ['CL:0000127', 'CL:0000617']
        ground_truth = ['CL:0000128', 'CL:0000679']

        mean_sim, median_sim = score_batch(self.graph, predictions, ground_truth,
                                           method='ic', ic_values=self.ic_values)
        self.assertGreater(mean_sim, 0.8)


if __name__ == '__main__':
    unittest.main()
