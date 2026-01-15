"""
Unit tests for evaluation_metrics.py module.
"""
import unittest
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_metrics import calculate_accuracy


class TestCalculateAccuracy(unittest.TestCase):
    """Test calculate_accuracy function."""

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        predictions = ['CL:0000001', 'CL:0000002', 'CL:0000003']
        ground_truth = ['CL:0000001', 'CL:0000002', 'CL:0000003']

        metrics = calculate_accuracy(predictions, ground_truth)

        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['f1_macro'], 1.0)
        self.assertEqual(metrics['f1_weighted'], 1.0)

    def test_zero_accuracy(self):
        """Test with all wrong predictions."""
        predictions = ['CL:0000001', 'CL:0000001', 'CL:0000001']
        ground_truth = ['CL:0000002', 'CL:0000003', 'CL:0000004']

        metrics = calculate_accuracy(predictions, ground_truth)

        self.assertEqual(metrics['accuracy'], 0.0)

    def test_partial_accuracy(self):
        """Test with some correct predictions."""
        predictions = ['CL:0000001', 'CL:0000002', 'CL:0000003', 'CL:0000001']
        ground_truth = ['CL:0000001', 'CL:0000002', 'CL:0000001', 'CL:0000001']

        metrics = calculate_accuracy(predictions, ground_truth)

        # 3 out of 4 correct
        self.assertEqual(metrics['accuracy'], 0.75)
        self.assertIn('f1_macro', metrics)
        self.assertIn('f1_weighted', metrics)

    def test_top_k_accuracy_all_correct(self):
        """Test top-k accuracy when all ground truth labels are in neighbors."""
        predictions = ['CL:0000001', 'CL:0000002']
        ground_truth = ['CL:0000003', 'CL:0000004']
        neighbor_labels = [
            ['CL:0000001', 'CL:0000002', 'CL:0000003'],  # CL:0000003 is in neighbors
            ['CL:0000004', 'CL:0000002', 'CL:0000001']   # CL:0000004 is in neighbors
        ]

        metrics = calculate_accuracy(predictions, ground_truth, neighbor_labels)

        # Top-k accuracy should be 1.0 (both ground truth in neighbors)
        self.assertEqual(metrics['top_k_accuracy'], 1.0)

    def test_top_k_accuracy_partial(self):
        """Test top-k accuracy with some ground truth labels in neighbors."""
        predictions = ['CL:0000001', 'CL:0000002', 'CL:0000003']
        ground_truth = ['CL:0000001', 'CL:0000005', 'CL:0000003']
        neighbor_labels = [
            ['CL:0000001', 'CL:0000002'],  # CL:0000001 in neighbors ✓
            ['CL:0000002', 'CL:0000003'],  # CL:0000005 NOT in neighbors ✗
            ['CL:0000003', 'CL:0000004']   # CL:0000003 in neighbors ✓
        ]

        metrics = calculate_accuracy(predictions, ground_truth, neighbor_labels)

        # 2 out of 3 correct
        self.assertAlmostEqual(metrics['top_k_accuracy'], 2/3, places=5)

    def test_top_k_accuracy_none_correct(self):
        """Test top-k accuracy when no ground truth labels are in neighbors."""
        predictions = ['CL:0000001', 'CL:0000002']
        ground_truth = ['CL:0000003', 'CL:0000004']
        neighbor_labels = [
            ['CL:0000001', 'CL:0000002'],  # CL:0000003 NOT in neighbors
            ['CL:0000001', 'CL:0000002']   # CL:0000004 NOT in neighbors
        ]

        metrics = calculate_accuracy(predictions, ground_truth, neighbor_labels)

        self.assertEqual(metrics['top_k_accuracy'], 0.0)

    def test_without_neighbor_labels(self):
        """Test that top-k accuracy is not computed when neighbor_labels is None."""
        predictions = ['CL:0000001', 'CL:0000002']
        ground_truth = ['CL:0000001', 'CL:0000003']

        metrics = calculate_accuracy(predictions, ground_truth, neighbor_labels=None)

        self.assertNotIn('top_k_accuracy', metrics)

    def test_f1_scores_with_imbalanced_classes(self):
        """Test F1 scores with imbalanced class distribution."""
        predictions = ['CL:0000001'] * 90 + ['CL:0000002'] * 10
        ground_truth = ['CL:0000001'] * 50 + ['CL:0000002'] * 50

        metrics = calculate_accuracy(predictions, ground_truth)

        # Overall accuracy: first 50 correct (CL:0000001), last 10 correct (CL:0000002) = 60/100 = 0.6
        self.assertEqual(metrics['accuracy'], 0.6)

        # F1 scores should handle class imbalance
        self.assertIn('f1_macro', metrics)
        self.assertIn('f1_weighted', metrics)
        self.assertGreater(metrics['f1_macro'], 0)
        self.assertGreater(metrics['f1_weighted'], 0)

    def test_single_prediction(self):
        """Test with single prediction and ground truth."""
        predictions = ['CL:0000001']
        ground_truth = ['CL:0000001']

        metrics = calculate_accuracy(predictions, ground_truth)

        self.assertEqual(metrics['accuracy'], 1.0)

    def test_empty_lists(self):
        """Test with empty prediction and ground truth lists."""
        predictions = []
        ground_truth = []

        # Should handle gracefully (sklearn handles this)
        try:
            metrics = calculate_accuracy(predictions, ground_truth)
            # If it doesn't raise an error, check that metrics exist
            self.assertIn('accuracy', metrics)
        except Exception:
            # It's also acceptable to raise an exception for empty inputs
            pass

    def test_mismatched_lengths(self):
        """Test that mismatched prediction and ground truth lengths raise error."""
        predictions = ['CL:0000001', 'CL:0000002']
        ground_truth = ['CL:0000001']

        # sklearn will raise ValueError for mismatched lengths
        with self.assertRaises(ValueError):
            calculate_accuracy(predictions, ground_truth)

    def test_all_metrics_present(self):
        """Test that all expected metrics are returned."""
        predictions = ['CL:0000001', 'CL:0000002', 'CL:0000003']
        ground_truth = ['CL:0000001', 'CL:0000002', 'CL:0000001']
        neighbor_labels = [
            ['CL:0000001', 'CL:0000002'],
            ['CL:0000002', 'CL:0000003'],
            ['CL:0000001', 'CL:0000002']
        ]

        metrics = calculate_accuracy(predictions, ground_truth, neighbor_labels)

        # Check all expected metrics
        expected_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'top_k_accuracy']
        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check all values are floats
        for value in metrics.values():
            self.assertIsInstance(value, float)

        # Check all values are in valid range [0, 1]
        for value in metrics.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == '__main__':
    unittest.main()
