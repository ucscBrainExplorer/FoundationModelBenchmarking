"""
Unit tests for main_benchmark.py module.
"""
import unittest
import os
import sys
import tempfile
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_benchmark import run_benchmark


class TestRunBenchmark(unittest.TestCase):
    """Test run_benchmark function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.index_paths = {
            "flat": os.path.join(self.test_dir, "indices/index_flat.faiss"),
            "ivfFlat": os.path.join(self.test_dir, "indices/index_ivfflat.faiss")
        }
        self.ref_annotation_path = os.path.join(self.test_dir, "reference_data/prediction_obs.tsv")
        self.test_data_dir = os.path.join(self.test_dir, "test_data")
        self.obo_path = os.path.join(self.test_dir, "reference_data/cl.obo")

    def tearDown(self):
        """Clean up after tests."""
        # Remove benchmark results file if it exists
        if os.path.exists("benchmark_results.csv"):
            os.remove("benchmark_results.csv")

    def test_run_benchmark_basic(self):
        """Test basic benchmark execution."""
        # Run benchmark without S3 download
        run_benchmark(
            index_paths=self.index_paths,
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        # Check that results file was created
        self.assertTrue(os.path.exists("benchmark_results.csv"))

        # Load and verify results
        results_df = pd.read_csv("benchmark_results.csv")

        # Should have results for all index-metric-dataset combinations
        self.assertGreater(len(results_df), 0)

        # Check that required columns exist
        expected_columns = [
            'Index', 'Metric', 'Dataset', 'Avg_Query_Time_ms',
            'accuracy', 'f1_macro', 'f1_weighted', 'top_k_accuracy',
            'mean_ontology_dist', 'median_ontology_dist'
        ]
        for col in expected_columns:
            self.assertIn(col, results_df.columns)

    def test_run_benchmark_multiple_metrics(self):
        """Test benchmark with multiple distance metrics."""
        run_benchmark(
            index_paths={"flat": self.index_paths["flat"]},
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean', 'cosine'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Should have results for both metrics
        metrics_in_results = results_df['Metric'].unique()
        self.assertIn('euclidean', metrics_in_results)
        self.assertIn('cosine', metrics_in_results)

    def test_run_benchmark_multiple_indices(self):
        """Test benchmark with multiple indices."""
        run_benchmark(
            index_paths=self.index_paths,
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Should have results for both indices
        indices_in_results = results_df['Index'].unique()
        self.assertIn('flat', indices_in_results)
        self.assertIn('ivfFlat', indices_in_results)

    def test_run_benchmark_multiple_datasets(self):
        """Test that benchmark processes multiple test datasets."""
        run_benchmark(
            index_paths={"flat": self.index_paths["flat"]},
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Should have multiple datasets
        datasets = results_df['Dataset'].unique()
        self.assertGreater(len(datasets), 1)

        # Check that our mock datasets are present
        dataset_list = datasets.tolist()
        self.assertTrue(
            any('organoid_test' in ds or 'brain_test' in ds or 'positive_control' in ds
                for ds in dataset_list)
        )

    def test_run_benchmark_without_ontology(self):
        """Test benchmark when ontology file is missing."""
        run_benchmark(
            index_paths={"flat": self.index_paths["flat"]},
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path="nonexistent.obo",
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Ontology metrics should be NaN
        self.assertTrue(results_df['mean_ontology_dist'].isna().all())
        self.assertTrue(results_df['median_ontology_dist'].isna().all())

    def test_run_benchmark_missing_index(self):
        """Test that missing index files are skipped gracefully."""
        index_paths = {
            "flat": self.index_paths["flat"],
            "nonexistent": "nonexistent_index.faiss"
        }

        run_benchmark(
            index_paths=index_paths,
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Should only have results for the valid index
        indices = results_df['Index'].unique()
        self.assertIn('flat', indices)
        self.assertNotIn('nonexistent', indices)

    def test_run_benchmark_missing_reference(self):
        """Test behavior when reference annotation file is missing."""
        # This should fail gracefully and return early
        run_benchmark(
            index_paths=self.index_paths,
            ref_annotation_path="nonexistent.tsv",
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        # Should not create results file or create empty one
        if os.path.exists("benchmark_results.csv"):
            results_df = pd.read_csv("benchmark_results.csv")
            self.assertEqual(len(results_df), 0)

    def test_run_benchmark_empty_test_dir(self):
        """Test behavior with empty test directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_benchmark(
                index_paths=self.index_paths,
                ref_annotation_path=self.ref_annotation_path,
                test_dir=tmp_dir,
                obo_path=self.obo_path,
                metrics=['euclidean'],
                k=10,
                download_s3=False
            )

            # Should not create results file
            if os.path.exists("benchmark_results.csv"):
                # Or results should be empty
                results_df = pd.read_csv("benchmark_results.csv")
                self.assertEqual(len(results_df), 0)

    def test_run_benchmark_different_k_values(self):
        """Test benchmark with different k values."""
        for k_val in [5, 10, 30]:
            # Clean up previous results
            if os.path.exists("benchmark_results.csv"):
                os.remove("benchmark_results.csv")

            run_benchmark(
                index_paths={"flat": self.index_paths["flat"]},
                ref_annotation_path=self.ref_annotation_path,
                test_dir=self.test_data_dir,
                obo_path=self.obo_path,
                metrics=['euclidean'],
                k=k_val,
                download_s3=False
            )

            # Should complete successfully
            self.assertTrue(os.path.exists("benchmark_results.csv"))

    def test_run_benchmark_with_ontology_metrics(self):
        """Test that ontology metrics are computed when ontology is available."""
        run_benchmark(
            index_paths={"flat": self.index_paths["flat"]},
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Ontology metrics should not be NaN
        self.assertFalse(results_df['mean_ontology_dist'].isna().all())
        self.assertFalse(results_df['median_ontology_dist'].isna().all())

        # Ontology distances should be non-negative
        self.assertTrue((results_df['mean_ontology_dist'] >= 0).all())
        self.assertTrue((results_df['median_ontology_dist'] >= 0).all())

    def test_run_benchmark_timing_metrics(self):
        """Test that query timing is recorded."""
        run_benchmark(
            index_paths={"flat": self.index_paths["flat"]},
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Query time should be recorded and positive
        self.assertTrue((results_df['Avg_Query_Time_ms'] > 0).all())

    def test_run_benchmark_accuracy_range(self):
        """Test that accuracy metrics are in valid range [0, 1]."""
        run_benchmark(
            index_paths={"flat": self.index_paths["flat"]},
            ref_annotation_path=self.ref_annotation_path,
            test_dir=self.test_data_dir,
            obo_path=self.obo_path,
            metrics=['euclidean'],
            k=10,
            download_s3=False
        )

        results_df = pd.read_csv("benchmark_results.csv")

        # Check accuracy metrics are in [0, 1]
        accuracy_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'top_k_accuracy']
        for metric in accuracy_metrics:
            if metric in results_df.columns:
                self.assertTrue((results_df[metric] >= 0).all())
                self.assertTrue((results_df[metric] <= 1).all())


if __name__ == '__main__':
    unittest.main()
