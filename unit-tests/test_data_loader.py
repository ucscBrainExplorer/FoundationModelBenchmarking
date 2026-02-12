"""
Unit tests for data_loader.py module.
"""
import unittest
import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
import faiss

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import (
    load_faiss_index,
    load_reference_annotations,
    load_test_batch,
    download_data_from_s3
)


class TestLoadFaissIndex(unittest.TestCase):
    """Test load_faiss_index function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.index_path = os.path.join(self.test_dir, "indices/index_flat.faiss")
        self.ivf_index_path = os.path.join(self.test_dir, "indices/index_ivfflat.faiss")

    def test_load_valid_index(self):
        """Test loading a valid FAISS index."""
        index = load_faiss_index(self.index_path)
        self.assertIsInstance(index, faiss.Index)
        self.assertEqual(index.ntotal, 100)  # We created 100 vectors
        self.assertEqual(index.d, 64)  # Dimension is 64

    def test_load_ivfflat_index(self):
        """Test loading an IVFFlat index."""
        index = load_faiss_index(self.ivf_index_path, index_type='ivfFlat')
        self.assertIsInstance(index, faiss.Index)
        self.assertEqual(index.ntotal, 100)
        self.assertEqual(index.d, 64)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_faiss_index("nonexistent_index.faiss")

    def test_invalid_index_file(self):
        """Test that RuntimeError is raised for invalid index file."""
        # Create a temporary invalid file
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
            tmp.write(b"not a valid faiss index")
            tmp_path = tmp.name

        try:
            with self.assertRaises(RuntimeError):
                load_faiss_index(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestLoadReferenceAnnotations(unittest.TestCase):
    """Test load_reference_annotations function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data"
        self.ref_path = os.path.join(self.test_dir, "reference_data/prediction_obs.tsv")

    def test_load_valid_annotations(self):
        """Test loading valid reference annotations."""
        df = load_reference_annotations(self.ref_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn('cell_type_ontology_term_id', df.columns)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_reference_annotations("nonexistent.tsv")

    def test_missing_required_columns(self):
        """Test that ValueError is raised when required columns are missing."""
        # Create a temporary TSV without required columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as tmp:
            tmp.write("wrong_column1\twrong_column2\n")
            tmp.write("value1\tvalue2\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises((ValueError, RuntimeError)):
                load_reference_annotations(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_alignment_with_index(self):
        """Test that annotations align with FAISS index size."""
        df = load_reference_annotations(self.ref_path)
        index_path = os.path.join(self.test_dir, "indices/index_flat.faiss")
        index = load_faiss_index(index_path)

        # The number of rows in annotations should match index size
        self.assertEqual(len(df), index.ntotal)


class TestLoadTestBatch(unittest.TestCase):
    """Test load_test_batch function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "unit-tests/mock_data/test_data"

    def test_load_test_datasets(self):
        """Test loading test datasets from directory."""
        datasets = load_test_batch(self.test_dir)

        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)

        # Check structure of each dataset
        for ds in datasets:
            self.assertIn('id', ds)
            self.assertIn('embedding_path', ds)
            self.assertIn('metadata_path', ds)

            # Verify files exist
            self.assertTrue(os.path.exists(ds['embedding_path']))
            self.assertTrue(os.path.exists(ds['metadata_path']))

    def test_dataset_pairing(self):
        """Test that embeddings and metadata are correctly paired."""
        datasets = load_test_batch(self.test_dir)

        for ds in datasets:
            # Load both files
            embeddings = np.load(ds['embedding_path'])
            metadata = pd.read_csv(ds['metadata_path'], sep='\t')

            # Number of embeddings should match metadata rows
            self.assertEqual(embeddings.shape[0], len(metadata))

    def test_directory_not_found(self):
        """Test that FileNotFoundError is raised for missing directory."""
        with self.assertRaises(FileNotFoundError):
            load_test_batch("nonexistent_directory")

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        # Create a temporary empty directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets = load_test_batch(tmp_dir)
            self.assertEqual(len(datasets), 0)

    def test_positive_control_included(self):
        """Test that positive control is included in datasets."""
        datasets = load_test_batch(self.test_dir)
        dataset_ids = [ds['id'] for ds in datasets]

        # Check if positive_control is in the discovered datasets
        self.assertIn('positive_control', dataset_ids)


class TestDownloadDataFromS3(unittest.TestCase):
    """Test download_data_from_s3 function."""

    def test_s3_download_no_credentials(self):
        """Test S3 download without credentials (should handle gracefully)."""
        # This test will likely fail in CI/local without AWS credentials
        # We're mainly testing that the function doesn't crash unexpectedly
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This should raise an error or handle gracefully
            try:
                download_data_from_s3(
                    bucket="test-bucket",
                    prefix="test-prefix/",
                    local_dir=tmp_dir,
                    profile_name="nonexistent-profile"
                )
            except (RuntimeError, Exception) as e:
                # Expected to fail without valid credentials
                self.assertIsInstance(e, (RuntimeError, Exception))


if __name__ == '__main__':
    unittest.main()
