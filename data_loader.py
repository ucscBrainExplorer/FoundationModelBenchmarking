import os
import logging
import faiss
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import boto3
from botocore.exceptions import NoCredentialsError

logger = logging.getLogger(__name__)

def load_faiss_index(path: str, index_type: str = 'ivfFlat') -> faiss.Index:
    """
    Securely load the FAISS index file.
    
    Args:
        path (str): Path to the FAISS index file.
        index_type (str): Type of index (e.g., 'ivfFlat', 'ivfPQ'). 
                          Currently primarily handled by faiss.read_index, 
                          but kept as an arg for future flexibility or validation.
                          
    Returns:
        faiss.Index: Loaded FAISS index.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index file not found at: {path}")
    
    try:
        index = faiss.read_index(path)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")

def load_reference_annotations(path: str) -> pd.DataFrame:
    """
    Load the specific prediction_obs.tsv file.
    
    Args:
        path (str): Path to the prediction_obs.tsv file.
        
    Returns:
        pd.DataFrame: DataFrame containing reference annotations.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference annotations file not found at: {path}")
        
    try:
        df = pd.read_csv(path, sep='\t')
        
        required_columns = ['cell_type_ontology_term_id', 'cell_type']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Annotation file must contain columns: {required_columns}")
            
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load reference annotations: {e}")

def load_test_batch(test_dir: str) -> List[Dict[str, str]]:
    """
    Discover test datasets in a directory by matching file pairs.

    Expected naming convention — for each dataset, two files:
        {dataset_id}_prediction_obs.tsv   — ground truth cell type labels
        {dataset_id}_*.npy                — foundation model embeddings

    The dataset_id is extracted from the TSV filename (everything before
    ``_prediction_obs.tsv``).  The corresponding .npy file is any ``.npy``
    file whose name starts with the same dataset_id.

    Example::

        test_data/
        ├── organoid_embeddings.npy
        ├── organoid_prediction_obs.tsv
        ├── adult_brain_embeddings.npy
        └── adult_brain_prediction_obs.tsv

    Args:
        test_dir: Path to the directory containing test files.

    Returns:
        A list of dicts, each with keys 'id', 'embedding_path', and
        'metadata_path'.
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    files = os.listdir(test_dir)
    npy_files = {f for f in files if f.endswith('.npy')}

    dataset_pairs = []
    for f in sorted(files):
        if not f.endswith('_prediction_obs.tsv'):
            continue
        dataset_id = f.replace('_prediction_obs.tsv', '')
        matching_npy = sorted(n for n in npy_files if n.startswith(dataset_id))
        if matching_npy:
            dataset_pairs.append({
                'id': dataset_id,
                'embedding_path': os.path.join(test_dir, matching_npy[0]),
                'metadata_path': os.path.join(test_dir, f)
            })

    return dataset_pairs

def download_data_from_s3(bucket: str, prefix: str, local_dir: str, profile_name: str = 'braingeneers'):
    """
    Download data from S3 bucket to local directory.
    mirroring the structure.
    
    Args:
        bucket (str): S3 bucket name.
        prefix (str): Prefix in the bucket (folder path).
        local_dir (str): Local directory to save files.
        profile_name (str): AWS CLI profile to use (optional, will use env vars if available).
    """
    # Try environment variables first (for Kubernetes secrets)
    if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("Using AWS credentials from environment variables...")
        try:
            s3 = boto3.client('s3',
                            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
        except Exception as e:
            raise RuntimeError(f"Could not connect to S3 with environment variables: {e}")
    else:
        # Fall back to profile-based authentication
        try:
            session = boto3.Session(profile_name=profile_name)
            s3 = session.client('s3')
        except Exception as e:
            print(f"Failed to create boto3 session with profile '{profile_name}': {e}")
            print("Falling back to default credentials/profile...")
            try:
                s3 = boto3.client('s3')
            except Exception as e2:
                raise RuntimeError(f"Could not connect to S3: {e2}")

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    print(f"Listing objects in s3://{bucket}/{prefix}")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # Skip if it behaves like a folder (ends with /)
                if key.endswith('/'):
                    continue
                # Skip previous benchmark results
                if 'benchmark_results/' in key:
                    continue
                    
                # Calculate local path relative to prefix
                # If prefix is "foo/", and key is "foo/bar.txt", rel is "bar.txt"
                if key.startswith(prefix):
                    rel_path = key[len(prefix):].lstrip('/')
                else:
                    rel_path = key 
                
                local_file_path = os.path.join(local_dir, rel_path)
                local_file_dir = os.path.dirname(local_file_path)
                
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)
                    
                if not os.path.exists(local_file_path):
                    print(f"Downloading {key} to {local_file_path}")
                    s3.download_file(bucket, key, local_file_path)
                else:
                    print(f"Skipping {key}, already exists.")
                
    except Exception as e:
        raise RuntimeError(f"Error downloading from S3: {e}")


def _get_s3_client(profile_name: str = 'braingeneers'):
    """Create an S3 client using env vars (cluster) or AWS profile (local)."""
    if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        return boto3.client(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    try:
        session = boto3.Session(profile_name=profile_name)
        return session.client('s3')
    except Exception:
        return boto3.client('s3')


def upload_results_to_s3(bucket: str, prefix: str, results_dir: str,
                         timestamp: str, profile_name: str = 'braingeneers') -> str:
    """
    Upload benchmark results to S3.

    Uploads all files in results_dir to
    s3://{bucket}/{prefix}/benchmark_results/{timestamp}/.

    Args:
        bucket (str): S3 bucket name.
        prefix (str): S3 key prefix (e.g. 'combined_UCE_5neuro/').
        results_dir (str): Local timestamped results directory to upload.
        timestamp (str): Run timestamp (matches the local directory name).
        profile_name (str): AWS profile for local runs.

    Returns:
        str: The S3 URI where results were uploaded.
    """
    s3 = _get_s3_client(profile_name)

    results_prefix = f"{prefix.rstrip('/')}/benchmark_results/{timestamp}"

    uploaded_count = 0
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            local_file = os.path.join(root, fname)
            rel = os.path.relpath(local_file, results_dir)
            s3_key = f"{results_prefix}/{rel}"
            logger.info(f"Uploading {local_file} -> s3://{bucket}/{s3_key}")
            print(f"  Uploading {rel}")
            s3.upload_file(local_file, bucket, s3_key)
            uploaded_count += 1

    s3_uri = f"s3://{bucket}/{results_prefix}/"
    print(f"  {uploaded_count} files uploaded to {s3_uri}")
    logger.info(f"Uploaded {uploaded_count} files to {s3_uri}")
    return s3_uri
