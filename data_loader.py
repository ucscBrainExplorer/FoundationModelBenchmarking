import os
import faiss
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import boto3
from botocore.exceptions import NoCredentialsError

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
    Iterate through the test/ directory to discover datasets.
    
    Args:
        test_dir (str): Path to the test directory.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dict contains paths 
                              for 'embedding' and 'metadata' for a dataset.
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
    datasets = []
    
    # Simple discovery logic: look for pairs of {id}_{embedding}.npy and {id}_prediction_obs.tsv
    # We'll list all .npy files and check for corresponding .tsv
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npy'):
            # Expected format: {dataset_id}_{embedding}.npy
            # Note: The plan says "{dataset_id}_{embedding}.npy" but also implies we need to find the ID.
            # Let's assume the suffix is always rigid or we split by first underscore if needed.
            # However, simpler approach: if filename is "foo_embedding.npy", ID might be "foo".
            # Plan says: "{dataset_id}_{embedding}.npy" and "{dataset_id}_prediction_obs.tsv"
            # It might be safer to just look for matching prefixes if we knew the suffix exactly.
            # Let's assume suffix is `_embeddings.npy` or just split extension. 
            # Re-reading plan: "Embedding is {dataset_id}_{embedding}.npy" and "ground truth labels in {dataset_id}_prediction_obs.tsv"
            # This is slightly ambiguous on the exact suffix part "{embedding}". 
            # Let's assume we pair any .npy with a .tsv that shares the prefix before the last underscore?
            # Or simpler: Look for files, if we find X.npy and Y.tsv, how do we match?
            # Let's try to infer dataset_id. 
            
            # Let's try to match exactly what is likely there. 
            # If we have `dataset1_embeddings.npy` and `dataset1_prediction_obs.tsv`. 
            # We can strip `.npy` and check if there is a corresponding tsv.
            
            # Actually, let's look for the TSV files first as they might be more distinguishable with `_prediction_obs.tsv`
            pass
            
    # Better approach given the ambiguity:
    # Iterate all files, find those ending in `_prediction_obs.tsv`.
    # Then look for a corresponding .npy file. 
    # The plan says "Embedding is {dataset_id}_{embedding}.npy". This `{embedding}` might be a variable string?
    # Or literal "embedding"? "Snapshot of s3://..." suggests standard naming.
    # Let's assume we find `X_prediction_obs.tsv` and look for `X.npy` or `X_embedding.npy`.
    # Ideally the function should be robust.
    
    files = os.listdir(test_dir)
    embeddings_map = {}
    metadata_map = {}
    
    for f in files:
        if f.endswith('.npy'):
            embeddings_map[f] = os.path.join(test_dir, f)
        elif f.endswith('.tsv') or f.endswith('.txt'): # assuming tsv content
            metadata_map[f] = os.path.join(test_dir, f)
            
    # This is still tricky without knowing the exact naming convention. 
    # Let's try to match by prefix. 
    # If we have `organoid_embeddings.npy` and `organoid_prediction_obs.tsv`
    # Common prefix `organoid`.
    
    # Going with a heuristic: 
    # Identify unique dataset IDs from the TSV filenames (assuming they end in _prediction_obs.tsv)
    
    dataset_pairs = []
    
    for meta_file in metadata_map:
        if 'prediction_obs' in meta_file:
            # extract dataset_id
            # Ref: "{dataset_id}_prediction_obs.tsv"
            dataset_id = meta_file.replace('_prediction_obs.tsv', '')
            
            # Now find the embedding file
            # Ref: "{dataset_id}_{embedding}.npy" - implies literal or variable.
            # We'll search for a .npy file that starts with dataset_id
            
            matching_npy = [k for k in embeddings_map.keys() if k.startswith(dataset_id) and k.endswith('.npy')]
            
            if matching_npy:
                # Take the first one found, or warn if multiple?
                # For now take the first.
                dataset_pairs.append({
                    'id': dataset_id,
                    'embedding_path': embeddings_map[matching_npy[0]],
                    'metadata_path': metadata_map[meta_file]
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
        profile_name (str): AWS CLI profile to use.
    """
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
