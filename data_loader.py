import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict
from prediction_module import validate_ref_columns

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
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")

    # IVF indices default to nprobe=1 (searches only 1 Voronoi cell),
    # which causes missing neighbors and inf distances in sparse cells.
    ivf = faiss.extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = 20
        print(f"  IVF index: searching {ivf.nprobe} of {ivf.nlist} cells (nprobe)")

    return index

def load_reference_annotations(path: str) -> pd.DataFrame:
    """
    Load the specific prediction_obs.tsv file.

    Args:
        path (str): Path to the prediction_obs.tsv file.

    Returns:
        pd.DataFrame: DataFrame containing reference annotations.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the required column is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference annotations file not found at: {path}")

    df = pd.read_csv(path, sep='\t')
    validate_ref_columns(df)
    return df

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

