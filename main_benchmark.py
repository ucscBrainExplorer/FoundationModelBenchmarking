import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
import time

from .data_loader import load_faiss_index, load_reference_annotations, load_test_batch, download_data_from_s3
from .prediction_module import execute_query, vote_neighbors
from .ontology_utils import load_ontology, score_batch
from .evaluation_metrics import calculate_accuracy

# Default Configuration
DEFAULT_TEST_DIR = "test_data/"
DEFAULT_INDEX_DIR = "indices/"
DEFAULT_REF_ANNOTATION = "reference_data/prediction_obs.tsv"
DEFAULT_OBO_PATH = "reference_data/cl.obo"

def run_benchmark(
    index_paths: Dict[str, str],
    ref_annotation_path: str,
    test_dir: str,
    obo_path: str,
    metrics: List[str] = ['euclidean', 'cosine'],
    k: int = 30,
    download_s3: bool = True,
    s3_bucket: str = "latentbrain",
    s3_prefix: str = "combined_UCE_5neuro/"
):
    """
    Orchestrate the benchmarking process.
    """
    # 0. S3 Download (Optional)
    if download_s3:
        print("Attempting to download data from S3...")
        try:
            # We assume we download everything to the parent directory of test_dir or specific locations
            # But the prompt was vague on exact mapping. Let's assume we sync to current directory or `test_dir` parent.
            # Usually test_dir is where we look for data.
            # If s3_prefix is "combined_UCE_5neuro/", we might want to dump it to `test_data` or root.
            # Let's download to `.` (current dir) which will create `combined_UCE_5neuro` or similar?
            # Or simpler: download content of prefix into `test_dir`?
            # The prompt S3 path: "s3://latentbrain/combined_UCE_5neuro/"
            # If we download to `data/`, we might get `data/combined_UCE_5neuro/...`
            
            # Let's use a dedicated data directory
            download_root = "data" 
            download_data_from_s3(s3_bucket, s3_prefix, download_root)
            
            # Update paths if we downloaded data
            # Assuming the structure in S3 matches what we expect in local paths.
            # If s3://.../ contains test/, indices/, reference_data/ folders:
            # We pointed defaults to "test_data/", "indices/", "reference_data/".
            # The user might need to adjust defaults or we assume S3 layout matches defaults.
            # Let's proceed with download and assume user will direct `test_dir` appropriately or we rely on defaults working if files exist there.
            
        except Exception as e:
            print(f"S3 Download failed: {e}")
            print("Proceeding with local files if available...")

    # 1. Load Resources
    print("Loading resources...")
    try:
        # Check files first
        if not os.path.exists(ref_annotation_path):
             print(f"Warning: Reference file not found: {ref_annotation_path}")
        
        ref_df = load_reference_annotations(ref_annotation_path)
        
        # Load Ontology if available
        try:
            ontology_graph = load_ontology(obo_path)
            print("Ontology loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load ontology ({e}). Ontology metrics will be skipped.")
            ontology_graph = None
            
        # Discover Datasets
        datasets = load_test_batch(test_dir)
        if not datasets:
            print(f"No datasets found in {test_dir}")
            return
            
        print(f"Found {len(datasets)} datasets.")
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    results = []

    # 2. Benchmark Loop
    for index_name, index_path in index_paths.items():
        print(f"\nProcessing Index: {index_name} ({index_path})")
        if not os.path.exists(index_path):
            print(f"  Index file not found, skipping.")
            continue
            
        try:
            index = load_faiss_index(index_path)
        except Exception as e:
            print(f"  Failed to load index: {e}")
            continue
            
        for metric in metrics:
            print(f"  Metric: {metric}")
            
            for dataset in datasets:
                ds_id = dataset['id']
                print(f"    Dataset: {ds_id}")
                
                try:
                    # Load Test Data
                    embeddings = np.load(dataset['embedding_path'])
                    metadata_df = pd.read_csv(dataset['metadata_path'], sep='\t')
                    
                    truth_labels = metadata_df['cell_type_ontology_term_id'].tolist()
                    
                    # Execute Prediction
                    start_time = time.time()
                    dists, neighbor_indices = execute_query(index, embeddings, k=k, metric=metric)
                    query_time = time.time() - start_time
                    
                    # Vote
                    predictions = vote_neighbors(neighbor_indices, ref_df)
                    
                    # Get neighbor text labels for Top-k
                    # We need to map neighbor_indices to labels for Top-k check
                    # Term IDs from ref df
                    term_ids = ref_df['cell_type_ontology_term_id'].values
                    neighbor_labels = []
                    for row in neighbor_indices:
                        neighbor_labels.append(term_ids[row].tolist())
                        
                    # Calculate Standard Metrics
                    metrics_scores = calculate_accuracy(predictions, truth_labels, neighbor_labels)
                    
                    # Calculate Ontology Metrics
                    if ontology_graph:
                        mean_dist, median_dist = score_batch(ontology_graph, predictions, truth_labels)
                        metrics_scores['mean_ontology_dist'] = mean_dist
                        metrics_scores['median_ontology_dist'] = median_dist
                    else:
                        metrics_scores['mean_ontology_dist'] = np.nan
                        metrics_scores['median_ontology_dist'] = np.nan
                        
                    # Record Result
                    res_entry = {
                        'Index': index_name,
                        'Metric': metric,
                        'Dataset': ds_id,
                        'Avg_Query_Time_ms': (query_time * 1000) / len(embeddings),
                        **metrics_scores
                    }
                    results.append(res_entry)
                    
                except Exception as e:
                    print(f"    Error processing dataset {ds_id}: {e}")

    # 3. Report
    if results:
        df_results = pd.DataFrame(results)
        print("\nBenchmark Results Summary:")
        print(df_results.to_string())
        
        # Save to file
        df_results.to_csv("benchmark_results.csv", index=False)
        print("\nResults saved to benchmark_results.csv")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmarking")
    parser.add_argument("--indices_config", type=str, help="Path to a text file mapping name:path for indices")
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--ref_annot", type=str, default=DEFAULT_REF_ANNOTATION)
    parser.add_argument("--obo", type=str, default=DEFAULT_OBO_PATH)
    parser.add_argument("--no-s3", action="store_true", help="Skip downloading data from S3")
    parser.add_argument("--s3_bucket", type=str, default="latentbrain")
    parser.add_argument("--s3_prefix", type=str, default="combined_UCE_5neuro/")
    
    args = parser.parse_args()
    
    # Simple default indices if no config provided
    # Ideally should read from args.indices_config or construct default paths
    index_paths = {
        "ivfFlat": os.path.join(DEFAULT_INDEX_DIR, "index_ivfflat.faiss"),
        # "ivfPQ": os.path.join(DEFAULT_INDEX_DIR, "index_ivfpq.faiss") # Example
    }
    
    run_benchmark(
        index_paths, 
        args.ref_annot, 
        args.test_dir, 
        args.obo, 
        download_s3=not args.no_s3,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )
