import os
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict
import time

from data_loader import load_faiss_index, load_reference_annotations, load_test_batch, download_data_from_s3, upload_results_to_s3
from prediction_module import execute_query, vote_neighbors
from ontology_utils import load_ontology, precompute_ic, score_batch, calculate_per_cell_distances, calculate_avg_neighbor_distances
from evaluation_metrics import calculate_accuracy
try:
    from visualization import generate_visualizations
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization dependencies not available. Install with: pip install matplotlib seaborn umap-learn")

# Default Configuration
# Use /data for Kubernetes, fallback to relative paths for local development
DATA_ROOT = "/data" if os.path.exists("/data") else "."

# Create timestamped results directory (mirrors S3 structure)
from datetime import datetime
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join(DATA_ROOT, "benchmark_results", TIMESTAMP)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure file logging (writes into the timestamped results directory)
LOG_PATH = os.path.join(RESULTS_DIR, "benchmark.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_TEST_DIR = os.path.join(DATA_ROOT, "test_data") if DATA_ROOT != "." else "test_data/"
DEFAULT_INDEX_DIR = os.path.join(DATA_ROOT, "indices") if DATA_ROOT != "." else "indices/"
DEFAULT_REF_ANNOTATION = os.path.join(DATA_ROOT, "reference_data", "prediction_obs.tsv") if DATA_ROOT != "." else "reference_data/prediction_obs.tsv"
DEFAULT_OBO_PATH = os.path.join(DATA_ROOT, "reference_data", "cl.obo") if DATA_ROOT != "." else "reference_data/cl.obo"

def run_benchmark(
    index_paths: Dict[str, str],
    ref_annotation_path: str,
    test_dir: str,
    obo_path: str,
    metrics: List[str] = ['euclidean'],
    k: int = 30,
    download_s3: bool = True,
    s3_bucket: str = "latentbrain",
    s3_prefix: str = "combined_UCE_5neuro/",
    generate_plots: bool = False,
    output_dir: str = None,
    ontology_method: str = 'ic',
    upload_s3: bool = True
):
    """
    Orchestrate the benchmarking process.
    """
    # 0. S3 Download (Optional)
    if download_s3:
        # Check if data already exists
        download_root = "/data" if os.path.exists("/data") else "data"
        indices_dir = os.path.join(download_root, "indices")
        reference_dir = os.path.join(download_root, "reference_data")
        test_data_dir = os.path.join(download_root, "test_data")
        
        index_file = os.path.join(indices_dir, "index_ivfflat.faiss")
        ref_file = os.path.join(reference_dir, "prediction_obs.tsv")
        
        # Check if required files already exist
        if os.path.exists(index_file) and os.path.exists(ref_file):
            print("✓ Required data files already exist, skipping S3 download.")
            print(f"  Index: {index_file}")
            print(f"  Reference: {ref_file}")
        else:
            print("Attempting to download data from S3...")
            try:
                # Test if download directory is writable
                os.makedirs(download_root, exist_ok=True)
                test_file = os.path.join(download_root, ".write_test")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                except Exception as e:
                    raise RuntimeError(f"Cannot write to {download_root}: {e}")
                
                # Download to a temporary location first, then reorganize
                temp_download_dir = os.path.join(download_root, "_temp_download")
                
                print(f"Downloading from s3://{s3_bucket}/{s3_prefix} to {temp_download_dir}...")
                download_data_from_s3(s3_bucket, s3_prefix, temp_download_dir)
            
                # Reorganize downloaded files to match expected structure
                print("Reorganizing downloaded files...")
                
                # Create expected directory structure
                os.makedirs(indices_dir, exist_ok=True)
                os.makedirs(reference_dir, exist_ok=True)
                os.makedirs(test_data_dir, exist_ok=True)
            
                # Move index file
                temp_index = os.path.join(temp_download_dir, "index_ivfflat.faiss")
                final_index = os.path.join(indices_dir, "index_ivfflat.faiss")
                if os.path.exists(temp_index) and not os.path.exists(final_index):
                    print(f"Moving index: {temp_index} -> {final_index}")
                    os.rename(temp_index, final_index)
                
                # Move reference annotations
                temp_ref = os.path.join(temp_download_dir, "prediction_obs.tsv")
                final_ref = os.path.join(reference_dir, "prediction_obs.tsv")
                if os.path.exists(temp_ref) and not os.path.exists(final_ref):
                    print(f"Moving reference annotations: {temp_ref} -> {final_ref}")
                    os.rename(temp_ref, final_ref)
                
                # Move test directory
                temp_test = os.path.join(temp_download_dir, "test")
                if os.path.exists(temp_test):
                    # Move contents of test/ to test_data/
                    for item in os.listdir(temp_test):
                        src = os.path.join(temp_test, item)
                        dst = os.path.join(test_data_dir, item)
                        if not os.path.exists(dst):
                            print(f"Moving test file: {src} -> {dst}")
                            os.rename(src, dst)
                    # Remove empty test directory
                    try:
                        os.rmdir(temp_test)
                    except:
                        pass
                
                # Clean up temp directory
                try:
                    # Remove any remaining files in temp directory
                    for item in os.listdir(temp_download_dir):
                        item_path = os.path.join(temp_download_dir, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            import shutil
                            shutil.rmtree(item_path)
                    os.rmdir(temp_download_dir)
                except Exception as e:
                    print(f"Error cleaning up temporary download directory: {e}")
                
            except Exception as e:
                print(f"❌ S3 Download failed: {e}")
                import traceback
                traceback.print_exc()
                print("\nProceeding with existing files if available...")
                print("If files don't exist, the benchmark will fail.")
        
        # Download ontology files if not present (always check, even if S3 download was skipped)
        download_root = "/data" if os.path.exists("/data") else "data"
        reference_dir = os.path.join(download_root, "reference_data")
        os.makedirs(reference_dir, exist_ok=True)
        
        obo_file = os.path.join(reference_dir, "cl.obo")
        basic_obo_file = os.path.join(reference_dir, "cl-basic.obo")
        
        if not os.path.exists(obo_file):
            print("Downloading Cell Ontology (cl.obo)...")
            try:
                import urllib.request
                obo_url = 'http://purl.obolibrary.org/obo/cl.obo'
                urllib.request.urlretrieve(obo_url, obo_file)
                print(f"✓ Downloaded: {obo_file}")
            except Exception as e:
                print(f"⚠ Could not download cl.obo: {e}")
        
        # Also download cl-basic.obo as fallback if not present
        if not os.path.exists(basic_obo_file):
            print("Downloading Cell Ontology subset (cl-basic.obo) as fallback...")
            try:
                import urllib.request
                basic_obo_url = 'http://purl.obolibrary.org/obo/cl/cl-basic.obo'
                urllib.request.urlretrieve(basic_obo_url, basic_obo_file)
                print(f"✓ Downloaded: {basic_obo_file}")
            except Exception as e:
                print(f"⚠ Could not download cl-basic.obo: {e}")
                print("  Ontology metrics will be skipped if no ontology file is available.")

    # 1. Load Resources
    print("Loading resources...")
    try:
        # Check files first
        if not os.path.exists(ref_annotation_path):
             print(f"Warning: Reference file not found: {ref_annotation_path}")
        
        ref_df = load_reference_annotations(ref_annotation_path)
        
        # Load Ontology if available (with fallback to cl-basic.obo)
        ontology_graph = None
        obo_paths_to_try = [obo_path]
        # If main obo_path ends with cl.obo, also try cl-basic.obo as fallback
        if obo_path.endswith('cl.obo'):
            obo_dir = os.path.dirname(obo_path)
            basic_obo = os.path.join(obo_dir, 'cl-basic.obo')
            if os.path.exists(basic_obo):
                obo_paths_to_try.append(basic_obo)
        
        for try_path in obo_paths_to_try:
            try:
                if os.path.exists(try_path):
                    ontology_graph = load_ontology(try_path)
                    print(f"Ontology loaded successfully from {try_path}.")
                    break
            except Exception as e:
                print(f"Warning: Could not load ontology from {try_path} ({e}).")
                continue
        
        if ontology_graph is None:
            print("Warning: Could not load ontology from any available path. Ontology metrics will be skipped.")

        # Precompute IC values if using IC-based ontology method
        ic_values = None
        if ontology_graph is not None and ontology_method == 'ic':
            print(f"Precomputing Information Content (Zhou k=0.5) for {ontology_graph.number_of_nodes()} terms...")
            ic_values = precompute_ic(ontology_graph, k=0.5)
            print(f"  IC precomputation complete.")

        print(f"Ontology scoring method: {ontology_method}")

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
    
    # Store data for visualizations (if enabled)
    embeddings_dict = {}
    predictions_dict = {}
    ground_truth_dict = {}
    best_config = None  # Track best performing configuration for visualization

    # 2. Benchmark Loop
    for index_name, index_path in index_paths.items():
        print(f"\nProcessing Index: {index_name} ({index_path})")
        if not os.path.exists(index_path):
            print(f"  Index file not found, skipping.")
            continue
            
        try:
            print(f"  Loading FAISS index from {index_path}...")
            index = load_faiss_index(index_path)
            print(f"  ✓ FAISS index loaded successfully (dimension: {index.d}, vectors: {index.ntotal})")
        except Exception as e:
            print(f"  Failed to load index: {e}")
            import traceback
            traceback.print_exc()
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
                    
                    # Vote (returns predictions and vote percentages)
                    predictions, vote_percentages = vote_neighbors(neighbor_indices, ref_df)
                    
                    # Get neighbor text labels for Top-k
                    # We need to map neighbor_indices to labels for Top-k check
                    # Term IDs from ref df
                    term_ids = ref_df['cell_type_ontology_term_id'].values
                    neighbor_labels = []
                    for row in neighbor_indices:
                        valid_row = row[row >= 0]
                        neighbor_labels.append(term_ids[valid_row].tolist())
                        
                    # Filter out cells with empty-string predictions before metrics.
                    # Empty predictions (from cells where all neighbors had invalid labels)
                    # would create a phantom '' class that drags down macro F1.
                    valid_mask = [p != '' for p in predictions]
                    filtered_preds = [p for p, v in zip(predictions, valid_mask) if v]
                    filtered_truth = [t for t, v in zip(truth_labels, valid_mask) if v]
                    filtered_neighbors = [n for n, v in zip(neighbor_labels, valid_mask) if v] if neighbor_labels else None

                    if len(filtered_preds) < len(predictions):
                        n_filtered = len(predictions) - len(filtered_preds)
                        # Identify which cells had empty predictions for investigation
                        if 'cell_id' in metadata_df.columns:
                            empty_cell_ids = [metadata_df['cell_id'].iloc[i] for i, v in enumerate(valid_mask) if not v]
                        else:
                            empty_cell_ids = [f"{ds_id}_cell_{i}" for i, v in enumerate(valid_mask) if not v]
                        logger.warning(
                            f"{ds_id}/{index_name}/{metric}: {n_filtered} cells had empty predictions "
                            f"(filtered from metrics). Cell IDs: {empty_cell_ids}"
                        )

                    # Calculate Standard Metrics
                    metrics_scores = calculate_accuracy(filtered_preds, filtered_truth, filtered_neighbors)
                    
                    # Calculate Ontology Metrics
                    per_cell_distances = None
                    avg_neighbor_distances = None
                    if ontology_graph:
                        mean_score, median_score = score_batch(
                            ontology_graph, predictions, truth_labels,
                            method=ontology_method, ic_values=ic_values)
                        if ontology_method == 'ic':
                            metrics_scores['mean_ontology_similarity'] = mean_score
                            metrics_scores['median_ontology_similarity'] = median_score
                        elif ontology_method == 'shortest_path':
                            metrics_scores['mean_ontology_dist'] = mean_score
                            metrics_scores['median_ontology_dist'] = median_score
                        else:
                            print(f"    ERROR: Unknown ontology method '{ontology_method}'. Only 'ic' and 'shortest_path' are supported.")
                        # Calculate per-cell scores for detailed output
                        per_cell_distances = calculate_per_cell_distances(
                            ontology_graph, predictions, truth_labels,
                            method=ontology_method, ic_values=ic_values)
                        # Calculate average neighbor scores (average across 30 neighbors for each cell)
                        avg_neighbor_distances = calculate_avg_neighbor_distances(
                            ontology_graph, neighbor_labels, truth_labels,
                            method=ontology_method, ic_values=ic_values)
                        # Add aggregate statistic for average neighbor distances
                        valid_avg_neighbor_dists = [d for d in avg_neighbor_distances if pd.notna(d)]
                        if len(valid_avg_neighbor_dists) > 0:
                            metrics_scores['mean_avg_neighbor_ontology_dist'] = float(np.mean(valid_avg_neighbor_dists))
                        else:
                            metrics_scores['mean_avg_neighbor_ontology_dist'] = np.nan
                    else:
                        metrics_scores['mean_ontology_dist'] = np.nan
                        metrics_scores['median_ontology_dist'] = np.nan
                        metrics_scores['mean_avg_neighbor_ontology_dist'] = np.nan
                        per_cell_distances = [np.nan] * len(predictions)
                        avg_neighbor_distances = [np.nan] * len(predictions)
                    
                    # Save per-cell results
                    # Get cell identifiers (use index if no cell_id column exists)
                    if 'cell_id' in metadata_df.columns:
                        cell_ids = metadata_df['cell_id'].tolist()
                    else:
                        # Use row index as cell identifier
                        cell_ids = [f"{ds_id}_cell_{i}" for i in range(len(predictions))]
                    
                    # Get human-readable labels if available
                    if 'cell_type' in metadata_df.columns:
                        true_labels_readable = metadata_df['cell_type'].tolist()
                    else:
                        true_labels_readable = truth_labels
                    
                    # Get readable labels for predictions (map prediction_label to readable name from ref_df)
                    prediction_labels_readable = []
                    if 'cell_type' in ref_df.columns:
                        # Create mapping from ontology_term_id to readable name
                        ref_label_map = dict(zip(ref_df['cell_type_ontology_term_id'], ref_df['cell_type']))
                        for pred_label in predictions:
                            if pred_label and pred_label in ref_label_map:
                                prediction_labels_readable.append(ref_label_map[pred_label])
                            else:
                                prediction_labels_readable.append('')
                    else:
                        prediction_labels_readable = predictions
                    
                    # Calculate euclidean distance metrics (FAISS distances)
                    # Mean distance to neighbors and distance to nearest neighbor
                    mean_euclidean_distances = []
                    nearest_neighbor_distances = []
                    for row_dists in dists:
                        # Filter out invalid distances (if any)
                        valid_dists = row_dists[np.isfinite(row_dists)]
                        if len(valid_dists) > 0:
                            mean_euclidean_distances.append(float(np.mean(valid_dists)))
                            nearest_neighbor_distances.append(float(np.min(valid_dists)))
                        else:
                            mean_euclidean_distances.append(np.nan)
                            nearest_neighbor_distances.append(np.nan)
                    
                    # Create per-cell results DataFrame
                    per_cell_df = pd.DataFrame({
                        'cell_id': cell_ids,
                        'true_label': truth_labels,
                        'true_label_readable': true_labels_readable,
                        'prediction_label': predictions,
                        'prediction_label_readable': prediction_labels_readable,
                        'vote_percentage': vote_percentages,
                        'mean_euclidean_distance': mean_euclidean_distances,
                        'nearest_neighbor_euclidean_distance': nearest_neighbor_distances,
                        'ontology_distance': per_cell_distances if per_cell_distances else [np.nan] * len(predictions),
                        'avg_neighbor_ontology_distance': avg_neighbor_distances if avg_neighbor_distances else [np.nan] * len(predictions),
                        'dataset': ds_id,
                        'index_type': index_name,
                        'metric': metric
                    })
                    
                    # Save per-cell results to CSV
                    print(f"    Preparing to save per-cell results for {len(predictions)} cells...")
                    try:
                        per_cell_output_dir = os.path.join(RESULTS_DIR, "per_cell_results")
                        os.makedirs(per_cell_output_dir, exist_ok=True)
                        per_cell_filename = f"{ds_id}_{index_name}_{metric}_per_cell_results.csv"
                        per_cell_path = os.path.join(per_cell_output_dir, per_cell_filename)
                        per_cell_df.to_csv(per_cell_path, index=False)
                        print(f"    ✓ Saved per-cell results to {per_cell_path} ({len(per_cell_df)} cells)")
                    except Exception as e:
                        print(f"    ⚠️  Warning: Failed to save per-cell results: {e}")
                        import traceback
                        traceback.print_exc()
                        
                    # Record Result
                    res_entry = {
                        'Index': index_name,
                        'Metric': metric,
                        'Dataset': ds_id,
                        'Avg_Query_Time_ms': (query_time * 1000) / len(embeddings),
                        **metrics_scores
                    }
                    results.append(res_entry)
                    
                    # Store data for visualizations (use best configuration or first one)
                    if generate_plots and VISUALIZATION_AVAILABLE:
                        if best_config is None or metrics_scores.get('accuracy', 0) > best_config.get('accuracy', 0):
                            best_config = {
                                'index_name': index_name,
                                'metric': metric,
                                'accuracy': metrics_scores.get('accuracy', 0)
                            }
                            embeddings_dict[ds_id] = embeddings
                            predictions_dict[ds_id] = predictions
                            ground_truth_dict[ds_id] = truth_labels
                    
                except Exception as e:
                    print(f"    Error processing dataset {ds_id}: {e}")

    # 3. Report
    if results:
        df_results = pd.DataFrame(results)
        print("\nBenchmark Results Summary:")
        print(df_results.to_string())
        
        # Save to file
        results_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
        df_results.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        # Generate visualizations if requested
        if generate_plots and VISUALIZATION_AVAILABLE:
            if embeddings_dict:
                viz_output_dir = output_dir if output_dir else os.path.join(RESULTS_DIR, "visualizations")
                print(f"\nGenerating visualizations...")
                print(f"Using best configuration: {best_config['index_name']} with {best_config['metric']} metric")
                
                # Create label map from reference annotations
                label_map = dict(zip(ref_df['cell_type_ontology_term_id'], ref_df['cell_type']))
                
                generate_visualizations(
                    df_results,
                    embeddings_dict,
                    predictions_dict,
                    ground_truth_dict,
                    ontology_graph,
                    viz_output_dir,
                    label_map
                )
            else:
                print("No data available for visualizations.")
        elif generate_plots and not VISUALIZATION_AVAILABLE:
            print("\nVisualizations requested but dependencies not available.")
            print("Install with: pip install matplotlib seaborn umap-learn")
        
        # Run ontology analysis if ontology was loaded and per-cell results exist
        if ontology_graph is not None:
            try:
                print("\nRunning ontology distance analysis...")
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from analyze_ontology_results import load_per_cell_results, calculate_ontology_statistics, generate_summary_report, analyze_distance_metric_relationship
                
                per_cell_dir = os.path.join(RESULTS_DIR, "per_cell_results")
                analysis_dir = os.path.join(RESULTS_DIR, "ontology_analysis")
                
                if os.path.exists(per_cell_dir):
                    df_per_cell = load_per_cell_results(per_cell_dir)
                    stats = calculate_ontology_statistics(df_per_cell)
                    report_path = os.path.join(analysis_dir, "ontology_analysis_report.txt")
                    os.makedirs(analysis_dir, exist_ok=True)
                    generate_summary_report(df_per_cell, stats, report_path)
                    analyze_distance_metric_relationship(df_per_cell, analysis_dir)
                    print(f"Ontology analysis complete. Results saved to {analysis_dir}")
                else:
                    print(f"Per-cell results directory not found: {per_cell_dir}")
            except Exception as e:
                print(f"Warning: Could not run ontology analysis: {e}")
                import traceback
                traceback.print_exc()
        
        # Upload results to S3
        if upload_s3:
            print(f"\nUploading results to s3://{s3_bucket}/{s3_prefix.rstrip('/')}/benchmark_results/{TIMESTAMP}/...")
            try:
                s3_uri = upload_results_to_s3(s3_bucket, s3_prefix, RESULTS_DIR, TIMESTAMP)
                print(f"Results available at {s3_uri}")
            except Exception as e:
                print(f"Warning: S3 upload failed: {e}")
                logger.error(f"S3 upload failed: {e}")
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
    parser.add_argument("--generate-plots", action="store_true", help="Generate UMAP plots and confusion matrices")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for visualization outputs (default: visualizations/)")
    parser.add_argument("--ontology-method", type=str, default="ic", choices=["ic", "shortest_path"],
                        help="Ontology scoring method: 'ic' for Lin similarity with Zhou IC (default), "
                             "'shortest_path' for shortest undirected path distance")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading results to S3 (upload is on by default when S3 is enabled)")
    
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
        s3_prefix=args.s3_prefix,
        generate_plots=args.generate_plots,
        output_dir=args.output_dir,
        ontology_method=args.ontology_method,
        upload_s3=not args.no_s3 and not args.no_upload
    )
