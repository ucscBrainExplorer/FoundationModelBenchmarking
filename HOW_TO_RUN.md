# How to Run the Benchmark

## Running on the Cluster

### 1. Configure the Job

Edit `benchmarking-job.yaml` to set your S3 bucket, prefix, and any other options:

```yaml
args: [
  "--test_dir", "test_data",
  "--ref_annot", "prediction_obs.tsv",
  "--obo", "cl.obo",
  "--s3_bucket", "latentbrain",
  "--s3_prefix", "combined_UCE_5neuro/"
]
```

All S3 data is relative to `s3://{bucket}/{prefix}/`. The job downloads input data from this root and uploads results back to `s3://{bucket}/{prefix}/benchmark_results/{timestamp}/`.

Available options:

| Argument | Description |
|----------|-------------|
| `--s3_bucket` | S3 bucket name (required) |
| `--s3_prefix` | S3 key prefix (e.g. `combined_UCE_5neuro/`). All S3 I/O is relative to `s3://{bucket}/{prefix}/`. |
| `--index` | Path to FAISS index file (default: `indices/index_ivfflat.faiss`). Downloaded from `{prefix}/*.faiss` on S3. |
| `--test_dir` | Path to test data directory. Downloaded from `{prefix}/test/` on S3. |
| `--ref_annot` | Path to reference annotations TSV. Downloaded from `{prefix}/prediction_obs.tsv` on S3. |
| `--obo` | Path to Cell Ontology OBO file. Downloaded from `{prefix}/cl.obo` on S3. |
| `--ontology-method` | `ic` (default) or `shortest_path` |
| `--no-s3` | Skip S3 download/upload (local files only) |
| `--generate-plots` | Generate UMAP plots and confusion matrices |

### 2. Deploy

```bash
kubectl delete job fm-benchmark-job -n braingeneers 2>/dev/null
kubectl apply -f benchmarking-job.yaml
```

### 3. Monitor

```bash
# Watch logs
kubectl logs -n braingeneers -l job-name=fm-benchmark-job -f

# Check status
kubectl get job fm-benchmark-job -n braingeneers
```

**What to look for in the logs:**
- `Precomputing Information Content...` — IC precomputation (once per run)
- `Saved per-cell results...` — results being written
- `Uploading results to s3://...` — S3 upload in progress
- `files uploaded to s3://...` — done

### 4. Get Results

Results are uploaded to S3 automatically:

```bash
# List available runs
aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/

# Download a specific run
aws s3 cp --recursive s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/ ./benchmark_results/{timestamp}/
```

### 5. Clean Up

```bash
kubectl delete job fm-benchmark-job -n braingeneers
```

---

## Running Locally

```bash
python3 -m main_benchmark \
    --index indices/index_ivfflat.faiss \
    --test_dir test_data/ \
    --ref_annot reference_data/prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --no-s3
```

Results are written to `./benchmark_results/{timestamp}/`.

---

## Understanding Results

Each run creates a timestamped directory (same structure locally and on S3):

```
benchmark_results/{timestamp}/
├── benchmark_results.csv              Summary metrics per index/metric/dataset
├── benchmark.log                      Run log (warnings, filtered cells, timing)
├── per_cell_results/
│   └── {dataset}_{index}_{metric}_per_cell_results.csv
├── ontology_analysis/
│   └── ontology_analysis_report.txt
└── visualizations/                    (if --generate-plots was used)
```

For column descriptions, ontology similarity interpretation, and output format details, see [Output Columns](README.md#output-columns) and [Ontology-Aware Metrics](README.md#ontology-aware-metrics) in README.

---

## Setup (One-Time)

These steps are only needed when setting up the cluster environment for the first time or updating the Docker image.

### Build and Push Docker Image

```bash
docker buildx build --platform linux/amd64 -t jzhu647/fm_benchmark:latest -f Dockerfile .
docker login
docker push jzhu647/fm_benchmark:latest
```

### Update Docker Image Reference

If you change the image name/tag, update `benchmarking-job.yaml`:

```yaml
image: jzhu647/fm_benchmark:latest
```

---

## Troubleshooting

### Pod Stuck in Pending
```bash
kubectl describe pod -n braingeneers -l job-name=fm-benchmark-job
```
Check Events for: node selector mismatch, resource limits.

### Job Fails
```bash
kubectl logs -n braingeneers -l job-name=fm-benchmark-job
```
Check for error messages. Common causes: missing data files, S3 credentials not configured.

### S3 Upload Failed
- Check AWS credentials in the Kubernetes secret (`latentbrain-aws-creds`)
- Results are still in the container's temp directory if S3 upload fails

### No Results Files
- Verify job completed: `kubectl get job fm-benchmark-job -n braingeneers`
- Check S3: `aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/`
