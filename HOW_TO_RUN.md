# Complete Guide: Running the Benchmark and Viewing Results

## Prerequisites

1. **Docker** installed and running
2. **kubectl** configured for your Kubernetes cluster
3. **Access to Kubernetes namespace**: `braingeneers`
4. **Docker Hub account** (for pushing images)

---

## Step 1: Build the Docker Image

### 1.1 Navigate to Project Directory
```bash
cd FoundationModelBenchmarking
```

### 1.2 Build Docker Image (with --no-cache to include latest code)
```bash
docker build --no-cache -t suhaso123/uce-benchmark:v1 .
```

**Expected Output**:
```
[+] Building ... FINISHED
 => naming to docker.io/suhaso123/uce-benchmark:v1
```

**Note**: This may take 5-10 minutes on first build. Subsequent builds (without --no-cache) will be faster.

---

## Step 2: Push Docker Image to Docker Hub

### 2.1 Login to Docker Hub
```bash
docker login
```

**Enter your Docker Hub username and password when prompted**

### 2.2 Push the Image
```bash
docker push suhaso123/uce-benchmark:v1
```

**Expected Output**:
```
The push refers to repository [docker.io/suhaso123/uce-benchmark]
...
v1: digest: sha256:... size: ...
```

---

## Step 3: Ensure PVC Exists

### 3.1 Check if PVC Exists
```bash
kubectl get pvc benchmark-data-pvc -n braingeneers
```

### 3.2 Create PVC if Missing
If the PVC doesn't exist, create it:
```bash
kubectl apply -f pvc.yaml
```

---

## Step 4: Delete Old Job (if exists)

### 4.1 Check for Existing Job
```bash
kubectl get job uce-benchmark-job -n braingeneers
```

### 4.2 Delete Old Job (if needed)
```bash
kubectl delete job uce-benchmark-job -n braingeneers
```

**Wait a few seconds for cleanup**:
```bash
sleep 5
```

---

## Step 5: Deploy the Benchmark Job

### 5.1 Apply the Job Configuration
```bash
kubectl apply -f benchmarking-job.yaml
```

### 5.2 Verify Job Created
```bash
kubectl get job uce-benchmark-job -n braingeneers
```

**Expected Output**:
```
NAME                 COMPLETIONS   DURATION   AGE
uce-benchmark-job    0/1           <unknown>  5s
```

---

## Step 6: Monitor Job Progress

### 6.1 Check Job Status
```bash
kubectl get job uce-benchmark-job -n braingeneers
```

### 6.2 Check Pod Status
```bash
kubectl get pods -n braingeneers -l job-name=uce-benchmark-job
```

**Statuses**:
- `Pending`: Waiting for resources
- `Running`: Job is executing
- `Succeeded`: Job completed successfully
- `Failed`: Job encountered an error

### 6.3 Watch Logs in Real-Time
```bash
kubectl logs -n braingeneers -l job-name=uce-benchmark-job -f
```

**What to Look For**:
- `Loading FAISS index...` - Index loading
- `Processing Index: ivfFlat` - Processing started
- `Precomputing Information Content...` - IC precomputation (once per run)
- `Saved per-cell results...` - Results being saved
- `Results saved to /data/benchmark_results/{timestamp}/benchmark_results.csv` - Summary complete
- `Ontology analysis complete` - Full pipeline complete
- `Uploading results to s3://...` - S3 upload in progress
- `files uploaded to s3://...` - S3 upload complete

**To exit logs**: Press `Ctrl+C`

### 6.4 Check Detailed Pod Information (if issues)
```bash
kubectl describe pod -n braingeneers -l job-name=uce-benchmark-job
```

---

## Step 7: Wait for Job Completion

### 7.1 Check Completion Status
```bash
kubectl get job uce-benchmark-job -n braingeneers
```

**When Complete**:
```
NAME                 COMPLETIONS   DURATION   AGE
uce-benchmark-job    1/1           15m        20m
```

### 7.2 Verify Pod Status
```bash
kubectl get pods -n braingeneers -l job-name=uce-benchmark-job
```

**Should show**: `STATUS: Succeeded`

---

## Step 8: Access Results

Results are available in three ways: S3 (easiest), PVC access pod, or kubectl cp.

### Option A: Download from S3 (Recommended)

Results are uploaded to S3 automatically after each run.

```bash
# List available runs
aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/

# Download a specific run's results
aws s3 cp --recursive s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/ ./results/

# View summary
cat ./results/benchmark_results.csv
```

### Option B: Access via PVC Pod

```bash
# Create access pod
kubectl apply -f pvc-access-pod.yaml
kubectl wait --for=condition=Ready pod/pvc-access-pod -n braingeneers --timeout=60s

# List available runs
kubectl exec -n braingeneers pvc-access-pod -- ls /mnt/data/benchmark_results/

# View a run's summary (replace {timestamp} with actual value, e.g. 20260210_143022)
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results/{timestamp}/benchmark_results.csv

# View ontology analysis
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results/{timestamp}/ontology_analysis/ontology_analysis_report.txt

# View per-cell results sample
kubectl exec -n braingeneers pvc-access-pod -- head -50 /mnt/data/benchmark_results/{timestamp}/per_cell_results/*_per_cell_results.csv

# View the run log (warnings, filtered cells, timing)
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results/{timestamp}/benchmark.log

# List all files in a run
kubectl exec -n braingeneers pvc-access-pod -- find /mnt/data/benchmark_results/{timestamp} -type f | sort
```

### Option C: Copy Results Locally via kubectl

```bash
mkdir -p local_results

# Copy entire run directory
kubectl cp braingeneers/pvc-access-pod:/mnt/data/benchmark_results/{timestamp}/ ./local_results/
```

---

## Step 9: Understand Results

### Results Directory Structure

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

### Summary CSV Columns (`benchmark_results.csv`)

| Column | Description |
|--------|-------------|
| Index | Index type (e.g., "ivfFlat") |
| Metric | Distance metric (e.g., "euclidean") |
| Dataset | Test dataset ID |
| accuracy | Overall prediction accuracy (0-1) |
| f1_macro | Macro-averaged F1 score |
| f1_weighted | Weighted F1 score |
| top_k_accuracy | Proportion where truth in top-k neighbors |
| mean_ontology_similarity | Mean Lin similarity (0-1, **higher = more similar**). Default IC method. |
| median_ontology_similarity | Median Lin similarity |
| Avg_Query_Time_ms | Average query time per cell (milliseconds) |

**Interpreting ontology similarity** (IC method, default):

| Score | Meaning |
|-------|---------|
| **1.0** | Identical terms (perfect prediction) |
| **> 0.8** | Closely related (e.g., neuron subtypes) |
| **0.4 - 0.7** | Moderately related (e.g., neuron vs. astrocyte) |
| **~0.0** | Unrelated |

If `--ontology-method shortest_path` was used instead, the columns are
`mean_ontology_dist` / `median_ontology_dist` (lower = more similar).

### Per-Cell CSV Columns

| Column | Description |
|--------|-------------|
| cell_id | Cell identifier |
| true_label | Ground truth ontology term ID |
| true_label_readable | Human-readable cell type name |
| prediction_label | Predicted ontology term ID |
| prediction_label_readable | Human-readable predicted name |
| vote_percentage | Fraction of k neighbors that voted for the prediction |
| mean_euclidean_distance | Mean FAISS distance to all k neighbors |
| nearest_neighbor_euclidean_distance | Distance to the closest neighbor |
| ontology_distance | Ontology score between prediction and truth |
| avg_neighbor_ontology_distance | Mean ontology score across all neighbors vs truth |

---

## Step 10: Clean Up

### 10.1 Delete Access Pod (when done viewing results)
```bash
kubectl delete pod pvc-access-pod -n braingeneers
```

### 10.2 Delete Completed Job (optional, to free resources)
```bash
kubectl delete job uce-benchmark-job -n braingeneers
```

**Note**: Results remain in PVC and S3, so you can access them anytime.

---

## Quick Reference: All Commands in One Place

```bash
# 1. Build and push
cd FoundationModelBenchmarking
docker build --no-cache -t suhaso123/uce-benchmark:v1 .
docker login
docker push suhaso123/uce-benchmark:v1

# 2. Deploy
kubectl delete job uce-benchmark-job -n braingeneers 2>/dev/null; sleep 5
kubectl apply -f benchmarking-job.yaml

# 3. Monitor
kubectl logs -n braingeneers -l job-name=uce-benchmark-job -f

# 4. Check status
kubectl get job uce-benchmark-job -n braingeneers

# 5. Get results from S3 (easiest)
aws s3 ls s3://latentbrain/combined_UCE_5neuro/benchmark_results/
aws s3 cp --recursive s3://latentbrain/combined_UCE_5neuro/benchmark_results/{timestamp}/ ./results/

# 6. Or access via PVC pod
kubectl apply -f pvc-access-pod.yaml
kubectl wait --for=condition=Ready pod/pvc-access-pod -n braingeneers --timeout=60s
kubectl exec -n braingeneers pvc-access-pod -- ls /mnt/data/benchmark_results/
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results/{timestamp}/benchmark_results.csv

# 7. Cleanup
kubectl delete pod pvc-access-pod -n braingeneers
```

---

## Troubleshooting

### Issue: Docker Build Fails
**Solution**:
- Check Docker is running: `docker ps`
- Check you're in the right directory
- Try: `docker build --no-cache -t suhaso123/uce-benchmark:v1 .`

### Issue: Docker Push Fails with "denied"
**Solution**:
- Run `docker login` first
- Check your Docker Hub username matches the image tag

### Issue: Pod Stuck in Pending
**Solution**:
```bash
kubectl describe pod -n braingeneers -l job-name=uce-benchmark-job
```
- Check Events section for errors
- Common: PVC not found, node selector mismatch, resource limits

### Issue: Job Fails
**Solution**:
```bash
kubectl logs -n braingeneers -l job-name=uce-benchmark-job
kubectl describe job uce-benchmark-job -n braingeneers
```
- Check logs for error messages
- Check job events for resource issues

### Issue: Access Pod Times Out
**Solution**:
```bash
kubectl get pod pvc-access-pod -n braingeneers
kubectl describe pod pvc-access-pod -n braingeneers
```
- Check if PVC exists: `kubectl get pvc benchmark-data-pvc -n braingeneers`
- Check node selector matches available nodes
- Try deleting and recreating: `kubectl delete pod pvc-access-pod -n braingeneers && kubectl apply -f pvc-access-pod.yaml`

### Issue: No Results Files Found
**Solution**:
- Check job completed successfully: `kubectl get job uce-benchmark-job -n braingeneers`
- Check pod logs for errors: `kubectl logs -n braingeneers -l job-name=uce-benchmark-job`
- Check the run log: look in `/mnt/data/benchmark_results/` for timestamped directories

### Issue: S3 Upload Failed
**Solution**:
- Check AWS credentials are configured in the Kubernetes secret
- Check pod logs for the specific error message
- Results are still available on the PVC even if S3 upload fails

---

## Expected Runtime

- **Docker Build**: 5-10 minutes (first time), 1-2 minutes (cached)
- **Docker Push**: 2-5 minutes (depends on network)
- **Benchmark Execution**: 10-20 minutes (depends on dataset size)
- **S3 Upload**: 1-2 minutes
- **Total**: ~20-35 minutes end-to-end
