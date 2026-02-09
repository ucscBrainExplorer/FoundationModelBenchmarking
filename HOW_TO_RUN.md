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
cd /Users/Suhas/FoundationModelBenchmarking
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
5feb82d2dbfd: Pushed
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

**Note**: You may need to create `pvc.yaml` if it doesn't exist. Basic format:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: benchmark-data-pvc
  namespace: braingeneers
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
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
- `✓ Saved per-cell results...` - Results being saved
- `Results saved to /data/benchmark_results.csv` - Summary complete
- `Ontology analysis complete` - Full pipeline complete

**To exit logs**: Press `Ctrl+C`

### 6.4 Check Detailed Pod Information (if issues)
```bash
kubectl describe pod -n braingeneers -l job-name=uce-benchmark-job
```

**Look for**:
- Events section (shows what's happening)
- Status conditions
- Resource usage

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

## Step 8: Access Results via Temporary Pod

### 8.1 Create Access Pod
```bash
kubectl apply -f pvc-access-pod.yaml
```

### 8.2 Wait for Pod to be Ready
```bash
kubectl wait --for=condition=Ready pod/pvc-access-pod -n braingeneers --timeout=60s
```

**If it times out**, check pod status:
```bash
kubectl get pod pvc-access-pod -n braingeneers
kubectl describe pod pvc-access-pod -n braingeneers
```

**Common Issues**:
- Pod stuck in `Pending`: Check node selector matches available nodes
- PVC not found: Create PVC (Step 3.2)

---

## Step 9: View Results

### 9.1 View Summary Results (Benchmark Metrics)
```bash
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results.csv
```

**Expected Output**:
```
Index,Metric,Dataset,Avg_Query_Time_ms,accuracy,f1_macro,f1_weighted,top_k_accuracy,mean_ontology_dist,median_ontology_dist
ivfFlat,euclidean,e4ddac12-f48f-4455-8e8d-c2a48a683437,0.32,0.8942,0.1499,0.9125,0.9129,0.0998,0.0
ivfFlat,cosine,e4ddac12-f48f-4455-8e8d-c2a48a683437,0.31,0.8942,0.1499,0.9125,0.9129,0.0998,0.0
```

### 9.2 View Ontology Analysis Report
```bash
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/ontology_analysis/ontology_analysis_report.txt
```

**Shows**:
- Overall statistics
- Distance distribution
- Relationship to accuracy metrics
- Correlation analysis

### 9.3 List All Generated Files
```bash
kubectl exec -n braingeneers pvc-access-pod -- find /mnt/data -type f | sort
```

**Expected Files**:
- `/mnt/data/benchmark_results.csv` - Summary metrics
- `/mnt/data/per_cell_results/*_per_cell_results.csv` - Per-cell predictions
- `/mnt/data/ontology_analysis/ontology_analysis_report.txt` - Detailed analysis
- `/mnt/data/ontology_analysis/ontology_distance_accuracy_relationship.csv` - Distance-accuracy correlation
- `/mnt/data/visualizations/*.png` - UMAP plots, confusion matrix
- `/mnt/data/visualizations/summary_table.csv` - Summary table

### 9.4 View Sample Per-Cell Results
```bash
kubectl exec -n braingeneers pvc-access-pod -- head -50 /mnt/data/per_cell_results/e4ddac12-f48f-4455-8e8d-c2a48a683437_ivfFlat_euclidean_per_cell_results.csv
```

**Shows**: First 50 rows with cell_id, true_label, prediction_label, ontology_distance

### 9.5 View Distance-Accuracy Relationship
```bash
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/ontology_analysis/ontology_distance_accuracy_relationship.csv
```

**Shows**: How accuracy varies by ontology distance

---

## Step 10: Run Detailed Analysis Script (Optional)

### 10.1 Install Required Packages in Pod
```bash
kubectl exec -n braingeneers pvc-access-pod -- pip install pandas numpy
```

### 10.2 Copy Analysis Script to Pod
```bash
kubectl cp analyze_per_cell_results.py braingeneers/pvc-access-pod:/tmp/analyze_per_cell_results.py
```

### 10.3 Run Analysis Script
```bash
kubectl exec -n braingeneers pvc-access-pod -- python3 /tmp/analyze_per_cell_results.py --results-dir /mnt/data/per_cell_results --benchmark-results /mnt/data/benchmark_results.csv --max-rows 200
```

**Output Includes**:
- Per-cell results table
- Ontology tree distance statistics
- Relationship between tree distance and metrics
- Distance distribution analysis

---

## Step 11: Download Results Locally (Optional)

### 11.1 Create Local Directory
```bash
mkdir -p local_results
```

### 11.2 Copy Results Files
```bash
# Copy benchmark results
kubectl cp braingeneers/pvc-access-pod:/mnt/data/benchmark_results.csv ./local_results/benchmark_results.csv

# Copy per-cell results (one file at a time)
kubectl cp braingeneers/pvc-access-pod:/mnt/data/per_cell_results/e4ddac12-f48f-4455-8e8d-c2a48a683437_ivfFlat_euclidean_per_cell_results.csv ./local_results/euclidean_per_cell_results.csv

kubectl cp braingeneers/pvc-access-pod:/mnt/data/per_cell_results/e4ddac12-f48f-4455-8e8d-c2a48a683437_ivfFlat_cosine_per_cell_results.csv ./local_results/cosine_per_cell_results.csv

# Copy ontology analysis
kubectl cp braingeneers/pvc-access-pod:/mnt/data/ontology_analysis/ontology_analysis_report.txt ./local_results/ontology_analysis_report.txt

# Copy visualizations (if needed)
kubectl cp braingeneers/pvc-access-pod:/mnt/data/visualizations ./local_results/visualizations
```

---

## Step 12: Clean Up

### 12.1 Delete Access Pod (when done viewing results)
```bash
kubectl delete pod pvc-access-pod -n braingeneers
```

### 12.2 Delete Completed Job (optional, to free resources)
```bash
kubectl delete job uce-benchmark-job -n braingeneers
```

**Note**: Results remain in PVC, so you can recreate the access pod anytime to view them again.

---

## Quick Reference: All Commands in One Place

```bash
# 1. Build
cd /Users/Suhas/FoundationModelBenchmarking
docker build --no-cache -t suhaso123/uce-benchmark:v1 .

# 2. Push
docker login
docker push suhaso123/uce-benchmark:v1

# 3. Deploy
kubectl delete job uce-benchmark-job -n braingeneers 2>/dev/null; sleep 5
kubectl apply -f benchmarking-job.yaml

# 4. Monitor
kubectl logs -n braingeneers -l job-name=uce-benchmark-job -f

# 5. Check Status
kubectl get job uce-benchmark-job -n braingeneers
kubectl get pods -n braingeneers -l job-name=uce-benchmark-job

# 6. Access Results
kubectl apply -f pvc-access-pod.yaml
kubectl wait --for=condition=Ready pod/pvc-access-pod -n braingeneers --timeout=60s

# 7. View Results
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/benchmark_results.csv
kubectl exec -n braingeneers pvc-access-pod -- cat /mnt/data/ontology_analysis/ontology_analysis_report.txt
kubectl exec -n braingeneers pvc-access-pod -- head -50 /mnt/data/per_cell_results/*_per_cell_results.csv

# 8. Cleanup
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
- Verify PVC is mounted correctly

### Issue: kubectl Commands Fail with Authentication Error
**Solution**:
- This is a network/authentication issue with your kubectl config
- Run commands in your local terminal (not through chat interface)
- Check your kubeconfig: `kubectl config view`

---

## Expected Runtime

- **Docker Build**: 5-10 minutes (first time), 1-2 minutes (cached)
- **Docker Push**: 2-5 minutes (depends on network)
- **Benchmark Execution**: 10-20 minutes (depends on dataset size)
- **Total**: ~20-35 minutes end-to-end

---

## What Gets Generated

### Summary Files
- `benchmark_results.csv`: Aggregate metrics per index/metric combination
- `summary_table.csv`: Summary statistics

### Detailed Files
- `*_per_cell_results.csv`: Predictions for each cell (one file per index/metric)
- `ontology_analysis_report.txt`: Comprehensive ontology distance analysis
- `ontology_distance_accuracy_relationship.csv`: Distance vs accuracy correlation

### Visualizations
- `*_umap_ground_truth.png`: UMAP colored by true cell types
- `*_umap_predictions.png`: UMAP colored by predictions
- `*_umap_errors.png`: UMAP colored by error magnitude
- `*_confusion_matrix.png`: Confusion matrix heatmap
- `ontology_distance_analysis.png`: Distance distribution plots

---

## Next Steps After Viewing Results

1. **Compare Results**: Compare new results with previous runs
2. **Analyze Errors**: Look at cells with distance ≥ 1 to understand error patterns
3. **Check Visualizations**: Review UMAP plots and confusion matrix
4. **Update Documentation**: Update results summary if needed
5. **Share Results**: Use results for presentations or reports

---

*This guide covers the complete workflow from building to viewing results. For technical details, see TECHNICAL_WALKTHROUGH.md*
