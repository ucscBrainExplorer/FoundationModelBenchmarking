#!/bin/bash

# Simple script to run the UCE benchmarking

echo "Starting UCE Benchmarking..."
echo "============================"
echo ""

# Check if FAISS index exists
if [ ! -f "indices/index_ivfflat.faiss" ]; then
    echo "ERROR: FAISS index not found at indices/index_ivfflat.faiss"
    echo "Please ensure the download is complete."
    exit 1
fi

# Check if reference annotations exist
if [ ! -f "reference_data/prediction_obs.tsv" ]; then
    echo "ERROR: Reference annotations not found at reference_data/prediction_obs.tsv"
    exit 1
fi

# Check if test data exists
if [ ! -d "test_data" ] || [ -z "$(ls -A test_data/*.npy 2>/dev/null)" ]; then
    echo "ERROR: No test data found in test_data/"
    exit 1
fi

echo "All required files found. Running benchmark..."
echo ""

# Run the benchmark
python3 main_benchmark.py \
    --test_dir test_data/ \
    --ref_annot reference_data/prediction_obs.tsv \
    --obo reference_data/cl.obo \
    --no-s3

echo ""
echo "============================"
echo "Benchmark complete!"
echo "Results saved to: benchmark_results.csv"
