FROM python:3.9-slim

WORKDIR /app

# Install system dependencies with retry logic
RUN for i in 1 2 3; do \
        apt-get update --fix-missing && break || sleep 5; \
    done && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for better wheel support
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
# Install numpy first (required for faiss-cpu)
RUN pip install --no-cache-dir "numpy<2.0"
# Install faiss-cpu with prefer-binary to use pre-built wheels
RUN pip install --no-cache-dir --prefer-binary faiss-cpu
# Install remaining dependencies
RUN pip install --no-cache-dir --prefer-binary pandas scikit-learn pronto networkx boto3 matplotlib seaborn umap-learn

# Copy dataset and index files if we want them baked in, 
# but likely they should be mounted or downloaded.
# For now, we assume the code is here.
COPY . .

# Set default command
CMD ["python3", "main_benchmark.py"]
