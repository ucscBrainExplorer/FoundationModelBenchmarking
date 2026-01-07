FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset and index files if we want them baked in, 
# but likely they should be mounted or downloaded.
# For now, we assume the code is here.
COPY . .

# Set default command
CMD ["python3", "main_benchmark.py"]
