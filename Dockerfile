FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for DVC with S3
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, config, and pipeline definition
COPY src/ ./src/
COPY conftest.py .
COPY params.yaml .
COPY dvc.yaml .
COPY pyproject.toml .

# Create output directories
RUN mkdir -p data/raw data/processed models reports mlruns

CMD ["dvc", "repro"]