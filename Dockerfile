FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and config
COPY src/ ./src/
COPY params.yaml .
COPY dvc.yaml .

# Create output directories
RUN mkdir -p data/raw data/processed models reports

CMD ["dvc", "repro"]