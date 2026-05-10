# Sensor Anomaly Detection — MLOps Pipeline

A fully reproducible ML pipeline for detecting bearing faults from vibration/IMU sensor data, built with production-grade MLOps practices.

One command reproduces every experiment:

```bash
dvc repro
```

## What It Does

Takes raw vibration signals and classifies them into 4 categories: **Normal**, **Inner Race Fault**, **Outer Race Fault**, or **Ball Fault**. Two model architectures are supported:

| Architecture | Test Accuracy | Input |
|---|---|---|
| FC (feed-forward) | 91.77% | Statistical features (mean, std, peak, RMS) |
| CNN (1D-CNN) | 99.76% | Raw signal windows |

The CNN is the default and recommended model.

## Stack

| Component | Tool | Purpose |
|---|---|---|
| Data versioning | DVC | Track datasets alongside code |
| Experiment tracking | MLflow | Log params, metrics, and models per run |
| Containerization | Docker + Dev Containers | Reproducible environments |
| CI/CD | GitHub Actions | Auto-test, lint, train on push |
| Training | PyTorch | FC and 1D-CNN classifiers |
| Hyperparameter tuning | Optuna | Bayesian search over lr, hidden_dim, batch_size |
| Serving | FastAPI | REST inference endpoint |
| Monitoring | Custom + Evidently | KS-test drift detection + classification quality |
| Code quality | ruff + mypy | Linting and type checking |

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate mlops

# Run the full pipeline
dvc repro

# Run unit tests
pytest tests/unit/ -v

# Switch to FC model
# Edit params.yaml: change model.arch from "cnn" to "fc", then dvc repro

# Switch to CWRU real dataset
# Edit params.yaml: change data_source.type from "synthetic" to "cwru", then dvc repro

# Serve via API
uvicorn src.serve:app --port 8000
# POST http://localhost:8000/predict  {"signal": [0.1, -0.3, ...]}
```

## Pipeline Stages

```
data → preprocess → validate → train → evaluate → tune → monitor
```

| Stage | Input | Output |
|---|---|---|
| `data` | params.yaml | `data/raw/signals.npy`, `labels.npy` |
| `preprocess` | raw data + params | features, windows, normalization stats |
| `validate` | processed data | `reports/data_validation.json` |
| `train` | processed data + params | `models/model.pth`, `reports/metrics.json` |
| `evaluate` | model + test data | `reports/evaluation_report.md` |
| `tune` | processed data + params | `reports/tuning_results.json` |
| `monitor` | train/test data + model | `reports/drift_metrics.json` |

All hyperparameters live in `params.yaml` — no magic numbers in code.

## Project Structure

```
sensor_mlops_pipeline/
├── .github/workflows/       # CI/CD (train.yml, evaluate.yml)
├── .devcontainer/            # VS Code Dev Container
├── data/                     # DVC-tracked (raw + processed)
├── src/
│   ├── generate_synthetic.py # Synthetic vibration data generator
│   ├── download_cwru.py      # CWRU bearing dataset downloader
│   ├── prepare_data.py       # Data dispatcher (synthetic or CWRU)
│   ├── preprocess.py         # Feature extraction + windowing
│   ├── dataset.py            # PyTorch datasets (FC + CNN)
│   ├── model.py              # Model factory (fc / cnn)
│   ├── models/               # FC and CNN implementations
│   ├── train.py              # Training with MLflow tracking
│   ├── evaluate.py           # Classification report + confusion matrix
│   ├── validate.py           # Data quality checks + drift detection
│   ├── tune.py               # Optuna hyperparameter search
│   ├── predict.py            # Single-sample inference
│   ├── serve.py              # FastAPI serving endpoint
│   └── monitor.py            # Drift + classification monitoring
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Pipeline-level tests
├── reports/                  # Auto-generated metrics and reports
├── models/                   # Trained model artifacts (DVC-tracked)
├── dvc.yaml                  # Pipeline definition
├── params.yaml               # All hyperparameters
├── Dockerfile                # Training container
├── Dockerfile.serve          # Serving container
├── docker-compose.yml        # train + mlflow + serve
├── pyproject.toml            # ruff + mypy + pytest config
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Classify a vibration signal |
| `GET` | `/health` | Check if model is loaded |

Example request:
```json
{"signal": [0.12, -0.34, 0.56, ...]}
```

Example response:
```json
{
  "predicted_class": "Normal",
  "class_index": 0,
  "probabilities": {"Normal": 0.94, "Inner Race Fault": 0.02, "Outer Race Fault": 0.02, "Ball Fault": 0.02}
}
```

## Data Sources

- **Synthetic** (default): Generated vibration data mirroring CWRU characteristics
- **CWRU Bearing Dataset**: Real bearing fault data from [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter)

Switch between them via `data_source.type` in `params.yaml`.

## Docker

```bash
# Training pipeline
docker compose up train

# Start MLflow tracking server
# (included in docker-compose, port 5000)

# Serve model
docker compose up serve
```

## CI/CD

- **On push** (main/develop): lint + type-check + unit tests + train + upload metrics
- **On PR** (to main): full `dvc repro` + comment metrics on PR

## Reproducibility

- `dvc.lock` pins exact data + code versions
- `params.yaml` is the single source of truth for all hyperparameters
- MLflow logs every run with params, metrics, and model artifacts
- Dev Container ensures identical environments