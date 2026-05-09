# Session 2 Update — 2026-05-09

## What Was Done

Upgraded the MLOps pipeline with 8 phases of improvements, all verified end-to-end.

## Phase 0: Initial Git Commit
- Committed all baseline files as first commit

## Phase 1: Bug Fixes (5 bugs)
- **Normalization order**: `preprocess.py` now splits data first, then computes normalization stats from the actual training set (was using a sequential slice before random split)
- **Stratified splitting**: `split_data` now uses `sklearn.model_selection.train_test_split` with `stratify=labels` instead of random permutation
- **Hardcoded input_dim**: `evaluate.py` now loads `preprocess_info.json` to infer input_dim dynamically
- **Train-serve skew**: `predict.py` now predicts per-window and aggregates probabilities (was averaging features before prediction)
- **Docstring fix**: `model.py` module docstring corrected from "1D CNN" to "fully-connected"

## Phase 2: CWRU Bearing Dataset Integration
- **New file**: `src/download_cwru.py` — downloads CWRU .mat files, converts to signals.npy/labels.npy
- **New file**: `src/prepare_data.py` — dispatcher that routes to synthetic or CWRU based on `data_source.type` in params.yaml
- **Modified**: `params.yaml` — added `data_source.type: synthetic|cwru`
- **Modified**: `dvc.yaml` — merged generate/download into single `data` stage (DVC doesn't allow overlapping outputs)
- **Modified**: `requirements.txt` — added `scipy>=1.11.0`

## Phase 3: 1D-CNN Model Architecture
- **New**: `src/models/__init__.py` — MODEL_REGISTRY mapping "fc" and "cnn" to model classes
- **New**: `src/models/fc.py` — moved `SensorClassifier` from model.py
- **New**: `src/models/cnn.py` — `SensorCNN1D` (Conv1d→BN→ReLU→Pool→Conv1d→BN→ReLU→Pool→AdaptiveAvgPool→FC)
- **Modified**: `src/model.py` — now a factory with `get_model(arch, **kwargs)`
- **Modified**: `src/preprocess.py` — added `extract_raw_windows()` and `create_window_dataset()` for CNN; outputs raw window .npy files alongside features
- **Modified**: `src/dataset.py` — added `SensorWindowDataset` (loads raw windows, shape N×1×W); `get_dataloaders()` accepts `model_type` parameter
- **Modified**: `src/train.py` — routes to FC or CNN based on `model.arch` param; logs `arch` to MLflow
- **Modified**: `src/evaluate.py` — routes to correct model/dataset based on `model.arch`
- **Modified**: `src/predict.py` — routes to FC (feature extraction) or CNN (raw windows) based on `model.arch`; CNN uses raw windows without feature normalization
- **Modified**: `params.yaml` — added `model.arch: cnn` and `model.cnn` section (channels, kernel_size, pool_size, fc_hidden, dropout)
- **Modified**: `dvc.yaml` — added window outputs to preprocess stage, CNN params to train/evaluate stages
- **New**: `tests/unit/test_cnn_model.py` — 8 tests (CNN output shape, gradient flow, factory pattern)
- **Updated**: `conftest.py` — added `sample_windows` fixture

**Performance comparison**:
- FC model: 91.77% test accuracy
- CNN model: 99.76% test accuracy

## Phase 4: Data Validation
- **New file**: `src/validate.py` — shape checks, label distribution, NaN/Inf detection, KS-test drift detection
- **Modified**: `dvc.yaml` — added `validate` stage between preprocess and train
- **Modified**: `params.yaml` — added `validate` section (min_per_class, drift_threshold)
- **New**: `tests/unit/test_validate.py` — 10 tests
- **Result**: Pipeline validation passed with 1 minor drift warning (1/4 features)

## Phase 5: Hyperparameter Tuning (Optuna)
- **New file**: `src/tune.py` — Optuna study with MLflow tracking; tunes lr, hidden_dim, batch_size
- **Modified**: `dvc.yaml` — added optional `tune` stage
- **Modified**: `requirements.txt` — added `optuna>=3.4.0`
- **Result**: Best trial found lr=0.00105, hidden_dim=256, batch_size=64 with 92.23% val accuracy

## Phase 6: FastAPI Model Serving
- **New file**: `src/serve.py` — FastAPI app with `/predict` and `/health` endpoints; uses lifespan pattern for model loading; supports both FC and CNN architectures
- **New file**: `Dockerfile.serve` — serving-specific container on port 8000
- **Modified**: `docker-compose.yml` — added `serve` service
- **Modified**: `requirements.txt` — added `fastapi>=0.104.0`, `uvicorn>=0.24.0`
- **New**: `tests/unit/test_serve.py` — 4 tests (pydantic models, class names)

## Phase 7: Data Monitoring
- **New file**: `src/monitor.py` — per-feature KS-test drift detection + classification quality metrics; saves JSON and optional HTML report
- **Modified**: `dvc.yaml` — added `monitor` stage after evaluate
- **Modified**: `requirements.txt` — added `evidently>=0.4.0` (uses scipy.stats.ks_2samp for drift; Evidently for future HTML reports)
- **Result**: 1/4 features drifted (minor), 99.76% classification accuracy

## Phase 8: DVC Remote Configuration
- **Modified**: `.dvc/config` — added S3 remote template (`s3://your-bucket/dvc-storage`)
- **Modified**: `requirements.txt` — changed `dvc` to `dvc[s3]>=3.36.0`

## Environment
- Cloned `gpu_base` conda env into `mlops` (Python 3.12, CUDA 12.4, PyTorch 2.5.1)
- All dependencies installed (scipy, optuna, fastapi, uvicorn, evidently)

## Test Results
- **35 unit tests passing**
- **7 DVC pipeline stages all green**: data → preprocess → validate → train → evaluate → tune → monitor
- **CNN model (default)**: 99.76% test accuracy, 99.62% val accuracy

## How to Run
```bash
cd sensor_mlops_pipeline
conda activate mlops
dvc repro                      # run full pipeline (CNN model)
dvc repro --force              # force re-run all stages
pytest tests/unit/ -v           # run unit tests
python src/predict.py --signal-file test_signal.npy  # single inference
# To switch to FC model: change model.arch to "fc" in params.yaml
# To use CWRU data: change data_source.type to "cwru" in params.yaml
# To serve via API: uvicorn src.serve:app --port 8000
```

## Key Files Changed/Created
- `src/model.py` — refactored to factory pattern
- `src/models/cnn.py` — new CNN model
- `src/models/fc.py` — moved FC model
- `src/preprocess.py` — fixed normalization, stratification, added window extraction
- `src/evaluate.py` — dynamic input_dim, multi-architecture support
- `src/predict.py` — fixed train-serve skew, multi-architecture support
- `src/train.py` — multi-architecture, MLflow arch logging
- `src/dataset.py` — added SensorWindowDataset, model_type parameter
- `src/validate.py` — new data validation module
- `src/tune.py` — new Optuna tuning module
- `src/serve.py` — new FastAPI serving module
- `src/monitor.py` — new drift monitoring module
- `src/download_cwru.py` — new CWRU data download
- `src/prepare_data.py` — new data dispatcher
- `dvc.yaml` — 7 stages (data, preprocess, validate, train, evaluate, tune, monitor)
- `params.yaml` — added data_source, validate, model.arch, model.cnn sections
- `docker-compose.yml` — added serve service
- `Dockerfile.serve` — new serving container