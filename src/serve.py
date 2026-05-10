"""FastAPI inference server for sensor anomaly detection."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import get_model

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]

# Global model state
_model = None
_norm_mean = None
_norm_std = None
_params = None


class PredictRequest(BaseModel):
    signal: list[float]


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model():
    """Load model and normalization stats."""
    global _model, _norm_mean, _norm_std, _params

    params_path = Path("params.yaml")
    if not params_path.exists():
        return

    with open(params_path) as f:
        _params = yaml.safe_load(f)

    data_dir = Path("data/processed")
    model_path = Path("models/model.pth")

    if not model_path.exists() or not data_dir.exists():
        return

    arch = _params["model"].get("arch", "fc")
    model_params = _params["model"]
    preprocess_params = _params["preprocess"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == "cnn":
        cnn_params = model_params.get("cnn", {})
        _model = get_model(
            arch,
            window_size=preprocess_params["window_size"],
            n_classes=model_params["n_classes"],
            channels=tuple(cnn_params.get("channels", [32, 64])),
            kernel_size=cnn_params.get("kernel_size", 7),
            pool_size=cnn_params.get("pool_size", 2),
            fc_hidden=cnn_params.get("fc_hidden", 128),
            dropout=cnn_params.get("dropout", 0.3),
        )
    else:
        with open(data_dir / "preprocess_info.json") as f:
            preprocess_info = json.load(f)
        input_dim = preprocess_info["n_features"]

        _model = get_model(
            arch,
            input_dim=input_dim,
            hidden_dim=model_params["hidden_dim"],
            n_classes=model_params["n_classes"],
        )

    _model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    _model.to(device)
    _model.eval()

    _norm_mean = np.load(data_dir / "norm_mean.npy")
    _norm_std = np.load(data_dir / "norm_std.npy")


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(title="Sensor Anomaly Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict fault type from raw vibration signal."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from src.preprocess import extract_features, extract_raw_windows

    signal = np.array(request.signal, dtype=np.float32)
    assert _params is not None  # guaranteed by model check above
    model_cfg = _params["model"]
    preprocess_cfg = _params["preprocess"]
    arch = model_cfg.get("arch", "fc")
    window_size = preprocess_cfg["window_size"]
    stride = preprocess_cfg["stride"]

    device = next(_model.parameters()).device

    if arch == "cnn":
        windows = extract_raw_windows(signal, window_size=window_size, stride=stride)
        # CNN uses raw windows directly (no normalization needed)
        x = torch.from_numpy(windows).float().unsqueeze(1).to(device)
    else:
        features = extract_features(signal, window_size=window_size, stride=stride)
        features_norm = (features - _norm_mean) / _norm_std
        x = torch.from_numpy(features_norm).float().to(device)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    mean_probs = probs.mean(axis=0)
    pred_class = int(mean_probs.argmax())

    return PredictResponse(
        predicted_class=CLASS_NAMES[pred_class],
        class_index=pred_class,
        probabilities={name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, mean_probs)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
