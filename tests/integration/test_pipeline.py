"""Integration tests — verify full pipeline and model serving end-to-end."""

import subprocess

import numpy as np
import pytest
import torch
import yaml


@pytest.mark.integration
def test_dvc_repro():
    """Full pipeline should complete without error via dvc repro."""
    result = subprocess.run(
        ["dvc", "repro"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"dvc repro failed: {result.stderr}"


@pytest.mark.integration
def test_fc_model_output_shape():
    """FC model should produce correct output shape."""
    from src.model import get_model

    model = get_model("fc", input_dim=4, hidden_dim=256, n_classes=4)
    model.load_state_dict(
        torch.load("models/model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    x = torch.randn(8, 4)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (8, 4)


@pytest.mark.integration
def test_cnn_model_output_shape():
    """CNN model should produce correct output shape from raw windows."""
    from src.model import get_model

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    cnn_params = params["model"]["cnn"]
    model = get_model(
        "cnn",
        window_size=params["preprocess"]["window_size"],
        n_classes=params["model"]["n_classes"],
        channels=tuple(cnn_params["channels"]),
        kernel_size=cnn_params["kernel_size"],
        pool_size=cnn_params["pool_size"],
        fc_hidden=cnn_params["fc_hidden"],
        dropout=cnn_params["dropout"],
    )
    model.load_state_dict(
        torch.load("models/model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    x = torch.randn(8, 1, params["preprocess"]["window_size"])
    with torch.no_grad():
        output = model(x)
    assert output.shape == (8, params["model"]["n_classes"])


@pytest.mark.integration
def test_evaluation_report_exists():
    """Evaluation report should be generated after pipeline run."""
    from pathlib import Path

    report_path = Path("reports/evaluation_report.md")
    assert report_path.exists(), "Evaluation report not found"
    content = report_path.read_text()
    assert "Classification Report" in content
    assert "Confusion Matrix" in content


@pytest.mark.integration
def test_serve_health_endpoint():
    """FastAPI health endpoint should return ok when model is loaded."""
    from fastapi.testclient import TestClient
    from src.serve import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "model_not_loaded")


@pytest.mark.integration
def test_serve_predict_endpoint():
    """FastAPI predict endpoint should return valid classification."""
    from fastapi.testclient import TestClient
    from src.serve import app

    client = TestClient(app)
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(2048).astype(float).tolist()

    response = client.post("/predict", json={"signal": signal})
    if response.status_code == 503:
        pytest.skip("Model not loaded — skipping predict test")
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
    assert data["class_index"] in range(4)
    assert len(data["probabilities"]) == 4