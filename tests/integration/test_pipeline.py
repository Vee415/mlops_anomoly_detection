"""Integration test — verifies full DVC pipeline runs end-to-end."""

import subprocess

import pytest


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
def test_model_output_shape():
    """Trained model should produce correct output shape."""
    import numpy as np
    import torch

    from src.model import SensorClassifier

    model = SensorClassifier(input_dim=4, hidden_dim=256, n_classes=4)
    model.load_state_dict(
        torch.load("models/model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    x = torch.randn(8, 4)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (8, 4)


@pytest.mark.integration
def test_evaluation_report_exists():
    """Evaluation report should be generated after pipeline run."""
    from pathlib import Path

    report_path = Path("reports/evaluation_report.md")
    assert report_path.exists(), "Evaluation report not found"
    content = report_path.read_text()
    assert "Classification Report" in content
    assert "Confusion Matrix" in content