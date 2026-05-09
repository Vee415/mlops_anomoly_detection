"""Unit tests for FastAPI serve module."""

import numpy as np
import pytest

from src.serve import app, CLASS_NAMES, PredictRequest, PredictResponse, HealthResponse


def test_predict_request_model():
    """Test PredictRequest pydantic model."""
    req = PredictRequest(signal=[1.0, 2.0, 3.0])
    assert len(req.signal) == 3
    assert req.signal[0] == 1.0


def test_predict_response_model():
    """Test PredictResponse pydantic model."""
    resp = PredictResponse(
        predicted_class="Normal",
        class_index=0,
        probabilities={"Normal": 0.9, "Inner Race Fault": 0.05, "Outer Race Fault": 0.03, "Ball Fault": 0.02},
    )
    assert resp.predicted_class == "Normal"
    assert resp.class_index == 0


def test_health_response_model():
    """Test HealthResponse pydantic model."""
    resp = HealthResponse(status="ok", model_loaded=True)
    assert resp.status == "ok"
    assert resp.model_loaded is True


def test_class_names():
    """Verify CLASS_NAMES has 4 entries."""
    assert len(CLASS_NAMES) == 4
    assert "Normal" in CLASS_NAMES