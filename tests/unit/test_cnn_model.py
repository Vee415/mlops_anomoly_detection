"""Unit tests for the 1D-CNN model and model factory."""

import pytest
import torch

from src.model import get_model, MODEL_REGISTRY
from src.models.cnn import SensorCNN1D
from src.models.fc import SensorClassifier


class TestSensorCNN1D:
    def test_output_shape(self):
        """CNN output should be (batch, n_classes)."""
        model = SensorCNN1D(window_size=128, n_classes=4)
        x = torch.randn(8, 1, 128)
        out = model(x)
        assert out.shape == (8, 4)

    def test_single_sample(self):
        """CNN should work with batch_size=1 in eval mode."""
        model = SensorCNN1D(window_size=128, n_classes=4)
        model.eval()
        x = torch.randn(1, 1, 128)
        out = model(x)
        assert out.shape == (1, 4)

    def test_gradient_flow(self):
        """Gradients should flow through the CNN."""
        model = SensorCNN1D(window_size=128, n_classes=4)
        x = torch.randn(4, 1, 128)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_different_window_sizes(self):
        """CNN should work with different window sizes."""
        for ws in [64, 128, 256]:
            model = SensorCNN1D(window_size=ws, n_classes=4)
            x = torch.randn(2, 1, ws)
            out = model(x)
            assert out.shape == (2, 4)


class TestModelFactory:
    def test_get_model_fc(self):
        """get_model('fc') should return SensorClassifier instance."""
        model = get_model("fc", input_dim=4, hidden_dim=256, n_classes=4)
        assert isinstance(model, SensorClassifier)

    def test_get_model_cnn(self):
        """get_model('cnn') should return SensorCNN1D instance."""
        model = get_model("cnn", window_size=128, n_classes=4)
        assert isinstance(model, SensorCNN1D)

    def test_get_model_unknown(self):
        """get_model with unknown arch should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown arch"):
            get_model("transformer", input_dim=4)

    def test_registry_keys(self):
        """MODEL_REGISTRY should have fc and cnn keys."""
        assert "fc" in MODEL_REGISTRY
        assert "cnn" in MODEL_REGISTRY