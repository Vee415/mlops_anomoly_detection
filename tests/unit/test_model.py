"""Unit tests for model module."""

import torch

from src.model import SensorClassifier


class TestSensorClassifier:
    def test_output_shape(self):
        """Model should output (batch, n_classes) tensor."""
        model = SensorClassifier(input_dim=4, hidden_dim=256, n_classes=4)
        x = torch.randn(8, 4)
        output = model(x)
        assert output.shape == (8, 4)

    def test_single_sample(self):
        """Model should handle batch size of 1 in eval mode."""
        model = SensorClassifier(input_dim=4, hidden_dim=256, n_classes=4)
        model.eval()
        x = torch.randn(1, 4)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 4)

    def test_output_is_logits(self):
        """Output should be logits (not probabilities)."""
        model = SensorClassifier(input_dim=4, hidden_dim=256, n_classes=4)
        model.eval()
        x = torch.randn(4, 4)
        with torch.no_grad():
            output = model(x)
        # Logits are not bounded to [0, 1]
        assert output.min() < 0 or output.max() > 1

    def test_different_input_dims(self):
        """Model should work with different input dimensions."""
        model = SensorClassifier(input_dim=12, hidden_dim=128, n_classes=4)
        x = torch.randn(4, 12)
        output = model(x)
        assert output.shape == (4, 4)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = SensorClassifier(input_dim=4, hidden_dim=256, n_classes=4)
        x = torch.randn(4, 4)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None