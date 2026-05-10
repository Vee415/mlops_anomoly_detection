"""Fully-connected classifier for bearing fault detection."""

import torch
import torch.nn as nn


class SensorClassifier(nn.Module):
    """FC classifier for sensor anomaly detection.

    Takes a feature vector from sliding-window preprocessing and classifies
    into n_classes fault categories.

    Architecture:
        FC input → BatchNorm → ReLU → FC hidden → BatchNorm → ReLU → FC output

    Args:
        input_dim: Number of input features (4 for single-axis stats).
        hidden_dim: Hidden layer dimension.
        n_classes: Number of fault categories.
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        n_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Logits tensor of shape (batch, n_classes)
        """
        return self.classifier(x)
