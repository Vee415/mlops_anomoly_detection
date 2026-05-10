"""1D-CNN classifier for bearing fault detection on raw signal windows."""

import torch
import torch.nn as nn


class SensorCNN1D(nn.Module):
    """1D-CNN classifier for sensor anomaly detection on raw signal windows.

    Takes raw signal windows (shape: batch, 1, window_size) and classifies
    into n_classes fault categories.

    Architecture:
        Conv1d(1, channels[0], kernel_size) → BN → ReLU → MaxPool
        Conv1d(channels[0], channels[1], kernel_size) → BN → ReLU → MaxPool
        AdaptiveAvgPool1d(1) → Flatten
        FC(channels[1], fc_hidden) → BN → ReLU → Dropout
        FC(fc_hidden, n_classes)

    Args:
        window_size: Length of input signal window.
        n_classes: Number of fault categories.
        channels: List of conv channel sizes.
        kernel_size: Conv kernel size.
        pool_size: MaxPool kernel size.
        fc_hidden: Hidden dim for FC layers after conv.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        window_size: int = 128,
        n_classes: int = 4,
        channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 7,
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        conv_layers = []
        in_channels = 1
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, window_size)

        Returns:
            Logits tensor of shape (batch, n_classes)
        """
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
