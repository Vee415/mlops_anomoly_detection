"""Unit tests for dataset module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import SensorDataset, get_dataloaders


class TestSensorDataset:
    def test_length_and_item(self, sample_features, tmp_path):
        """Dataset should return correct length and (features, label) tuples."""
        features, labels = sample_features
        feat_path = tmp_path / "feat.npy"
        label_path = tmp_path / "label.npy"
        np.save(feat_path, features)
        np.save(label_path, labels)

        ds = SensorDataset(str(feat_path), str(label_path))
        assert len(ds) == len(labels)

        feat_item, label_item = ds[0]
        assert isinstance(feat_item, torch.Tensor)
        assert isinstance(label_item, torch.Tensor)
        assert feat_item.dtype == torch.float32
        assert label_item.dtype == torch.int64

    def test_dataloader_creation(self, sample_features, tmp_path):
        """get_dataloaders should create working DataLoaders."""
        features, labels = sample_features
        for split in ["train", "val", "test"]:
            np.save(tmp_path / f"{split}_features.npy", features)
            np.save(tmp_path / f"{split}_labels.npy", labels)

        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=str(tmp_path), batch_size=16
        )

        batch_features, batch_labels = next(iter(train_loader))
        assert batch_features.shape[1] == 4
        assert batch_labels.dtype == torch.int64