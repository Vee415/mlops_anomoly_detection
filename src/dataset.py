"""PyTorch Dataset for sensor classification features."""

import torch
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    """Dataset wrapping preprocessed .npy feature/label arrays."""

    def __init__(self, features_path: str, labels_path: str):
        import numpy as np

        self.features = torch.from_numpy(np.load(features_path)).float()
        self.labels = torch.from_numpy(np.load(labels_path)).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def get_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 64,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test DataLoaders from processed data."""
    from pathlib import Path

    data_path = Path(data_dir)

    train_ds = SensorDataset(str(data_path / "train_features.npy"), str(data_path / "train_labels.npy"))
    val_ds = SensorDataset(str(data_path / "val_features.npy"), str(data_path / "val_labels.npy"))
    test_ds = SensorDataset(str(data_path / "test_features.npy"), str(data_path / "test_labels.npy"))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader