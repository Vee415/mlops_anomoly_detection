"""Preprocess raw sensor data into feature vectors for training.

Sliding-window feature extraction with statistical features:
mean, std, peak, RMS per window → 4 features per window.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load_raw_data(raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw signals and labels from .npy files."""
    signals = np.load(Path(raw_dir) / "signals.npy")
    labels = np.load(Path(raw_dir) / "labels.npy")
    return signals, labels


def extract_features(
    signal: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
) -> np.ndarray:
    """Extract statistical features from a single signal using sliding windows.

    For 1D signal input (shape: [signal_length]), computes:
    mean, std, peak (max abs), RMS per window → 4 features per window.

    Args:
        signal: 1D array of shape (signal_length,)
        window_size: size of sliding window
        stride: step between windows

    Returns:
        features: array of shape (n_windows, 4)
    """
    features = []
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i : i + window_size]
        features.append(
            [
                window.mean(),
                window.std(),
                np.abs(window).max(),
                np.sqrt(np.mean(window**2)),  # RMS
            ]
        )
    return np.array(features, dtype=np.float32)


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features. Compute stats from training set if not provided."""
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero
    normalized = (features - mean) / std
    return normalized, mean, std


def create_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for all samples.

    Each sample's windows get the sample's label. Then all windows
    are aggregated across all samples.

    Returns:
        all_features: shape (total_windows, 4)
        all_labels: shape (total_windows,)
    """
    all_features = []
    all_labels = []

    for signal, label in zip(signals, labels):
        features = extract_features(signal, window_size=window_size, stride=stride)
        all_features.append(features)
        all_labels.append(np.full(len(features), label))

    return np.concatenate(all_features), np.concatenate(all_labels)


def extract_raw_windows(
    signal: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
) -> np.ndarray:
    """Extract raw signal windows from a single signal.

    Args:
        signal: 1D array of shape (signal_length,)
        window_size: size of sliding window
        stride: step between windows

    Returns:
        windows: array of shape (n_windows, window_size)
    """
    windows = []
    for i in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[i : i + window_size])
    return np.array(windows, dtype=np.float32)


def create_window_dataset(
    signals: np.ndarray,
    labels: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw signal windows and labels for all samples.

    Returns:
        all_windows: shape (total_windows, window_size)
        all_labels: shape (total_windows,)
    """
    all_windows = []
    all_labels = []

    for signal, label in zip(signals, labels):
        windows = extract_raw_windows(signal, window_size=window_size, stride=stride)
        all_windows.append(windows)
        all_labels.append(np.full(len(windows), label))

    return np.concatenate(all_windows), np.concatenate(all_labels)


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split data into train/val/test sets with stratification.

    Uses stratified splitting to preserve class proportions across all splits.
    """
    from sklearn.model_selection import train_test_split

    # First split off test set
    remaining_features, test_features, remaining_labels, test_labels = (
        train_test_split(features, labels, test_size=test_split, stratify=labels, random_state=seed)
    )

    # Then split remaining into train and val
    val_ratio = val_split / (1 - test_split)
    train_features, val_features, train_labels, val_labels = (
        train_test_split(remaining_features, remaining_labels, test_size=val_ratio, stratify=remaining_labels, random_state=seed)
    )

    splits = {
        "train": (train_features, train_labels),
        "val": (val_features, val_labels),
        "test": (test_features, test_labels),
    }

    for name, (x, y) in splits.items():
        print(f"  {name}: {len(x)} samples, class distribution: {np.bincount(y)}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Preprocess sensor data")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--params", default="params.yaml", help="Params file")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    window_size = params["preprocess"]["window_size"]
    stride = params["preprocess"]["stride"]
    val_split = params["preprocess"]["val_split"]
    test_split = params["preprocess"]["test_split"]

    print("Loading raw data...")
    signals, labels = load_raw_data(args.input)
    print(f"  Loaded {len(signals)} samples")

    print("Extracting features...")
    features, feature_labels = create_dataset(signals, labels, window_size, stride)
    print(f"  Extracted {len(features)} feature windows")

    print("Extracting raw windows...")
    windows, window_labels = create_window_dataset(signals, labels, window_size, stride)
    print(f"  Extracted {len(windows)} raw windows")

    print("Splitting data...")
    # Split first to get the actual train set, then normalize using its statistics
    splits = split_data(features, feature_labels, val_split, test_split)

    # Split raw windows using the same indices (need same split structure)
    # Since split_data is stratified, we split windows separately but with same params
    window_splits = split_data(windows, window_labels, val_split, test_split)

    print("Normalizing features...")
    # Compute normalization stats from the actual training split only
    train_features_raw = splits["train"][0]
    _, mean, std = normalize_features(train_features_raw)

    # Apply normalization to all splits
    normalized_splits = {}
    for name, (x, y) in splits.items():
        x_norm, _, _ = normalize_features(x, mean, std)
        normalized_splits[name] = (x_norm, y)
    splits = normalized_splits

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (x, y) in splits.items():
        np.save(output_dir / f"{name}_features.npy", x)
        np.save(output_dir / f"{name}_labels.npy", y)

    # Save raw windows for CNN model
    for name, (w, _) in window_splits.items():
        np.save(output_dir / f"{name}_windows.npy", w)

    # Save normalization stats for inference
    np.save(output_dir / "norm_mean.npy", mean)
    np.save(output_dir / "norm_std.npy", std)

    # Save split info as metrics metadata
    info = {
        "n_train": len(splits["train"][0]),
        "n_val": len(splits["val"][0]),
        "n_test": len(splits["test"][0]),
        "n_features": features.shape[1],
        "window_size": window_size,
        "stride": stride,
    }
    with open(output_dir / "preprocess_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved to {output_dir}/")
    print(f"  Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")


if __name__ == "__main__":
    main()