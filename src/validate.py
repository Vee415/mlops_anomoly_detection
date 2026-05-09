"""Data validation — checks shapes, labels, NaN, and drift before training."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate_shapes(features: np.ndarray, labels: np.ndarray, expected_n_features: int) -> list[str]:
    """Verify feature and label array shapes match expectations."""
    errors = []
    if features.ndim != 2:
        errors.append(f"Features should be 2D, got {features.ndim}D")
    if labels.ndim != 1:
        errors.append(f"Labels should be 1D, got {labels.ndim}D")
    if features.shape[0] != labels.shape[0]:
        errors.append(f"Features and labels count mismatch: {features.shape[0]} vs {labels.shape[0]}")
    if features.shape[1] != expected_n_features:
        errors.append(f"Expected {expected_n_features} features, got {features.shape[1]}")
    return errors


def validate_label_distribution(labels: np.ndarray, min_per_class: int = 10) -> list[str]:
    """Check each class has minimum representation."""
    warnings = []
    counts = np.bincount(labels)
    for cls, count in enumerate(counts):
        if count < min_per_class:
            warnings.append(f"Class {cls} has only {count} samples (minimum: {min_per_class})")
    return warnings


def validate_no_nan(features: np.ndarray, labels: np.ndarray) -> list[str]:
    """Check for NaN/Inf values."""
    errors = []
    if np.any(np.isnan(features)):
        errors.append(f"Features contain {np.isnan(features).sum()} NaN values")
    if np.any(np.isinf(features)):
        errors.append(f"Features contain {np.isinf(features).sum()} Inf values")
    if np.any(np.isnan(labels)):
        errors.append(f"Labels contain {np.isnan(labels).sum()} NaN values")
    return errors


def validate_drift(
    train_features: np.ndarray,
    test_features: np.ndarray,
    threshold: float = 0.05,
) -> list[str]:
    """Detect distribution drift between train and test using KS test."""
    from scipy.stats import ks_2samp

    warnings = []
    n_features = train_features.shape[1]
    drifting = []
    for i in range(n_features):
        stat, pvalue = ks_2samp(train_features[:, i], test_features[:, i])
        if pvalue < threshold:
            drifting.append((i, stat, pvalue))

    if drifting:
        warnings.append(
            f"Drift detected in {len(drifting)}/{n_features} features (KS test, p<{threshold}): "
            + ", ".join(f"feat_{i}(p={p:.4f})" for i, _, p in drifting)
        )
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed data")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    validate_params = params.get("validate", {})
    min_per_class = validate_params.get("min_per_class", 10)
    drift_threshold = validate_params.get("drift_threshold", 0.05)

    data_dir = Path(args.data_dir)
    all_errors = []
    all_warnings = []

    # Load data
    train_features = np.load(data_dir / "train_features.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    val_features = np.load(data_dir / "val_features.npy")
    val_labels = np.load(data_dir / "val_labels.npy")
    test_features = np.load(data_dir / "test_features.npy")
    test_labels = np.load(data_dir / "test_labels.npy")

    # Load preprocess info for expected feature count
    with open(data_dir / "preprocess_info.json") as f:
        preprocess_info = json.load(f)
    expected_n_features = preprocess_info["n_features"]

    # Shape validation
    for name, feats, lbls in [
        ("train", train_features, train_labels),
        ("val", val_features, val_labels),
        ("test", test_features, test_labels),
    ]:
        errors = validate_shapes(feats, lbls, expected_n_features)
        for e in errors:
            all_errors.append(f"[{name}] {e}")

    # Label distribution
    for name, lbls in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        warnings = validate_label_distribution(lbls, min_per_class)
        for w in warnings:
            all_warnings.append(f"[{name}] {w}")

    # NaN check
    for name, feats, lbls in [
        ("train", train_features, train_labels),
        ("val", val_features, val_labels),
        ("test", test_features, test_labels),
    ]:
        errors = validate_no_nan(feats, lbls)
        for e in errors:
            all_errors.append(f"[{name}] {e}")

    # Drift detection
    warnings = validate_drift(train_features, test_features, drift_threshold)
    all_warnings.extend(warnings)

    # Print results
    print("=== Data Validation ===")
    if all_errors:
        print("ERRORS:")
        for e in all_errors:
            print(f"  {e}")
    if all_warnings:
        print("WARNINGS:")
        for w in all_warnings:
            print(f"  {w}")
    if not all_errors and not all_warnings:
        print("All checks passed!")

    # Save report
    report = {
        "passed": len(all_errors) == 0,
        "n_errors": len(all_errors),
        "n_warnings": len(all_warnings),
        "errors": all_errors,
        "warnings": all_warnings,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_validation.json", "w") as f:
        json.dump(report, f, indent=2)

    if all_errors:
        print(f"\nValidation FAILED with {len(all_errors)} errors")
        sys.exit(1)
    else:
        print(f"\nValidation passed ({len(all_warnings)} warnings)")


if __name__ == "__main__":
    main()