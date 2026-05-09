"""Data drift and model performance monitoring."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import ks_2samp


def compute_drift_metrics(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.05,
) -> dict:
    """Compute per-feature drift using KS test.

    Args:
        reference_features: Training features (reference distribution).
        current_features: Test features (current distribution).
        feature_names: Optional names for features.
        threshold: P-value threshold for drift detection.

    Returns:
        Dict with drift metrics.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(reference_features.shape[1])]

    drifting = []
    for i, name in enumerate(feature_names):
        stat, pvalue = ks_2samp(reference_features[:, i], current_features[:, i])
        drifting.append({
            "name": name,
            "statistic": round(float(stat), 4),
            "pvalue": round(float(pvalue), 4),
            "is_drift": pvalue < threshold,
        })

    n_drifted = sum(1 for d in drifting if d["is_drift"])
    return {
        "n_features": len(feature_names),
        "n_drifted": n_drifted,
        "dataset_drift": n_drifted > 0,
        "threshold": threshold,
        "features": drifting,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Compute classification quality metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Optional class name mapping.

    Returns:
        Dict with classification metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)

    return {
        "accuracy": round(acc, 4),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor data drift and model performance")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    train_features = np.load(data_dir / "train_features.npy")
    test_features = np.load(data_dir / "test_features.npy")
    test_labels = np.load(data_dir / "test_labels.npy")

    # Drift metrics
    print("Computing drift metrics...")
    drift_metrics = compute_drift_metrics(train_features, test_features)

    # Load model and predict
    import torch
    from src.model import get_model

    with open(args.params) as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = params["model"].get("arch", "fc")
    model_params = params["model"]

    if arch == "cnn":
        from src.dataset import SensorWindowDataset

        cnn_params = model_params.get("cnn", {})
        model = get_model(
            arch,
            window_size=params["preprocess"]["window_size"],
            n_classes=model_params["n_classes"],
            channels=tuple(cnn_params.get("channels", [32, 64])),
            kernel_size=cnn_params.get("kernel_size", 7),
            pool_size=cnn_params.get("pool_size", 2),
            fc_hidden=cnn_params.get("fc_hidden", 128),
            dropout=cnn_params.get("dropout", 0.3),
        )
        test_ds = SensorWindowDataset(str(data_dir / "test_windows.npy"), str(data_dir / "test_labels.npy"))
    else:
        from src.dataset import SensorDataset

        model = get_model(
            arch,
            input_dim=train_features.shape[1],
            hidden_dim=model_params["hidden_dim"],
            n_classes=model_params["n_classes"],
        )
        test_ds = SensorDataset(str(data_dir / "test_features.npy"), str(data_dir / "test_labels.npy"))

    model.load_state_dict(torch.load("models/model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
    class_metrics = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)

    # Save metrics
    Path("reports").mkdir(exist_ok=True)
    metrics = {
        "drift": drift_metrics,
        "classification": class_metrics,
    }
    with open("reports/drift_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Drift metrics saved to reports/drift_metrics.json")
    print(f"  Dataset drift: {drift_metrics['dataset_drift']} ({drift_metrics['n_drifted']}/{drift_metrics['n_features']} features)")
    print(f"  Accuracy: {class_metrics['accuracy']}")


if __name__ == "__main__":
    main()