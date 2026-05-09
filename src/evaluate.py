"""Evaluation script — generates per-class metrics and evaluation report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import subprocess
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import SensorDataset
from src.model import SensorClassifier

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]


def get_git_short_hash() -> str:
    """Get short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate(
    model: torch.nn.Module,
    dataset: SensorDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on dataset. Returns (predictions, labels)."""
    model.eval()
    all_preds = []
    all_labels = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: dict,
) -> str:
    """Generate markdown evaluation report."""
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=2, output_dict=False
    )
    cm = confusion_matrix(y_true, y_pred)
    git_hash = get_git_short_hash()

    # Per-class metrics as dict for JSON
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    md = f"""# Evaluation Report — Sensor Anomaly Detection

## Run Info
- **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Commit:** {git_hash}
- **Params:** window_size={params['preprocess']['window_size']}, lr={params['train']['lr']}, epochs={params['train']['epochs']}

## Classification Report

```
{report}
```

## Confusion Matrix

| | Normal | Inner Race | Outer Race | Ball Fault |
|---|---|---|---|---|
"""

    for i, name in enumerate(CLASS_NAMES):
        row = f"| {name} | " + " | ".join(str(cm[i][j]) for j in range(4)) + " |"
        md += row + "\n"

    md += f"""
## Reproducibility
- Run `dvc repro` to reproduce these exact results.
- DVC lock file: `dvc.lock` (committed)
- MLflow experiment: `sensor_anomaly_detection`
"""

    # Write metrics JSON for DVC
    metrics = {
        "val_acc": round(report_dict["accuracy"], 4),
        "macro_precision": round(report_dict["macro avg"]["precision"], 4),
        "macro_recall": round(report_dict["macro avg"]["recall"], 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return md


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SensorClassifier(
        input_dim=4,  # matches preprocessing output
        hidden_dim=params["model"]["hidden_dim"],
        n_classes=params["model"]["n_classes"],
    )
    model.load_state_dict(torch.load("models/model.pth", map_location=device, weights_only=True))
    model.to(device)

    # Load test data
    test_ds = SensorDataset("data/processed/test_features.npy", "data/processed/test_labels.npy")

    # Evaluate
    y_pred, y_true = evaluate(model, test_ds, device)

    # Generate report
    report = generate_report(y_true, y_pred, params)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/evaluation_report.md", "w") as f:
        f.write(report)

    print("Evaluation report saved to reports/evaluation_report.md")
    print(f"Test accuracy: {(y_pred == y_true).mean():.4f}")


if __name__ == "__main__":
    main()