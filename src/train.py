"""Training script with MLflow experiment tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import SensorClassifier


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]
    model_params = params["model"]
    preprocess_params = params["preprocess"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=train_params["batch_size"],
    )

    # Determine input_dim from data
    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[1]

    # Model
    model = SensorClassifier(
        input_dim=input_dim,
        hidden_dim=model_params["hidden_dim"],
        n_classes=model_params["n_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"])
    criterion = nn.CrossEntropyLoss()

    # MLflow tracking
    mlflow.set_experiment("sensor_anomaly_detection")

    with mlflow.start_run():
        mlflow.log_params({
            **train_params,
            **model_params,
            **preprocess_params,
            "input_dim": input_dim,
        })

        best_val_acc = 0.0
        patience_counter = 0
        patience = train_params["patience"]

        for epoch in range(train_params["epochs"]):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{train_params['epochs']}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                Path("models").mkdir(exist_ok=True)
                torch.save(model.state_dict(), "models/model.pth")
                mlflow.pytorch.log_model(model, "best_model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Write metrics.json for DVC
        Path("reports").mkdir(exist_ok=True)
        metrics = {
            "val_acc": round(best_val_acc, 4),
            "val_loss": round(val_loss, 4),
            "best_epoch": epoch - patience_counter + 1,
        }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
        print(f"Model saved to models/model.pth")


if __name__ == "__main__":
    main()