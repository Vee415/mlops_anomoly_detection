"""Optuna-based hyperparameter tuning with MLflow tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import optuna
import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import get_model


def objective(trial: optuna.Trial, params: dict, data_dir: str = "data/processed") -> float:
    """Optuna objective: train model with trial params, return best val accuracy."""
    train_params = params["train"]
    model_params = params["model"]
    preprocess_params = params["preprocess"]

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    arch = model_params.get("arch", "fc")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        model_type=arch,
    )

    # Model
    if arch == "cnn":
        cnn_params = model_params.get("cnn", {})
        model = get_model(
            arch,
            window_size=preprocess_params["window_size"],
            n_classes=model_params["n_classes"],
            channels=tuple(cnn_params.get("channels", [32, 64])),
            kernel_size=cnn_params.get("kernel_size", 7),
            pool_size=cnn_params.get("pool_size", 2),
            fc_hidden=cnn_params.get("fc_hidden", 128),
            dropout=cnn_params.get("dropout", 0.3),
        )
    else:
        sample_features, _ = next(iter(train_loader))
        input_dim = sample_features.shape[1]
        model = get_model(
            arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_classes=model_params["n_classes"],
        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = train_params["patience"]
    patience_counter = 0

    for epoch in range(train_params["epochs"]):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--apply", action="store_true", help="Apply best params to params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, params), n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  Value (val_acc): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Save results
    results = {
        "best_val_acc": round(study.best_value, 4),
        "best_params": {
            k: (v if not isinstance(v, float) or not v.is_integer() else int(v))
            for k, v in study.best_params.items()
        },
        "n_trials": args.n_trials,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if args.apply:
        # Update params.yaml with best params
        if "lr" in study.best_params:
            params["train"]["lr"] = study.best_params["lr"]
        if "hidden_dim" in study.best_params:
            params["model"]["hidden_dim"] = study.best_params["hidden_dim"]
        if "batch_size" in study.best_params:
            params["train"]["batch_size"] = study.best_params["batch_size"]

        with open(args.params, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        print(f"\nApplied best params to {args.params}")


if __name__ == "__main__":
    main()
