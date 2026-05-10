"""Single-sample inference script."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.model import get_model
from src.preprocess import extract_features, extract_raw_windows, normalize_features

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]


def predict(signal: np.ndarray, model_path: str, data_dir: str, params_path: str) -> dict:
    """Predict fault type for a single vibration signal.

    Predicts per-window probabilities and aggregates via mean to match
    how training uses individual windows as separate samples.

    Args:
        signal: 1D numpy array of raw vibration data
        model_path: Path to saved model weights
        data_dir: Directory containing normalization stats
        params_path: Path to params.yaml

    Returns:
        Dict with predicted class name, class index, and probabilities.
    """
    with open(params_path) as f:
        params = yaml.safe_load(f)

    arch = params["model"].get("arch", "fc")
    model_params = params["model"]
    preprocess_params = params["preprocess"]
    window_size = preprocess_params["window_size"]
    stride = preprocess_params["stride"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == "cnn":
        # CNN: use raw signal windows directly (no feature normalization needed)
        windows = extract_raw_windows(signal, window_size=window_size, stride=stride)
        x = torch.from_numpy(windows).float().unsqueeze(1).to(device)  # (N, 1, W)

        cnn_params = model_params.get("cnn", {})
        model = get_model(
            arch,
            window_size=window_size,
            n_classes=model_params["n_classes"],
            channels=tuple(cnn_params.get("channels", [32, 64])),
            kernel_size=cnn_params.get("kernel_size", 7),
            pool_size=cnn_params.get("pool_size", 2),
            fc_hidden=cnn_params.get("fc_hidden", 128),
            dropout=cnn_params.get("dropout", 0.3),
        )
    else:
        # FC: use extracted features per window
        features = extract_features(signal, window_size=window_size, stride=stride)
        mean = np.load(f"{data_dir}/norm_mean.npy")
        std = np.load(f"{data_dir}/norm_std.npy")
        features_norm, _, _ = normalize_features(features, mean, std)
        x = torch.from_numpy(features_norm).float().to(device)  # (N, 4)

        # Load preprocess info to infer input_dim
        import json
        with open(f"{data_dir}/preprocess_info.json") as f:
            preprocess_info = json.load(f)
        input_dim = preprocess_info["n_features"]

        model = get_model(
            arch,
            input_dim=input_dim,
            hidden_dim=model_params["hidden_dim"],
            n_classes=model_params["n_classes"],
        )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Predict per-window and aggregate probabilities (matches training)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Average probabilities across all windows
    mean_probs = probs.mean(axis=0)
    pred_class = int(mean_probs.argmax())
    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "class_index": pred_class,
        "probabilities": {
            name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, mean_probs)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict fault type for a vibration signal")
    parser.add_argument("--signal-file", help="Path to .npy file with vibration signal")
    parser.add_argument("--model-path", default="models/model.pth")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    signal = np.load(args.signal_file)
    result = predict(signal, args.model_path, args.data_dir, args.params)

    print(f"Predicted: {result['predicted_class']} (class {result['class_index']})")
    print("Probabilities:")
    for name, prob in result["probabilities"].items():
        print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
