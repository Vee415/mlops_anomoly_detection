"""Single-sample inference script."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.model import SensorClassifier
from src.preprocess import extract_features, normalize_features

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]


def predict(signal: np.ndarray, model_path: str, data_dir: str, params_path: str) -> dict:
    """Predict fault type for a single vibration signal.

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

    # Extract features
    features = extract_features(
        signal,
        window_size=params["preprocess"]["window_size"],
        stride=params["preprocess"]["stride"],
    )

    # Aggregate window features (mean across windows)
    feature_vector = features.mean(axis=0, keepdims=True)  # shape: (1, 4)

    # Normalize using training stats
    mean = np.load(f"{data_dir}/norm_mean.npy")
    std = np.load(f"{data_dir}/norm_std.npy")
    feature_vector, _, _ = normalize_features(feature_vector, mean, std)

    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorClassifier(
        input_dim=feature_vector.shape[1],
        hidden_dim=params["model"]["hidden_dim"],
        n_classes=params["model"]["n_classes"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(feature_vector).float().to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(probs.argmax())
    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "class_index": pred_class,
        "probabilities": {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)},
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