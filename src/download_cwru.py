"""Download and prepare CWRU Bearing Dataset for the pipeline.

Downloads .mat files from the Case Western Reserve University Bearing
Data Center and converts them to the signals.npy/labels.npy format
used by the rest of the pipeline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# CWRU data URLs for 4 conditions at 0 HP load, drive end
# Normal: 97 mat file, Inner Race (0.007): 105, Outer Race (0.007): 130, Ball (0.007): 118
# Using 12kHz sample rate, drive end data
CWRU_URLS = {
    0: "https://engineering.case.edu/sites/default/files/97.mat",   # Normal baseline
    1: "https://engineering.case.edu/sites/default/files/105.mat",   # Inner Race Fault 0.007"
    2: "https://engineering.case.edu/sites/default/files/130.mat",   # Outer Race Fault 0.007"
    3: "https://engineering.case.edu/sites/default/files/118.mat",   # Ball Fault 0.007"
}

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]

# Field names in .mat files (drive end accelerometer data)
CWRU_FIELDS = {
    0: "X097_DE_time",    # Normal
    1: "X105_DE_time",   # Inner Race
    2: "X130_DE_time",   # Outer Race
    3: "X118_DE_time",   # Ball Fault
}


def download_and_load(class_id: int, target_length: int = 2048) -> list[np.ndarray]:
    """Download a single CWRU .mat file and extract signal windows.

    Args:
        class_id: Class index (0-3).
        target_length: Target signal window length.

    Returns:
        List of 1D numpy arrays, each of length target_length.
    """
    from urllib.request import urlretrieve

    from scipy.io import loadmat

    url = CWRU_URLS[class_id]
    field = CWRU_FIELDS[class_id]
    cache_dir = Path("data/cwru_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    mat_path = cache_dir / f"{class_id}.mat"

    if not mat_path.exists():
        print(f"  Downloading {CLASS_NAMES[class_id]} data from {url}...")
        urlretrieve(url, mat_path)
    else:
        print(f"  Using cached {CLASS_NAMES[class_id]} data")

    mat = loadmat(str(mat_path))
    signal = mat[field].flatten().astype(np.float32)

    # Split into windows of target_length
    n_windows = len(signal) // target_length
    windows = [
        signal[i * target_length : (i + 1) * target_length]
        for i in range(n_windows)
    ]
    return windows


def main():
    parser = argparse.ArgumentParser(description="Download and prepare CWRU Bearing Dataset")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--params", default="params.yaml", help="Params file")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    target_length = params["generate"]["signal_length"]
    n_samples = params["generate"]["n_samples"]

    print("Downloading CWRU Bearing Dataset...")
    all_signals = []
    all_labels = []

    for class_id in range(4):
        windows = download_and_load(class_id, target_length)
        # Sample n_samples windows per class
        rng = np.random.default_rng(params["generate"]["seed"])
        indices = rng.choice(len(windows), size=min(n_samples, len(windows)), replace=False)
        for idx in indices:
            all_signals.append(windows[idx])
            all_labels.append(class_id)

    signals = np.array(all_signals, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    rng = np.random.default_rng(params["generate"]["seed"])
    shuffle_idx = rng.permutation(len(labels))
    signals = signals[shuffle_idx]
    labels = labels[shuffle_idx]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "signals.npy", signals)
    np.save(output_dir / "labels.npy", labels)

    print(f"Saved {len(signals)} samples to {output_dir}/")
    print(f"  Signal shape: {signals.shape}")
    print(f"  Class distribution: {np.bincount(labels)}")


if __name__ == "__main__":
    main()
