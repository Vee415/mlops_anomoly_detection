"""Generate synthetic bearing vibration data for pipeline development.

Produces 4-class sensor data mirroring CWRU Bearing Dataset structure:
  0 = Normal, 1 = Inner Race Fault, 2 = Outer Race Fault, 3 = Ball Fault
"""

import argparse
from pathlib import Path

import numpy as np


def generate_sample(
    label: int,
    length: int = 2048,
    sample_rate: int = 12000,
    noise_std: float = 0.3,
) -> np.ndarray:
    """Generate a single vibration signal for a given fault type.

    Each fault type has a characteristic frequency signature:
    - Normal: low-amplitude broadband noise
    - Inner Race: high-frequency periodic impulses
    - Outer Race: mid-frequency periodic impulses
    - Ball Fault: low-frequency periodic impulses
    """
    t = np.linspace(0, length / sample_rate, length)
    signal = np.random.randn(length) * noise_std

    if label == 0:  # Normal — mostly noise
        signal += 0.1 * np.sin(2 * np.pi * 30 * t)

    elif label == 1:  # Inner Race Fault — high-freq impulses
        f_bpid = 90.0  # ball pass frequency inner race
        signal += 0.8 * np.sin(2 * np.pi * f_bpid * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))

    elif label == 2:  # Outer Race Fault — mid-freq impulses
        f_bpfo = 60.0  # ball pass frequency outer race
        signal += 0.7 * np.sin(2 * np.pi * f_bpfo * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))

    elif label == 3:  # Ball Fault — low-freq impulses
        f_bsf = 35.0  # ball spin frequency
        signal += 0.6 * np.sin(2 * np.pi * f_bsf * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))

    return signal


def generate_dataset(
    n_samples_per_class: int = 200,
    signal_length: int = 2048,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate full synthetic dataset with labels.

    Returns:
        signals: shape (n_samples_per_class * 4, signal_length)
        labels: shape (n_samples_per_class * 4,)
    """
    rng = np.random.default_rng(seed)
    signals = []
    labels = []

    for label in range(4):
        for _ in range(n_samples_per_class):
            sample = generate_sample(label, length=signal_length)
            sample += rng.normal(0, 0.1, size=signal_length)  # per-sample jitter
            signals.append(sample)
            labels.append(label)

    signals = np.array(signals, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(labels))
    return signals[idx], labels[idx]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bearing vibration data")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per class")
    parser.add_argument("--signal-length", type=int, default=2048, help="Signal length per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    signals, labels = generate_dataset(
        n_samples_per_class=args.n_samples,
        signal_length=args.signal_length,
        seed=args.seed,
    )

    np.save(output_dir / "signals.npy", signals)
    np.save(output_dir / "labels.npy", labels)

    print(f"Generated {len(labels)} samples (4 classes x {args.n_samples})")
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()