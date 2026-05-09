"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_signal():
    """Single 1D vibration signal for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(2048).astype(np.float32)


@pytest.fixture
def sample_signals():
    """Batch of signals with labels for testing."""
    rng = np.random.default_rng(42)
    n = 20
    signals = rng.standard_normal((n, 2048)).astype(np.float32)
    labels = rng.integers(0, 4, size=n).astype(np.int64)
    return signals, labels


@pytest.fixture
def sample_features():
    """Small feature array for testing."""
    rng = np.random.default_rng(42)
    n = 100
    features = rng.standard_normal((n, 4)).astype(np.float32)
    labels = rng.integers(0, 4, size=n).astype(np.int64)
    return features, labels


@pytest.fixture
def sample_windows():
    """Small raw window array for CNN testing."""
    rng = np.random.default_rng(42)
    n = 50
    window_size = 128
    windows = rng.standard_normal((n, window_size)).astype(np.float32)
    labels = rng.integers(0, 4, size=n).astype(np.int64)
    return windows, labels