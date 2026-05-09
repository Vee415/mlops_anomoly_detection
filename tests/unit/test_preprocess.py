"""Unit tests for preprocessing module."""

import numpy as np
import pytest

from src.preprocess import extract_features, normalize_features, create_dataset


class TestExtractFeatures:
    def test_output_shape(self, sample_signal):
        """Features should have shape (n_windows, 4)."""
        features = extract_features(sample_signal, window_size=128, stride=64)
        assert features.ndim == 2
        assert features.shape[1] == 4

    def test_window_count(self):
        """Sliding window should produce correct number of windows."""
        signal = np.random.randn(512).astype(np.float32)
        features = extract_features(signal, window_size=128, stride=64)
        expected = (512 - 128) // 64 + 1  # inclusive of last aligned window
        assert features.shape[0] == expected

    def test_feature_values(self):
        """First feature (mean) should equal window mean."""
        signal = np.ones(256).astype(np.float32) * 3.0
        features = extract_features(signal, window_size=128, stride=64)
        # All windows of constant signal should have mean = 3.0
        np.testing.assert_allclose(features[:, 0], 3.0, atol=1e-5)


class TestNormalizeFeatures:
    def test_zero_mean_unit_var(self, sample_features):
        """Normalized features should have approximately zero mean and unit variance."""
        features, _ = sample_features
        normalized, mean, std = normalize_features(features)
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=0.1)

    def test_given_stats(self, sample_features):
        """Normalization with given stats should apply them correctly."""
        features, _ = sample_features
        mean = np.array([1.0, 2.0, 3.0, 4.0])
        std = np.array([0.5, 0.5, 0.5, 0.5])
        normalized, _, _ = normalize_features(features, mean, std)
        expected = (features - mean) / std
        np.testing.assert_allclose(normalized, expected)


class TestCreateDataset:
    def test_output_shapes(self, sample_signals):
        """create_dataset should produce valid feature and label arrays."""
        signals, labels = sample_signals
        features, feature_labels = create_dataset(signals, labels, window_size=128, stride=64)
        assert features.ndim == 2
        assert features.shape[1] == 4
        assert len(features) == len(feature_labels)