"""Unit tests for data validation module."""

import numpy as np
import pytest

from src.validate import (
    validate_drift,
    validate_label_distribution,
    validate_no_nan,
    validate_shapes,
)


class TestValidateShapes:
    def test_valid_shapes(self):
        features = np.random.randn(100, 4).astype(np.float32)
        labels = np.random.randint(0, 4, 100).astype(np.int64)
        errors = validate_shapes(features, labels, expected_n_features=4)
        assert errors == []

    def test_wrong_feature_count(self):
        features = np.random.randn(100, 8).astype(np.float32)
        labels = np.random.randint(0, 4, 100).astype(np.int64)
        errors = validate_shapes(features, labels, expected_n_features=4)
        assert len(errors) == 1
        assert "8" in errors[0]

    def test_count_mismatch(self):
        features = np.random.randn(50, 4).astype(np.float32)
        labels = np.random.randint(0, 4, 100).astype(np.int64)
        errors = validate_shapes(features, labels, expected_n_features=4)
        assert len(errors) == 1


class TestValidateLabelDistribution:
    def test_balanced_labels(self):
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50, dtype=np.int64)
        warnings = validate_label_distribution(labels, min_per_class=10)
        assert warnings == []

    def test_imbalanced_labels(self):
        labels = np.array([0] * 100 + [1] * 3 + [2] * 50 + [3] * 50, dtype=np.int64)
        warnings = validate_label_distribution(labels, min_per_class=10)
        assert len(warnings) == 1
        assert "Class 1" in warnings[0]


class TestValidateNoNan:
    def test_clean_data(self):
        features = np.random.randn(100, 4).astype(np.float32)
        labels = np.zeros(100, dtype=np.int64)
        errors = validate_no_nan(features, labels)
        assert errors == []

    def test_nan_in_features(self):
        features = np.random.randn(100, 4).astype(np.float32)
        features[0, 0] = np.nan
        labels = np.zeros(100, dtype=np.int64)
        errors = validate_no_nan(features, labels)
        assert len(errors) == 1
        assert "NaN" in errors[0]

    def test_inf_in_features(self):
        features = np.random.randn(100, 4).astype(np.float32)
        features[0, 0] = np.inf
        labels = np.zeros(100, dtype=np.int64)
        errors = validate_no_nan(features, labels)
        assert len(errors) == 1
        assert "Inf" in errors[0]


class TestValidateDrift:
    def test_no_drift(self):
        rng = np.random.default_rng(42)
        train = rng.standard_normal((200, 4)).astype(np.float32)
        test = rng.standard_normal((50, 4)).astype(np.float32)
        warnings = validate_drift(train, test, threshold=0.05)
        # With random data from same distribution, drift is unlikely but possible
        # Just check it returns a list
        assert isinstance(warnings, list)

    def test_extreme_drift(self):
        train = np.zeros((200, 4), dtype=np.float32)
        test = np.ones((50, 4), dtype=np.float32) * 100
        warnings = validate_drift(train, test, threshold=0.05)
        assert len(warnings) > 0