"""Tests for Dinkelbach expected F-beta optimization method."""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.optimizers import _dinkelbach_expected_fbeta


class TestDinkelbachMethod:
    """Test Dinkelbach expected F-beta optimization."""

    def test_dinkelbach_basic_functionality(self):
        """Test that Dinkelbach method produces valid thresholds."""
        # Simple binary classification case
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        threshold = _dinkelbach_expected_fbeta(y_true, y_prob, beta=1.0)

        # Should return a valid threshold
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_through_get_optimal_threshold(self):
        """Test Dinkelbach method through the main API."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Should work for F1 metric and return a tuple
        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f1_score = result
        assert isinstance(threshold, float)
        assert isinstance(f1_score, float)
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f1_score <= 1.0

    def test_dinkelbach_supports_multiple_metrics(self):
        """Test that mode='expected' supports multiple metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        # Should work with different metrics
        for metric in ["f1", "accuracy"]:
            result = get_optimal_threshold(
                y_true, y_prob, metric=metric, mode="expected"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            threshold, score = result
            assert 0.0 <= threshold <= 1.0
            assert 0.0 <= score <= 1.0

    def test_dinkelbach_supports_sample_weights(self):
        """Test that mode='expected' supports sample weights."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        sample_weight = np.array([1.0, 2.0, 1.0, 1.0])

        # Should work with sample weights
        result = get_optimal_threshold(
            y_true,
            y_prob,
            metric="f1",
            mode="expected",
            sample_weight=sample_weight,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f1_score = result
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f1_score <= 1.0

    def test_dinkelbach_edge_cases(self):
        """Test Dinkelbach with edge cases."""
        # All negative labels
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.2, 0.5, 0.8])
        threshold = _dinkelbach_expected_fbeta(y_true, y_prob, beta=1.0)
        assert isinstance(threshold, float)

        # All positive labels
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.2, 0.5, 0.8])
        threshold = _dinkelbach_expected_fbeta(y_true, y_prob, beta=1.0)
        assert isinstance(threshold, float)

    def test_dinkelbach_tied_probabilities(self):
        """Test Dinkelbach with tied probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.5, 0.5, 0.7])  # Two samples with prob 0.5

        threshold = _dinkelbach_expected_fbeta(y_true, y_prob, beta=1.0)
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_different_beta_values(self):
        """Test Dinkelbach with different beta values."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Test different beta values
        for beta in [0.5, 1.0, 2.0]:
            threshold = _dinkelbach_expected_fbeta(y_true, y_prob, beta=beta)
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_consistency(self):
        """Test that Dinkelbach is consistent for the same input."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.2, 0.4, 0.5, 0.6, 0.8])

        # Multiple calls should return same result
        threshold1 = _dinkelbach_expected_fbeta(y_true, y_prob, beta=1.0)
        threshold2 = _dinkelbach_expected_fbeta(y_true, y_prob, beta=1.0)

        assert threshold1 == threshold2

    def test_dinkelbach_vs_other_methods(self):
        """Compare Dinkelbach results with other methods."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 50, p=[0.6, 0.4])
        y_prob = np.random.beta(2, 2, 50)

        # Get thresholds from different methods
        result_dinkelbach = get_optimal_threshold(
            y_true, y_prob, mode="expected", metric="f1"
        )
        threshold_brute = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="unique_scan"
        )

        # Expected mode returns tuple, extract threshold
        assert isinstance(result_dinkelbach, tuple)
        threshold_dinkelbach, f1_score_dinkelbach = result_dinkelbach

        # Both should be valid thresholds
        assert 0.0 <= threshold_dinkelbach <= 1.0
        assert 0.0 <= threshold_brute <= 1.0
        assert 0.0 <= f1_score_dinkelbach <= 1.0

        # Dinkelbach might differ from brute force (it optimizes expected F-beta)
        # but both should be reasonable
        assert isinstance(threshold_dinkelbach, float)
        assert isinstance(threshold_brute, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
