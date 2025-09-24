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

        # Should work for F1 metric
        threshold = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="dinkelbach"
        )
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_unsupported_metric(self):
        """Test that Dinkelbach raises error for unsupported metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        with pytest.raises(
            ValueError, match="dinkelbach method currently only supports F1 metric"
        ):
            get_optimal_threshold(
                y_true, y_prob, metric="accuracy", method="dinkelbach"
            )

    def test_dinkelbach_no_sample_weights(self):
        """Test that Dinkelbach raises error when sample weights provided."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        sample_weight = np.array([1.0, 2.0, 1.0, 1.0])

        with pytest.raises(
            ValueError, match="dinkelbach method does not support sample weights"
        ):
            get_optimal_threshold(
                y_true,
                y_prob,
                metric="f1",
                method="dinkelbach",
                sample_weight=sample_weight,
            )

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
        threshold_dinkelbach = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="dinkelbach"
        )
        threshold_brute = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="smart_brute"
        )

        # Both should be valid thresholds
        assert 0.0 <= threshold_dinkelbach <= 1.0
        assert 0.0 <= threshold_brute <= 1.0

        # Dinkelbach might differ from brute force (it optimizes expected F-beta)
        # but both should be reasonable
        assert isinstance(threshold_dinkelbach, float)
        assert isinstance(threshold_brute, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
