"""Tests for Dinkelbach expected F-beta optimization method."""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.expected import dinkelbach_expected_fbeta_binary


class TestDinkelbachMethod:
    """Test Dinkelbach expected F-beta optimization."""

    def test_dinkelbach_basic_functionality(self):
        """Test that Dinkelbach method produces valid thresholds."""
        # Test probability distribution
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
        threshold = result.threshold

        # Should return a valid threshold
        assert isinstance(threshold, (float, np.number)) or (isinstance(threshold, np.ndarray) and threshold.size == 1)
        assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_through_get_optimal_threshold(self):
        """Test Dinkelbach method through the main API."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Should work for F1 metric and return a tuple
        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold = result.threshold
        assert hasattr(result, "threshold") and hasattr(result, "score")
        threshold_dinkelbach, f1_score_dinkelbach = result.threshold, result.score
        # Both should be valid thresholds
        assert 0.0 <= threshold_dinkelbach <= 1.0
        assert 0.0 <= f1_score_dinkelbach <= 1.0

        # Dinkelbach should produce reasonable results
        assert isinstance(threshold_dinkelbach, (float, np.number)) or (isinstance(threshold_dinkelbach, np.ndarray) and threshold_dinkelbach.size == 1)
        assert isinstance(f1_score_dinkelbach, (float, np.number))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
