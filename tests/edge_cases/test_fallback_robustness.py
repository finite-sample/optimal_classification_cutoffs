"""Tests for fallback robustness in optimization algorithms.

This module tests that optimization methods properly fall back to alternative
approaches when primary methods fail or encounter edge cases.
"""

import warnings

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import compute_metric_at_threshold
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)


class TestFallbackTieHandling:
    """Test that fallback methods handle ties correctly."""

    def test_ties_with_inclusive_semantics(self):
        """Test ties with >= comparison are handled correctly."""
        # All samples have same probability - pure tie scenario
        y_true = np.array([1, 0, 1, 0])
        p_tied = np.array([0.5, 0.5, 0.5, 0.5])

        # Should not crash and return a valid threshold
        result = get_optimal_threshold(
            y_true, p_tied, metric="f1", method="sort_scan", comparison=">="
        )
        threshold = result.threshold
        assert_valid_threshold(threshold)

        # Test with main function too
        result_main = get_optimal_threshold(
            y_true, p_tied, metric="f1", method="sort_scan", comparison=">="
        )
        threshold_main = result_main.threshold
        assert_valid_threshold(threshold_main)

    def test_ties_with_exclusive_semantics(self):
        """Test ties with > comparison are handled correctly."""
        y_true = np.array([1, 0, 1, 0])
        p_tied = np.array([0.5, 0.5, 0.5, 0.5])

        result = get_optimal_threshold(
            y_true, p_tied, metric="f1", method="sort_scan", comparison=">"
        )
        threshold = result.threshold
        assert_valid_threshold(threshold)

        # Should get valid metric score
        score = compute_metric_at_threshold(
            y_true, p_tied, threshold, "f1", comparison=">"
        )
        assert_valid_metric_score(score, "f1")


class TestDegenerateCaseFallbacks:
    """Test fallback behavior for degenerate cases."""

    def test_all_same_class_fallback(self):
        """Test fallback when all samples have same class."""
        # All positive case
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.2, 0.5, 0.7, 0.9])

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

        # All negative case
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.3, 0.6, 0.8])

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        # F1 should be 0 for all-negative case
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_extreme_imbalance_fallback(self):
        """Test fallback with extreme class imbalance."""
        # 99:1 imbalance
        y_true = np.concatenate([np.zeros(99), np.ones(1)])
        y_prob = np.random.uniform(0, 1, 100)

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")


class TestNumericalStabilityFallbacks:
    """Test fallback behavior for numerical stability issues."""

    def test_near_zero_probability_fallback(self):
        """Test fallback with probabilities near zero."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([1e-15, 1e-14, 1e-13, 1e-12])

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_near_one_probability_fallback(self):
        """Test fallback with probabilities near one."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([1 - 1e-15, 1 - 1e-14, 1 - 1e-13, 1 - 1e-12])

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_machine_epsilon_differences_fallback(self):
        """Test fallback with machine epsilon level differences."""
        eps = np.finfo(float).eps
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5 + eps, 0.5 + 2 * eps, 0.5 + 3 * eps])

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")


class TestOptimizationMethodFallbacks:
    """Test fallbacks between different optimization methods."""

    def test_method_specific_fallbacks(self):
        """Test that methods fall back gracefully when they encounter issues."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        # Test that different methods all work
        methods = ["sort_scan", "unique_scan"]

        for method in methods:
            try:
                result = get_optimal_threshold(
                    y_true, y_prob, metric="f1", method=method
                )
                threshold = result.threshold
                score = result.score
                assert_valid_threshold(threshold)
                assert_valid_metric_score(score, "f1")
            except (ValueError, NotImplementedError):
                # Some methods might not be implemented
                continue

    def test_metric_specific_fallbacks(self):
        """Test fallbacks for metrics that might have edge cases."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        # Test different metrics that might require fallbacks
        metrics = ["f1", "precision", "recall", "accuracy"]

        for metric in metrics:
            result = get_optimal_threshold(y_true, y_prob, metric=metric)
            threshold = result.threshold
            score = result.score
            assert_valid_threshold(threshold)
            assert_valid_metric_score(score, metric, allow_nan=True)


class TestWarningSuppressionInFallbacks:
    """Test that appropriate warnings are handled in fallback scenarios."""

    def test_convergence_warning_handling(self):
        """Test that optimization convergence warnings are handled appropriately."""
        # Create a case that might trigger convergence warnings
        y_true = np.array([0, 1] * 50)  # Alternating pattern
        y_prob = np.random.uniform(0.4, 0.6, 100)  # Probabilities in narrow range

        # Should complete without unhandled warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            result = get_optimal_threshold(y_true, y_prob, metric="f1")
            threshold = result.threshold
            score = result.score

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_numerical_warning_handling(self):
        """Test handling of numerical precision warnings."""
        # Create case that might trigger numerical warnings
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([1e-100, 1 - 1e-100, 1e-100, 1 - 1e-100])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = get_optimal_threshold(y_true, y_prob, metric="f1")
            threshold = result.threshold
            score = result.score

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")


class TestRobustnessUnderAdversarialConditions:
    """Test robustness under adversarial conditions."""

    def test_pathological_probability_distributions(self):
        """Test robustness with pathological probability distributions."""
        # Extremely peaked distribution
        y_true = np.random.binomial(1, 0.5, 100)
        y_prob = np.random.beta(0.1, 0.1, 100)  # U-shaped distribution

        # Ensure both classes present
        y_true[0] = 0
        y_true[1] = 1

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_high_precision_requirements(self):
        """Test behavior when high precision is required."""
        # Create data requiring high precision threshold
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array(
            [0.4999999, 0.5000000, 0.5000001, 0.5000002, 0.5000003, 0.5000004]
        )

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_stress_testing_large_ties(self):
        """Stress test with large numbers of tied values."""
        n_samples = 1000
        y_true = np.random.binomial(1, 0.5, n_samples)

        # Create large blocks of tied values
        y_prob = np.zeros(n_samples)
        y_prob[: n_samples // 3] = 0.3
        y_prob[n_samples // 3 : 2 * n_samples // 3] = 0.5
        y_prob[2 * n_samples // 3 :] = 0.7

        # Ensure both classes present
        y_true[0] = 0
        y_true[1] = 1

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")
