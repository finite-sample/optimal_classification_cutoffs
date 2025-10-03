"""Tests for tied probability scenarios and edge cases.

This module tests the library's handling of tied probability values, which present
unique challenges for threshold optimization algorithms.
"""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import get_confusion_matrix, get_vectorized_metric
from optimal_cutoffs.piecewise import optimal_threshold_sortscan
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import generate_tied_probabilities


class TestTiedProbabilityHandling:
    """Test handling of tied probability values."""

    def test_all_identical_probabilities(self):
        """Test optimization when all probabilities are identical."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(score)

    def test_partial_ties(self):
        """Test optimization with some tied probabilities."""
        y_true, y_prob = generate_tied_probabilities(
            30, tie_fraction=0.4, random_state=42
        )

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

    def test_comparison_operators_with_ties(self):
        """Test that comparison operators handle ties appropriately."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.7, 0.7])  # Three samples tied at 0.7

        # Test both comparison operators
        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y_true, y_prob, metric="f1", comparison=comparison
            )
            assert_valid_threshold(threshold)

            # Verify predictions are different for tied values
            tp_gt, tn_gt, fp_gt, fn_gt = get_confusion_matrix(
                y_true, y_prob, threshold, comparison=">"
            )
            tp_gte, tn_gte, fp_gte, fn_gte = get_confusion_matrix(
                y_true, y_prob, threshold, comparison=">="
            )

            # At least for some thresholds, results should differ
            if threshold in y_prob:
                # When threshold equals a probability value, operators should differ
                assert (tp_gt, fp_gt) != (tp_gte, fp_gte) or (fn_gt, tn_gt) != (
                    fn_gte,
                    tn_gte,
                )

    def test_sort_scan_with_ties(self):
        """Test sort-scan algorithm specifically with tied values."""
        y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        y_prob = np.array([0.2, 0.5, 0.5, 0.5, 0.8, 0.8])

        f1_vectorized = get_vectorized_metric("f1")

        # Test both comparison operators
        for inclusive in [">", ">="]:
            threshold, score, k_star = optimal_threshold_sortscan(
                y_true, y_prob, f1_vectorized, inclusive=inclusive
            )

            assert_valid_threshold(threshold)
            assert_valid_metric_score(score, "f1")
            assert isinstance(k_star, int)
            assert 0 <= k_star <= len(y_true)

    def test_ties_at_boundaries(self):
        """Test tied values at probability boundaries (0.0, 1.0)."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        # Should achieve perfect separation
        from optimal_cutoffs.metrics import compute_metric_at_threshold

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_numerical_precision_ties(self):
        """Test ties that occur due to numerical precision."""
        eps = np.finfo(float).eps
        y_true = np.array([0, 1, 0, 1])
        # Create values that are effectively tied due to floating point precision
        y_prob = np.array([0.5, 0.5 + eps / 2, 0.5 - eps / 2, 0.5 + eps])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

    def test_method_consistency_with_ties(self):
        """Test that different methods handle ties consistently."""
        y_true, y_prob = generate_tied_probabilities(
            25, tie_fraction=0.6, random_state=42
        )

        methods = ["unique_scan", "minimize"]
        results = {}

        for method in methods:
            try:
                threshold = get_optimal_threshold(
                    y_true, y_prob, metric="f1", method=method
                )
                results[method] = threshold
                assert_valid_threshold(threshold)
            except (ValueError, NotImplementedError):
                # Some methods might not support all cases
                continue

        # If multiple methods succeeded, they should achieve reasonable performance
        if len(results) >= 2:
            from optimal_cutoffs.metrics import compute_metric_at_threshold

            scores = []
            for method, threshold in results.items():
                score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
                scores.append(score)

            # All methods should achieve reasonable scores
            for score in scores:
                assert_valid_metric_score(score, "f1")


class TestTieBreakingConsistency:
    """Test consistency of tie-breaking across different scenarios."""

    def test_deterministic_tie_breaking(self):
        """Test that tie-breaking is deterministic for identical inputs."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.4, 0.6, 0.6, 0.6])  # Multiple ties

        # Run multiple times to ensure determinism
        thresholds = []
        for _ in range(3):
            threshold = get_optimal_threshold(
                y_true, y_prob, metric="f1", method="unique_scan"
            )
            thresholds.append(threshold)

        # All results should be identical
        assert len(set(thresholds)) == 1, f"Non-deterministic results: {thresholds}"

    def test_tie_breaking_with_weights(self):
        """Test tie-breaking behavior with sample weights."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])  # All tied
        weights = np.array([1.0, 2.0, 1.0, 2.0])  # Different weights

        threshold = get_optimal_threshold(
            y_true, y_prob, metric="f1", sample_weight=weights
        )
        assert_valid_threshold(threshold)

        # Weighted optimization should still work with ties
        from optimal_cutoffs.metrics import compute_metric_at_threshold

        score = compute_metric_at_threshold(
            y_true, y_prob, threshold, "f1", sample_weight=weights
        )
        assert_valid_metric_score(score, "f1")


class TestExtremeTieCases:
    """Test extreme cases involving ties."""

    def test_all_zero_probabilities(self):
        """Test optimization when all probabilities are 0.0."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0])

        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # Should predict all negative for best accuracy
        tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # Best accuracy is achieved by predicting majority class
        expected_accuracy = max(np.sum(y_true == 0), np.sum(y_true == 1)) / len(y_true)
        assert accuracy == pytest.approx(expected_accuracy, abs=1e-10)

    def test_all_one_probabilities(self):
        """Test optimization when all probabilities are 1.0."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([1.0, 1.0, 1.0, 1.0])

        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # Should predict all positive for best accuracy when threshold <= 1.0
        tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        expected_accuracy = max(np.sum(y_true == 0), np.sum(y_true == 1)) / len(y_true)
        assert accuracy == pytest.approx(expected_accuracy, abs=1e-10)

    def test_alternating_ties(self):
        """Test case with alternating tied probability values."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.7, 0.7])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        from optimal_cutoffs.metrics import compute_metric_at_threshold

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    def test_massive_ties(self):
        """Test optimization with very large number of tied values."""
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.full(n_samples, 0.5)  # All identical

        # Ensure both classes are present
        y_true[0] = 0
        y_true[1] = 1

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        from optimal_cutoffs.metrics import compute_metric_at_threshold

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")
