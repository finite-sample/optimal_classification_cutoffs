"""Tests for tied probability scenarios and edge cases.

This module tests the library's handling of tied probability values, which present
unique challenges for threshold optimization algorithms. Tied probabilities occur
when multiple samples have identical predicted probabilities, requiring careful
handling to ensure consistent and optimal threshold selection.

Key test categories:
- All identical probabilities (extreme tie case)
- Partially tied probabilities (some ties)
- Comparison operator effects with ties (">" vs ">=")
- Sort-scan algorithm with tied values
- Sample weight handling with ties
- Numerical precision near ties
- Cross-method consistency with tied data

The tests ensure that:
1. Tied probabilities don't break optimization algorithms
2. Comparison operators are handled consistently
3. Different optimization methods produce reasonable results
4. Edge cases like all-zero or all-one probabilities work correctly
"""

import numpy as np
import pytest

from optimal_cutoffs import ThresholdOptimizer, get_optimal_threshold
from optimal_cutoffs.metrics import get_confusion_matrix
from optimal_cutoffs.piecewise import get_vectorized_metric, optimal_threshold_sortscan


class TestTiedProbabilities:
    """Test handling of tied probability values."""

    def test_all_identical_probabilities(self):
        """Test optimization when all probabilities are identical."""
        # All probabilities are 0.5
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [0.5, 0.5, 0.5, 0.5, 0.5]

        # Should still return a valid threshold
        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

        # Test with different comparison operators
        threshold_gt = get_optimal_threshold(
            y_true, pred_prob, metric="f1", comparison=">"
        )
        threshold_gte = get_optimal_threshold(
            y_true, pred_prob, metric="f1", comparison=">="
        )

        # Thresholds should be different due to tied handling
        assert threshold_gt != threshold_gte

    def test_partially_tied_probabilities(self):
        """Test optimization with some tied probabilities."""
        y_true = [0, 0, 1, 1, 1, 0]
        pred_prob = [0.2, 0.5, 0.5, 0.5, 0.8, 0.2]  # Ties at 0.2 and 0.5

        for method in ["smart_brute", "sort_scan", "minimize"]:
            if method == "sort_scan" and not _has_vectorized_metric("f1"):
                continue

            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method=method
            )
            assert 0.0 <= threshold <= 1.0

            # Verify the threshold produces a valid confusion matrix
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == len(y_true)

    def test_tied_probabilities_comparison_operators(self):
        """Test that tied probabilities are handled correctly with > vs >=."""
        y_true = [0, 1, 1, 0]
        pred_prob = [0.4, 0.6, 0.6, 0.4]  # Two tied pairs

        # Test both comparison operators
        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", comparison=comparison
            )

            # Apply threshold with the specified comparison
            if comparison == ">":
                predictions = np.array(pred_prob) > threshold
            else:
                predictions = np.array(pred_prob) >= threshold

            # Should produce a valid prediction array
            assert predictions.dtype == bool
            assert len(predictions) == len(y_true)

    def test_sort_scan_with_ties(self):
        """Test sort-scan algorithm specifically with tied probabilities."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        pred_prob = np.array([0.3, 0.7, 0.3, 0.7, 0.7, 0.3])

        # Test with vectorized F1
        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert 0.0 <= threshold <= 1.0
        assert score >= 0.0
        assert 0 <= k_star <= len(y_true)

        # Test both comparison operators
        for inclusive in [">", ">="]:
            threshold, score, k_star = optimal_threshold_sortscan(
                y_true, pred_prob, f1_vectorized, inclusive=inclusive
            )
            assert 0.0 <= threshold <= 1.0

    def test_extreme_tied_cases(self):
        """Test extreme cases of tied probabilities."""
        # Case 1: All probabilities are 0
        y_true = [0, 1, 0, 1]
        pred_prob = [0.0, 0.0, 0.0, 0.0]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="accuracy")
        assert 0.0 <= threshold <= 1.0

        # Case 2: All probabilities are 1
        pred_prob = [1.0, 1.0, 1.0, 1.0]
        threshold = get_optimal_threshold(y_true, pred_prob, metric="accuracy")
        assert 0.0 <= threshold <= 1.0

        # Case 3: Mix of 0 and 1 only
        pred_prob = [0.0, 1.0, 0.0, 1.0]
        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

    def test_tied_probabilities_with_sample_weights(self):
        """Test tied probability handling with sample weights."""
        y_true = [0, 1, 1, 0]
        pred_prob = [0.5, 0.5, 0.5, 0.5]  # All tied
        sample_weight = [1.0, 2.0, 1.5, 0.5]

        threshold = get_optimal_threshold(
            y_true, pred_prob, metric="f1", sample_weight=sample_weight
        )
        assert 0.0 <= threshold <= 1.0

        # Test with sort_scan method if available
        if _has_vectorized_metric("f1"):
            threshold_sort = get_optimal_threshold(
                y_true,
                pred_prob,
                metric="f1",
                method="sort_scan",
                sample_weight=sample_weight,
            )
            assert 0.0 <= threshold_sort <= 1.0

    def test_threshold_optimizer_with_ties(self):
        """Test ThresholdOptimizer class with tied probabilities."""
        y_true = [0, 1, 0, 1, 1, 0]
        pred_prob = [0.4, 0.6, 0.4, 0.6, 0.6, 0.4]

        optimizer = ThresholdOptimizer(objective="f1", method="smart_brute")
        optimizer.fit(y_true, pred_prob)

        predictions = optimizer.predict(pred_prob)
        assert len(predictions) == len(y_true)
        assert predictions.dtype == bool

    def test_multiclass_with_tied_probabilities(self):
        """Test multiclass scenarios with tied probabilities."""
        y_true = [0, 1, 2, 0, 1, 2]
        pred_prob = np.array(
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.5, 0.2],
                [0.2, 0.3, 0.5],
                [0.5, 0.3, 0.2],  # Duplicate of first row
                [0.3, 0.5, 0.2],  # Duplicate of second row
                [0.2, 0.3, 0.5],  # Duplicate of third row
            ]
        )

        thresholds = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert len(thresholds) == 3
        assert all(0.0 <= t <= 1.0 for t in thresholds)


class TestNumericalEdgeCases:
    """Test numerical edge cases that could cause instability."""

    def test_probabilities_near_machine_epsilon(self):
        """Test with probabilities very close to 0 or 1."""
        eps = np.finfo(float).eps

        y_true = [0, 1, 0, 1]
        pred_prob = [eps, 1 - eps, eps, 1 - eps]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

    def test_very_close_probabilities(self):
        """Test with probabilities that differ by tiny amounts."""
        eps = 1e-15
        base = 0.5

        y_true = [0, 1, 0, 1]
        pred_prob = [base, base + eps, base - eps, base + 2 * eps]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

    def test_threshold_boundary_conditions(self):
        """Test that thresholds respect [0,1] bounds."""
        y_true = [0, 1, 0, 1]

        # Case 1: Very high probabilities
        pred_prob = [0.999999, 0.999998, 0.999997, 0.999996]
        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

        # Case 2: Very low probabilities
        pred_prob = [0.000001, 0.000002, 0.000003, 0.000004]
        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0


def _has_vectorized_metric(metric_name: str) -> bool:
    """Check if a metric has vectorized implementation."""
    try:
        get_vectorized_metric(metric_name)
        return True
    except ValueError:
        return False


class TestConsistencyAcrossMethods:
    """Test that different methods handle tied probabilities consistently."""

    @pytest.mark.parametrize("metric", ["f1", "accuracy", "precision", "recall"])
    def test_method_consistency_with_ties(self, metric):
        """Test that different optimization methods give reasonable results with ties."""
        y_true = [0, 1, 0, 1, 1, 0, 1, 0]
        pred_prob = [0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7, 0.3]  # Many ties

        methods = ["smart_brute", "minimize", "gradient"]
        if _has_vectorized_metric(metric):
            methods.append("sort_scan")

        thresholds = {}
        scores = {}

        for method in methods:
            thresholds[method] = get_optimal_threshold(
                y_true, pred_prob, metric=metric, method=method
            )

            # Compute actual score achieved
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, thresholds[method])

            if metric == "f1":
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                scores[method] = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else 0.0
                )
            elif metric == "accuracy":
                scores[method] = (tp + tn) / (tp + tn + fp + fn)
            elif metric == "precision":
                scores[method] = tp / (tp + fp) if tp + fp > 0 else 0.0
            elif metric == "recall":
                scores[method] = tp / (tp + fn) if tp + fn > 0 else 0.0

        # All methods should produce valid thresholds
        for method, threshold in thresholds.items():
            assert 0.0 <= threshold <= 1.0, (
                f"Invalid threshold for {method}: {threshold}"
            )

        # Scores should be non-negative and finite
        for method, score in scores.items():
            assert score >= 0.0, f"Negative score for {method}: {score}"
            assert np.isfinite(score), f"Non-finite score for {method}: {score}"
