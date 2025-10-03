"""Unit tests for core optimization functions.

This module tests the individual optimization algorithms and their
basic functionality without complex integration scenarios.
"""

import time

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import compute_metric_at_threshold, is_piecewise_metric
from optimal_cutoffs.optimize import find_optimal_threshold
from tests.fixtures.assertions import (
    assert_method_consistency,
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_extreme_probabilities,
    generate_tied_probabilities,
)


class TestBasicOptimization:
    """Test basic optimization functionality."""

    @pytest.mark.parametrize("method", ["unique_scan", "minimize", "gradient"])
    def test_optimization_methods_basic(self, method):
        """Test that all optimization methods work and return valid results."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

        threshold = get_optimal_threshold(y_true, y_prob, method=method, metric="f1")

        assert_valid_threshold(threshold)

        # Check that the method achieves reasonable performance
        f1_score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(f1_score, "f1")
        assert f1_score >= 0.8, f"Method {method} achieved F1={f1_score:.6f}"

    @pytest.mark.parametrize("metric", ["f1", "accuracy", "precision", "recall"])
    def test_optimization_different_metrics(self, metric):
        """Test optimization with different metrics."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        threshold = get_optimal_threshold(y_true, y_prob, metric=metric)

        assert_valid_threshold(threshold)

        # Verify the threshold optimizes the requested metric
        score = compute_metric_at_threshold(y_true, y_prob, threshold, metric)
        assert_valid_metric_score(score, metric)

    def test_method_auto_selection(self):
        """Test automatic method selection."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        # Auto method should work without errors
        threshold_auto = get_optimal_threshold(y_true, y_prob, method="auto")
        threshold_default = get_optimal_threshold(y_true, y_prob)  # Default is auto

        assert_valid_threshold(threshold_auto)
        assert_valid_threshold(threshold_default)
        assert threshold_auto == threshold_default


class TestOptimizationEdgeCases:
    """Test optimization with edge cases."""

    def test_single_sample_optimization(self):
        """Test optimization with single sample."""
        for method in ["unique_scan", "minimize"]:
            threshold = get_optimal_threshold([1], [0.7], method=method, metric="f1")
            assert_valid_threshold(threshold)

    def test_all_same_class_optimization(self):
        """Test optimization when all samples are same class."""
        # All negative - should predict all negative
        threshold_neg = get_optimal_threshold(
            [0, 0, 0], [0.1, 0.5, 0.9], metric="accuracy"
        )
        assert_valid_threshold(threshold_neg)

        # All positive - should predict all positive
        threshold_pos = get_optimal_threshold(
            [1, 1, 1], [0.1, 0.5, 0.9], metric="accuracy"
        )
        assert_valid_threshold(threshold_pos)

    def test_all_same_probabilities(self):
        """Test optimization when all probabilities are identical."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.5, 0.5, 0.5, 0.5]

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

    def test_extreme_probabilities(self):
        """Test optimization with extreme probability values."""
        y_true, y_prob = generate_extreme_probabilities(20, random_state=42)

        for method in ["unique_scan", "minimize"]:
            threshold = get_optimal_threshold(y_true, y_prob, method=method, metric="f1")
            assert_valid_threshold(threshold)

    def test_tied_probabilities(self):
        """Test optimization with many tied probability values."""
        y_true, y_prob = generate_tied_probabilities(50, random_state=42)

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)


class TestPiecewiseOptimization:
    """Test piecewise (sort-scan) optimization specifically."""

    def test_piecewise_vs_brute_force_consistency(self):
        """Test that piecewise optimization matches brute force for piecewise metrics."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        for metric in ["f1", "accuracy", "precision", "recall"]:
            if is_piecewise_metric(metric):
                # Get result from piecewise optimization
                threshold_piecewise, _ = find_optimal_threshold(
                    y_true, y_prob, metric, strategy="sort_scan"
                )

                # Get result from unique_scan (exhaustive on unique values)
                threshold_unique = get_optimal_threshold(
                    y_true, y_prob, metric, method="unique_scan"
                )

                assert_valid_threshold(threshold_piecewise)
                assert_valid_threshold(threshold_unique)

                # They should achieve the same optimal score
                score_piecewise = compute_metric_at_threshold(
                    y_true, y_prob, threshold_piecewise, metric
                )
                score_unique = compute_metric_at_threshold(
                    y_true, y_prob, threshold_unique, metric
                )

                assert abs(score_piecewise - score_unique) < 1e-10, (
                    f"Score mismatch for {metric}: piecewise={score_piecewise:.10f}, "
                    f"unique_scan={score_unique:.10f}"
                )

    def test_piecewise_known_optimal_case(self):
        """Test piecewise optimization on case with known optimal solution."""
        # Construct case where optimal threshold is exactly between two values
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.7])

        # For F1, optimal threshold should be around 0.5 (between 0.4 and 0.6)
        threshold, score = find_optimal_threshold(y_true, y_prob, "f1", strategy="sort_scan")

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

        # Should achieve perfect F1 score
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_piecewise_performance(self):
        """Test that piecewise optimization is reasonably fast."""
        # Large dataset to test performance
        y_true, y_prob = generate_binary_data(1000, random_state=42)

        start_time = time.time()
        threshold, score = find_optimal_threshold(y_true, y_prob, "f1", strategy="sort_scan")
        end_time = time.time()

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")
        assert end_time - start_time < 1.0  # Should complete in reasonable time


class TestMethodConsistency:
    """Test consistency between different optimization methods."""

    def test_methods_on_separable_data(self):
        """Test that methods produce consistent results on separable data."""
        # Perfect separation case
        y_true = [0, 0, 0, 1, 1, 1]
        pred_prob = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

        methods = ["unique_scan", "minimize", "gradient"]
        results = {}

        for method in methods:
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method=method
            )
            score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
            results[method] = (threshold, score)

        # All methods should achieve high performance on separable data
        for method, (threshold, score) in results.items():
            assert_valid_threshold(threshold)
            assert score >= 0.9, f"Method {method} achieved low F1 score {score}"

    def test_vectorized_vs_scalar_methods(self):
        """Test that vectorized and scalar methods produce consistent results."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        # Compare sort_scan (vectorized) with unique_scan for piecewise metrics
        metrics_to_test = ["f1", "accuracy"]

        for metric in metrics_to_test:
            if is_piecewise_metric(metric):
                threshold_vector = get_optimal_threshold(
                    y_true, y_prob, metric=metric, method="sort_scan"
                )
                threshold_scalar = get_optimal_threshold(
                    y_true, y_prob, metric=metric, method="unique_scan"
                )

                assert_method_consistency(
                    threshold_vector, threshold_scalar,
                    "sort_scan", "unique_scan", tolerance=1e-8
                )


class TestOptimizationParameters:
    """Test optimization with different parameters."""

    @pytest.mark.parametrize("comparison", [">", ">="])
    def test_comparison_operators(self, comparison):
        """Test optimization with different comparison operators."""
        y_true, y_prob = generate_tied_probabilities(30, random_state=42)

        threshold = get_optimal_threshold(
            y_true, y_prob, metric="f1", comparison=comparison
        )

        assert_valid_threshold(threshold)

    def test_sample_weights_optimization(self):
        """Test optimization with sample weights."""
        y_true, y_prob = generate_binary_data(50, random_state=42)
        weights = np.random.uniform(0.5, 2.0, len(y_true))

        # Test different methods with weights
        for method in ["unique_scan", "minimize"]:
            threshold = get_optimal_threshold(
                y_true, y_prob, metric="f1", method=method, sample_weight=weights
            )
            assert_valid_threshold(threshold)

    def test_optimization_with_mode_parameter(self):
        """Test optimization with different modes."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        # Test empirical mode (default)
        threshold_emp = get_optimal_threshold(y_true, y_prob, mode="empirical")
        assert_valid_threshold(threshold_emp)

        # Test expected mode (returns tuple)
        result_exp = get_optimal_threshold(y_true, y_prob, mode="expected")
        assert isinstance(result_exp, tuple)
        threshold_exp, expected_score = result_exp
        assert_valid_threshold(threshold_exp)
        assert_valid_metric_score(expected_score, "expected_f1")


class TestOptimizationErrors:
    """Test error handling in optimization functions."""

    def test_empty_arrays_error(self):
        """Test that empty arrays raise appropriate errors."""
        with pytest.raises(ValueError):
            get_optimal_threshold([], [], metric="f1")

    def test_mismatched_lengths_error(self):
        """Test that mismatched array lengths raise errors."""
        with pytest.raises(ValueError):
            get_optimal_threshold([0, 1], [0.1], metric="f1")

    def test_invalid_metric_error(self):
        """Test that invalid metrics raise errors."""
        y_true, y_prob = generate_binary_data(20, random_state=42)

        with pytest.raises(ValueError, match="Unknown metric"):
            get_optimal_threshold(y_true, y_prob, metric="invalid_metric")

    def test_invalid_method_error(self):
        """Test that invalid methods raise errors."""
        y_true, y_prob = generate_binary_data(20, random_state=42)

        with pytest.raises(ValueError, match="Invalid optimization method"):
            get_optimal_threshold(y_true, y_prob, method="invalid_method")

    def test_invalid_comparison_error(self):
        """Test that invalid comparison operators raise errors."""
        y_true, y_prob = generate_binary_data(20, random_state=42)

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_optimal_threshold(y_true, y_prob, comparison="<")


class TestOptimizationPerformance:
    """Test performance characteristics of optimization methods."""

    def test_optimization_scaling(self):
        """Test that optimization scales reasonably with dataset size."""
        sizes = [100, 500, 1000]
        times = {}

        for size in sizes:
            y_true, y_prob = generate_binary_data(size, random_state=42)

            start_time = time.time()
            get_optimal_threshold(y_true, y_prob, method="unique_scan", metric="f1")
            end_time = time.time()

            times[size] = end_time - start_time

            # Should complete in reasonable time
            assert times[size] < 5.0, f"Optimization took {times[size]:.2f}s for {size} samples"

    def test_method_performance_comparison(self):
        """Test relative performance of different methods."""
        y_true, y_prob = generate_binary_data(500, random_state=42)

        methods = ["unique_scan", "minimize", "gradient"]
        times = {}

        for method in methods:
            start_time = time.time()
            get_optimal_threshold(y_true, y_prob, method=method, metric="f1")
            end_time = time.time()

            times[method] = end_time - start_time

            # All methods should complete in reasonable time
            assert times[method] < 10.0, f"Method {method} took {times[method]:.2f}s"
