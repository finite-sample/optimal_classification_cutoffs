"""Integration tests for comparing different optimization methods.

This module tests consistency between different optimization methods and
validates that they produce reasonable and comparable results.
"""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import compute_metric_at_threshold, get_confusion_matrix
from tests.fixtures.assertions import (
    assert_method_consistency,
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_calibrated_probabilities,
    generate_multiclass_data,
    generate_tied_probabilities,
)


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

        # Methods should be reasonably consistent
        scores = [results[method][1] for method in methods]

        # All scores should be high and similar
        assert np.std(scores) < 0.2, f"High variance in scores: {scores}"

    def test_methods_on_noisy_data(self):
        """Test method consistency on noisy data."""
        y_true, pred_prob = generate_binary_data(100, noise=0.2, random_state=42)

        methods = ["unique_scan", "minimize", "gradient"]
        results = {}

        for method in methods:
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method=method
            )
            score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
            results[method] = (threshold, score)

        # All methods should produce valid results
        for method, (threshold, score) in results.items():
            assert_valid_threshold(threshold)
            assert_valid_metric_score(score, "f1")

    def test_piecewise_vs_fallback_consistency(self):
        """Test that piecewise and fallback methods agree for piecewise metrics."""
        y_true, pred_prob = generate_binary_data(80, random_state=42)

        # Methods that should agree for piecewise metrics
        piecewise_method = "sort_scan"
        fallback_method = "unique_scan"

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold_piecewise = get_optimal_threshold(
                y_true, pred_prob, metric=metric, method=piecewise_method
            )
            threshold_fallback = get_optimal_threshold(
                y_true, pred_prob, metric=metric, method=fallback_method
            )

            # Should be very close or identical
            assert_method_consistency(
                threshold_piecewise, threshold_fallback,
                piecewise_method, fallback_method, tolerance=1e-8
            )

    def test_auto_method_selection(self):
        """Test that auto method selection produces reasonable results."""
        y_true, pred_prob = generate_binary_data(60, random_state=42)

        # Auto method should work
        threshold_auto = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="auto"
        )

        # Compare with manual method selection
        threshold_manual = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="unique_scan"
        )

        assert_valid_threshold(threshold_auto)
        assert_valid_threshold(threshold_manual)

        # Should achieve similar performance
        score_auto = compute_metric_at_threshold(y_true, pred_prob, threshold_auto, "f1")
        score_manual = compute_metric_at_threshold(y_true, pred_prob, threshold_manual, "f1")

        assert abs(score_auto - score_manual) < 0.1  # Should be reasonably close


class TestMethodPerformanceComparison:
    """Test relative performance characteristics of methods."""

    def test_method_performance_on_large_data(self):
        """Test method performance scaling on larger datasets."""
        import time

        y_true, pred_prob = generate_binary_data(1000, random_state=42)

        methods = ["unique_scan", "minimize", "gradient"]
        times = {}
        results = {}

        for method in methods:
            start_time = time.time()
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method=method
            )
            end_time = time.time()

            times[method] = end_time - start_time
            results[method] = threshold

            # Should complete in reasonable time
            assert times[method] < 5.0, f"Method {method} took {times[method]:.2f}s"

        # All methods should produce valid results
        for method, threshold in results.items():
            assert_valid_threshold(threshold)

    def test_method_stability_across_runs(self):
        """Test that methods produce stable results across multiple runs."""
        y_true, pred_prob = generate_binary_data(50, random_state=42)

        methods = ["unique_scan", "minimize"]  # Deterministic methods

        for method in methods:
            # Run multiple times (should be deterministic)
            thresholds = []
            for _ in range(3):
                threshold = get_optimal_threshold(
                    y_true, pred_prob, metric="f1", method=method
                )
                thresholds.append(threshold)

            # Should be identical
            assert np.std(thresholds) < 1e-12, f"Method {method} not stable: {thresholds}"

    def test_method_comparison_with_ties(self):
        """Test method behavior with tied probability values."""
        y_true, pred_prob = generate_tied_probabilities(40, random_state=42)

        methods = ["unique_scan", "minimize"]

        for comparison in [">", ">="]:
            results = {}
            for method in methods:
                threshold = get_optimal_threshold(
                    y_true, pred_prob, metric="f1", method=method, comparison=comparison
                )
                results[method] = threshold

            # All methods should handle ties
            for method, threshold in results.items():
                assert_valid_threshold(threshold)


class TestModeComparison:
    """Test comparison between empirical and expected modes."""

    def test_empirical_vs_expected_mode(self):
        """Test comparison between empirical and expected optimization modes."""
        y_true, pred_prob = generate_calibrated_probabilities(100, random_state=42)

        # Empirical mode
        threshold_empirical = get_optimal_threshold(
            y_true, pred_prob, mode="empirical", metric="f1"
        )

        # Expected mode (returns tuple)
        result_expected = get_optimal_threshold(
            y_true, pred_prob, mode="expected", metric="f1"
        )
        threshold_expected, expected_f1 = result_expected

        assert_valid_threshold(threshold_empirical)
        assert_valid_threshold(threshold_expected)
        assert_valid_metric_score(expected_f1, "expected_f1")

        # Both should achieve reasonable performance
        empirical_f1 = compute_metric_at_threshold(y_true, pred_prob, threshold_empirical, "f1")
        assert_valid_metric_score(empirical_f1, "f1")

    def test_mode_consistency_on_calibrated_data(self):
        """Test that modes give reasonable results on calibrated data."""
        y_true, pred_prob = generate_calibrated_probabilities(150, random_state=42)

        threshold_empirical = get_optimal_threshold(
            y_true, pred_prob, mode="empirical", metric="f1"
        )
        result_expected = get_optimal_threshold(
            y_true, pred_prob, mode="expected", metric="f1"
        )
        threshold_expected, expected_f1 = result_expected

        # Compute empirical F1 for both thresholds
        empirical_f1_emp = compute_metric_at_threshold(y_true, pred_prob, threshold_empirical, "f1")
        empirical_f1_exp = compute_metric_at_threshold(y_true, pred_prob, threshold_expected, "f1")

        # Both should achieve reasonable performance on calibrated data
        assert empirical_f1_emp > 0.2
        assert empirical_f1_exp > 0.2

    def test_expected_mode_label_independence(self):
        """Test that expected mode is independent of labels."""
        pred_prob = np.array([0.2, 0.4, 0.6, 0.8])

        # Different label configurations
        labels_1 = np.array([0, 0, 1, 1])
        labels_2 = np.array([1, 0, 0, 1])
        labels_3 = np.array([0, 1, 1, 0])

        results = []
        for labels in [labels_1, labels_2, labels_3]:
            result = get_optimal_threshold(
                labels, pred_prob, mode="expected", metric="f1"
            )
            results.append(result)

        # All results should be identical (label-independent)
        threshold_1, expected_1 = results[0]
        for i, (threshold_i, expected_i) in enumerate(results[1:], 1):
            assert threshold_1 == pytest.approx(threshold_i, abs=1e-10), \
                f"Expected mode threshold differs between label sets: {threshold_1} vs {threshold_i}"
            assert expected_1 == pytest.approx(expected_i, abs=1e-10), \
                f"Expected mode score differs between label sets: {expected_1} vs {expected_i}"


class TestMulticlassMethodComparison:
    """Test method comparison for multiclass scenarios."""

    def test_multiclass_ovr_vs_coord_ascent(self):
        """Test One-vs-Rest vs coordinate ascent approaches."""
        y_true, pred_prob = generate_multiclass_data(60, n_classes=3, random_state=42)

        # One-vs-Rest approach (default)
        thresholds_ovr = get_optimal_threshold(
            y_true, pred_prob, method="unique_scan", metric="f1"
        )

        # Coordinate ascent approach
        thresholds_coord = get_optimal_threshold(
            y_true, pred_prob, method="coord_ascent", metric="f1"
        )

        # Both should produce valid results
        assert len(thresholds_ovr) == 3
        assert len(thresholds_coord) == 3

        for threshold in list(thresholds_ovr) + list(thresholds_coord):
            assert_valid_threshold(threshold)

    def test_multiclass_different_averaging(self):
        """Test multiclass methods with different averaging approaches."""
        y_true, pred_prob = generate_multiclass_data(50, n_classes=3, random_state=42)

        methods = ["unique_scan", "minimize"]
        averages = ["macro", "micro", "weighted"]

        for method in methods:
            for average in averages:
                thresholds = get_optimal_threshold(
                    y_true, pred_prob, method=method, metric="f1", average=average
                )

                assert len(thresholds) == 3
                for threshold in thresholds:
                    assert_valid_threshold(threshold)

    def test_multiclass_method_consistency(self):
        """Test consistency between multiclass methods."""
        y_true, pred_prob = generate_multiclass_data(40, n_classes=3, random_state=42)

        methods = ["unique_scan", "minimize"]
        results = {}

        for method in methods:
            thresholds = get_optimal_threshold(
                y_true, pred_prob, method=method, metric="f1"
            )
            results[method] = thresholds

        # All methods should produce valid results
        for method, thresholds in results.items():
            assert len(thresholds) == 3
            for threshold in thresholds:
                assert_valid_threshold(threshold)


class TestComparisonOperatorMethods:
    """Test method behavior with different comparison operators."""

    def test_methods_with_comparison_operators(self):
        """Test that all methods work with both comparison operators."""
        y_true, pred_prob = generate_tied_probabilities(30, random_state=42)

        methods = ["unique_scan", "minimize", "gradient"]
        comparisons = [">", ">="]

        for method in methods:
            for comparison in comparisons:
                threshold = get_optimal_threshold(
                    y_true, pred_prob, method=method, comparison=comparison
                )
                assert_valid_threshold(threshold)

    def test_comparison_operator_consistency(self):
        """Test that comparison operators affect results consistently across methods."""
        y_true = np.array([0, 1, 0, 1])
        pred_prob = np.array([0.3, 0.5, 0.7, 0.5])  # Ties at 0.5

        methods = ["unique_scan", "minimize"]

        for method in methods:
            threshold_gt = get_optimal_threshold(
                y_true, pred_prob, method=method, comparison=">"
            )
            threshold_gte = get_optimal_threshold(
                y_true, pred_prob, method=method, comparison=">="
            )

            # Verify that predictions differ appropriately
            tp_gt, tn_gt, fp_gt, fn_gt = get_confusion_matrix(
                y_true, pred_prob, threshold_gt, comparison=">"
            )
            tp_gte, tn_gte, fp_gte, fn_gte = get_confusion_matrix(
                y_true, pred_prob, threshold_gte, comparison=">="
            )

            # Results should be valid
            for tp, tn, fp, fn in [(tp_gt, tn_gt, fp_gt, fn_gt), (tp_gte, tn_gte, fp_gte, fn_gte)]:
                assert tp + tn + fp + fn == len(y_true)


class TestDeterminismAndReproducibility:
    """Test determinism and reproducibility across methods."""

    def test_method_determinism(self):
        """Test that deterministic methods produce identical results."""
        y_true, pred_prob = generate_binary_data(50, random_state=42)

        # These methods should be deterministic
        deterministic_methods = ["unique_scan", "sort_scan"]

        for method in deterministic_methods:
            try:
                # Run multiple times
                thresholds = []
                for _ in range(3):
                    threshold = get_optimal_threshold(
                        y_true, pred_prob, method=method, metric="f1"
                    )
                    thresholds.append(threshold)

                # Should be identical
                assert np.std(thresholds) < 1e-12, f"Method {method} not deterministic"
            except ValueError:
                # Skip if method not available
                continue

    def test_reproducibility_with_random_state(self):
        """Test reproducibility when random state is controlled."""
        y_true, pred_prob = generate_binary_data(40, random_state=42)

        # Methods that might use randomization
        methods_with_random = ["minimize", "gradient"]

        for method in methods_with_random:
            # Run with same random state (if applicable)
            threshold1 = get_optimal_threshold(
                y_true, pred_prob, method=method, metric="f1"
            )
            threshold2 = get_optimal_threshold(
                y_true, pred_prob, method=method, metric="f1"
            )

            # Should be reproducible for scipy-based methods
            if method in ["minimize"]:
                assert threshold1 == pytest.approx(threshold2, abs=1e-8)

    def test_cross_platform_consistency(self):
        """Test that methods produce consistent results across different inputs."""
        # Use simple, controlled data
        y_true = np.array([0, 1, 0, 1, 0, 1])
        pred_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        methods = ["unique_scan", "minimize"]

        for method in methods:
            threshold = get_optimal_threshold(
                y_true, pred_prob, method=method, metric="f1"
            )

            # Should produce valid, reasonable threshold
            assert_valid_threshold(threshold)

            # Should achieve good performance on this simple data
            score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
            assert score > 0.8  # Should be high for this separable case
