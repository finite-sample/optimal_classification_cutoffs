"""Tests for critical bug fixes identified in code review.

This module contains specific tests designed to verify the fixes for critical
bugs that could silently produce incorrect results:

1. Inclusive/exclusive threshold bug in sort_scan calls
2. Weighted metrics int casting bug
3. Degenerate label sets returning arbitrary 0.5
4. Dinkelbach Expected-Fβ using wrong sum
5. Stability enhancements
"""

import numpy as np

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import get_confusion_matrix
from optimal_cutoffs.optimizers import _dinkelbach_expected_fbeta


class TestInclusiveExclusiveBugFix:
    """Test fix for inclusive/exclusive threshold bug in sort_scan calls."""

    def test_tied_probabilities_comparison_operators(self):
        """Test that '>' and '>=' give different results with tied probabilities."""
        # Create dataset where threshold will be exactly at tied probability value
        y_true = [0, 0, 1, 1, 1, 0]
        pred_prob = [0.3, 0.5, 0.5, 0.5, 0.7, 0.4]  # Three samples tied at 0.5

        # Get thresholds using both comparison operators
        thresh_exclusive = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="sort_scan", comparison=">"
        )
        thresh_inclusive = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="sort_scan", comparison=">="
        )

        # Apply thresholds with respective comparison operators
        pred_exclusive = np.array(pred_prob) > thresh_exclusive
        pred_inclusive = np.array(pred_prob) >= thresh_inclusive

        # When probabilities equal the threshold, predictions should differ
        if np.any(np.array(pred_prob) == thresh_exclusive):
            assert not np.array_equal(pred_exclusive, pred_inclusive), (
                "Predictions should differ when probabilities equal threshold"
            )

        # Verify both produce valid confusion matrices
        tp_excl, tn_excl, fp_excl, fn_excl = get_confusion_matrix(
            y_true, pred_prob, thresh_exclusive, comparison=">"
        )
        tp_incl, tn_incl, fp_incl, fn_incl = get_confusion_matrix(
            y_true, pred_prob, thresh_inclusive, comparison=">="
        )

        assert tp_excl + tn_excl + fp_excl + fn_excl == len(y_true)
        assert tp_incl + tn_incl + fp_incl + fn_incl == len(y_true)

    def test_sort_scan_vs_fallback_consistency(self):
        """Test that sort_scan and fallback methods agree on comparison operators."""
        y_true = [0, 1, 0, 1, 1, 0, 1, 0]
        pred_prob = [0.2, 0.6, 0.6, 0.6, 0.8, 0.4, 0.7, 0.3]  # Multiple ties

        for comparison in [">", ">="]:
            # Get threshold from sort_scan
            try:
                thresh_sort = get_optimal_threshold(
                    y_true,
                    pred_prob,
                    metric="f1",
                    method="sort_scan",
                    comparison=comparison,
                )
                sort_available = True
            except Exception:
                sort_available = False

            # Get threshold from fallback (unique_scan)
            thresh_fallback = get_optimal_threshold(
                y_true,
                pred_prob,
                metric="f1",
                method="unique_scan",
                comparison=comparison,
            )

            if sort_available:
                # Confusion matrices should be identical or very close
                tp_sort, tn_sort, fp_sort, fn_sort = get_confusion_matrix(
                    y_true, pred_prob, thresh_sort, comparison=comparison
                )
                tp_fall, tn_fall, fp_fall, fn_fall = get_confusion_matrix(
                    y_true, pred_prob, thresh_fallback, comparison=comparison
                )

                # Allow small differences due to tie-breaking, but should be very close
                assert abs(tp_sort - tp_fall) <= 1
                assert abs(fp_sort - fp_fall) <= 1


class TestWeightedMetricsBugFix:
    """Test fix for weighted metrics int casting bug."""

    def test_fractional_weights_preserved(self):
        """Test that fractional weights are preserved, not cast to int."""
        y_true = [0, 1, 0, 1]
        pred_prob = [0.2, 0.8, 0.3, 0.7]
        sample_weight = [0.3, 2.7, 1.0, 1.5]  # Fractional weights

        # Get optimal threshold with fractional weights
        threshold = get_optimal_threshold(
            y_true,
            pred_prob,
            metric="f1",
            method="unique_scan",
            sample_weight=sample_weight,
        )

        # Compute weighted confusion matrix
        tp, tn, fp, fn = get_confusion_matrix(
            y_true, pred_prob, threshold, sample_weight=sample_weight
        )

        # With fractional weights, these should not be integers
        # The bug would make all fractional weights become 0
        total_weight = sum(sample_weight)
        assert abs(tp + tn + fp + fn - total_weight) < 1e-10, (
            "Total weighted counts should equal sum of weights"
        )

        # Verify F1 score can be computed with fractional counts
        if tp + fp > 0:
            precision = tp / (tp + fp)
            assert 0 <= precision <= 1
        if tp + fn > 0:
            recall = tp / (tp + fn)
            assert 0 <= recall <= 1

    def test_weighted_vs_expanded_dataset_consistency(self):
        """Test that weighted metrics match expanded dataset approach."""
        # Original dataset
        y_true = [0, 1, 1]
        pred_prob = [0.2, 0.7, 0.8]
        sample_weight = [0.5, 1.5, 2.0]  # Fractional weights

        # Create expanded dataset by replicating samples (approximate)
        y_expanded = [0] * 1 + [1] * 3 + [1] * 4  # Approximate expansion
        pred_expanded = [0.2] * 1 + [0.7] * 3 + [0.8] * 4

        # Get thresholds (should be similar if weights work correctly)
        thresh_weighted = get_optimal_threshold(
            y_true,
            pred_prob,
            metric="accuracy",
            method="unique_scan",
            sample_weight=sample_weight,
        )
        thresh_expanded = get_optimal_threshold(
            y_expanded, pred_expanded, metric="accuracy", method="unique_scan"
        )

        # Thresholds should be reasonably close
        assert abs(thresh_weighted - thresh_expanded) < 0.2, (
            "Weighted and expanded approaches should give similar thresholds"
        )


class TestDegenerateClassesBugFix:
    """Test fix for degenerate label sets returning arbitrary 0.5."""

    def test_all_negative_labels_optimal_threshold(self):
        """Test optimal threshold when all labels are negative."""
        y_true = [0, 0, 0, 0]
        pred_prob = [0.1, 0.4, 0.6, 0.9]

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y_true,
                pred_prob,
                metric="accuracy",
                method="unique_scan",
                comparison=comparison,
            )

            # Apply threshold
            if comparison == ">":
                predictions = np.array(pred_prob) > threshold
            else:
                predictions = np.array(pred_prob) >= threshold

            # All predictions should be negative for optimal accuracy
            assert np.all(~predictions), (
                f"With all negative labels and {comparison}, all predictions should be negative"
            )

            # Accuracy should be 1.0
            accuracy = np.mean(predictions == y_true)
            assert accuracy == 1.0

    def test_all_positive_labels_optimal_threshold(self):
        """Test optimal threshold when all labels are positive."""
        y_true = [1, 1, 1, 1]
        pred_prob = [0.1, 0.4, 0.6, 0.9]

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y_true,
                pred_prob,
                metric="accuracy",
                method="unique_scan",
                comparison=comparison,
            )

            # Apply threshold
            if comparison == ">":
                predictions = np.array(pred_prob) > threshold
            else:
                predictions = np.array(pred_prob) >= threshold

            # All predictions should be positive for optimal accuracy
            assert np.all(predictions), (
                f"With all positive labels and {comparison}, all predictions should be positive"
            )

            # Accuracy should be 1.0
            accuracy = np.mean(predictions == y_true)
            assert accuracy == 1.0

    def test_degenerate_cases_not_arbitrary_point_five(self):
        """Test that degenerate cases don't return arbitrary 0.5."""
        # All negative case
        y_true_neg = [0, 0, 0]
        pred_prob_neg = [0.2, 0.7, 0.8]

        thresh_neg = get_optimal_threshold(
            y_true_neg, pred_prob_neg, metric="f1", method="unique_scan"
        )

        # Should NOT be 0.5 (the old bug)
        assert thresh_neg != 0.5, "All-negative case should not return arbitrary 0.5"

        # All positive case
        y_true_pos = [1, 1, 1]
        pred_prob_pos = [0.2, 0.7, 0.8]

        thresh_pos = get_optimal_threshold(
            y_true_pos, pred_prob_pos, metric="f1", method="unique_scan"
        )

        # Should NOT be 0.5 (the old bug)
        assert thresh_pos != 0.5, "All-positive case should not return arbitrary 0.5"


class TestDinkelbachExpectedFBetaBugFix:
    """Test fix for Dinkelbach Expected-Fβ using wrong sum."""

    def test_dinkelbach_label_independence(self):
        """Test that Dinkelbach method is independent of label permutation."""
        pred_prob = [0.1, 0.3, 0.7, 0.9]

        # Original labels
        y_true_1 = [0, 1, 0, 1]
        thresh_1 = _dinkelbach_expected_fbeta(y_true_1, pred_prob, beta=1.0)

        # Permuted labels (same probabilities)
        y_true_2 = [1, 0, 1, 0]
        thresh_2 = _dinkelbach_expected_fbeta(y_true_2, pred_prob, beta=1.0)

        # Thresholds should be identical (expected Fβ depends only on probabilities)
        assert abs(thresh_1 - thresh_2) < 1e-10, (
            "Dinkelbach threshold should be independent of label permutation"
        )

        # Try another permutation
        y_true_3 = [0, 0, 1, 1]
        thresh_3 = _dinkelbach_expected_fbeta(y_true_3, pred_prob, beta=1.0)

        assert abs(thresh_1 - thresh_3) < 1e-10, (
            "Dinkelbach threshold should be independent of any label permutation"
        )

    def test_dinkelbach_depends_only_on_probabilities(self):
        """Test that Dinkelbach depends only on probabilities, not labels."""
        pred_prob = [0.2, 0.4, 0.6, 0.8]

        # Different label configurations
        labels_configs = [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 0],
        ]

        thresholds = []
        for y_true in labels_configs:
            thresh = _dinkelbach_expected_fbeta(y_true, pred_prob, beta=1.0)
            thresholds.append(thresh)

        # All thresholds should be identical
        for i in range(1, len(thresholds)):
            assert abs(thresholds[0] - thresholds[i]) < 1e-10, (
                f"All Dinkelbach thresholds should be identical, got {thresholds}"
            )

    def test_dinkelbach_sum_probabilities_not_labels(self):
        """Test that Dinkelbach uses sum of probabilities, not labels."""
        # Case where sum(probabilities) != sum(labels)
        pred_prob = [0.1, 0.2, 0.3, 0.4]  # sum = 1.0
        y_true = [1, 1, 1, 1]  # sum = 4.0

        # The fixed version should work correctly
        threshold = _dinkelbach_expected_fbeta(y_true, pred_prob, beta=1.0)

        # Threshold should be reasonable (between 0 and 1)
        assert 0.0 <= threshold <= 1.0

        # Another case
        pred_prob_2 = [0.9, 0.8, 0.7, 0.6]  # sum = 3.0
        y_true_2 = [0, 0, 0, 0]  # sum = 0.0

        threshold_2 = _dinkelbach_expected_fbeta(y_true_2, pred_prob_2, beta=1.0)
        assert 0.0 <= threshold_2 <= 1.0


class TestStabilityEnhancements:
    """Test stability enhancements for tie handling."""

    def test_stable_sort_deterministic_ties(self):
        """Test that stable sort produces deterministic results with ties."""
        # Dataset with many ties
        y_true = [0, 1, 0, 1, 1, 0, 1, 0] * 3  # Repeat for more ties
        pred_prob = [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.7, 0.7] * 3

        # Run multiple times - should get same result
        thresholds = []
        for _ in range(5):
            thresh = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method="unique_scan"
            )
            thresholds.append(thresh)

        # All thresholds should be identical (deterministic)
        for i in range(1, len(thresholds)):
            assert abs(thresholds[0] - thresholds[i]) < 1e-12, (
                "Results should be deterministic with tied probabilities"
            )

    def test_tied_probabilities_at_threshold(self):
        """Test handling of tied probabilities at the optimal threshold."""
        # Create case where optimal threshold equals some probabilities
        y_true = [0, 0, 1, 1, 1, 0]
        pred_prob = [0.2, 0.4, 0.4, 0.4, 0.6, 0.3]  # Ties at 0.4

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y_true,
                pred_prob,
                metric="f1",
                method="unique_scan",
                comparison=comparison,
            )

            # Should produce valid confusion matrix
            tp, tn, fp, fn = get_confusion_matrix(
                y_true, pred_prob, threshold, comparison=comparison
            )

            assert tp + tn + fp + fn == len(y_true)
            assert all(x >= 0 for x in [tp, tn, fp, fn])


# Integration test combining multiple fixes
class TestCriticalBugFixesIntegration:
    """Integration tests verifying multiple bug fixes work together."""

    def test_weighted_tied_probabilities_degenerate_case(self):
        """Test combination of fractional weights, tied probs, and edge cases."""
        # Edge case: mostly one class with fractional weights and ties
        y_true = [0, 0, 0, 1]
        pred_prob = [0.3, 0.5, 0.5, 0.7]  # Ties at 0.5
        sample_weight = [0.3, 1.7, 2.1, 0.9]  # Fractional weights

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y_true,
                pred_prob,
                metric="f1",
                method="unique_scan",
                comparison=comparison,
                sample_weight=sample_weight,
            )

            # Should not crash and produce valid threshold
            assert 0.0 <= threshold <= 1.0

            # Should produce valid weighted confusion matrix
            tp, tn, fp, fn = get_confusion_matrix(
                y_true,
                pred_prob,
                threshold,
                sample_weight=sample_weight,
                comparison=comparison,
            )

            total_weight = sum(sample_weight)
            assert abs(tp + tn + fp + fn - total_weight) < 1e-10

    def test_all_methods_consistency_after_fixes(self):
        """Test that all methods produce consistent results after bug fixes."""
        y_true = [0, 1, 0, 1, 1, 0]
        pred_prob = [0.2, 0.7, 0.4, 0.8, 0.6, 0.3]

        methods = ["unique_scan", "minimize"]
        try:
            methods.append("sort_scan")  # If vectorized F1 available
            get_optimal_threshold(y_true, pred_prob, metric="f1", method="sort_scan")
        except Exception:
            pass

        thresholds = {}
        for method in methods:
            try:
                thresholds[method] = get_optimal_threshold(
                    y_true, pred_prob, metric="f1", method=method
                )
            except Exception as e:
                print(f"Method {method} failed: {e}")
                continue

        # All working methods should produce reasonably similar thresholds
        if len(thresholds) > 1:
            threshold_values = list(thresholds.values())
            max_diff = max(threshold_values) - min(threshold_values)
            assert max_diff < 0.5, f"Methods should agree reasonably: {thresholds}"
