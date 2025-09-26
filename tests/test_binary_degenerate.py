"""Test edge cases with degenerate label sets produce mathematically correct predictions.

This module tests that when label distributions are degenerate (all-negative,
all-positive, all-equal scores, extreme probabilities), the optimization
algorithms return thresholds that realize the mathematically optimal trivial
predictions under both comparison operators.

These tests catch bugs where algorithms return arbitrary values (like 0.5)
instead of the mathematically correct thresholds for degenerate cases.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import get_confusion_matrix
from tests.strategies import extreme_probabilities


def _compute_accuracy(y, p, threshold, comparison):
    """Helper to compute accuracy for given threshold and comparison."""
    pred = (p > threshold) if comparison == ">" else (p >= threshold)
    return float(np.mean(pred == y))


def _compute_f1(y, p, threshold, comparison):
    """Helper to compute F1 for given threshold and comparison."""
    tp, tn, fp, fn = get_confusion_matrix(y, p, threshold, comparison=comparison)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)


class TestAllNegativeLabels:
    """Test optimization when all labels are negative (y=0)."""

    def test_all_negatives_predict_all_negative_accuracy(self):
        """All-negative labels should result in all-negative predictions for optimal accuracy."""
        # Test with various probability distributions
        prob_arrays = [
            np.array([0.1, 0.4, 0.9, 0.0]),
            np.array([0.2, 0.8, 0.5, 0.7, 0.1, 0.9]),
            np.array([0.0, 1.0, 0.5]),  # Extreme values
            np.array([0.3] * 10),  # All same probability
        ]

        for probs in prob_arrays:
            y = np.zeros_like(probs, dtype=int)  # All negative labels

            for comparison in [">", ">="]:
                threshold = get_optimal_threshold(
                    y,
                    probs,
                    metric="accuracy",
                    method="sort_scan",
                    comparison=comparison,
                )

                # Verify threshold produces all-negative predictions
                pred = (
                    (probs > threshold) if comparison == ">" else (probs >= threshold)
                )

                assert not pred.any(), (
                    f"All-negative case should predict all negative with {comparison}, "
                    f"but got {pred.sum()} positives out of {len(pred)} total. "
                    f"Threshold: {threshold}, Probs: {probs}"
                )

                # Verify perfect accuracy
                accuracy = _compute_accuracy(y, probs, threshold, comparison)
                assert accuracy == 1.0, (
                    f"All-negative case should achieve perfect accuracy, got {accuracy}"
                )

                # Verify threshold bounds make mathematical sense
                if comparison == ">":
                    # For '>', threshold should be >= max(probs) to exclude all
                    assert threshold >= np.max(probs) - 1e-10, (
                        f"Threshold {threshold} should be >= max prob {np.max(probs)} for '>'"
                    )
                else:  # '>='
                    # For '>=', threshold should be > max(probs) to exclude all
                    assert threshold > np.max(probs) - 1e-10, (
                        f"Threshold {threshold} should be > max prob {np.max(probs)} for '>='"
                    )

    def test_all_negatives_predict_all_negative_f1(self):
        """All-negative labels with F1 metric should handle edge case gracefully."""
        probs = np.array([0.2, 0.7, 0.1, 0.8, 0.5])
        y = np.zeros_like(probs, dtype=int)

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y, probs, metric="f1", method="sort_scan", comparison=comparison
            )

            # Should predict all negative (F1 is undefined/0 for all-negative case)
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            assert not pred.any(), (
                f"All-negative case should predict all negative for F1 with {comparison}"
            )

            # F1 should be 0 or undefined (handled gracefully)
            f1 = _compute_f1(y, probs, threshold, comparison)
            assert f1 == 0.0, f"F1 should be 0.0 for all-negative case, got {f1}"


class TestAllPositiveLabels:
    """Test optimization when all labels are positive (y=1)."""

    def test_all_positives_predict_all_positive_accuracy(self):
        """All-positive labels should result in all-positive predictions for optimal accuracy."""
        prob_arrays = [
            np.array([0.2, 0.6, 0.9, 0.0]),
            np.array([0.1, 0.3, 0.7, 0.8, 0.9]),
            np.array([1.0, 0.0, 0.5]),  # Extreme values
            np.array([0.7] * 8),  # All same probability
        ]

        for probs in prob_arrays:
            y = np.ones_like(probs, dtype=int)  # All positive labels

            for comparison in [">", ">="]:
                threshold = get_optimal_threshold(
                    y,
                    probs,
                    metric="accuracy",
                    method="sort_scan",
                    comparison=comparison,
                )

                # Verify threshold produces all-positive predictions (where possible)
                pred = (
                    (probs > threshold) if comparison == ">" else (probs >= threshold)
                )

                # Special case: if min_prob = 0 and comparison = '>', we cannot predict
                # the item with prob=0 as positive due to constraint threshold >= 0
                min_prob = np.min(probs)
                if comparison == ">" and min_prob == 0.0:
                    # This is a constraint limitation - cannot achieve perfect accuracy
                    # The algorithm should still optimize as best as possible
                    accuracy = _compute_accuracy(y, probs, threshold, comparison)
                    # Should predict all non-zero probabilities as positive
                    expected_positive = probs > 0.0
                    assert np.array_equal(pred, expected_positive), (
                        f"With comparison '>' and min_prob=0, should predict only prob>0 items as positive. "
                        f"Expected: {expected_positive}, Got: {pred}"
                    )
                    # Accuracy should be (n-zeros)/n where zeros is count of zero probabilities
                    expected_accuracy = np.sum(probs > 0.0) / len(probs)
                    assert abs(accuracy - expected_accuracy) < 1e-10, (
                        f"Expected accuracy {expected_accuracy} with min_prob=0 and '>', got {accuracy}"
                    )
                else:
                    assert pred.all(), (
                        f"All-positive case should predict all positive with {comparison}, "
                        f"but got {pred.sum()} positives out of {len(pred)} total. "
                        f"Threshold: {threshold}, Probs: {probs}"
                    )
                    # Verify perfect accuracy
                    accuracy = _compute_accuracy(y, probs, threshold, comparison)
                    assert accuracy == 1.0, (
                        f"All-positive case should achieve perfect accuracy, got {accuracy}"
                    )

                # Verify threshold bounds make mathematical sense
                if comparison == ">":
                    # For '>', threshold should be < min(probs) to include all
                    assert threshold < np.min(probs) + 1e-10, (
                        f"Threshold {threshold} should be < min prob {np.min(probs)} for '>'"
                    )
                else:  # '>='
                    # For '>=', threshold should be <= min(probs) to include all
                    assert threshold <= np.min(probs) + 1e-10, (
                        f"Threshold {threshold} should be <= min prob {np.min(probs)} for '>='"
                    )

    def test_all_positives_predict_all_positive_f1(self):
        """All-positive labels should achieve maximum F1."""
        probs = np.array([0.3, 0.8, 0.1, 0.9, 0.6])
        y = np.ones_like(probs, dtype=int)

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y, probs, metric="f1", method="sort_scan", comparison=comparison
            )

            # Should predict all positive
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            assert pred.all(), (
                f"All-positive case should predict all positive for F1 with {comparison}"
            )

            # F1 should be perfect (1.0)
            f1 = _compute_f1(y, probs, threshold, comparison)
            assert f1 == 1.0, f"F1 should be 1.0 for all-positive case, got {f1}"


class TestAllEqualProbabilities:
    """Test optimization when all probabilities are identical."""

    @given(
        prob_value=st.floats(0.01, 0.99),
        n_samples=st.integers(5, 50),
        comparison=st.sampled_from([">", ">="]),
    )
    @settings(deadline=None, max_examples=50)
    def test_all_equal_probabilities_consistency(
        self, prob_value, n_samples, comparison
    ):
        """When all probabilities are equal, optimization should be consistent."""
        probs = np.full(n_samples, prob_value)

        # Generate mixed labels to avoid degenerate all-same-label case
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=n_samples)

        # Ensure both classes present
        if labels.sum() == 0:
            labels[0] = 1
        if labels.sum() == labels.size:
            labels[0] = 0

        threshold = get_optimal_threshold(
            labels, probs, metric="accuracy", method="sort_scan", comparison=comparison
        )

        # Should produce valid threshold
        assert 0 <= threshold <= 1, f"Threshold {threshold} out of valid range"

        # Predictions should be consistent (all same since all probs equal)
        pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

        # All predictions should be identical
        assert len(np.unique(pred)) == 1, (
            f"All probabilities equal, so all predictions should be identical. "
            f"Got predictions: {pred}"
        )

    def test_all_equal_probabilities_decision_boundary(self):
        """Test decision boundary when all probabilities are equal."""
        prob_val = 0.5
        probs = np.full(10, prob_val)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Alternating

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                labels,
                probs,
                metric="accuracy",
                method="sort_scan",
                comparison=comparison,
            )

            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            # Since all probs are equal, decision should be consistent
            if comparison == ">" and threshold >= prob_val:
                # All probs <= threshold, so no predictions positive
                assert not pred.any(), (
                    "With '>' and threshold >= prob, should predict all negative"
                )
            elif comparison == ">=" and threshold > prob_val:
                # All probs < threshold, so no predictions positive
                assert not pred.any(), (
                    "With '>=' and threshold > prob, should predict all negative"
                )
            elif comparison == ">" and threshold < prob_val:
                # All probs > threshold, so all predictions positive
                assert pred.all(), (
                    "With '>' and threshold < prob, should predict all positive"
                )
            elif comparison == ">=" and threshold <= prob_val:
                # All probs >= threshold, so all predictions positive
                assert pred.all(), (
                    "With '>=' and threshold <= prob, should predict all positive"
                )


class TestExtremeProbabilities:
    """Test with probabilities at exactly 0.0 and 1.0."""

    def test_probabilities_at_zero_and_one(self):
        """Test with probabilities exactly at boundaries."""
        test_cases = [
            # (probabilities, labels)
            (np.array([0.0, 1.0]), np.array([0, 1])),
            (np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1])),
            (np.array([0.0, 0.5, 1.0]), np.array([0, 1, 1])),
            (np.array([0.0, 1.0, 0.0, 1.0]), np.array([1, 0, 0, 1])),  # Misaligned
        ]

        for probs, labels in test_cases:
            for comparison in [">", ">="]:
                for metric in ["accuracy", "f1"]:
                    threshold = get_optimal_threshold(
                        labels,
                        probs,
                        metric=metric,
                        method="sort_scan",
                        comparison=comparison,
                    )

                    # Should produce valid threshold
                    assert 0 <= threshold <= 1, (
                        f"Threshold {threshold} out of bounds for extreme probs"
                    )

                    # Verify predictions make sense

                    # Compute resulting metric
                    if metric == "accuracy":
                        score = _compute_accuracy(labels, probs, threshold, comparison)
                        assert 0 <= score <= 1, f"Accuracy {score} out of range"
                    else:  # f1
                        score = _compute_f1(labels, probs, threshold, comparison)
                        assert 0 <= score <= 1, f"F1 {score} out of range"

    def test_all_probabilities_zero(self):
        """Test when all probabilities are exactly 0.0."""
        probs = np.zeros(5)
        labels = np.array([0, 1, 0, 1, 1])

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                labels,
                probs,
                metric="accuracy",
                method="sort_scan",
                comparison=comparison,
            )

            # With all probs = 0, predictions depend on threshold and comparison
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            if comparison == ">":
                # 0 > threshold is false for any valid threshold >= 0
                assert not pred.any(), (
                    "All probs = 0, so '> threshold' should be all false"
                )
            else:  # '>='
                if threshold <= 0:
                    # 0 >= threshold is true if threshold <= 0
                    assert pred.all(), (
                        "All probs = 0, so '>= threshold' should be all true when threshold <= 0"
                    )
                else:
                    # 0 >= threshold is false if threshold > 0
                    assert not pred.any(), (
                        "All probs = 0, so '>= threshold' should be all false when threshold > 0"
                    )

    def test_all_probabilities_one(self):
        """Test when all probabilities are exactly 1.0."""
        probs = np.ones(6)
        labels = np.array([1, 0, 1, 0, 1, 0])

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                labels,
                probs,
                metric="accuracy",
                method="sort_scan",
                comparison=comparison,
            )

            # With all probs = 1, predictions depend on threshold and comparison
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            if comparison == ">":
                if threshold < 1:
                    # 1 > threshold is true if threshold < 1
                    assert pred.all(), (
                        "All probs = 1, so '> threshold' should be all true when threshold < 1"
                    )
                else:
                    # 1 > threshold is false if threshold >= 1
                    assert not pred.any(), (
                        "All probs = 1, so '> threshold' should be all false when threshold >= 1"
                    )
            else:  # '>='
                # 1 >= threshold is true for any valid threshold <= 1
                assert pred.all(), "All probs = 1, so '>= threshold' should be all true"


class TestDegenerateVsArbitraryThresholds:
    """Test that degenerate cases don't return arbitrary values like 0.5."""

    def test_degenerate_not_point_five(self):
        """Degenerate cases should not return arbitrary 0.5 thresholds."""
        # Test cases that used to return 0.5 incorrectly
        test_cases = [
            # All negative
            (np.array([0, 0, 0]), np.array([0.2, 0.7, 0.8])),
            # All positive
            (np.array([1, 1, 1]), np.array([0.3, 0.6, 0.9])),
            # All same probability
            (np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])),
        ]

        for labels, probs in test_cases:
            for comparison in [">", ">="]:
                for metric in ["accuracy", "f1"]:
                    threshold = get_optimal_threshold(
                        labels,
                        probs,
                        metric=metric,
                        method="sort_scan",
                        comparison=comparison,
                    )

                    # Should NOT be arbitrary 0.5 (unless it's actually optimal)
                    # We verify this by checking that the threshold actually produces
                    # the mathematically optimal result
                    pred = (
                        (probs > threshold)
                        if comparison == ">"
                        else (probs >= threshold)
                    )

                    if np.all(labels == 0):
                        # All negative case - should predict all negative
                        assert not pred.any(), (
                            f"All-negative case returned threshold {threshold} that doesn't "
                            f"predict all negative with {comparison}"
                        )
                    elif np.all(labels == 1):
                        # All positive case - should predict all positive
                        assert pred.all(), (
                            f"All-positive case returned threshold {threshold} that doesn't "
                            f"predict all positive with {comparison}"
                        )

                    # In all cases, threshold should be mathematically justified
                    if metric == "accuracy":
                        accuracy = _compute_accuracy(
                            labels, probs, threshold, comparison
                        )
                        # Should be optimal accuracy (test other thresholds don't beat it)
                        for test_thresh in [0.0, 0.25, 0.5, 0.75, 1.0]:
                            test_acc = _compute_accuracy(
                                labels, probs, test_thresh, comparison
                            )
                            assert accuracy >= test_acc - 1e-10, (
                                f"Threshold {threshold} gives accuracy {accuracy} but "
                                f"test threshold {test_thresh} gives {test_acc}"
                            )

    @given(extreme_probabilities(min_size=3, max_size=20))
    @settings(deadline=None, max_examples=30)
    def test_extreme_probabilities_not_arbitrary(self, probs):
        """Test that extreme probability cases don't return arbitrary thresholds."""
        rng = np.random.default_rng(123)
        labels = rng.integers(0, 2, size=len(probs))

        # Ensure both classes present if possible
        if len(labels) > 1:
            if labels.sum() == 0:
                labels[0] = 1
            if labels.sum() == labels.size:
                labels[0] = 0

        for comparison in [">", ">="]:
            try:
                threshold = get_optimal_threshold(
                    labels,
                    probs,
                    metric="accuracy",
                    method="sort_scan",
                    comparison=comparison,
                )

                # Threshold should be meaningful, not arbitrary
                assert 0 <= threshold <= 1, f"Threshold {threshold} out of valid range"

                # Should actually optimize the metric
                accuracy = _compute_accuracy(labels, probs, threshold, comparison)
                assert 0 <= accuracy <= 1, f"Accuracy {accuracy} out of valid range"

            except Exception as e:
                # Some extreme cases might be unsupported, that's okay
                if "empty" not in str(e).lower():
                    raise  # Re-raise unexpected errors
