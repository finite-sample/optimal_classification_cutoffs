"""Tests for extreme edge cases in threshold optimization."""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import get_confusion_matrix


class TestExtremeLabeDistributions:
    """Test optimization with extreme label distributions."""

    def test_all_positive_labels(self):
        """Test when all true labels are positive (y=1)."""
        y_true = [1, 1, 1, 1, 1]
        pred_prob = [0.1, 0.3, 0.5, 0.7, 0.9]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # Verify confusion matrix makes sense
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tn == 0  # No true negatives when all labels are 1
            assert fp == 0  # No false positives when all labels are 1
            assert tp + fn == len(y_true)  # All samples are actually positive

    def test_all_negative_labels(self):
        """Test when all true labels are negative (y=0)."""
        y_true = [0, 0, 0, 0, 0]
        pred_prob = [0.1, 0.3, 0.5, 0.7, 0.9]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # Verify confusion matrix makes sense
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp == 0  # No true positives when all labels are 0
            assert fn == 0  # No false negatives when all labels are 0
            assert tn + fp == len(y_true)  # All samples are actually negative

    def test_single_positive_in_negatives(self):
        """Test extreme imbalance: one positive among many negatives."""
        y_true = [0] * 99 + [1]  # 1% positive class
        pred_prob = np.random.RandomState(42).uniform(0, 1, 100)

        for metric in ["f1", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # Should still produce meaningful results
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == 100

    def test_single_negative_in_positives(self):
        """Test extreme imbalance: one negative among many positives."""
        y_true = [1] * 99 + [0]  # 99% positive class
        pred_prob = np.random.RandomState(42).uniform(0, 1, 100)

        for metric in ["f1", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # Should still produce meaningful results
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == 100


class TestExtremeProbabilityDistributions:
    """Test optimization with extreme probability distributions."""

    def test_all_zero_probabilities(self):
        """Test when all predicted probabilities are 0."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # With all probabilities at 0, optimal strategy depends on comparison operator
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == len(y_true)

    def test_all_one_probabilities(self):
        """Test when all predicted probabilities are 1."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [1.0, 1.0, 1.0, 1.0, 1.0]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # With all probabilities at 1, optimal strategy depends on comparison operator
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == len(y_true)

    def test_binary_probabilities_only(self):
        """Test with probabilities only at 0 and 1."""
        y_true = [0, 1, 0, 1, 1, 0, 1, 0]
        pred_prob = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

            # Should find optimal threshold between 0 and 1
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == len(y_true)

    def test_very_small_probabilities(self):
        """Test with very small but non-zero probabilities."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

        # Should handle small probabilities without numerical issues
        tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
        assert tp + tn + fp + fn == len(y_true)

    def test_very_large_probabilities_near_one(self):
        """Test with probabilities very close to 1."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [1 - 1e-10, 1 - 2e-10, 1 - 3e-10, 1 - 4e-10, 1 - 5e-10]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

        # Should handle probabilities near 1 without numerical issues
        tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
        assert tp + tn + fp + fn == len(y_true)


class TestSingleSampleCases:
    """Test edge cases with very small datasets."""

    def test_single_positive_sample(self):
        """Test with just one positive sample."""
        y_true = [1]
        pred_prob = [0.7]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

    def test_single_negative_sample(self):
        """Test with just one negative sample."""
        y_true = [0]
        pred_prob = [0.3]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0

    def test_two_samples_same_class(self):
        """Test with two samples of the same class."""
        # Two positives
        y_true = [1, 1]
        pred_prob = [0.3, 0.7]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

        # Two negatives
        y_true = [0, 0]
        pred_prob = [0.3, 0.7]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0

    def test_two_samples_different_class(self):
        """Test with two samples of different classes."""
        y_true = [0, 1]
        pred_prob = [0.3, 0.7]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(y_true, pred_prob, metric=metric)
            assert 0.0 <= threshold <= 1.0


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_machine_epsilon_probabilities(self):
        """Test with probabilities at machine epsilon."""
        eps = np.finfo(float).eps
        y_true = [0, 1, 0, 1]
        pred_prob = [eps, eps * 2, eps * 3, eps * 4]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0
        assert np.isfinite(threshold)

    def test_probabilities_near_one_minus_epsilon(self):
        """Test with probabilities very close to 1."""
        eps = np.finfo(float).eps
        y_true = [0, 1, 0, 1]
        pred_prob = [1 - eps, 1 - eps * 2, 1 - eps * 3, 1 - eps * 4]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0
        assert np.isfinite(threshold)

    def test_identical_tiny_differences(self):
        """Test with probabilities that differ by tiny amounts."""
        base = 0.5
        tiny_diff = 1e-15

        y_true = [0, 1, 0, 1]
        pred_prob = [base, base + tiny_diff, base - tiny_diff, base + 2 * tiny_diff]

        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= threshold <= 1.0
        assert np.isfinite(threshold)


class TestConsistencyWithExtremeCases:
    """Test that different methods handle extreme cases consistently."""

    def test_methods_agree_on_trivial_cases(self):
        """Test that methods agree on trivial optimization problems."""
        # Perfect separation case
        y_true = [0, 0, 1, 1]
        pred_prob = [0.1, 0.2, 0.8, 0.9]

        methods = ["smart_brute", "minimize", "gradient"]
        thresholds = {}

        for method in methods:
            thresholds[method] = get_optimal_threshold(
                y_true, pred_prob, metric="accuracy", method=method
            )

        # All methods should find thresholds that separate classes perfectly
        for method, threshold in thresholds.items():
            tp, tn, fp, fn = get_confusion_matrix(y_true, pred_prob, threshold)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            assert accuracy == 1.0, f"Method {method} didn't achieve perfect accuracy"

    def test_methods_handle_impossible_cases(self):
        """Test that methods handle impossible-to-separate cases gracefully."""
        # Impossible case: labels are anti-correlated with probabilities
        y_true = [1, 1, 0, 0]
        pred_prob = [0.1, 0.2, 0.8, 0.9]

        methods = ["smart_brute", "minimize"]

        for method in methods:
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method=method
            )
            assert 0.0 <= threshold <= 1.0
            assert np.isfinite(threshold)


class TestMulticlassExtremeCases:
    """Test extreme cases in multiclass scenarios."""

    def test_single_class_multiclass(self):
        """Test multiclass optimization with highly imbalanced classes."""
        # Use 2 classes but heavily imbalanced (95% class 0, 5% class 1)
        y_true = [0] * 19 + [1] * 1  # Only 1 sample of class 1
        pred_prob = np.random.RandomState(42).uniform(0, 1, (20, 2))
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)  # Normalize

        thresholds = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert len(thresholds) == 2
        assert all(0.0 <= t <= 1.0 for t in thresholds)

    def test_extreme_multiclass_imbalance(self):
        """Test multiclass with extreme class imbalance."""
        # 98% class 0, 1% class 1, 1% class 2
        y_true = [0] * 98 + [1] + [2]
        np.random.seed(42)
        pred_prob = np.random.uniform(0, 1, (100, 3))
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)  # Normalize

        thresholds = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert len(thresholds) == 3
        assert all(0.0 <= t <= 1.0 for t in thresholds)

    def test_multiclass_with_zero_probabilities(self):
        """Test multiclass with some zero probability columns."""
        y_true = [0, 1, 2, 0, 1, 2]
        pred_prob = np.array(
            [
                [1.0, 0.0, 0.0],  # Only class 0 has probability
                [0.0, 1.0, 0.0],  # Only class 1 has probability
                [0.0, 0.0, 1.0],  # Only class 2 has probability
                [0.5, 0.5, 0.0],  # Classes 0,1 split probability
                [0.0, 0.5, 0.5],  # Classes 1,2 split probability
                [0.3, 0.3, 0.4],  # All classes have some probability
            ]
        )

        thresholds = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert len(thresholds) == 3
        assert all(0.0 <= t <= 1.0 for t in thresholds)


if __name__ == "__main__":
    pytest.main([__file__])
