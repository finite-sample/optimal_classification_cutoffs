"""Comprehensive edge case testing for boundary conditions and extreme scenarios."""

import warnings

import numpy as np
import pytest

from optimal_cutoffs import get_confusion_matrix, get_optimal_threshold
from optimal_cutoffs.optimizers import _optimal_threshold_piecewise
from optimal_cutoffs.wrapper import ThresholdOptimizer


class TestLabelDistributionEdgeCases:
    """Test extreme label distributions."""

    def test_all_zeros_labels(self):
        """Test with all negative labels."""
        labels = np.array([0, 0, 0, 0, 0])
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Should handle gracefully - return default threshold
        threshold = _optimal_threshold_piecewise(labels, probabilities, "f1")
        assert threshold == 0.5  # Default for degenerate case

        # Test with get_optimal_threshold
        threshold = get_optimal_threshold(labels, probabilities, "accuracy")
        assert 0 <= threshold <= 1

        # Confusion matrix should be valid
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp == 0  # No true positives possible
        assert fn == 0  # No false negatives possible
        assert tn + fp == len(labels)  # All samples are negative

    def test_all_ones_labels(self):
        """Test with all positive labels."""
        labels = np.array([1, 1, 1, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Should handle gracefully - return default threshold
        threshold = _optimal_threshold_piecewise(labels, probabilities, "f1")
        assert threshold == 0.5  # Default for degenerate case

        # Test with get_optimal_threshold
        threshold = get_optimal_threshold(labels, probabilities, "recall")
        assert 0 <= threshold <= 1

        # Confusion matrix should be valid
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tn == 0  # No true negatives possible
        assert fp == 0  # No false positives possible
        assert tp + fn == len(labels)  # All samples are positive

    def test_single_positive_in_negatives(self):
        """Test extreme class imbalance - one positive in many negatives."""
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        probabilities = np.array(
            [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.95]
        )

        # Should find a reasonable threshold
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0 <= threshold <= 1

        # The optimal threshold should likely be between 0.5 and 0.95
        # to capture the single positive example
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)

        # Should be able to achieve some reasonable performance
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        # At least one of precision or recall should be reasonable
        assert precision > 0 or recall > 0

    def test_perfectly_balanced_labels(self):
        """Test perfectly balanced dataset."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold = get_optimal_threshold(labels, probabilities, metric)
            assert 0 <= threshold <= 1

            # Should achieve reasonable performance
            score = self._compute_metric_score(labels, probabilities, threshold, metric)
            assert score > 0, f"Zero score for {metric}"

    def test_extreme_imbalance_99_to_1(self):
        """Test extreme imbalance (99:1 ratio)."""
        n_negative = 99
        n_positive = 1

        # Create extremely imbalanced dataset
        labels = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])

        # Probabilities slightly favor the positive class
        neg_probs = np.random.uniform(0.0, 0.4, n_negative)
        pos_probs = np.random.uniform(0.6, 1.0, n_positive)
        probabilities = np.concatenate([neg_probs, pos_probs])

        # Should handle extreme imbalance
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0 <= threshold <= 1

        # Test confusion matrix validity
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    def _compute_metric_score(self, labels, probabilities, threshold, metric):
        """Helper to compute metric score."""
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)

        if metric == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        elif metric == "precision":
            return tp / (tp + fp) if tp + fp > 0 else 0
        elif metric == "recall":
            return tp / (tp + fn) if tp + fn > 0 else 0
        elif metric == "f1":
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            return (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")


class TestProbabilityDistributionEdgeCases:
    """Test extreme probability distributions."""

    def test_all_identical_probabilities(self):
        """Test when all probabilities are identical."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Should handle gracefully - any threshold in the vicinity should work
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0 <= threshold <= 1

        # Test confusion matrix
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

        # Test with comparison operators - should show differences
        tp_gt, tn_gt, fp_gt, fn_gt = get_confusion_matrix(
            labels, probabilities, 0.5, comparison=">"
        )
        tp_gte, tn_gte, fp_gte, fn_gte = get_confusion_matrix(
            labels, probabilities, 0.5, comparison=">="
        )

        # Results should be different due to tie-breaking
        assert (tp_gt, tn_gt, fp_gt, fn_gt) != (tp_gte, tn_gte, fp_gte, fn_gte)

    def test_perfectly_separated_classes(self):
        """Test with no overlap between class probability distributions."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        # Should achieve perfect or near-perfect performance
        for metric in ["accuracy", "f1", "precision", "recall"]:
            threshold = get_optimal_threshold(labels, probabilities, metric)

            # Threshold should be reasonable for perfect separation
            # For recall, optimal threshold might be very low to capture all positives
            # For precision, optimal threshold might be high to avoid false positives
            if metric == "recall":
                assert 0.05 <= threshold <= 0.75, (
                    f"Unexpected threshold {threshold} for {metric}"
                )
            elif metric == "precision":
                assert 0.25 <= threshold <= 0.95, (
                    f"Unexpected threshold {threshold} for {metric}"
                )
            else:
                assert 0.25 <= threshold <= 0.75, (
                    f"Unexpected threshold {threshold} for {metric}"
                )

            # Should achieve high performance
            score = self._compute_metric_score(labels, probabilities, threshold, metric)
            assert score >= 0.9, (
                f"Low score {score} for {metric} with perfect separation"
            )

    def test_boundary_probabilities(self):
        """Test with probabilities at 0.0 and 1.0."""
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.0, 0.0, 1.0, 1.0])

        threshold = get_optimal_threshold(labels, probabilities, "accuracy")

        # Should achieve perfect accuracy
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        assert accuracy == 1.0, f"Expected perfect accuracy, got {accuracy}"

    def test_narrow_probability_range(self):
        """Test with probabilities in very narrow range."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.49, 0.50, 0.51, 0.49, 0.51, 0.50])

        # Should handle narrow ranges gracefully
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0.48 <= threshold <= 0.52

        # Should produce valid confusion matrix
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    def test_extreme_probability_skew(self):
        """Test with extremely skewed probability distribution."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Heavily skewed toward low probabilities
        probabilities = np.array([0.01, 0.02, 0.03, 0.04, 0.95, 0.96, 0.97, 0.98])

        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0 <= threshold <= 1

        # Should achieve good separation
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)

        # With such clear separation, should have good performance
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        assert precision > 0.5 and recall > 0.5

    def _compute_metric_score(self, labels, probabilities, threshold, metric):
        """Helper to compute metric score."""
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)

        if metric == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        elif metric == "precision":
            return tp / (tp + fp) if tp + fp > 0 else 0
        elif metric == "recall":
            return tp / (tp + fn) if tp + fn > 0 else 0
        elif metric == "f1":
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            return (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")


class TestNumericalEdgeCases:
    """Test numerical edge cases and extreme scenarios."""

    def test_very_small_datasets(self):
        """Test with minimal viable datasets."""
        # Single positive, single negative
        labels = np.array([0, 1])
        probabilities = np.array([0.3, 0.7])

        threshold = get_optimal_threshold(labels, probabilities, "accuracy")
        assert 0 <= threshold <= 1

        # Should be able to achieve perfect classification
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn == 2  # Perfect classification possible

        # Test with 3 samples
        labels = np.array([0, 1, 0])
        probabilities = np.array([0.2, 0.8, 0.3])

        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0 <= threshold <= 1

    def test_very_large_datasets(self):
        """Test with large datasets to check scalability."""
        n_samples = 10000

        # Create large balanced dataset
        labels = np.array([i % 2 for i in range(n_samples)])
        probabilities = np.random.beta(2, 2, n_samples)  # Bell-shaped distribution

        # Should handle large datasets efficiently
        import time

        start_time = time.time()
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        end_time = time.time()

        assert 0 <= threshold <= 1
        assert end_time - start_time < 5.0  # Should complete in reasonable time

        # Test confusion matrix
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn + fp + fn == n_samples

    def test_probabilities_near_machine_epsilon(self):
        """Test with probabilities very close to 0."""
        labels = np.array([0, 0, 1, 1])

        # Probabilities near machine epsilon
        eps = np.finfo(float).eps
        probabilities = np.array([eps, 2 * eps, 1 - 2 * eps, 1 - eps])

        # Should handle near-zero probabilities
        threshold = get_optimal_threshold(labels, probabilities, "accuracy")
        assert 0 <= threshold <= 1

        # Should achieve perfect separation
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        assert accuracy == 1.0

    def test_probabilities_differing_by_epsilon(self):
        """Test with probabilities that differ by machine epsilon."""
        labels = np.array([0, 1, 0, 1])

        eps = np.finfo(float).eps
        base_prob = 0.5
        probabilities = np.array(
            [base_prob - eps, base_prob + eps, base_prob - 2 * eps, base_prob + 2 * eps]
        )

        # Should handle tiny differences gracefully
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        assert 0 <= threshold <= 1

        # Should produce valid confusion matrix
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    def test_probability_precision_limits(self):
        """Test behavior at floating point precision limits."""
        labels = np.array([0, 1, 0, 1, 0, 1])

        # Create probabilities with varying precision
        probabilities = np.array(
            [
                0.1,
                0.1 + 1e-15,  # Near machine precision limit
                0.5,
                0.5 + 1e-14,
                0.9,
                0.9 - 1e-15,
            ]
        )

        # Should handle precision limits gracefully
        threshold = get_optimal_threshold(labels, probabilities, "accuracy")
        assert 0 <= threshold <= 1


class TestErrorConditionEdgeCases:
    """Test error conditions and their handling."""

    def test_nan_in_inputs_clear_error(self):
        """Test that NaN in inputs produces clear error messages."""
        # NaN in labels
        with pytest.raises(ValueError, match="true_labs contains NaN or infinite"):
            get_optimal_threshold([0, np.nan, 1], [0.1, 0.5, 0.9], "f1")

        # NaN in probabilities
        with pytest.raises(ValueError, match="pred_prob contains NaN or infinite"):
            get_optimal_threshold([0, 1, 0], [0.1, np.nan, 0.9], "f1")

    def test_inf_in_inputs_clear_error(self):
        """Test that infinity in inputs produces clear error messages."""
        # Inf in labels
        with pytest.raises(ValueError, match="true_labs contains NaN or infinite"):
            get_optimal_threshold([0, np.inf, 1], [0.1, 0.5, 0.9], "f1")

        # Inf in probabilities
        with pytest.raises(ValueError, match="pred_prob contains NaN or infinite"):
            get_optimal_threshold([0, 1, 0], [0.1, np.inf, 0.9], "f1")

    def test_empty_arrays_clear_error(self):
        """Test that empty arrays produce clear error messages."""
        with pytest.raises(ValueError, match="true_labs cannot be empty"):
            get_optimal_threshold([], [0.5], "f1")

        with pytest.raises(ValueError, match="pred_prob cannot be empty"):
            get_optimal_threshold([0], [], "f1")

    def test_mismatched_lengths_clear_error(self):
        """Test that mismatched array lengths produce clear error messages."""
        with pytest.raises(ValueError, match="Length mismatch"):
            get_optimal_threshold([0, 1], [0.5], "f1")

        with pytest.raises(ValueError, match="Length mismatch"):
            get_optimal_threshold([0], [0.1, 0.5], "f1")

    def test_invalid_data_types_clear_error(self):
        """Test that invalid data types produce clear error messages."""
        # String labels should be handled (converted to numeric if possible)
        # but non-numeric strings should fail with clear message
        with pytest.raises((ValueError, TypeError)):
            get_optimal_threshold(["a", "b", "c"], [0.1, 0.5, 0.9], "f1")

    def test_out_of_range_probabilities_clear_error(self):
        """Test that probabilities outside [0,1] produce clear errors."""
        with pytest.raises(ValueError, match="Probabilities must be in \\[0, 1\\]"):
            get_optimal_threshold([0, 1, 0], [-0.1, 0.5, 0.9], "f1")

        with pytest.raises(ValueError, match="Probabilities must be in \\[0, 1\\]"):
            get_optimal_threshold([0, 1, 0], [0.1, 0.5, 1.1], "f1")


class TestWrapperEdgeCases:
    """Test ThresholdOptimizer wrapper with edge cases."""

    def test_wrapper_with_edge_cases(self):
        """Test that the wrapper handles edge cases properly."""
        # Test with all same class
        labels = np.array([0, 0, 0, 0])
        probabilities = np.array([0.1, 0.3, 0.5, 0.7])

        optimizer = ThresholdOptimizer(objective="accuracy")

        # Should handle gracefully (might issue warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            optimizer.fit(labels, probabilities)

        assert optimizer.threshold_ is not None
        assert 0 <= optimizer.threshold_ <= 1

        # Predictions should work
        predictions = optimizer.predict(probabilities)
        assert len(predictions) == len(probabilities)
        assert all(isinstance(p, (bool, np.bool_)) for p in predictions)

    def test_wrapper_multiclass_edge_cases(self):
        """Test wrapper with multiclass edge cases."""
        # Single class multiclass (degenerate)
        labels = np.array([0, 0, 0])
        probabilities = np.array([[1.0], [1.0], [1.0]])

        optimizer = ThresholdOptimizer(objective="f1")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(labels, probabilities)

        assert optimizer.threshold_ is not None

        predictions = optimizer.predict(probabilities)
        assert len(predictions) == len(labels)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)


class TestPerformanceEdgeCases:
    """Test performance characteristics with edge cases."""

    def test_worst_case_performance(self):
        """Test performance with worst-case scenarios."""
        # Many unique probability values (worst case for brute force)
        n_samples = 1000
        labels = np.random.randint(0, 2, n_samples)
        probabilities = np.linspace(0, 1, n_samples)  # All unique values

        import time

        start_time = time.time()
        threshold = get_optimal_threshold(labels, probabilities, "f1")
        end_time = time.time()

        assert 0 <= threshold <= 1
        assert end_time - start_time < 10.0  # Should complete in reasonable time

    def test_memory_usage_edge_cases(self):
        """Test that memory usage stays reasonable with edge cases."""
        # Large dataset with many unique values
        n_samples = 5000
        labels = np.random.randint(0, 2, n_samples)
        probabilities = np.random.random(n_samples)

        # Should handle without excessive memory usage
        threshold = get_optimal_threshold(labels, probabilities, "accuracy")
        assert 0 <= threshold <= 1

        # Test confusion matrix doesn't explode memory
        tp, tn, fp, fn = get_confusion_matrix(labels, probabilities, threshold)
        assert tp + tn + fp + fn == n_samples
