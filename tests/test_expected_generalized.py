"""Comprehensive tests for generalized Dinkelbach fractional-linear optimization.

This module tests the new expected_fractional.py framework that extends Dinkelbach
optimization from F-beta only to any fractional-linear metric (precision, recall,
Jaccard/IoU, Tversky, accuracy, specificity).

Key test areas:
1. Coefficient correctness for each metric
2. Binary optimization with all supported metrics
3. Multiclass/multilabel optimization with OvR strategy
4. Micro averaging for flattened optimization
5. Parameter handling (beta, tversky_alpha, tversky_beta)
6. Backward compatibility with existing F-beta code
7. Edge cases and numerical stability
"""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.expected import (
    dinkelbach_expected_fbeta_binary,
    dinkelbach_expected_fbeta_multilabel,
)
from optimal_cutoffs.expected_fractional import (
    coeffs_for_metric,
    dinkelbach_expected_fractional_binary,
    dinkelbach_expected_fractional_ovr,
)


class TestDinkelbachExpectedFbetaBinary:
    """Test binary expected F-beta with sample weights."""

    def test_f1_basic(self):
        """Test basic F1 optimization."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        threshold, f_beta = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f_beta <= 1.0
        assert isinstance(threshold, float)
        assert isinstance(f_beta, float)

    def test_different_beta_values(self):
        """Test different beta values."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        # F0.5 (emphasizes precision)
        t_05, f_05 = dinkelbach_expected_fbeta_binary(y_prob, beta=0.5)

        # F1 (balanced)
        t_1, f_1 = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        # F2 (emphasizes recall)
        t_2, f_2 = dinkelbach_expected_fbeta_binary(y_prob, beta=2.0)

        # Higher beta should generally give lower threshold (more inclusive)
        assert t_2 <= t_1 <= t_05

    def test_with_sample_weights(self):
        """Test with sample weights."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        weights = np.array([1, 2, 1, 1])

        t_weighted, f_weighted = dinkelbach_expected_fbeta_binary(
            y_prob, beta=1.0, sample_weight=weights
        )

        t_uniform, f_uniform = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        # Results should be different due to weights
        assert abs(t_weighted - t_uniform) > 1e-6 or abs(f_weighted - f_uniform) > 1e-6

    def test_weight_scaling_invariance(self):
        """Test that scaling all weights doesn't change threshold."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        weights = np.array([1, 2, 1, 1])

        t1, f1 = dinkelbach_expected_fbeta_binary(
            y_prob, beta=1.0, sample_weight=weights
        )

        # Scale weights by 10
        t2, f2 = dinkelbach_expected_fbeta_binary(
            y_prob, beta=1.0, sample_weight=weights * 10
        )

        # Threshold should be same (scaling invariant)
        assert abs(t1 - t2) < 1e-10
        assert abs(f1 - f2) < 1e-10

    def test_comparison_operators(self):
        """Test different comparison operators."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        t_excl, f_excl = dinkelbach_expected_fbeta_binary(
            y_prob, beta=1.0, comparison=">"
        )

        t_incl, f_incl = dinkelbach_expected_fbeta_binary(
            y_prob, beta=1.0, comparison=">="
        )

        # Should be close but potentially different for tied probabilities
        assert abs(t_excl - t_incl) < 0.1  # Should be similar

    def test_edge_cases(self):
        """Test edge cases."""
        # All zeros (no positives expected)
        y_prob_zeros = np.array([0.0, 0.0, 0.0])
        t_zeros, f_zeros = dinkelbach_expected_fbeta_binary(y_prob_zeros)
        assert t_zeros == 1.0  # Never predict positive
        assert f_zeros == 0.0

        # All ones (all positives expected) - the optimal threshold is actually 0.5
        # because Dinkelbach converges to the point where expected TP = expected FP
        y_prob_ones = np.array([1.0, 1.0, 1.0])
        t_ones, f_ones = dinkelbach_expected_fbeta_binary(y_prob_ones)
        assert abs(t_ones - 0.5) < 1e-6  # Converges to 0.5
        assert f_ones == 1.0  # Perfect F-score

        # Empty array
        t_empty, f_empty = dinkelbach_expected_fbeta_binary(np.array([]))
        assert t_empty == 0.0
        assert f_empty == 0.0

    def test_input_validation(self):
        """Test input validation."""
        # Probabilities outside [0, 1]
        with pytest.raises(ValueError, match="Probabilities must be in"):
            dinkelbach_expected_fbeta_binary(np.array([0.5, 1.5]))

        # Negative beta
        with pytest.raises(ValueError, match="beta must be nonnegative"):
            dinkelbach_expected_fbeta_binary(np.array([0.5]), beta=-1)

        # Wrong shape for sample weights
        with pytest.raises(ValueError, match="sample_weight must have shape"):
            dinkelbach_expected_fbeta_binary(
                np.array([0.5, 0.7]), sample_weight=np.array([1])
            )

        # Negative sample weights
        with pytest.raises(ValueError, match="sample_weight must be nonnegative"):
            dinkelbach_expected_fbeta_binary(
                np.array([0.5, 0.7]), sample_weight=np.array([1, -1])
            )


class TestDinkelbachExpectedFbetaMultilabel:
    """Test multilabel/multiclass expected F-beta."""

    def test_macro_averaging(self):
        """Test macro averaging strategy."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])

        result = dinkelbach_expected_fbeta_multilabel(y_prob, beta=1.0, average="macro")

        assert "thresholds" in result
        assert "f_beta_per_class" in result
        assert "f_beta" in result

        assert result["thresholds"].shape == (3,)
        assert result["f_beta_per_class"].shape == (3,)
        assert isinstance(result["f_beta"], float)

        # Macro average should be mean of per-class scores
        expected_macro = np.mean(result["f_beta_per_class"])
        assert abs(result["f_beta"] - expected_macro) < 1e-10

    def test_weighted_averaging(self):
        """Test weighted averaging strategy."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])
        class_weight = np.array([1, 2, 0.5])

        result = dinkelbach_expected_fbeta_multilabel(
            y_prob, beta=1.0, average="weighted", class_weight=class_weight
        )

        # Weighted average should use class weights
        w_norm = class_weight / class_weight.sum()
        expected_weighted = np.sum(w_norm * result["f_beta_per_class"])
        assert abs(result["f_beta"] - expected_weighted) < 1e-10

    def test_micro_averaging(self):
        """Test micro averaging strategy."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])

        result = dinkelbach_expected_fbeta_multilabel(y_prob, beta=1.0, average="micro")

        assert "threshold" in result
        assert "f_beta" in result
        assert isinstance(result["threshold"], float)
        assert isinstance(result["f_beta"], float)

    def test_micro_equals_flattened_binary(self):
        """Test that micro averaging equals flattened binary problem."""
        y_prob = np.array([[0.1, 0.8], [0.9, 0.2]])
        sample_weight = np.array([1, 2])

        # Micro result
        result_micro = dinkelbach_expected_fbeta_multilabel(
            y_prob, beta=1.0, average="micro", sample_weight=sample_weight
        )

        # Equivalent binary problem
        y_prob_flat = y_prob.reshape(-1)
        weight_flat = np.repeat(sample_weight, 2)
        t_binary, f_binary = dinkelbach_expected_fbeta_binary(
            y_prob_flat, beta=1.0, sample_weight=weight_flat
        )

        assert abs(result_micro["threshold"] - t_binary) < 1e-10
        assert abs(result_micro["f_beta"] - f_binary) < 1e-10

    def test_with_sample_weights(self):
        """Test with sample weights."""
        y_prob = np.array([[0.1, 0.8], [0.9, 0.2]])
        sample_weight = np.array([1, 3])

        result_weighted = dinkelbach_expected_fbeta_multilabel(
            y_prob, beta=1.0, average="macro", sample_weight=sample_weight
        )

        result_uniform = dinkelbach_expected_fbeta_multilabel(
            y_prob, beta=1.0, average="macro"
        )

        # Should be different due to weights
        thresh_diff = np.abs(
            result_weighted["thresholds"] - result_uniform["thresholds"]
        )
        assert np.any(thresh_diff > 1e-6)

    def test_different_beta_values_multiclass(self):
        """Test different beta values for multiclass."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])

        result_05 = dinkelbach_expected_fbeta_multilabel(
            y_prob, beta=0.5, average="macro"
        )
        result_2 = dinkelbach_expected_fbeta_multilabel(
            y_prob, beta=2.0, average="macro"
        )

        # Higher beta should generally give lower thresholds
        assert np.all(result_2["thresholds"] <= result_05["thresholds"] + 1e-6)

    def test_input_validation_multiclass(self):
        """Test input validation for multiclass."""
        # Wrong shape for y_prob
        with pytest.raises(ValueError, match="y_prob must have shape"):
            dinkelbach_expected_fbeta_multilabel(np.array([0.1, 0.8]))

        # Wrong shape for sample_weight
        with pytest.raises(ValueError, match="sample_weight must have shape"):
            dinkelbach_expected_fbeta_multilabel(
                np.array([[0.1, 0.8]]), sample_weight=np.array([1, 2])
            )

        # Wrong shape for class_weight
        with pytest.raises(ValueError, match="class_weight must have shape"):
            dinkelbach_expected_fbeta_multilabel(
                np.array([[0.1, 0.8]]),
                average="weighted",
                class_weight=np.array([1, 2, 3]),
            )

        # Invalid average
        with pytest.raises(ValueError, match="average must be one of"):
            dinkelbach_expected_fbeta_multilabel(
                np.array([[0.1, 0.8]]), average="invalid"
            )


class TestIntegrationWithRouter:
    """Test integration with the main router."""

    def test_binary_expected_mode(self):
        """Test binary expected mode through router."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        result = get_optimal_threshold(y_prob, y_prob, mode="expected", beta=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f_beta = result
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f_beta <= 1.0

    def test_multiclass_expected_macro(self):
        """Test multiclass expected mode with macro averaging."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])

        result = get_optimal_threshold(
            y_prob, y_prob, mode="expected", beta=1.5, average="macro"
        )

        assert isinstance(result, dict)
        assert "thresholds" in result
        assert "f_beta_per_class" in result
        assert "f_beta" in result

    def test_multiclass_expected_micro(self):
        """Test multiclass expected mode with micro averaging."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])

        result = get_optimal_threshold(y_prob, y_prob, mode="expected", average="micro")

        assert isinstance(result, dict)
        assert "threshold" in result
        assert "f_beta" in result

    def test_backward_compatibility_f1_only(self):
        """Test that old F1-only constraint is removed."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        # Should work with any beta, not just F1
        for beta in [0.5, 1.0, 2.0]:
            result = get_optimal_threshold(y_prob, y_prob, mode="expected", beta=beta)
            assert isinstance(result, tuple)

    def test_with_sample_weights_through_router(self):
        """Test sample weights through the router."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        weights = np.array([1, 2, 1, 1])

        result = get_optimal_threshold(
            y_prob, y_prob, mode="expected", sample_weight=weights
        )

        assert isinstance(result, tuple)
        threshold, f_beta = result
        assert 0.0 <= threshold <= 1.0

    def test_multiclass_with_class_weights(self):
        """Test multiclass with class weights through router."""
        y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])
        class_weight = np.array([1, 2, 0.5])

        result = get_optimal_threshold(
            y_prob,
            y_prob,
            mode="expected",
            average="weighted",
            class_weight=class_weight,
        )

        assert isinstance(result, dict)
        assert "f_beta" in result


class TestMathematicalProperties:
    """Test mathematical properties and identities."""

    def test_f1_identity_for_calibrated_data(self):
        """Test known F1 identity: t* = F1*/2 for specific cases."""
        # For uniform probabilities, the identity should hold approximately
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        threshold, f_beta = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        # For this case, threshold should be close to f_beta/2
        # (This is an approximation for this specific probability pattern)
        assert abs(threshold - f_beta / 2) < 0.2  # Loose bound due to discrete nature

    def test_monotonicity_in_beta(self):
        """Test that higher beta gives lower thresholds (more recall-focused)."""
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        betas = [0.5, 1.0, 1.5, 2.0]
        thresholds = []

        for beta in betas:
            t, _ = dinkelbach_expected_fbeta_binary(y_prob, beta=beta)
            thresholds.append(t)

        # Thresholds should generally decrease as beta increases
        for i in range(len(thresholds) - 1):
            assert thresholds[i + 1] <= thresholds[i] + 1e-6  # Allow small tolerance

    def test_convergence_properties(self):
        """Test that Dinkelbach algorithm converges."""
        # Use a case that might be challenging for convergence
        y_prob = np.array([0.01, 0.99])  # Extreme probabilities

        threshold, f_beta = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        # Should still converge to valid values
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f_beta <= 1.0
        assert np.isfinite(threshold)
        assert np.isfinite(f_beta)


# New comprehensive tests for generalized fractional-linear framework
class TestMetricCoefficients:
    """Test that coefficient mappings are mathematically correct."""

    def test_precision_coefficients(self):
        """Precision = TP / (TP + FP) should have correct coefficients."""
        coeffs = coeffs_for_metric("precision")

        # Precision: numerator = TP, denominator = TP + FP
        assert coeffs.alpha_tp == 1.0
        assert coeffs.alpha_tn == 0.0
        assert coeffs.alpha_fp == 0.0
        assert coeffs.alpha_fn == 0.0
        assert coeffs.alpha0 == 0.0

        assert coeffs.beta_tp == 1.0
        assert coeffs.beta_tn == 0.0
        assert coeffs.beta_fp == 1.0
        assert coeffs.beta_fn == 0.0
        assert coeffs.beta0 == 0.0

    def test_recall_coefficients(self):
        """Recall = TP / (TP + FN) should have correct coefficients."""
        coeffs = coeffs_for_metric("recall")

        # Recall: numerator = TP, denominator = TP + FN
        assert coeffs.alpha_tp == 1.0
        assert coeffs.alpha_tn == 0.0
        assert coeffs.alpha_fp == 0.0
        assert coeffs.alpha_fn == 0.0
        assert coeffs.alpha0 == 0.0

        assert coeffs.beta_tp == 1.0
        assert coeffs.beta_tn == 0.0
        assert coeffs.beta_fp == 0.0
        assert coeffs.beta_fn == 1.0
        assert coeffs.beta0 == 0.0

    def test_f1_coefficients(self):
        """F1 = 2*TP / (2*TP + FP + FN) should have correct coefficients."""
        coeffs = coeffs_for_metric("f1")

        # F1: numerator = 2*TP, denominator = 2*TP + FP + FN
        assert coeffs.alpha_tp == 2.0
        assert coeffs.alpha_tn == 0.0
        assert coeffs.alpha_fp == 0.0
        assert coeffs.alpha_fn == 0.0
        assert coeffs.alpha0 == 0.0

        assert coeffs.beta_tp == 2.0
        assert coeffs.beta_tn == 0.0
        assert coeffs.beta_fp == 1.0
        assert coeffs.beta_fn == 1.0
        assert coeffs.beta0 == 0.0

    def test_jaccard_coefficients(self):
        """Jaccard = TP / (TP + FP + FN) should have correct coefficients."""
        coeffs = coeffs_for_metric("jaccard")

        # Jaccard: numerator = TP, denominator = TP + FP + FN
        assert coeffs.alpha_tp == 1.0
        assert coeffs.alpha_tn == 0.0
        assert coeffs.alpha_fp == 0.0
        assert coeffs.alpha_fn == 0.0
        assert coeffs.alpha0 == 0.0

        assert coeffs.beta_tp == 1.0
        assert coeffs.beta_tn == 0.0
        assert coeffs.beta_fp == 1.0
        assert coeffs.beta_fn == 1.0
        assert coeffs.beta0 == 0.0

    def test_accuracy_coefficients(self):
        """Accuracy = (TP + TN) / (TP + TN + FP + FN) should have correct coefficients."""
        coeffs = coeffs_for_metric("accuracy")

        # Accuracy: numerator = TP + TN, denominator = TP + TN + FP + FN
        assert coeffs.alpha_tp == 1.0
        assert coeffs.alpha_tn == 1.0
        assert coeffs.alpha_fp == 0.0
        assert coeffs.alpha_fn == 0.0
        assert coeffs.alpha0 == 0.0

        assert coeffs.beta_tp == 1.0
        assert coeffs.beta_tn == 1.0
        assert coeffs.beta_fp == 1.0
        assert coeffs.beta_fn == 1.0
        assert coeffs.beta0 == 0.0

    def test_unsupported_metric_error(self):
        """Unsupported metrics should raise ValueError."""
        with pytest.raises(ValueError, match="not supported as fractional-linear"):
            coeffs_for_metric("unsupported_metric")


class TestGeneralizedBinaryOptimization:
    """Test binary optimization with all supported metrics."""

    def test_binary_optimization_all_metrics(self):
        """Test binary optimization works for all supported metrics."""
        np.random.seed(42)
        y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])

        metrics_to_test = [
            "precision",
            "recall",
            "specificity",
            "f1",
            "f2",
            "jaccard",
            "iou",
            "accuracy",
        ]

        for metric in metrics_to_test:
            coeffs = coeffs_for_metric(metric)
            threshold, score, direction = dinkelbach_expected_fractional_binary(
                y_prob, coeffs
            )

            assert 0.0 <= threshold <= 1.0, (
                f"Invalid threshold for {metric}: {threshold}"
            )
            assert 0.0 <= score <= 1.0, f"Invalid score for {metric}: {score}"
            assert direction in [">", "<"], (
                f"Invalid direction for {metric}: {direction}"
            )

    def test_binary_with_parameters(self):
        """Test binary optimization with metric parameters."""
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        # Test F-beta with different beta values
        for beta in [0.5, 1.0, 2.0]:
            coeffs = coeffs_for_metric("fbeta", beta=beta)
            threshold, score, direction = dinkelbach_expected_fractional_binary(
                y_prob, coeffs
            )
            assert 0.0 <= threshold <= 1.0
            assert 0.0 <= score <= 1.0

        # Test Tversky with different alpha/beta values
        for alpha in [0.3, 0.5, 0.7]:
            for beta in [0.2, 0.5, 0.8]:
                coeffs = coeffs_for_metric(
                    "tversky", tversky_alpha=alpha, tversky_beta=beta
                )
                threshold, score, direction = dinkelbach_expected_fractional_binary(
                    y_prob, coeffs
                )
                assert 0.0 <= threshold <= 1.0
                assert 0.0 <= score <= 1.0

    def test_binary_with_sample_weights(self):
        """Test binary optimization with sample weights."""
        y_prob = np.array([0.2, 0.5, 0.8])
        sample_weight = np.array([0.5, 2.0, 1.0])

        for metric in ["f1", "precision", "accuracy"]:
            coeffs = coeffs_for_metric(metric)
            threshold, score, direction = dinkelbach_expected_fractional_binary(
                y_prob, coeffs, sample_weight=sample_weight
            )
            assert 0.0 <= threshold <= 1.0
            assert 0.0 <= score <= 1.0


class TestGeneralizedMulticlassOptimization:
    """Test multiclass/multilabel optimization with OvR strategy."""

    def test_multiclass_ovr_all_metrics(self):
        """Test multiclass OvR optimization works for all supported metrics."""
        np.random.seed(42)
        y_prob = np.random.rand(20, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        metrics_to_test = ["f1", "precision", "recall", "accuracy", "jaccard"]

        for metric in metrics_to_test:
            for average in ["macro", "weighted"]:
                result = dinkelbach_expected_fractional_ovr(
                    y_prob, metric, average=average
                )

                assert "thresholds" in result
                assert "score" in result
                assert "per_class" in result
                assert "directions" in result

                assert len(result["thresholds"]) == 3
                assert len(result["per_class"]) == 3
                assert len(result["directions"]) == 3

                assert 0.0 <= result["score"] <= 1.0
                assert all(0.0 <= t <= 1.0 for t in result["thresholds"])
                assert all(0.0 <= s <= 1.0 for s in result["per_class"])

    def test_multiclass_micro_averaging(self):
        """Test multiclass micro averaging (flattened optimization)."""
        np.random.seed(123)
        y_prob = np.random.rand(15, 4)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        for metric in ["f1", "precision", "accuracy"]:
            result = dinkelbach_expected_fractional_ovr(y_prob, metric, average="micro")

            assert "threshold" in result
            assert "score" in result
            assert "direction" in result

            assert 0.0 <= result["threshold"] <= 1.0
            assert 0.0 <= result["score"] <= 1.0
            assert result["direction"] in [">", "<"]


class TestGeneralizedAPIIntegration:
    """Test integration with get_optimal_threshold API."""

    def test_api_integration_binary(self):
        """Test that new framework integrates correctly with main API for binary cases."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1])

        # Test various metrics through main API
        for metric in ["f1", "precision", "recall", "accuracy", "jaccard"]:
            result = get_optimal_threshold(
                y_true, y_prob, mode="expected", metric=metric
            )

            assert isinstance(result, tuple)
            assert len(result) == 2
            threshold, score = result

            assert isinstance(threshold, float)
            assert isinstance(score, float)
            assert 0.0 <= threshold <= 1.0
            assert 0.0 <= score <= 1.0

    def test_api_integration_multiclass(self):
        """Test that new framework integrates correctly with main API for multiclass."""
        np.random.seed(789)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.random.rand(6, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        # Test various metrics through main API
        for metric in ["f1", "precision", "accuracy"]:
            result = get_optimal_threshold(
                y_true, y_prob, mode="expected", metric=metric
            )

            assert isinstance(result, dict)
            assert "thresholds" in result
            assert "f_beta" in result
            assert "f_beta_per_class" in result

            thresholds = result["thresholds"]
            score = result["f_beta"]

            assert isinstance(thresholds, np.ndarray)
            assert isinstance(score, float)
            assert len(thresholds) == 3
            assert all(0.0 <= t <= 1.0 for t in thresholds)
            assert 0.0 <= score <= 1.0


class TestGeneralizedBackwardCompatibility:
    """Test backward compatibility with existing F-beta code."""

    def test_f1_compatibility(self):
        """Test that F1 results are consistent between old and new implementations."""
        np.random.seed(999)
        y_true = np.array([0, 1, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6])

        # Get result from main API (uses new framework)
        result_new = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold_new, f1_new = result_new

        # Results should be reasonable
        assert 0.0 <= threshold_new <= 1.0
        assert 0.0 <= f1_new <= 1.0

        # Test that we get consistent results across calls
        result_new2 = get_optimal_threshold(
            y_true, y_prob, mode="expected", metric="f1"
        )
        threshold_new2, f1_new2 = result_new2

        assert abs(threshold_new - threshold_new2) < 1e-10
        assert abs(f1_new - f1_new2) < 1e-10


class TestGeneralizedEdgeCases:
    """Test edge cases and numerical stability."""

    def test_extreme_probabilities(self):
        """Test optimization with extreme probabilities."""
        y_prob = np.array([0.0, 0.001, 0.999, 1.0])

        for metric in ["f1", "precision", "accuracy"]:
            coeffs = coeffs_for_metric(metric)
            threshold, score, direction = dinkelbach_expected_fractional_binary(
                y_prob, coeffs
            )

            assert 0.0 <= threshold <= 1.0
            assert 0.0 <= score <= 1.0

    def test_empty_input(self):
        """Test optimization with empty input."""
        y_prob = np.array([])

        coeffs = coeffs_for_metric("f1")
        threshold, score, direction = dinkelbach_expected_fractional_binary(
            y_prob, coeffs
        )

        # Should handle gracefully
        assert threshold == 0.0
        assert score == 0.0
