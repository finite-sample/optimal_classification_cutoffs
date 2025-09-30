"""Tests for generalized expected F-beta functionality."""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.expected import (
    dinkelbach_expected_fbeta_binary,
    dinkelbach_expected_fbeta_multilabel,
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
