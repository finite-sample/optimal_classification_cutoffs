"""Tests for multiclass Bayes functionality."""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.bayes import (
    bayes_decision_from_utility_matrix,
    bayes_threshold_from_costs_scalar,
    bayes_thresholds_from_costs_vector,
)


class TestBayesDecisionFromUtilityMatrix:
    """Test multiclass Bayes decisions from utility matrices."""

    def test_standard_classification(self):
        """Test standard K-way classification with identity utility matrix."""
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
        U = np.eye(3)  # Correct prediction = +1, wrong = 0

        decisions = bayes_decision_from_utility_matrix(y_prob, U)
        expected = np.array([0, 1, 2])  # Argmax of each row

        np.testing.assert_array_equal(decisions, expected)

    def test_with_abstain_option(self):
        """Test classification with abstain option."""
        y_prob = np.array([[0.4, 0.3, 0.3], [0.1, 0.8, 0.1]])
        # Identity matrix plus abstain row with moderate utility
        U = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.6, 0.6, 0.6]])
        labels = [0, 1, 2, -1]  # -1 for abstain

        decisions = bayes_decision_from_utility_matrix(y_prob, U, labels=labels)

        # First sample: max prob is 0.4, abstain gives 0.6*0.4 + 0.6*0.3 + 0.6*0.3 = 0.6
        # Class 0 gives 1*0.4 = 0.4, so should abstain
        assert decisions[0] == -1

        # Second sample: class 1 has prob 0.8, gives utility 0.8 > 0.6, so predict class 1
        assert decisions[1] == 1

    def test_return_scores(self):
        """Test returning expected utility scores."""
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        U = np.eye(3)

        decisions, scores = bayes_decision_from_utility_matrix(
            y_prob, U, return_scores=True
        )

        # Scores should be the probabilities themselves (since U is identity)
        np.testing.assert_array_almost_equal(scores, y_prob)
        np.testing.assert_array_equal(decisions, [0, 1])

    def test_custom_labels(self):
        """Test with custom decision labels."""
        y_prob = np.array([[0.7, 0.2, 0.1]])
        U = np.eye(3)
        labels = ["A", "B", "C"]

        decisions = bayes_decision_from_utility_matrix(y_prob, U, labels=labels)
        assert decisions[0] == "A"

    def test_input_validation(self):
        """Test input validation."""
        y_prob = np.array([[0.7, 0.2, 0.1]])

        # Wrong number of columns in U
        U_wrong = np.array([[1, 0], [0, 1]])
        with pytest.raises(
            ValueError, match="U has 2 columns but y_prob has 3 classes"
        ):
            bayes_decision_from_utility_matrix(y_prob, U_wrong)

        # Wrong shape for y_prob
        with pytest.raises(ValueError, match="y_prob must have shape"):
            bayes_decision_from_utility_matrix(np.array([0.7, 0.2, 0.1]), np.eye(3))

        # Wrong number of labels
        U = np.eye(3)
        with pytest.raises(ValueError, match="labels must have length"):
            bayes_decision_from_utility_matrix(y_prob, U, labels=["A", "B"])


class TestBayesThresholdsFromCostsVector:
    """Test per-class Bayes thresholds for OvR."""

    def test_equal_costs_different_fn(self):
        """Test with equal FP costs but different FN costs."""
        fp_cost = [-1, -1, -1]
        fn_cost = [-5, -3, -2]

        thresholds = bayes_thresholds_from_costs_vector(fp_cost, fn_cost)

        # Expected: τ_k = |fp| / (|fp| + |fn|) = 1 / (1 + |fn|)
        expected = np.array([1 / 6, 1 / 4, 1 / 3])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_with_benefits(self):
        """Test with benefits for correct predictions."""
        fp_cost = [-1, -1, -1]
        fn_cost = [-5, -3, -2]
        tp_benefit = [2, 2, 2]
        tn_benefit = [1, 1, 1]

        thresholds = bayes_thresholds_from_costs_vector(
            fp_cost, fn_cost, tp_benefit, tn_benefit
        )

        # τ_k = (tn - fp) / [(tn - fp) + (tp - fn)]
        # = (1 - (-1)) / [(1 - (-1)) + (2 - fn)]
        # = 2 / (2 + 2 - fn) = 2 / (4 + |fn|)
        expected = np.array([2 / 9, 2 / 7, 2 / 6])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_degenerate_cases(self):
        """Test degenerate cases where denominator is zero."""
        # Case where tp - fn <= 0: should get tau = 1 (never predict positive)
        fp_cost = [-1, -1]
        fn_cost = [-2, -3]
        tp_benefit = [1, 1]  # tp - fn = 1 - (-2) = 3, 1 - (-3) = 4

        thresholds = bayes_thresholds_from_costs_vector(fp_cost, fn_cost, tp_benefit)

        # τ_k = (tn - fp) / [(tn - fp) + (tp - fn)]
        # = (0 - (-1)) / [(0 - (-1)) + (1 - (-2))] = 1 / (1 + 3) = 1/4
        # = (0 - (-1)) / [(0 - (-1)) + (1 - (-3))] = 1 / (1 + 4) = 1/5
        expected = np.array([1 / 4, 1 / 5])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_always_positive_case(self):
        """Test case where should always predict positive."""
        fp_cost = [-1, -1]
        fn_cost = [-1, -1]
        tp_benefit = [10, 10]  # Very high benefit for TP

        thresholds = bayes_thresholds_from_costs_vector(fp_cost, fn_cost, tp_benefit)

        # τ_k = (tn - fp) / [(tn - fp) + (tp - fn)]
        # = (0 - (-1)) / [(0 - (-1)) + (10 - (-1))] = 1 / (1 + 11) = 1/12
        expected = np.array([1 / 12, 1 / 12])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_shape_validation(self):
        """Test that all arrays must have same shape."""
        fp_cost = [-1, -1, -1]
        fn_cost = [-5, -3]  # Wrong length

        with pytest.raises(
            ValueError, match="All cost/benefit arrays must have the same shape"
        ):
            bayes_thresholds_from_costs_vector(fp_cost, fn_cost)

    def test_clipping_to_unit_interval(self):
        """Test that thresholds are clipped to [0, 1]."""
        # Create scenario that would give threshold > 1
        fp_cost = [10]  # Large positive cost (becomes negative utility)
        fn_cost = [-1]

        thresholds = bayes_thresholds_from_costs_vector(fp_cost, fn_cost)

        # Should be clipped to [0, 1]
        assert 0.0 <= thresholds[0] <= 1.0


class TestBayesThresholdFromCostsScalar:
    """Test scalar Bayes threshold for backward compatibility."""

    def test_equivalent_to_vector_version(self):
        """Test that scalar version matches vector version for single class."""
        fp_cost = -1.0
        fn_cost = -5.0

        scalar_threshold = bayes_threshold_from_costs_scalar(fp_cost, fn_cost)
        vector_threshold = bayes_thresholds_from_costs_vector([fp_cost], [fn_cost])[0]

        assert abs(scalar_threshold - vector_threshold) < 1e-12

    def test_comparison_operators(self):
        """Test comparison operator handling."""
        fp_cost = -1.0
        fn_cost = -5.0

        threshold_excl = bayes_threshold_from_costs_scalar(
            fp_cost, fn_cost, comparison=">"
        )
        threshold_incl = bayes_threshold_from_costs_scalar(
            fp_cost, fn_cost, comparison=">="
        )

        # Inclusive should be slightly smaller for edge case handling
        assert threshold_incl <= threshold_excl

    def test_with_benefits(self):
        """Test with benefits included."""
        threshold = bayes_threshold_from_costs_scalar(
            fp_cost=-1, fn_cost=-5, tp_benefit=2, tn_benefit=1
        )

        # (tn - fp) / [(tn - fp) + (tp - fn)] = (1 - (-1)) / [2 + (2 - (-5))] = 2/9
        expected = 2.0 / 9.0
        assert abs(threshold - expected) < 1e-10


class TestIntegrationWithRouter:
    """Test integration of Bayes functionality with main router."""

    def test_multiclass_bayes_with_utility_matrix(self):
        """Test mode='bayes' with utility_matrix."""
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        U = np.eye(3)

        decisions = get_optimal_threshold(None, y_prob, utility_matrix=U, mode="bayes")

        expected = np.array([0, 1])
        np.testing.assert_array_equal(decisions, expected)

    def test_multiclass_bayes_with_cost_vectors(self):
        """Test mode='bayes' with per-class cost vectors."""
        y_prob = np.random.rand(5, 3)  # 5 samples, 3 classes
        utility = {
            "fp": [-1, -2, -1],
            "fn": [-5, -3, -4],
        }

        thresholds = get_optimal_threshold(None, y_prob, utility=utility, mode="bayes")

        assert thresholds.shape == (3,)
        assert np.all((thresholds >= 0) & (thresholds <= 1))

    def test_binary_bayes_backward_compatibility(self):
        """Test that binary Bayes still works as before."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        utility = {"fp": -1, "fn": -5}

        threshold = get_optimal_threshold(None, y_prob, utility=utility, mode="bayes")

        expected = 1.0 / 6.0  # Classic result
        assert abs(threshold - expected) < 1e-10

    def test_error_messages(self):
        """Test clear error messages for invalid usage."""
        y_prob = np.array([[0.7, 0.2, 0.1]])

        # No utility specified
        with pytest.raises(
            ValueError,
            match="mode='bayes' requires utility parameter or utility_matrix",
        ):
            get_optimal_threshold(None, y_prob, mode="bayes")

        # Multiclass without proper vectors
        with pytest.raises(
            ValueError, match="Multiclass Bayes requires 'fp' and 'fn' as arrays"
        ):
            get_optimal_threshold(None, y_prob, utility={"fp": -1}, mode="bayes")
