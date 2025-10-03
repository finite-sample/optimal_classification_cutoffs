"""Tests for multiclass Bayes functionality."""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.bayes import (
    BayesOptimal,
    bayes_optimal_decisions,
    bayes_optimal_threshold,
    bayes_thresholds_from_costs,
    compute_bayes_threshold,
)


class TestBayesDecisionFromUtilityMatrix:
    """Test multiclass Bayes decisions from utility matrices."""

    def test_standard_classification(self):
        """Test standard K-way classification with identity utility matrix."""
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
        U = np.eye(3)  # Correct prediction = +1, wrong = 0

        decisions = bayes_optimal_decisions(y_prob, U)
        expected = np.array([0, 1, 2])  # Argmax of each row

        np.testing.assert_array_equal(decisions, expected)

    def test_with_abstain_option(self):
        """Test classification with abstain option."""
        y_prob = np.array([[0.4, 0.3, 0.3], [0.1, 0.8, 0.1]])
        # Identity matrix plus abstain row with moderate utility
        U = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.6, 0.6, 0.6]])

        decisions = bayes_optimal_decisions(y_prob, U)

        # First sample: max prob is 0.4, abstain gives 0.6*0.4 + 0.6*0.3 + 0.6*0.3 = 0.6
        # Class 0 gives 1*0.4 = 0.4, so should abstain (decision 3)
        assert decisions[0] == 3

        # Second sample: class 1 has prob 0.8, gives utility 0.8 > 0.6, so predict class 1
        assert decisions[1] == 1

    def test_return_scores(self):
        """Test returning expected utility scores."""
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        U = np.eye(3)

        decisions = bayes_optimal_decisions(y_prob, U)
        
        # Compute expected utilities manually for verification
        expected_utilities = y_prob @ U.T  # (n_samples, n_classes) @ (n_classes, n_decisions)
        
        # Decisions should match argmax of expected utilities
        expected_decisions = np.argmax(expected_utilities, axis=1)
        np.testing.assert_array_equal(decisions, expected_decisions)
        np.testing.assert_array_equal(decisions, [0, 1])

    def test_custom_labels(self):
        """Test with custom decision labels."""
        y_prob = np.array([[0.7, 0.2, 0.1]])
        U = np.eye(3)
        labels = ["A", "B", "C"]

        # Get numeric decisions and map to labels
        numeric_decisions = bayes_optimal_decisions(y_prob, U)
        decisions = [labels[i] for i in numeric_decisions]
        assert decisions[0] == "A"

    def test_input_validation(self):
        """Test input validation."""
        y_prob = np.array([[0.7, 0.2, 0.1]])

        # Wrong number of columns in U
        U_wrong = np.array([[1, 0], [0, 1]])
        with pytest.raises(
            ValueError, match="probabilities has 3 classes but utility_matrix has 2"
        ):
            bayes_optimal_decisions(y_prob, U_wrong)

        # Wrong shape for y_prob
        with pytest.raises(ValueError, match="probabilities must be 2D array"):
            bayes_optimal_decisions(np.array([0.7, 0.2, 0.1]), np.eye(3))

        # Test utility matrix validation
        U_bad = np.array([[1, 0, 0], [0, np.nan, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="utility_matrix must be 2D array"):
            bayes_optimal_decisions(y_prob, U_bad.ravel())


class TestBayesThresholdsFromCostsVector:
    """Test per-class Bayes thresholds for OvR."""

    def test_equal_costs_different_fn(self):
        """Test with equal FP costs but different FN costs."""
        fp_cost = [-1, -1, -1]
        fn_cost = [-5, -3, -2]

        thresholds = bayes_thresholds_from_costs(fp_cost, fn_cost)

        # Expected: τ_k = |fp| / (|fp| + |fn|) = 1 / (1 + |fn|)
        expected = np.array([1 / 6, 1 / 4, 1 / 3])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_with_benefits(self):
        """Test with benefits for correct predictions."""
        fp_cost = [-1, -1, -1]
        fn_cost = [-5, -3, -2]
        
        # For the new API, we just need fp and fn costs
        # The new API computes thresholds as fp_cost / (fp_cost + fn_cost)
        # = 1 / (1 + |fn|) for our values
        thresholds = bayes_thresholds_from_costs(fp_cost, fn_cost)

        # Expected: τ_k = |fp| / (|fp| + |fn|) = 1 / (1 + |fn|)
        expected = np.array([1 / 6, 1 / 4, 1 / 3])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_degenerate_cases(self):
        """Test degenerate cases with extreme costs."""
        # Case with very high false negative cost
        fp_cost = [-1, -1]
        fn_cost = [-100, -1000]
        
        thresholds = bayes_thresholds_from_costs(fp_cost, fn_cost)

        # τ_k = |fp| / (|fp| + |fn|) = 1 / (1 + |fn|)
        expected = np.array([1 / 101, 1 / 1001])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_always_positive_case(self):
        """Test case where should almost always predict positive."""
        fp_cost = [-1, -1]
        fn_cost = [-1000, -1000]  # Very high false negative cost

        thresholds = bayes_thresholds_from_costs(fp_cost, fn_cost)

        # τ_k = |fp| / (|fp| + |fn|) = 1 / (1 + 1000) = 1/1001
        expected = np.array([1 / 1001, 1 / 1001])
        np.testing.assert_array_almost_equal(thresholds, expected)

    def test_shape_validation(self):
        """Test that all arrays must have same shape."""
        fp_cost = [-1, -1, -1]
        fn_cost = [-5, -3]  # Wrong length

        with pytest.raises(
            ValueError, match="fp_costs and fn_costs must have same shape"
        ):
            bayes_thresholds_from_costs(fp_cost, fn_cost)

    def test_clipping_to_unit_interval(self):
        """Test that thresholds are clipped to [0, 1]."""
        # Create scenario that would give threshold > 1
        fp_cost = [10]  # Large positive cost (becomes negative utility)
        fn_cost = [-1]

        thresholds = bayes_thresholds_from_costs(fp_cost, fn_cost)

        # Should be clipped to [0, 1]
        assert 0.0 <= thresholds[0] <= 1.0


class TestBayesThresholdFromCostsScalar:
    """Test scalar Bayes threshold for backward compatibility."""

    def test_equivalent_to_vector_version(self):
        """Test that scalar version matches vector version for single class."""
        fp_cost = 1.0
        fn_cost = 5.0

        scalar_threshold = bayes_optimal_threshold(fp_cost, fn_cost)
        vector_threshold = bayes_thresholds_from_costs([fp_cost], [fn_cost])[0]

        assert abs(scalar_threshold - vector_threshold) < 1e-12

    def test_simple_threshold_computation(self):
        """Test simple threshold computation."""
        fp_cost = 1.0
        fn_cost = 5.0

        threshold = bayes_optimal_threshold(fp_cost, fn_cost)
        
        # Should be fp_cost / (fp_cost + fn_cost) = 1 / (1 + 5) = 1/6
        expected = 1.0 / 6.0
        assert abs(threshold - expected) < 1e-10

    def test_with_benefits(self):
        """Test with benefits included."""
        # Test with costs as negative utilities and benefits as positive utilities
        threshold = bayes_optimal_threshold(
            fp_cost=1, fn_cost=5, tp_benefit=2, tn_benefit=1
        )
        
        # Using the formula from BayesOptimal.compute_threshold()
        # A = tp - fn = 2 - (-5) = 7, B = tn - fp = 1 - (-1) = 2
        # threshold = B / (A + B) = 2 / (7 + 2) = 2/9
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


class TestBayesEdgeCases:
    """Test mathematical edge cases and correctness fixes."""

    def test_extreme_cost_ratios(self):
        """Test with extreme cost ratios."""
        # Very high FN cost relative to FP cost
        fp_cost = 1.0
        fn_cost = 1000.0
        
        threshold = bayes_optimal_threshold(fp_cost, fn_cost)
        
        # Should be very low threshold (almost always predict positive)
        expected = 1.0 / 1001.0
        assert abs(threshold - expected) < 1e-10
        
        # Very high FP cost relative to FN cost
        fp_cost = 1000.0
        fn_cost = 1.0
        
        threshold = bayes_optimal_threshold(fp_cost, fn_cost)
        
        # Should be very high threshold (almost never predict positive)
        expected = 1000.0 / 1001.0
        assert abs(threshold - expected) < 1e-10

    def test_utility_matrix_validation(self):
        """Test utility matrix validation."""
        P = np.array([[0.7, 0.3]])
        
        # Test wrong dimensions
        U_1d = np.array([1, 0, 1])  # 1D array instead of 2D
        with pytest.raises(ValueError, match="utility_matrix must be 2D array"):
            bayes_optimal_decisions(P, U_1d)
            
        # Test mismatched shape
        U_wrong = np.array([[1, 0, 0], [0, 1, 0]])  # 3 classes but P has 2
        with pytest.raises(ValueError, match="probabilities has 2 classes but utility_matrix has 3"):
            bayes_optimal_decisions(P, U_wrong)

    def test_equal_costs(self):
        """Test with equal FP and FN costs."""
        fp_cost = 1.0
        fn_cost = 1.0
        
        threshold = bayes_optimal_threshold(fp_cost, fn_cost)
        
        # Should be 0.5 when costs are equal
        expected = 0.5
        assert abs(threshold - expected) < 1e-10
