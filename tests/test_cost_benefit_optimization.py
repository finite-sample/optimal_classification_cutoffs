"""Tests for cost/benefit-aware threshold optimization."""

import numpy as np
import pytest

from optimal_cutoffs import (
    bayes_threshold_from_costs_scalar,
    get_optimal_threshold,
    make_cost_metric,
    make_linear_counts_metric,
)
from optimal_cutoffs.metrics import get_confusion_matrix


class TestLinearUtilityMetrics:
    """Test the linear utility metric factories."""

    def test_make_linear_counts_metric_basic(self):
        """Test basic linear counts metric creation."""
        metric = make_linear_counts_metric(w_tp=1.0, w_tn=0.5, w_fp=-2.0, w_fn=-5.0)

        # Test with simple values
        tp, tn, fp, fn = 10, 20, 5, 3
        result = metric(tp, tn, fp, fn)
        expected = 1.0 * 10 + 0.5 * 20 + (-2.0) * 5 + (-5.0) * 3
        expected = 10 + 10 - 10 - 15  # = -5
        assert result == expected

    def test_make_linear_counts_metric_vectorized(self):
        """Test vectorized operation of linear counts metric."""
        metric = make_linear_counts_metric(w_tp=2.0, w_fp=-1.0)

        # Test with arrays
        tp = np.array([1, 2, 3])
        tn = np.array([4, 5, 6])
        fp = np.array([1, 1, 2])
        fn = np.array([0, 1, 0])

        result = metric(tp, tn, fp, fn)
        expected = 2.0 * tp + (-1.0) * fp  # Only tp and fp have non-zero weights
        np.testing.assert_array_equal(result, expected)

    def test_make_cost_metric(self):
        """Test cost metric convenience wrapper."""
        metric = make_cost_metric(fp_cost=1.0, fn_cost=5.0, tp_benefit=2.0)

        # Should be equivalent to make_linear_counts_metric(w_tp=2.0, w_fp=-1.0, w_fn=-5.0)
        tp, tn, fp, fn = 10, 20, 3, 2
        result = metric(tp, tn, fp, fn)
        expected = 2.0 * 10 + 0.0 * 20 + (-1.0) * 3 + (-5.0) * 2
        expected = 20 + 0 - 3 - 10  # = 7
        assert result == expected


class TestBayesThresholds:
    """Test Bayes-optimal threshold calculations."""

    def test_bayes_threshold_classic_cost_case(self):
        """Test classic cost case: C_FP=1, C_FN=5."""
        # Expected threshold: C_FP / (C_FP + C_FN) = 1/(1+5) = 1/6 ≈ 0.1667
        threshold = bayes_threshold_from_costs_scalar(fp_cost=-1, fn_cost=-5)
        expected = 1.0 / 6.0
        assert abs(threshold - expected) < 1e-10

    def test_bayes_threshold_from_costs_equivalent(self):
        """Test that costs wrapper gives same result as utilities."""
        threshold1 = bayes_threshold_from_costs_scalar(fp_cost=-1, fn_cost=-5)
        threshold2 = bayes_threshold_from_costs_scalar(fp_cost=-1.0, fn_cost=-5.0)
        assert abs(threshold1 - threshold2) < 1e-12

    def test_bayes_threshold_with_benefits(self):
        """Test threshold with benefits for correct predictions."""
        # U_tp=2, U_tn=1, U_fp=-1, U_fn=-5
        # t* = (U_tn - U_fp) / [(U_tn - U_fp) + (U_tp - U_fn)]
        # t* = (1 - (-1)) / [(1 - (-1)) + (2 - (-5))] = 2 / (2 + 7) = 2/9
        threshold = bayes_threshold_from_costs_scalar(
            fp_cost=-1, fn_cost=-5, tp_benefit=2, tn_benefit=1
        )
        expected = 2.0 / 9.0
        assert abs(threshold - expected) < 1e-10

    def test_bayes_threshold_degenerate_cases(self):
        """Test degenerate cases where one action dominates."""
        # Case 1: Positive always better (very high TP benefit, no costs)
        threshold = bayes_threshold_from_costs_scalar(
            fp_cost=0, fn_cost=0, tp_benefit=100, tn_benefit=0
        )
        # Should predict all as positive -> very low threshold
        assert threshold < 1e-10

        # Case 2: Negative always better (U_fn >= U_tp to get threshold >= 1.0)
        # Make false negative utility higher than true positive utility
        threshold = bayes_threshold_from_costs_scalar(
            fp_cost=-100, fn_cost=0, tp_benefit=-10, tn_benefit=1
        )
        # Should predict all as negative -> very high threshold
        assert threshold >= 1.0

    def test_bayes_threshold_comparison_operators(self):
        """Test that comparison operators are handled correctly."""
        # Simple case where threshold should be exactly 0.5
        # U_tp = U_tn = 1.0, U_fp = U_fn = 0.0
        # t* = (1-0) / [(1-0) + (1-0)] = 1/2 = 0.5

        thresh_excl = bayes_threshold_from_costs_scalar(
            fp_cost=0, fn_cost=0, tp_benefit=1, tn_benefit=1, comparison=">"
        )
        thresh_incl = bayes_threshold_from_costs_scalar(
            fp_cost=0, fn_cost=0, tp_benefit=1, tn_benefit=1, comparison=">="
        )

        # Both should be close to 0.5, with slight differences for tie handling
        assert abs(thresh_excl - 0.5) < 1e-10
        assert abs(thresh_incl - 0.5) < 1e-10


class TestUtilityOptimization:
    """Test utility-based threshold optimization through get_optimal_threshold."""

    def test_basic_utility_optimization(self):
        """Test basic utility optimization vs manual calculation."""
        # Generate simple test data
        np.random.seed(42)
        n = 1000
        p = np.random.uniform(0, 1, size=n)
        y = (np.random.uniform(0, 1, size=n) < p).astype(int)  # Calibrated

        # Simple cost case: FP=1, FN=5
        threshold = get_optimal_threshold(
            y, p, utility={"fp": -1.0, "fn": -5.0}, comparison=">="
        )

        # Should be reasonable threshold (not extreme)
        assert 0.01 < threshold < 0.99

        # Verify it actually optimizes the utility
        tp, tn, fp, fn = get_confusion_matrix(y, p, threshold, comparison=">=")
        utility_score = 0 * tp + 0 * tn + (-1) * fp + (-5) * fn

        # Test nearby thresholds should give worse utility
        for delta in [-0.01, 0.01]:
            test_thresh = np.clip(threshold + delta, 0, 1)
            tp_test, tn_test, fp_test, fn_test = get_confusion_matrix(
                y, p, test_thresh, comparison=">="
            )
            utility_test = 0 * tp_test + 0 * tn_test + (-1) * fp_test + (-5) * fn_test
            # Allow for small numerical differences due to discrete nature
            assert utility_test <= utility_score + 1e-10

    def test_bayes_vs_empirical_on_calibrated_data(self):
        """Test that Bayes and empirical give similar results on calibrated data."""
        np.random.seed(123)
        n = 20000  # Large sample for good calibration
        p = np.random.uniform(0, 1, size=n)
        y = (np.random.uniform(0, 1, size=n) < p).astype(int)  # Calibrated

        # Cost case: FP=1, FN=5
        thresh_empirical = get_optimal_threshold(
            y, p, utility={"fp": -1.0, "fn": -5.0}, comparison=">="
        )
        thresh_bayes = get_optimal_threshold(
            None, p, utility={"fp": -1.0, "fn": -5.0}, mode="bayes", comparison=">="
        )

        # Should be close on well-calibrated data
        assert abs(thresh_empirical - thresh_bayes) < 0.01

        # Bayes should be exactly 1/(1+5) = 1/6
        expected_bayes = 1.0 / 6.0
        assert abs(thresh_bayes - expected_bayes) < 1e-10

    def test_minimize_cost_flag(self):
        """Test minimize_cost flag behavior."""
        np.random.seed(42)
        n = 500
        p = np.random.uniform(0.2, 0.8, size=n)
        y = (np.random.uniform(0, 1, size=n) < 0.5).astype(int)

        # These should be equivalent
        thresh1 = get_optimal_threshold(y, p, utility={"fp": -1.0, "fn": -5.0})
        thresh2 = get_optimal_threshold(
            y, p, utility={"fp": 1.0, "fn": 5.0}, minimize_cost=True
        )

        assert abs(thresh1 - thresh2) < 1e-12

    def test_utility_with_sample_weights(self):
        """Test utility optimization with sample weights."""
        np.random.seed(456)
        n = 200
        p = np.random.uniform(0, 1, size=n)
        y = (p > 0.5).astype(int)  # Simple deterministic relationship
        weights = np.random.uniform(0.5, 2.0, size=n)  # Varying weights

        # Should not raise an error
        threshold = get_optimal_threshold(
            y, p, utility={"tp": 1.0, "fp": -1.0}, sample_weight=weights
        )
        assert 0 <= threshold <= 1

    def test_utility_multiclass_not_implemented(self):
        """Test that multiclass utility optimization raises appropriate error."""
        n = 100
        p = np.random.uniform(0, 1, size=(n, 3))  # 3 classes
        y = np.random.randint(0, 3, size=n)

        with pytest.raises(
            NotImplementedError,
            match="Utility/cost-based optimization not yet implemented for multiclass",
        ):
            get_optimal_threshold(y, p, utility={"fp": -1.0, "fn": -5.0})


class TestUtilityMetricIntegration:
    """Test integration with existing optimization methods."""

    def test_linear_utility_reduces_to_f1_alignment(self):
        """Test that linear utility can approximate F1 optimization."""
        # F1 maximizes TP while minimizing FP+FN. A utility that rewards TP
        # and penalizes FP/FN should give similar results on many datasets.
        np.random.seed(789)
        n = 2000
        p = np.random.uniform(0, 1, size=n)
        y = (np.random.uniform(0, 1, size=n) < 0.4).astype(int)

        # F1 optimization
        thresh_f1 = get_optimal_threshold(y, p, metric="f1", method="sort_scan")

        # Utility optimization that rewards TP and penalizes FP/FN equally
        thresh_util = get_optimal_threshold(
            y, p, utility={"tp": 1.0, "fp": -0.5, "fn": -0.5}
        )

        # Check that predictions are mostly the same
        pred_f1 = (p > thresh_f1).astype(int)
        pred_util = (p > thresh_util).astype(int)
        agreement = np.mean(pred_f1 == pred_util)

        # Should agree on most samples (heuristic test)
        assert agreement > 0.95

    def test_scale_invariance(self):
        """Test that scaling all utilities by positive constant doesn't change optimum."""
        np.random.seed(100)
        n = 500
        p = np.random.uniform(0, 1, size=n)
        y = (np.random.uniform(0, 1, size=n) < p).astype(int)

        # Base utilities
        base_util = {"tp": 2.0, "tn": 1.0, "fp": -1.0, "fn": -3.0}
        thresh1 = get_optimal_threshold(y, p, utility=base_util)

        # Scaled utilities (multiply by 10)
        scaled_util = {k: v * 10 for k, v in base_util.items()}
        thresh2 = get_optimal_threshold(y, p, utility=scaled_util)

        # Should give same threshold (up to numerical precision)
        assert abs(thresh1 - thresh2) < 1e-10

    def test_utility_respects_comparison_operator(self):
        """Test that utility optimization respects comparison operator."""
        # Create data with probabilities exactly at potential threshold
        p = np.array([0.1, 0.3, 0.5, 0.5, 0.7, 0.9])
        y = np.array([0, 0, 1, 0, 1, 1])

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                y, p, utility={"tp": 1.0, "fn": -1.0}, comparison=comparison
            )

            # Apply threshold and check consistency
            if comparison == ">":
                pred = (p > threshold).astype(int)
            else:
                pred = (p >= threshold).astype(int)

            # Should produce valid predictions (not a strong test, but checks basic functionality)
            assert len(pred) == len(y)
            assert all(pred_val in [0, 1] for pred_val in pred)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_bayes_requires_no_true_labels(self):
        """Test that Bayes optimization doesn't need true labels."""
        p = np.array([0.1, 0.5, 0.9])

        # This should work (no true_labs needed)
        threshold = get_optimal_threshold(
            None, p, utility={"fp": -1, "fn": -5}, mode="bayes"
        )
        assert 0 <= threshold <= 1

    def test_empirical_requires_true_labels(self):
        """Test that empirical optimization requires true labels."""
        p = np.array([0.1, 0.5, 0.9])

        with pytest.raises(
            ValueError, match="true_labs is required for empirical utility optimization"
        ):
            get_optimal_threshold(
                None, p, utility={"fp": -1, "fn": -5}, mode="empirical"
            )

    def test_empty_utility_dict(self):
        """Test with empty or minimal utility specification."""
        np.random.seed(42)
        n = 100
        p = np.random.uniform(0, 1, size=n)
        y = np.random.randint(0, 2, size=n)

        # Empty dict should work (all utilities = 0)
        threshold = get_optimal_threshold(y, p, utility={})
        assert 0 <= threshold <= 1

        # Single utility should work
        threshold = get_optimal_threshold(y, p, utility={"tp": 1.0})
        assert 0 <= threshold <= 1


if __name__ == "__main__":
    pytest.main([__file__])
