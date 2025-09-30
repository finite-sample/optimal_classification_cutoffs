"""Tests for the code coherence improvements implemented in v0.4.0."""

import numpy as np
import pytest

from optimal_cutoffs import (
    ThresholdOptimizer,
    bayes_threshold_from_costs_scalar,
    cv_threshold_optimization,
    get_optimal_threshold,
    nested_cv_threshold_optimization,
)


class TestModeParameter:
    """Test the new mode parameter functionality."""

    def test_mode_empirical_default(self):
        """Test that mode='empirical' works as default."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # These should be equivalent
        threshold1 = get_optimal_threshold(y_true, y_prob, metric="f1")
        threshold2 = get_optimal_threshold(
            y_true, y_prob, metric="f1", mode="empirical"
        )

        assert abs(threshold1 - threshold2) < 1e-10

    def test_mode_bayes_requires_utility(self):
        """Test that mode='bayes' requires utility parameter."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError, match="mode='bayes' requires utility parameter"):
            get_optimal_threshold(None, y_prob, mode="bayes")

    def test_mode_bayes_with_utility(self):
        """Test that mode='bayes' works with utility parameter."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 0, "tn": 0, "fp": -1, "fn": -5}

        threshold = get_optimal_threshold(None, y_prob, mode="bayes", utility=utility)
        expected = bayes_threshold_from_costs_scalar(fp_cost=1, fn_cost=5)

        assert abs(threshold - expected) < 1e-10

    def test_mode_expected_f1_only(self):
        """Test that mode='expected' only works with F1 metric."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Should work with F1 and return a tuple (threshold, f1_score)
        result = get_optimal_threshold(y_true, y_prob, metric="f1", mode="expected")
        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f1_score = result
        assert isinstance(threshold, float)
        assert isinstance(f1_score, float)
        assert 0 <= threshold <= 1
        assert 0 <= f1_score <= 1

        # Note: mode='expected' now supports other metrics as well
        # Test that it works with accuracy too
        result_acc = get_optimal_threshold(
            y_true, y_prob, metric="accuracy", mode="expected"
        )
        assert isinstance(result_acc, tuple)
        assert len(result_acc) == 2

    def test_mode_expected_supports_multiclass(self):
        """Test that mode='expected' works with multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.random.rand(6, 3)

        # Should work with multiclass and return a dict
        result = get_optimal_threshold(y_true, y_prob, metric="f1", mode="expected")
        assert isinstance(result, dict)
        assert "thresholds" in result
        assert "f_beta_per_class" in result
        assert "f_beta" in result
        assert isinstance(result["thresholds"], np.ndarray)
        assert len(result["thresholds"]) == 3  # 3 classes

    def test_mode_expected_supports_sample_weights(self):
        """Test that mode='expected' supports sample weights."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        sample_weight = np.array([1, 1, 2, 2, 1, 2])

        # Should work with sample weights
        result = get_optimal_threshold(
            y_true,
            y_prob,
            metric="f1",
            mode="expected",
            sample_weight=sample_weight,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f1_score = result
        assert 0 <= threshold <= 1
        assert 0 <= f1_score <= 1

    def test_mode_expected_works_without_true_labs(self):
        """Test that mode='expected' works without true_labs."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Should work without true_labs
        result = get_optimal_threshold(None, y_prob, metric="f1", mode="expected")
        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f1_score = result
        assert 0 <= threshold <= 1
        assert 0 <= f1_score <= 1


class TestDeprecatedParameterRejection:
    """Test that deprecated parameters are properly rejected."""

    def test_bayes_parameter_rejected(self):
        """Test that bayes=True raises TypeError (parameter no longer exists)."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 0, "tn": 0, "fp": -1, "fn": -5}

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            get_optimal_threshold(None, y_prob, utility=utility, bayes=True)

    def test_deprecated_dinkelbach_method_rejected(self):
        """Test that deprecated method='dinkelbach' raises ValueError (method no longer exists)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError, match="Invalid optimization method"):
            get_optimal_threshold(y_true, y_prob, metric="f1", method="dinkelbach")

    def test_deprecated_smart_brute_method_rejected(self):
        """Test that deprecated method='smart_brute' raises ValueError (method no longer exists)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError, match="Invalid optimization method"):
            get_optimal_threshold(y_true, y_prob, metric="f1", method="smart_brute")

    def test_deprecated_objective_parameter_rejected_in_wrapper(self):
        """Test that deprecated objective parameter in ThresholdOptimizer raises TypeError."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ThresholdOptimizer(objective="f1")


class TestMethodEquivalence:
    """Test that different methods produce equivalent results."""

    def test_unique_scan_vs_sort_scan_equivalence(self):
        """Test that unique_scan gives same results as sort_scan for piecewise metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85])

        threshold_unique_scan = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="unique_scan"
        )

        threshold_sort_scan = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="sort_scan"
        )

        assert abs(threshold_unique_scan - threshold_sort_scan) < 1e-10

    def test_unique_scan_vs_sort_scan_on_piecewise_metrics(self):
        """Test that unique_scan gives same results as sort_scan for piecewise metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85])

        threshold_unique_scan = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="unique_scan"
        )

        threshold_sort_scan = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="sort_scan"
        )

        assert abs(threshold_unique_scan - threshold_sort_scan) < 1e-10


class TestThresholdOptimizerWrapper:
    """Test the enhanced ThresholdOptimizer wrapper."""

    def test_metric_parameter_works(self):
        """Test that metric parameter works in ThresholdOptimizer."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_prob)

        assert optimizer.metric == "f1"
        assert optimizer.threshold_ is not None

    def test_valid_metric_parameter(self):
        """Test that valid metric parameter works."""
        optimizer = ThresholdOptimizer(metric="f1")
        assert optimizer.metric == "f1"

    def test_mode_parameter_in_wrapper(self):
        """Test that mode parameter works in ThresholdOptimizer."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        optimizer = ThresholdOptimizer(metric="f1", mode="expected")
        optimizer.fit(y_true, y_prob)

        assert optimizer.mode == "expected"
        assert optimizer.threshold_ is not None

    def test_utility_parameter_in_wrapper(self):
        """Test that utility parameter works in ThresholdOptimizer."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 1, "tn": 1, "fp": -2, "fn": -5}

        optimizer = ThresholdOptimizer(utility=utility)
        optimizer.fit(y_true, y_prob)

        assert optimizer.utility == utility
        assert optimizer.threshold_ is not None

    def test_bayes_mode_in_wrapper(self):
        """Test that mode='bayes' works in ThresholdOptimizer."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 0, "tn": 0, "fp": -1, "fn": -5}

        optimizer = ThresholdOptimizer(mode="bayes", utility=utility)
        optimizer.fit(None, y_prob)  # No true_labs needed for Bayes

        assert optimizer.mode == "bayes"
        assert optimizer.threshold_ is not None


class TestCVDefaultMethods:
    """Test that CV functions use 'auto' as default method."""

    def test_cv_threshold_optimization_default_method(self):
        """Test that cv_threshold_optimization uses 'auto' by default."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25])

        # Should not raise an error and should return valid results
        thresholds, scores = cv_threshold_optimization(y_true, y_prob, cv=3)

        assert len(thresholds) == 3
        assert len(scores) == 3
        assert all(isinstance(t, (int, float)) for t in thresholds)
        assert all(isinstance(s, (int, float)) for s in scores)

    def test_nested_cv_threshold_optimization_default_method(self):
        """Test that nested_cv_threshold_optimization uses 'auto' by default."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25])

        # Should not raise an error and should return valid results
        thresholds, scores = nested_cv_threshold_optimization(
            y_true, y_prob, inner_cv=3, outer_cv=3
        )

        assert len(thresholds) == 3
        assert len(scores) == 3
        assert all(isinstance(t, (int, float)) for t in thresholds)
        assert all(isinstance(s, (int, float)) for s in scores)


class TestGoldenTests:
    """Golden tests for equivalence across regimes."""

    def test_bayes_closed_form_vs_utility_api(self):
        """Test that Bayes closed-form equals get_optimal_threshold with utility."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 2, "tn": 1, "fp": -1, "fn": -5}

        # Direct call to Bayes function (use negative costs to match utility convention)
        threshold1 = bayes_threshold_from_costs_scalar(
            fp_cost=-1, fn_cost=-5, tp_benefit=2, tn_benefit=1
        )

        # Via get_optimal_threshold API
        threshold2 = get_optimal_threshold(None, y_prob, utility=utility, mode="bayes")

        assert abs(threshold1 - threshold2) < 1e-12

    def test_method_consistency(self):
        """Test that methods give consistent results."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85])

        threshold1 = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="unique_scan"
        )

        threshold2 = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="minimize"
        )

        # Different methods may give slightly different results, but should be close
        assert abs(threshold1 - threshold2) < 0.1

    def test_expected_mode_works(self):
        """Test that mode='expected' works correctly."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Both calls should work and give same result
        result1 = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")

        result2 = get_optimal_threshold(y_true, y_prob, metric="f1", mode="expected")

        # Both should return tuples
        assert isinstance(result1, tuple)  # Expected mode returns (threshold, f1_score)
        assert isinstance(result2, tuple)  # Expected mode returns (threshold, f1_score)

        # Extract thresholds and compare
        threshold1, f1_score1 = result1
        threshold2, f1_score2 = result2
        assert abs(threshold1 - threshold2) < 1e-12
        assert abs(f1_score1 - f1_score2) < 1e-12


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_mode_bayes_error_message(self):
        """Test clear error message for mode='bayes' without utility."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError) as exc_info:
            get_optimal_threshold(None, y_prob, mode="bayes")

        assert "mode='bayes' requires utility parameter" in str(exc_info.value)

    def test_mode_expected_supports_multiple_metrics(self):
        """Test that mode='expected' supports multiple metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Should work with different metrics
        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = get_optimal_threshold(
                y_true, y_prob, metric=metric, mode="expected"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            threshold, score = result
            assert 0 <= threshold <= 1
            assert 0 <= score <= 1

    def test_mode_expected_multiclass_support(self):
        """Test that mode='expected' supports multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.random.rand(6, 3)

        # Should work with multiclass and return a dict
        result = get_optimal_threshold(y_true, y_prob, metric="f1", mode="expected")
        assert isinstance(result, dict)
        assert "thresholds" in result
        assert "f_beta_per_class" in result
        assert "f_beta" in result
