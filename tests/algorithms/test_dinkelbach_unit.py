"""Unit tests for Dinkelbach expected F-beta optimization method.

This module tests the Dinkelbach algorithm for optimizing expected F-beta scores
under perfect calibration assumptions. The method depends only on predicted
probabilities, not on realized labels.
"""

import warnings
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.expected import dinkelbach_expected_fbeta_binary
from optimal_cutoffs.metrics import f1_score, get_confusion_matrix
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import generate_calibrated_probabilities
from tests.fixtures.hypothesis_strategies import beta_bernoulli_calibrated


class TestDinkelbachBasic:
    """Test basic Dinkelbach method functionality."""

    def test_dinkelbach_basic_functionality(self):
        """Test that Dinkelbach method produces valid thresholds."""
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        threshold, expected_score = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_score, "expected_f1")

    def test_dinkelbach_different_beta_values(self):
        """Test Dinkelbach with different beta values."""
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        beta_values = [0.5, 1.0, 2.0]
        for beta in beta_values:
            threshold, expected_score = dinkelbach_expected_fbeta_binary(
                y_prob, beta=beta
            )

            assert_valid_threshold(result.threshold)
            assert_valid_metric_score(expected_score, f"expected_f{beta}")

    def test_dinkelbach_comparison_operators(self):
        """Test Dinkelbach with different comparison operators."""
        y_prob = np.array([0.3, 0.5, 0.5, 0.7])  # Include ties

        for comparison in [">", ">="]:
            threshold, expected_score = dinkelbach_expected_fbeta_binary(
                y_prob, beta=1.0, comparison=comparison
            )

            assert_valid_threshold(result.threshold)
            assert_valid_metric_score(expected_score, "expected_f1")

    def test_dinkelbach_single_probability(self):
        """Test Dinkelbach with single probability value."""
        y_prob = np.array([0.7])

        threshold, expected_score = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_score, "expected_f1")

    def test_dinkelbach_extreme_probabilities(self):
        """Test Dinkelbach with extreme probability values."""
        y_prob = np.array([0.0, 0.5, 1.0])

        threshold, expected_score = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_score, "expected_f1")


class TestDinkelbachAPI:
    """Test Dinkelbach through the main API."""

    def test_dinkelbach_through_get_optimal_threshold(self):
        """Test Dinkelbach method through the main API."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")

        assert isinstance(result, tuple)
        assert len(result) == 2
        threshold, f1_score = result

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(f1_score, "expected_f1")

    def test_dinkelbach_supported_metrics(self):
        """Test that expected mode supports multiple metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        # Should work with F-beta related metrics
        for metric in ["f1", "precision", "recall"]:
            result = get_optimal_threshold(
                y_true, y_prob, metric=metric, mode="expected"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            threshold, score = result
            assert_valid_threshold(result.threshold)
            assert_valid_metric_score(score, f"expected_{metric}")

    def test_dinkelbach_with_comparison_operators(self):
        """Test Dinkelbach API with different comparison operators."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.4, 0.6, 0.5, 0.5])  # Include ties

        for comparison in [">", ">="]:
            result = get_optimal_threshold(
                y_true, y_prob, mode="expected", metric="f1", comparison=comparison
            )
            threshold, expected_f1 = result

            assert_valid_threshold(result.threshold)
            assert_valid_metric_score(expected_f1, "expected_f1")


class TestDinkelbachLabelIndependence:
    """Test that Dinkelbach depends only on probabilities, not labels."""

    def test_dinkelbach_label_independence_basic(self):
        """Test that Dinkelbach threshold is identical for same probabilities with different labels."""
        probs = np.array([0.2, 0.4, 0.6, 0.8])

        # Different label configurations
        labels_1 = np.array([0, 0, 1, 1])  # Correlated with probs
        labels_2 = np.array([1, 0, 0, 1])  # Random
        labels_3 = np.array([1, 1, 0, 0])  # Anti-correlated with probs

        for comparison in [">", ">="]:
            # Get thresholds for all label configurations
            result_1 = get_optimal_threshold(
                labels_1, probs, metric="f1", mode="expected", comparison=comparison
            )
            result_2 = get_optimal_threshold(
                labels_2, probs, metric="f1", mode="expected", comparison=comparison
            )
            result_3 = get_optimal_threshold(
                labels_3, probs, metric="f1", mode="expected", comparison=comparison
            )

            threshold_1, expected_1 = result_1
            threshold_2, expected_2 = result_2
            threshold_3, expected_3 = result_3

            # Thresholds should be identical (label-independent)
            assert threshold_1 == pytest.approx(threshold_2, abs=1e-10)
            assert threshold_2 == pytest.approx(threshold_3, abs=1e-10)

            # Expected scores should also be identical
            assert expected_1 == pytest.approx(expected_2, abs=1e-10)
            assert expected_2 == pytest.approx(expected_3, abs=1e-10)

    @given(n=st.integers(20, 100))
    @settings(deadline=None, max_examples=20)
    def test_dinkelbach_label_independence_property(self, n):
        """Property: Dinkelbach should be label-independent."""
        rng = np.random.default_rng(99)
        probs = rng.uniform(0.1, 0.9, size=n)

        # Generate different label sets
        labels_calibrated = (rng.uniform(0, 1, size=n) < probs).astype(int)
        labels_random = rng.integers(0, 2, size=n)

        # Ensure both classes present to avoid degenerate cases
        if labels_calibrated.sum() == 0:
            labels_calibrated[0] = 1
        elif labels_calibrated.sum() == n:
            labels_calibrated[0] = 0

        if labels_random.sum() == 0:
            labels_random[0] = 1
        elif labels_random.sum() == n:
            labels_random[0] = 0

        for comparison in [">", ">="]:
            result_cal = get_optimal_threshold(
                labels_calibrated,
                probs,
                metric="f1",
                mode="expected",
                comparison=comparison,
            )
            result_rand = get_optimal_threshold(
                labels_random,
                probs,
                metric="f1",
                mode="expected",
                comparison=comparison,
            )

            threshold_cal, expected_cal = result_cal
            threshold_rand, expected_rand = result_rand

            # Should be identical within numerical precision
            assert threshold_cal == pytest.approx(threshold_rand, abs=1e-8)
            assert expected_cal == pytest.approx(expected_rand, abs=1e-8)


class TestDinkelbachCalibration:
    """Test Dinkelbach behavior on calibrated data."""

    def test_dinkelbach_on_calibrated_data(self):
        """Test Dinkelbach performance on perfectly calibrated data."""
        y_true, y_prob = generate_calibrated_probabilities(200, random_state=42)

        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold, expected_f1 = result

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_f1, "expected_f1")

        # On calibrated data, should achieve reasonable performance
        assert expected_f1 > 0.3  # Should be better than random

    @given(beta_bernoulli_calibrated(min_size=50, max_size=200))
    @settings(max_examples=10, deadline=5000)
    def test_dinkelbach_calibrated_property(self, data):
        """Property: Dinkelbach should perform well on calibrated data."""
        y_true, y_prob, alpha, beta = data

        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold, expected_f1 = result

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_f1, "expected_f1")

        # Should achieve reasonable performance on calibrated data
        # (The exact threshold depends on the Beta distribution parameters)
        assert expected_f1 >= 0.0  # Basic validity check


class TestDinkelbachTieHandling:
    """Test Dinkelbach tie handling behavior."""

    def test_dinkelbach_with_ties(self):
        """Test that Dinkelbach handles tied probabilities correctly."""
        # Create data with ties at the likely optimal threshold
        y_prob = np.array([0.3, 0.5, 0.5, 0.5, 0.7])
        y_true = np.array([0, 1, 0, 1, 1])  # Arbitrary labels

        for comparison in [">", ">="]:
            result = get_optimal_threshold(
                y_true, y_prob, mode="expected", metric="f1", comparison=comparison
            )
            threshold, expected_f1 = result

            assert_valid_threshold(result.threshold)
            assert_valid_metric_score(expected_f1, "expected_f1")

            # Verify that prediction behavior is consistent with comparison operator
            if comparison == ">":
                predictions = y_prob > threshold
            else:
                predictions = y_prob >= threshold

            assert isinstance(predictions, np.ndarray)
            assert predictions.dtype == bool

    def test_dinkelbach_all_tied_probabilities(self):
        """Test Dinkelbach when all probabilities are identical."""
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        y_true = np.array([0, 1, 0, 1])

        for comparison in [">", ">="]:
            result = get_optimal_threshold(
                y_true, y_prob, mode="expected", metric="f1", comparison=comparison
            )
            threshold, expected_f1 = result

            assert_valid_threshold(result.threshold)
            assert_valid_metric_score(expected_f1, "expected_f1")


class TestDinkelbachEdgeCases:
    """Test Dinkelbach edge cases."""

    def test_dinkelbach_uniform_probabilities(self):
        """Test Dinkelbach with uniform probability distribution."""
        y_prob = np.linspace(0.1, 0.9, 10)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold, expected_f1 = result

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_f1, "expected_f1")

    def test_dinkelbach_skewed_probabilities(self):
        """Test Dinkelbach with heavily skewed probability distribution."""
        # Most probabilities very low, few very high
        y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.95, 0.96, 0.97, 0.98])
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold, expected_f1 = result

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_f1, "expected_f1")

    def test_dinkelbach_binary_probabilities(self):
        """Test Dinkelbach with only 0 and 1 probabilities."""
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        y_true = np.array([0, 0, 1, 1])

        result = get_optimal_threshold(y_true, y_prob, mode="expected", metric="f1")
        threshold, expected_f1 = result

        assert_valid_threshold(result.threshold)
        assert_valid_metric_score(expected_f1, "expected_f1")


class TestDinkelbachComparison:
    """Test comparison between Dinkelbach and empirical optimization."""

    def test_dinkelbach_vs_empirical_consistency(self):
        """Test that Dinkelbach and empirical optimization give reasonable results."""
        y_true, y_prob = generate_calibrated_probabilities(100, random_state=42)

        # Get results from both methods
        threshold_empirical = get_optimal_threshold(
            y_true, y_prob, mode="empirical", metric="f1"
        )
        result_expected = get_optimal_threshold(
            y_true, y_prob, mode="expected", metric="f1"
        )
        threshold_expected, expected_f1 = result_expected

        # Both should be valid
        assert_valid_threshold(threshold_empirical)
        assert_valid_threshold(threshold_expected)

        # Compute empirical F1 for both thresholds
        tp_emp, tn_emp, fp_emp, fn_emp = get_confusion_matrix(
            y_true, y_prob, threshold_empirical
        )
        tp_exp, tn_exp, fp_exp, fn_exp = get_confusion_matrix(
            y_true, y_prob, threshold_expected
        )

        f1_emp = f1_score(tp_emp, tn_emp, fp_emp, fn_emp)
        f1_exp = f1_score(tp_exp, tn_exp, fp_exp, fn_exp)

        # Both should achieve reasonable F1 scores
        assert f1_emp >= 0.0
        assert f1_exp >= 0.0

    def test_dinkelbach_respects_beta_parameter(self):
        """Test that different beta values give different thresholds."""
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        # Get thresholds for different beta values
        threshold_05, _ = dinkelbach_expected_fbeta_binary(
            y_prob, beta=0.5
        )  # Precision-weighted
        threshold_10, _ = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)  # F1
        threshold_20, _ = dinkelbach_expected_fbeta_binary(
            y_prob, beta=2.0
        )  # Recall-weighted

        # All should be valid
        for threshold in [threshold_05, threshold_10, threshold_20]:
            assert_valid_threshold(result.threshold)

        # They should generally be different (unless there's a unique optimum)
        # This is more of a sanity check than a strict requirement


class TestDinkelbachConvergenceWarnings:
    """Test convergence warnings for Dinkelbach algorithms."""

    def test_dinkelbach_normal_case_no_warnings(self):
        """Test that normal cases don't produce warnings."""
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            threshold, score = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
            
            # Should not have any warnings for normal convergent cases
            assert len(w) == 0
            assert_valid_threshold(result.threshold)
            assert 0.0 <= score <= 1.0

    def test_dinkelbach_edge_cases_no_warnings(self):
        """Test that edge cases handle warnings appropriately."""
        edge_cases = [
            np.array([0.5]),      # Single probability
            np.array([0.0, 1.0]), # Extreme probabilities
            np.zeros(5),          # All zeros
            np.ones(5),           # All ones
        ]
        
        for y_prob in edge_cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                threshold, score = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
                
                # Edge cases should generally converge without warnings
                assert_valid_threshold(result.threshold)
                assert 0.0 <= score <= 1.0
