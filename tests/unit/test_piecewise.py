"""Unit tests for the O(n log n) piecewise sort-scan algorithm.

This module tests the core piecewise optimization algorithm that provides
significant performance improvements for piecewise-constant metrics.
"""

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs.metrics import get_vectorized_metric
from optimal_cutoffs.piecewise import (
    _compute_threshold_midpoint,
    _validate_inputs,
    _validate_sample_weights,
    _vectorized_counts,
    optimal_threshold_sortscan,
)
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_sample_weights,
    generate_tied_probabilities,
)


class TestPiecewiseValidation:
    """Test input validation functions for piecewise module."""

    def test_validate_inputs_basic(self):
        """Test basic input validation."""
        y_true = [0, 1, 0, 1]
        pred_prob = [0.1, 0.7, 0.3, 0.9]

        y, p = _validate_inputs(y_true, pred_prob)

        assert y.dtype == np.int8
        assert p.dtype == np.float64
        assert len(y) == len(p) == 4
        np.testing.assert_array_equal(y, [0, 1, 0, 1])
        np.testing.assert_array_almost_equal(p, [0.1, 0.7, 0.3, 0.9])

    def test_validate_inputs_errors(self):
        """Test input validation error cases."""
        # Wrong dimensions
        with pytest.raises(ValueError, match="true_labs must be 1D"):
            _validate_inputs([[0, 1]], [0.5])

        with pytest.raises(ValueError, match="2D pred_prob not allowed"):
            _validate_inputs([0], [[0.5]])

        # Length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            _validate_inputs([0, 1], [0.5])

        # Empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_inputs([], [])

        # Non-binary labels
        with pytest.raises(ValueError, match="Binary labels must be from"):
            _validate_inputs([0, 1, 2], [0.1, 0.5, 0.9])

    def test_validate_inputs_probability_range(self):
        """Test probability range validation."""
        # Out of range - low
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            _validate_inputs([0, 1], [-0.1, 0.5])

        # Out of range - high
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            _validate_inputs([0, 1], [0.5, 1.1])

        # Non-finite values
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            _validate_inputs([0, 1], [0.5, np.nan])

        with pytest.raises(ValueError, match="contains NaN or infinite"):
            _validate_inputs([0, 1], [0.5, np.inf])

    def test_validate_inputs_scores_mode(self):
        """Test validation with scores outside [0,1] range."""
        y_true = [0, 1, 0, 1]
        scores = [-1.5, 2.3, -0.8, 1.9]  # Outside [0,1]

        # Should work with require_proba=False
        y, s = _validate_inputs(y_true, scores, require_proba=False)
        assert y.dtype == np.int8
        assert s.dtype == np.float64
        np.testing.assert_array_equal(y, y_true)
        np.testing.assert_array_almost_equal(s, scores)

    def test_validate_sample_weights_basic(self):
        """Test sample weight validation."""
        # None case
        w = _validate_sample_weights(None, 4)
        np.testing.assert_array_equal(w, [1.0, 1.0, 1.0, 1.0])

        # Array case
        w = _validate_sample_weights([0.5, 1.0, 1.5, 2.0], 4)
        np.testing.assert_array_almost_equal(w, [0.5, 1.0, 1.5, 2.0])

    def test_validate_sample_weights_errors(self):
        """Test sample weight validation errors."""
        with pytest.raises(ValueError, match="sample_weight must be 1D"):
            _validate_sample_weights([[1.0]], 1)

        with pytest.raises(ValueError, match="Length mismatch"):
            _validate_sample_weights([1.0, 2.0], 3)

        with pytest.raises(ValueError, match="sample_weight must be non-negative"):
            _validate_sample_weights([-1.0, 1.0], 2)

        with pytest.raises(ValueError, match="contains NaN or infinite"):
            _validate_sample_weights([np.nan, 1.0], 2)

        with pytest.raises(ValueError, match="sample_weight cannot sum to zero"):
            _validate_sample_weights([0.0, 0.0], 2)


class TestPiecewiseAlgorithm:
    """Test the core piecewise optimization algorithm."""

    def test_optimal_threshold_sortscan_basic(self):
        """Test basic sort-scan algorithm functionality."""
        y_true = np.array([0, 1, 0, 1])
        pred_prob = np.array([0.1, 0.8, 0.3, 0.6])

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")
        assert isinstance(k_star, int)
        assert 0 <= k_star <= len(y_true)

    def test_optimal_threshold_sortscan_comparison_operators(self):
        """Test sort-scan with different comparison operators."""
        y_true, pred_prob = generate_tied_probabilities(30, random_state=42)
        f1_vectorized = get_vectorized_metric("f1")

        for comparison in [">", ">="]:
            threshold, score, k_star = optimal_threshold_sortscan(
                y_true, pred_prob, f1_vectorized, inclusive=comparison
            )

            assert_valid_threshold(threshold)
            assert_valid_metric_score(score, "f1")

    def test_optimal_threshold_sortscan_with_weights(self):
        """Test sort-scan algorithm with sample weights."""
        y_true, pred_prob = generate_binary_data(50, random_state=42)
        weights = generate_sample_weights(len(y_true), "random", random_state=42)

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized, sample_weight=weights
        )

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_optimal_threshold_sortscan_perfect_case(self):
        """Test sort-scan on perfectly separable case."""
        y_true = np.array([0, 0, 1, 1])
        pred_prob = np.array([0.2, 0.4, 0.6, 0.8])

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        assert score == pytest.approx(1.0, abs=1e-10)  # Perfect F1

    def test_optimal_threshold_sortscan_single_sample(self):
        """Test sort-scan with single sample."""
        y_true = np.array([1])
        pred_prob = np.array([0.7])

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        assert k_star == 0 or k_star == 1


class TestVectorizedCounts:
    """Test the vectorized count computation function."""

    def test_vectorized_counts_basic(self):
        """Test basic vectorized count computation."""
        y_true = np.array([0, 1, 0, 1], dtype=np.int8)
        pred_prob = np.array([0.1, 0.8, 0.3, 0.6], dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)

        # Sort by probabilities
        sort_idx = np.argsort(pred_prob)
        y_sorted = y_true[sort_idx]
        w_sorted = weights[sort_idx]

        tp_cumsum, fn_cumsum = _vectorized_counts(y_sorted, w_sorted)

        # Check dimensions
        assert len(tp_cumsum) == len(y_sorted) + 1
        assert len(fn_cumsum) == len(y_sorted) + 1

        # Check that counts are non-negative and non-decreasing
        assert np.all(tp_cumsum >= 0)
        assert np.all(fn_cumsum >= 0)
        assert np.all(np.diff(tp_cumsum) >= 0)
        assert np.all(np.diff(fn_cumsum) <= 0)  # fn decreases

    def test_vectorized_counts_with_weights(self):
        """Test vectorized counts with non-uniform weights."""
        y_true = np.array([0, 1, 1, 0], dtype=np.int8)
        weights = np.array([0.5, 2.0, 1.5, 1.0], dtype=np.float64)

        tp_cumsum, fn_cumsum = _vectorized_counts(y_true, weights)

        # Total positive weight should be preserved
        total_pos_weight = weights[y_true == 1].sum()
        assert fn_cumsum[0] == pytest.approx(total_pos_weight)
        assert tp_cumsum[-1] == pytest.approx(total_pos_weight)


class TestThresholdMidpoint:
    """Test threshold midpoint computation."""

    def test_compute_threshold_midpoint_basic(self):
        """Test basic threshold midpoint computation."""
        sorted_probs = np.array([0.1, 0.3, 0.7, 0.9])

        # Test different k values
        for k in range(len(sorted_probs) + 1):
            threshold = _compute_threshold_midpoint(sorted_probs, k)
            assert_valid_threshold(threshold)

    def test_compute_threshold_midpoint_edge_cases(self):
        """Test threshold computation edge cases."""
        sorted_probs = np.array([0.5])

        # k=0: threshold should be <= min prob
        threshold_0 = _compute_threshold_midpoint(sorted_probs, 0)
        assert threshold_0 <= sorted_probs[0]

        # k=1: threshold should be > max prob
        threshold_1 = _compute_threshold_midpoint(sorted_probs, 1)
        assert threshold_1 > sorted_probs[0]

    def test_compute_threshold_midpoint_tied_values(self):
        """Test threshold computation with tied probability values."""
        sorted_probs = np.array([0.3, 0.5, 0.5, 0.5, 0.7])

        for k in range(len(sorted_probs) + 1):
            threshold = _compute_threshold_midpoint(sorted_probs, k)
            assert_valid_threshold(threshold)


class TestPiecewisePerformance:
    """Test performance characteristics of piecewise optimization."""

    def test_piecewise_performance_scaling(self):
        """Test that piecewise algorithm scales well."""
        sizes = [100, 500, 1000]
        times = {}

        f1_vectorized = get_vectorized_metric("f1")

        for size in sizes:
            y_true, pred_prob = generate_binary_data(size, random_state=42)

            start_time = time.time()
            optimal_threshold_sortscan(y_true, pred_prob, f1_vectorized)
            end_time = time.time()

            times[size] = end_time - start_time

            # Should be reasonably fast
            assert times[size] < 2.0, f"Piecewise took {times[size]:.2f}s for {size} samples"

    def test_piecewise_vs_brute_force_performance(self):
        """Test that piecewise is faster than brute force on large data."""
        # This is more of a sanity check - we can't easily implement
        # true brute force here, but we can verify reasonable performance
        y_true, pred_prob = generate_binary_data(1000, random_state=42)
        f1_vectorized = get_vectorized_metric("f1")

        start_time = time.time()
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )
        end_time = time.time()

        # Should complete quickly for large dataset
        assert end_time - start_time < 1.0
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")


class TestPiecewisePropertyBased:
    """Property-based tests for piecewise optimization."""

    @given(
        st.integers(min_value=10, max_value=100).flatmap(
            lambda n: st.tuples(
                st.just(n),
                st.lists(st.integers(0, 1), min_size=n, max_size=n),
                st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
                        min_size=n, max_size=n),
            )
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_piecewise_always_finds_valid_solution(self, data):
        """Property: piecewise always finds a valid threshold and score."""
        n, y_true, pred_prob = data

        # Ensure both classes are present
        if sum(y_true) == 0:
            y_true[0] = 1
        elif sum(y_true) == len(y_true):
            y_true[0] = 0

        y_true = np.array(y_true, dtype=np.int8)
        pred_prob = np.array(pred_prob, dtype=np.float64)

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")
        assert 0 <= k_star <= len(y_true)

    @given(
        st.integers(min_value=5, max_value=50).flatmap(
            lambda n: st.tuples(
                st.just(n),
                st.lists(st.integers(0, 1), min_size=n, max_size=n),
                st.lists(st.floats(0.1, 0.9, allow_nan=False, allow_infinity=False),
                        min_size=n, max_size=n),
                st.lists(st.floats(0.1, 2.0, allow_nan=False, allow_infinity=False),
                        min_size=n, max_size=n),
            )
        )
    )
    @settings(max_examples=15, deadline=5000)
    def test_piecewise_with_weights_property(self, data):
        """Property: piecewise with weights always produces valid results."""
        n, y_true, pred_prob, weights = data

        # Ensure both classes are present
        if sum(y_true) == 0:
            y_true[0] = 1
        elif sum(y_true) == len(y_true):
            y_true[0] = 0

        y_true = np.array(y_true, dtype=np.int8)
        pred_prob = np.array(pred_prob, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized, sample_weight=weights
        )

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")


class TestPiecewiseEdgeCases:
    """Test edge cases for piecewise optimization."""

    def test_all_same_probabilities(self):
        """Test piecewise with all identical probabilities."""
        y_true = np.array([0, 1, 0, 1])
        pred_prob = np.array([0.5, 0.5, 0.5, 0.5])

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")

    def test_all_same_class(self):
        """Test piecewise with all samples from same class."""
        # All negative
        y_true = np.array([0, 0, 0])
        pred_prob = np.array([0.2, 0.5, 0.8])

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        # F1 should be 0 for all negative case (no true positives possible)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_extreme_probabilities(self):
        """Test piecewise with extreme probability values."""
        y_true = np.array([0, 1, 0, 1])
        pred_prob = np.array([0.0, 1.0, 1e-10, 1.0 - 1e-10])

        f1_vectorized = get_vectorized_metric("f1")
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "f1")
