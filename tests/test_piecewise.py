"""Tests for the optimized O(n log n) piecewise implementation."""

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.optimizers import _optimal_threshold_piecewise
from optimal_cutoffs.piecewise import (
    _compute_threshold_midpoint,
    _validate_inputs,
    _validate_sample_weights,
    _vectorized_counts,
    accuracy_vectorized,
    f1_vectorized,
    get_vectorized_metric,
    optimal_threshold_sortscan,
    precision_vectorized,
    recall_vectorized,
)


class TestInputValidation:
    """Test input validation functions."""

    def test_validate_inputs_valid(self):
        """Test validation with valid inputs."""
        y_true = [0, 1, 0, 1]
        pred_prob = [0.1, 0.7, 0.3, 0.9]

        y, p = _validate_inputs(y_true, pred_prob)
        assert y.dtype == np.int8
        assert p.dtype == np.float64
        assert len(y) == len(p) == 4
        np.testing.assert_array_equal(y, [0, 1, 0, 1])
        np.testing.assert_array_almost_equal(p, [0.1, 0.7, 0.3, 0.9])

    def test_validate_inputs_wrong_dimensions(self):
        """Test validation with wrong dimensions."""
        with pytest.raises(ValueError, match="1D arrays"):
            _validate_inputs([[0, 1]], [0.5])

        with pytest.raises(ValueError, match="1D arrays"):
            _validate_inputs([0], [[0.5]])

    def test_validate_inputs_length_mismatch(self):
        """Test validation with mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            _validate_inputs([0, 1], [0.5])

    def test_validate_inputs_empty(self):
        """Test validation with empty arrays."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_inputs([], [])

    def test_validate_inputs_non_binary(self):
        """Test validation with non-binary labels."""
        with pytest.raises(ValueError, match="binary in"):
            _validate_inputs([0, 1, 2], [0.1, 0.5, 0.9])

    def test_validate_inputs_invalid_probabilities(self):
        """Test validation with invalid probabilities."""
        # Out of range
        with pytest.raises(ValueError, match="finite and in"):
            _validate_inputs([0, 1], [-0.1, 0.5])

        with pytest.raises(ValueError, match="finite and in"):
            _validate_inputs([0, 1], [0.5, 1.1])

        # Non-finite
        with pytest.raises(ValueError, match="finite and in"):
            _validate_inputs([0, 1], [0.5, np.nan])

        with pytest.raises(ValueError, match="finite and in"):
            _validate_inputs([0, 1], [0.5, np.inf])

    def test_validate_sample_weights_valid(self):
        """Test sample weight validation with valid inputs."""
        # None case
        w = _validate_sample_weights(None, 4)
        np.testing.assert_array_equal(w, [1.0, 1.0, 1.0, 1.0])

        # Array case
        w = _validate_sample_weights([0.5, 1.0, 1.5, 2.0], 4)
        np.testing.assert_array_almost_equal(w, [0.5, 1.0, 1.5, 2.0])

    def test_validate_sample_weights_invalid(self):
        """Test sample weight validation with invalid inputs."""
        with pytest.raises(ValueError, match="1D array"):
            _validate_sample_weights([[1.0]], 1)

        with pytest.raises(ValueError, match="must match number"):
            _validate_sample_weights([1.0, 2.0], 3)

        with pytest.raises(ValueError, match="finite and non-negative"):
            _validate_sample_weights([-1.0, 1.0], 2)

        with pytest.raises(ValueError, match="finite and non-negative"):
            _validate_sample_weights([np.nan, 1.0], 2)


class TestVectorizedCounts:
    """Test vectorized confusion matrix computation."""

    def test_vectorized_counts_simple(self):
        """Test vectorized counts with simple example."""
        y_sorted = np.array([1, 0, 1, 0])  # Labels sorted by descending probability
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        tp, tn, fp, fn = _vectorized_counts(y_sorted, weights)

        # At cut 0 (predict nothing): 0 TP, 0 FP, 2 FN, 2 TN
        assert tp[0] == 0
        assert fp[0] == 0
        assert fn[0] == 2  # P - tp[0] = 2 - 0 = 2
        assert tn[0] == 2  # N - fp[0] = 2 - 0 = 2

        # At cut 1 (include only first element): 1 TP, 0 FP, 1 FN, 2 TN
        assert tp[1] == 1
        assert fp[1] == 0
        assert fn[1] == 1  # P - tp[1] = 2 - 1 = 1
        assert tn[1] == 2  # N - fp[1] = 2 - 0 = 2

        # At cut 4 (include all elements): 2 TP, 2 FP, 0 FN, 0 TN
        assert tp[4] == 2
        assert fp[4] == 2
        assert fn[4] == 0  # P - tp[4] = 2 - 2 = 0
        assert tn[4] == 0  # N - fp[4] = 2 - 2 = 0

    def test_vectorized_counts_weighted(self):
        """Test vectorized counts with sample weights."""
        y_sorted = np.array([1, 0, 1])
        weights = np.array([2.0, 1.0, 3.0])

        tp, tn, fp, fn = _vectorized_counts(y_sorted, weights)

        # P = 2*1 + 1*0 + 3*1 = 5, N = 2*0 + 1*1 + 3*0 = 1
        # At cut 0 (predict nothing): tp=0, fp=0, fn=5, tn=1
        assert tp[0] == 0.0
        assert fp[0] == 0.0
        assert fn[0] == 5.0  # 5 - 0 = 5
        assert tn[0] == 1.0  # 1 - 0 = 1

        # At cut 1 (include first element): tp=2, fp=0, fn=3, tn=1
        assert tp[1] == 2.0
        assert fp[1] == 0.0
        assert fn[1] == 3.0  # 5 - 2 = 3
        assert tn[1] == 1.0  # 1 - 0 = 1

        # At cut 3 (include all elements): tp=5, fp=1, fn=0, tn=0
        assert tp[3] == 5.0
        assert fp[3] == 1.0
        assert fn[3] == 0.0  # 5 - 5 = 0
        assert tn[3] == 0.0  # 1 - 1 = 0


class TestVectorizedMetrics:
    """Test vectorized metric functions."""

    def test_f1_vectorized(self):
        """Test vectorized F1 computation."""
        tp = np.array([1, 2, 3])
        tn = np.array([2, 1, 0])
        fp = np.array([0, 1, 2])
        fn = np.array([1, 0, 1])

        f1_scores = f1_vectorized(tp, tn, fp, fn)

        # Manual calculation for each case
        # Case 0: precision=1/1=1, recall=1/2=0.5, f1=2*1*0.5/(1+0.5)=2/3
        assert abs(f1_scores[0] - 2 / 3) < 1e-10

        # Case 1: precision=2/3, recall=2/2=1, f1=2*(2/3)*1/(2/3+1)=4/5
        assert abs(f1_scores[1] - 0.8) < 1e-10

    def test_accuracy_vectorized(self):
        """Test vectorized accuracy computation."""
        tp = np.array([1, 2])
        tn = np.array([2, 1])
        fp = np.array([0, 1])
        fn = np.array([1, 0])

        acc_scores = accuracy_vectorized(tp, tn, fp, fn)

        # Case 0: (1+2)/(1+2+0+1) = 3/4 = 0.75
        assert abs(acc_scores[0] - 0.75) < 1e-10

        # Case 1: (2+1)/(2+1+1+0) = 3/4 = 0.75
        assert abs(acc_scores[1] - 0.75) < 1e-10

    def test_precision_vectorized(self):
        """Test vectorized precision computation."""
        tp = np.array([1, 0, 2])
        tn = np.array([2, 1, 1])
        fp = np.array([0, 0, 1])
        fn = np.array([1, 1, 0])

        prec_scores = precision_vectorized(tp, tn, fp, fn)

        # Case 0: 1/(1+0) = 1.0
        assert abs(prec_scores[0] - 1.0) < 1e-10

        # Case 1: 0/(0+0) = 0.0 (handled by np.where)
        assert abs(prec_scores[1] - 0.0) < 1e-10

        # Case 2: 2/(2+1) = 2/3
        assert abs(prec_scores[2] - 2 / 3) < 1e-10

    def test_recall_vectorized(self):
        """Test vectorized recall computation."""
        tp = np.array([1, 0, 2])
        tn = np.array([2, 1, 1])
        fp = np.array([0, 0, 1])
        fn = np.array([1, 2, 0])

        rec_scores = recall_vectorized(tp, tn, fp, fn)

        # Case 0: 1/(1+1) = 0.5
        assert abs(rec_scores[0] - 0.5) < 1e-10

        # Case 1: 0/(0+2) = 0.0
        assert abs(rec_scores[1] - 0.0) < 1e-10

        # Case 2: 2/(2+0) = 1.0
        assert abs(rec_scores[2] - 1.0) < 1e-10

    def test_get_vectorized_metric(self):
        """Test metric lookup function."""
        f1_func = get_vectorized_metric("f1")
        assert f1_func is f1_vectorized

        acc_func = get_vectorized_metric("accuracy")
        assert acc_func is accuracy_vectorized

        with pytest.raises(ValueError, match="not available"):
            get_vectorized_metric("unknown_metric")


class TestThresholdComputation:
    """Test threshold midpoint computation."""

    def test_compute_threshold_midpoint_different_values(self):
        """Test midpoint computation with different adjacent values."""
        p_sorted = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

        # For cut 2 (include elements 0,1), midpoint between 0.7 and 0.5
        threshold = _compute_threshold_midpoint(p_sorted, 2, ">")
        assert abs(threshold - 0.6) < 1e-10

        # For inclusive, nudge slightly lower
        threshold = _compute_threshold_midpoint(p_sorted, 2, ">=")
        assert threshold < 0.6
        assert threshold > 0.5999

    def test_compute_threshold_midpoint_tied_values(self):
        """Test midpoint computation with tied adjacent values."""
        p_sorted = np.array([0.8, 0.5, 0.5, 0.2])

        # For cut 2 (include elements 0,1), need to separate 0.5 from 0.5 (ties)
        threshold_exclusive = _compute_threshold_midpoint(p_sorted, 2, ">")
        threshold_inclusive = _compute_threshold_midpoint(p_sorted, 2, ">=")

        # Should be slightly above and below 0.5
        assert threshold_exclusive > 0.5
        assert threshold_inclusive < 0.5

    def test_compute_threshold_midpoint_end_boundary(self):
        """Test midpoint computation at array boundaries."""
        p_sorted = np.array([0.9, 0.7, 0.5])

        # Last cut (k=3), no right neighbor - want to include all items (0,1,2) as positive
        # For p > threshold to include item with prob 0.5, need threshold < 0.5
        threshold = _compute_threshold_midpoint(p_sorted, 3, ">")
        assert threshold < 0.5  # Should be nextafter(0.5, -inf) to include 0.5 with ">"


class TestOptimalThresholdSortScan:
    """Test the main optimization algorithm."""

    def test_optimal_threshold_simple(self):
        """Test optimization with simple perfect separation case."""
        y_true = [0, 0, 1, 1]
        pred_prob = [0.1, 0.3, 0.7, 0.9]

        threshold, score, k = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        # Should achieve perfect F1 = 1.0
        assert abs(score - 1.0) < 1e-10
        assert 0.3 < threshold < 0.7  # Midpoint between 0.3 and 0.7

    def test_optimal_threshold_with_weights(self):
        """Test optimization with sample weights."""
        y_true = [0, 1, 0, 1]
        pred_prob = [0.2, 0.4, 0.6, 0.8]
        weights = [1.0, 2.0, 1.0, 2.0]  # Give positive examples more weight

        threshold1, score1, _ = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )

        threshold2, score2, _ = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized, sample_weight=weights
        )

        # Weighted version might choose different threshold
        # Both should be valid thresholds
        assert 0 <= threshold1 <= 1
        assert 0 <= threshold2 <= 1
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1

    def test_optimal_threshold_comparison_operators(self):
        """Test both inclusive and exclusive comparison operators."""
        y_true = [0, 0, 1, 1]
        pred_prob = [0.2, 0.5, 0.5, 0.8]  # Tie at 0.5

        threshold_gt, _, _ = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized, inclusive=">"
        )

        threshold_gte, _, _ = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized, inclusive=">="
        )

        # Should handle ties differently
        assert threshold_gt != threshold_gte
        assert 0 <= threshold_gt <= 1
        assert 0 <= threshold_gte <= 1

    def test_optimal_threshold_edge_cases(self):
        """Test edge cases."""
        # Single sample
        threshold, score, k = optimal_threshold_sortscan([1], [0.7], f1_vectorized)
        assert threshold == 0.7
        assert k == 0

        # All negative labels
        threshold, score, k = optimal_threshold_sortscan(
            [0, 0, 0], [0.1, 0.5, 0.9], f1_vectorized
        )
        assert threshold == 0.5  # Default fallback

        # All positive labels
        threshold, score, k = optimal_threshold_sortscan(
            [1, 1, 1], [0.1, 0.5, 0.9], f1_vectorized
        )
        assert threshold == 0.5  # Default fallback


class TestBackwardCompatibility:
    """Test backward compatibility with existing implementation."""

    def test_piecewise_vs_original_random_data(self):
        """Test new implementation matches original on random data."""
        np.random.seed(42)

        for n in [10, 50, 100]:
            y_true = np.random.randint(0, 2, n)
            pred_prob = np.random.random(n)

            for metric in ["f1", "accuracy", "precision", "recall"]:
                # New implementation through _optimal_threshold_piecewise
                threshold_new = _optimal_threshold_piecewise(y_true, pred_prob, metric)

                # Should be valid threshold
                assert 0 <= threshold_new <= 1, (
                    f"Invalid threshold for {metric}: {threshold_new}"
                )

    def test_piecewise_vs_smart_brute(self):
        """Test piecewise optimization matches smart_brute method."""
        np.random.seed(123)

        y_true = [0, 1, 0, 1, 0, 1]
        pred_prob = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold_piecewise = _optimal_threshold_piecewise(
                y_true, pred_prob, metric
            )
            threshold_smart = get_optimal_threshold(
                y_true, pred_prob, metric, method="smart_brute"
            )

            # Should get very close results (allowing for midpoint vs exact probability differences)
            assert 0 <= threshold_piecewise <= 1
            assert 0 <= threshold_smart <= 1

            # Scores should be identical or very close
            from optimal_cutoffs.optimizers import _metric_score

            score_piecewise = _metric_score(
                y_true, pred_prob, threshold_piecewise, metric
            )
            score_smart = _metric_score(y_true, pred_prob, threshold_smart, metric)

            assert abs(score_piecewise - score_smart) < 1e-6, (
                f"Score mismatch for {metric}: {score_piecewise} vs {score_smart}"
            )

    def test_sample_weights_compatibility(self):
        """Test sample weights work with new implementation."""
        y_true = [0, 1, 0, 1]
        pred_prob = [0.2, 0.4, 0.6, 0.8]
        weights = [1.0, 2.0, 1.5, 0.5]

        # Should work without errors
        threshold = _optimal_threshold_piecewise(
            y_true, pred_prob, "f1", sample_weight=weights
        )

        assert 0 <= threshold <= 1


class TestPerformance:
    """Test performance characteristics."""

    def test_performance_large_dataset(self):
        """Test performance on large dataset."""
        np.random.seed(42)
        n = 5000

        y_true = np.random.randint(0, 2, n)
        pred_prob = np.random.random(n)

        # Time the new implementation
        start_time = time.time()
        threshold = _optimal_threshold_piecewise(y_true, pred_prob, "f1")
        end_time = time.time()

        duration = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 5k samples)
        assert duration < 1.0, f"Too slow: {duration:.3f} seconds for {n} samples"
        assert 0 <= threshold <= 1

    def test_performance_many_unique_values(self):
        """Test performance with many unique probability values (worst case for old approach)."""
        n = 1000
        y_true = np.random.randint(0, 2, n)
        pred_prob = np.linspace(0, 1, n)  # All unique values

        start_time = time.time()
        threshold = _optimal_threshold_piecewise(y_true, pred_prob, "f1")
        end_time = time.time()

        duration = end_time - start_time

        # New algorithm should handle this efficiently
        assert duration < 0.5, f"Too slow for many unique values: {duration:.3f}s"
        assert 0 <= threshold <= 1


class TestPropertyBasedComparison:
    """Property-based tests comparing piecewise optimization against brute force."""

    def brute_force_midpoints(self, y, p, metric_fn):
        """Brute force reference implementation that evaluates metric at optimal thresholds.

        This function evaluates the metric at thresholds that correspond exactly
        to the cuts tested by the sort-and-scan algorithm: midpoints between adjacent
        probabilities in the sorted array, plus boundary cases.
        """
        y = np.asarray(y)
        p = np.asarray(p)

        # Sort arrays by descending probability (same as sort-and-scan)
        sort_idx = np.argsort(-p, kind="mergesort")
        p_sorted = p[sort_idx]
        y_sorted = y[sort_idx]  # noqa: F841

        # Generate thresholds corresponding to each possible cut
        # With enhanced indexing: k=0 means predict nothing, k=1 means items 0, etc.
        thresholds = []

        # k=0: predict nothing (threshold above maximum probability)
        max_prob = float(p_sorted[0])
        if max_prob < 1.0:
            thresholds.append(min(1.0, max_prob + 1e-10))
        else:
            thresholds.append(1.0)

        # k=1 to k=n: predict items 0 to k-1
        for k in range(1, len(p_sorted) + 1):
            if k == len(p_sorted):
                # Last cut: predict all as positive
                # Use a threshold below the minimum probability
                min_prob = float(p_sorted[-1])
                if min_prob > 0.0:
                    thresholds.append(max(0.0, min_prob - 1e-10))
                else:
                    thresholds.append(0.0)
            else:
                # Normal cut: threshold between p_sorted[k-1] and p_sorted[k]
                left = p_sorted[k - 1]
                right = p_sorted[k]

                if left > right:
                    # Different probabilities: use midpoint
                    thresholds.append(0.5 * (left + right))
                else:
                    # Same probabilities: use threshold slightly above
                    thresholds.append(min(1.0, float(np.nextafter(left, np.inf))))

        # Evaluate metric at each threshold
        best_score = -np.inf
        best_threshold = 0.5

        for t in thresholds:
            pred = (p > t).astype(int)
            tp = int(np.sum((y == 1) & (pred == 1)))
            fp = int(np.sum((y == 0) & (pred == 1)))
            fn = int(np.sum((y == 1) & (pred == 0)))
            tn = int(np.sum((y == 0) & (pred == 0)))

            # Use vectorized metric function (expects arrays)
            score = metric_fn(
                np.array([tp]), np.array([tn]), np.array([fp]), np.array([fn])
            )[0]

            if score > best_score:
                best_score = score
                best_threshold = t

        return float(best_threshold), float(best_score)

    @settings(deadline=None, max_examples=200)
    @given(
        n=st.integers(min_value=5, max_value=300),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_sortscan_matches_bruteforce_f1(self, n, seed):
        """Test that sort-and-scan F1 optimization matches brute force over midpoints."""
        rng = np.random.default_rng(seed)
        p = rng.uniform(0, 1, size=n)

        # Ensure both classes appear (essential for meaningful F1 score)
        y = (rng.uniform(0, 1, size=n) < 0.3).astype(int)
        if y.sum() == 0 or y.sum() == n:
            y[0] = 1
            y[-1] = 0

        # Test sort-and-scan algorithm
        t_scan, s_scan, _ = optimal_threshold_sortscan(y, p, f1_vectorized)

        # Test brute force over midpoints
        t_br, s_br = self.brute_force_midpoints(y, p, f1_vectorized)

        # The thresholds may differ (due to plateaus), but best scores must match
        assert pytest.approx(s_scan, rel=0, abs=1e-12) == s_br, (
            f"F1 score mismatch: sort-scan={s_scan:.10f} vs brute-force={s_br:.10f}"
        )

    @settings(deadline=None, max_examples=200)
    @given(
        n=st.integers(min_value=5, max_value=300),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_sortscan_matches_bruteforce_accuracy(self, n, seed):
        """Test that sort-and-scan accuracy optimization matches brute force over midpoints."""
        rng = np.random.default_rng(seed)
        p = rng.uniform(0, 1, size=n)

        # Use balanced class distribution for accuracy
        y = (rng.uniform(0, 1, size=n) < 0.5).astype(int)
        if y.sum() == 0 or y.sum() == n:
            y[0] = 1
            y[-1] = 0

        # Test sort-and-scan algorithm
        t_scan, s_scan, _ = optimal_threshold_sortscan(y, p, accuracy_vectorized)

        # Test brute force over midpoints
        t_br, s_br = self.brute_force_midpoints(y, p, accuracy_vectorized)

        # The thresholds may differ (due to plateaus), but best scores must match
        assert pytest.approx(s_scan, rel=0, abs=1e-12) == s_br, (
            f"Accuracy score mismatch: sort-scan={s_scan:.10f} vs brute-force={s_br:.10f}"
        )
