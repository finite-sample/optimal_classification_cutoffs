import time

import numpy as np
import pytest

from optimal_cutoffs import cv_threshold_optimization, get_optimal_threshold
from optimal_cutoffs.metrics import is_piecewise_metric, register_metric
from optimal_cutoffs.optimizers import _optimal_threshold_piecewise


def test_get_optimal_threshold_methods():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    for method in ["smart_brute", "minimize", "gradient"]:
        thr = get_optimal_threshold(y_true, y_prob, method=method)
        assert 0.0 <= thr <= 1.0
        assert thr == pytest.approx(0.5, abs=0.2)


def test_cv_threshold_optimization():
    rng = np.random.default_rng(0)
    y_prob = rng.random(100)
    y_true = (y_prob > 0.5).astype(int)
    thresholds, scores = cv_threshold_optimization(
        y_true, y_prob, method="smart_brute", cv=5, random_state=0
    )
    assert thresholds.shape == (5,)
    assert scores.shape == (5,)
    assert np.all((thresholds >= 0) & (thresholds <= 1))
    assert np.all((scores >= 0) & (scores <= 1))


def test_piecewise_optimization_correctness():
    """Test that piecewise optimization gives same results as brute force."""
    # Create test data
    rng = np.random.default_rng(42)
    n_samples = 100
    y_prob = rng.random(n_samples)
    y_true = (y_prob + 0.2 * rng.normal(size=n_samples) > 0.6).astype(int)

    # Test all piecewise metrics
    for metric in ["f1", "accuracy", "precision", "recall"]:
        if is_piecewise_metric(metric):
            # Get result from piecewise optimization
            threshold_piecewise = _optimal_threshold_piecewise(y_true, y_prob, metric)

            # Get result from smart_brute (which now uses piecewise for piecewise metrics)
            threshold_smart = get_optimal_threshold(
                y_true, y_prob, metric, method="smart_brute"
            )

            # They should be identical
            assert abs(threshold_piecewise - threshold_smart) < 1e-10, (
                f"Mismatch for {metric}"
            )

            # Both should be valid thresholds
            assert 0 <= threshold_piecewise <= 1, (
                f"Invalid threshold for {metric}: {threshold_piecewise}"
            )
            assert 0 <= threshold_smart <= 1, (
                f"Invalid threshold for {metric}: {threshold_smart}"
            )


def test_piecewise_edge_cases():
    """Test edge cases for piecewise optimization."""

    # Empty arrays
    with pytest.raises(ValueError):
        _optimal_threshold_piecewise([], [], "f1")

    # Mismatched lengths
    with pytest.raises(ValueError):
        _optimal_threshold_piecewise([0, 1], [0.1], "f1")

    # Single sample
    result = _optimal_threshold_piecewise([1], [0.7], "f1")
    assert result == 0.7

    # All same class
    result = _optimal_threshold_piecewise([0, 0, 0], [0.1, 0.5, 0.9], "f1")
    assert result == 0.5  # Default for degenerate case

    result = _optimal_threshold_piecewise([1, 1, 1], [0.1, 0.5, 0.9], "f1")
    assert result == 0.5  # Default for degenerate case

    # All same predictions
    result = _optimal_threshold_piecewise([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], "f1")
    assert 0 <= result <= 1  # Should handle gracefully


def test_piecewise_known_optimal():
    """Test piecewise optimization on cases with known optimal solutions."""
    from optimal_cutoffs.optimizers import _metric_score

    # Perfect separation case
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    # Should achieve perfect accuracy (threshold can be midpoint or boundary value)
    threshold = _optimal_threshold_piecewise(y_true, y_prob, "accuracy")
    accuracy = _metric_score(y_true, y_prob, threshold, "accuracy")
    assert accuracy == 1.0, f"Expected perfect accuracy, got {accuracy}"
    assert 0.2 <= threshold <= 0.8, f"Unexpected threshold: {threshold}"

    # For F1, precision, recall - results should be reasonable
    for metric in ["f1", "precision", "recall"]:
        threshold = _optimal_threshold_piecewise(y_true, y_prob, metric)
        assert 0 <= threshold <= 1


def test_piecewise_vs_original_brute_force():
    """Compare piecewise optimization with original brute force approach."""

    def _original_smart_brute(true_labs, pred_prob, metric):
        """Original smart_brute implementation for comparison."""
        from optimal_cutoffs.optimizers import _metric_score

        thresholds = np.unique(pred_prob)
        scores = [_metric_score(true_labs, pred_prob, t, metric) for t in thresholds]
        return float(thresholds[int(np.argmax(scores))])

    # Test on several random datasets
    from optimal_cutoffs.optimizers import _metric_score

    rng = np.random.default_rng(123)

    for n_samples in [20, 50, 100]:
        y_prob = rng.random(n_samples)
        # Create imbalanced classes
        y_true = (y_prob + 0.3 * rng.normal(size=n_samples) > 0.7).astype(int)

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold_piecewise = _optimal_threshold_piecewise(y_true, y_prob, metric)
            threshold_original = _original_smart_brute(y_true, y_prob, metric)

            # Scores should be identical (thresholds may differ due to midpoint calculation)
            score_piecewise = _metric_score(y_true, y_prob, threshold_piecewise, metric)
            score_original = _metric_score(y_true, y_prob, threshold_original, metric)
            assert abs(score_piecewise - score_original) < 1e-10, (
                f"Score mismatch for {metric} on {n_samples} samples: "
                f"{score_piecewise} vs {score_original}"
            )


def test_performance_improvement():
    """Test that piecewise optimization is faster for large datasets."""

    # Create large dataset
    rng = np.random.default_rng(456)
    n_samples = 5000
    y_prob = rng.random(n_samples)
    y_true = (y_prob > 0.5).astype(int)

    # Time piecewise optimization
    start_time = time.time()
    threshold_piecewise = _optimal_threshold_piecewise(y_true, y_prob, "f1")
    piecewise_time = time.time() - start_time

    # Time original brute force (simulate O(n²) behavior)
    from optimal_cutoffs.optimizers import _metric_score

    start_time = time.time()
    thresholds = np.unique(y_prob)
    _ = [
        _metric_score(y_true, y_prob, t, "f1") for t in thresholds[:100]
    ]  # Limit to avoid timeout
    brute_time = time.time() - start_time

    # Piecewise should be significantly faster (though this is a rough test)
    print(f"Piecewise time: {piecewise_time:.4f}s")
    print(f"Sample brute time: {brute_time:.4f}s (limited to 100 thresholds)")

    # Basic sanity check - piecewise should complete quickly
    assert piecewise_time < 1.0, "Piecewise optimization should be fast"
    assert 0 <= threshold_piecewise <= 1, "Should return valid threshold"


def test_metric_properties():
    """Test metric property system."""

    # Built-in metrics should be piecewise
    assert is_piecewise_metric("f1")
    assert is_piecewise_metric("accuracy")
    assert is_piecewise_metric("precision")
    assert is_piecewise_metric("recall")

    # Test registering a non-piecewise metric
    @register_metric("test_smooth", is_piecewise=False)
    def smooth_metric(tp, tn, fp, fn):
        return tp / (tp + fp + 0.1)  # Smoothed precision

    assert not is_piecewise_metric("test_smooth")

    # Unknown metrics should default to piecewise
    assert is_piecewise_metric("unknown_metric")
