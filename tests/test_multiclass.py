"""Tests for multiclass classification threshold optimization."""

import numpy as np
import pytest

from optimal_cutoffs import (
    ThresholdOptimizer,
    get_multiclass_confusion_matrix,
    get_optimal_multiclass_thresholds,
    get_optimal_threshold,
    multiclass_metric,
)


def test_multiclass_confusion_matrix():
    """Test multiclass confusion matrix computation."""
    # 3-class problem
    true_labs = np.array([0, 1, 2, 0, 1, 2])
    pred_prob = np.array(
        [
            [0.8, 0.1, 0.1],  # True: 0, should predict 0
            [0.2, 0.7, 0.1],  # True: 1, should predict 1
            [0.1, 0.2, 0.7],  # True: 2, should predict 2
            [0.6, 0.3, 0.1],  # True: 0, should predict 0
            [0.3, 0.6, 0.1],  # True: 1, should predict 1
            [0.2, 0.2, 0.6],  # True: 2, should predict 2
        ]
    )
    thresholds = np.array([0.5, 0.5, 0.5])

    cms = get_multiclass_confusion_matrix(true_labs, pred_prob, thresholds)

    assert len(cms) == 3
    # Each confusion matrix should have 4 elements: (tp, tn, fp, fn)
    for cm in cms:
        assert len(cm) == 4
        assert all(isinstance(x, (int, np.integer)) for x in cm)


def test_multiclass_confusion_matrix_binary_fallback():
    """Test that multiclass confusion matrix works with binary input."""
    true_labs = np.array([0, 1, 1, 0])
    pred_prob = np.array([0.2, 0.8, 0.7, 0.3])
    threshold = np.array([0.5])

    cms = get_multiclass_confusion_matrix(true_labs, pred_prob, threshold)

    assert len(cms) == 1
    assert len(cms[0]) == 4


def test_multiclass_metrics():
    """Test multiclass metric computation."""
    # Create confusion matrices for 3 classes
    cms = [
        (2, 4, 0, 0),  # Class 0: perfect precision/recall
        (1, 3, 1, 1),  # Class 1: some errors
        (1, 4, 0, 1),  # Class 2: some errors
    ]

    # Test macro averaging
    f1_macro = multiclass_metric(cms, "f1", "macro")
    assert 0 <= f1_macro <= 1

    # Test micro averaging
    f1_micro = multiclass_metric(cms, "f1", "micro")
    assert 0 <= f1_micro <= 1

    # Test weighted averaging
    f1_weighted = multiclass_metric(cms, "f1", "weighted")
    assert 0 <= f1_weighted <= 1


def test_get_optimal_multiclass_thresholds():
    """Test multiclass threshold optimization."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 3

    # Generate synthetic data
    true_labs = np.random.randint(0, n_classes, n_samples)
    pred_prob = np.random.rand(n_samples, n_classes)
    # Normalize to make probabilities sum to 1 (more realistic)
    pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

    # Test different methods
    for method in ["smart_brute", "minimize", "gradient"]:
        thresholds = get_optimal_multiclass_thresholds(
            true_labs, pred_prob, metric="f1", method=method
        )

        assert len(thresholds) == n_classes
        assert all(0 <= t <= 1 for t in thresholds)


def test_get_optimal_threshold_multiclass_auto():
    """Test that get_optimal_threshold automatically detects multiclass."""
    np.random.seed(42)
    true_labs = np.array([0, 1, 2, 0, 1, 2])
    pred_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    thresholds = get_optimal_threshold(true_labs, pred_prob, metric="f1")

    assert isinstance(thresholds, np.ndarray)
    assert len(thresholds) == 3
    assert all(0 <= t <= 1 for t in thresholds)


def test_threshold_optimizer_multiclass():
    """Test ThresholdOptimizer with multiclass data."""
    np.random.seed(42)
    true_labs = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    pred_prob = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )

    optimizer = ThresholdOptimizer(objective="f1")
    optimizer.fit(true_labs, pred_prob)

    # Check that multiclass mode was detected
    assert optimizer.is_multiclass_
    assert isinstance(optimizer.threshold_, np.ndarray)
    assert len(optimizer.threshold_) == 3

    # Test prediction
    predictions = optimizer.predict(pred_prob)
    assert len(predictions) == len(true_labs)
    assert all(0 <= p < 3 for p in predictions)
    assert predictions.dtype == int


def test_threshold_optimizer_binary_compatibility():
    """Test that ThresholdOptimizer still works with binary data."""
    true_labs = np.array([0, 1, 1, 0, 1])
    pred_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.9])

    optimizer = ThresholdOptimizer(objective="f1")
    optimizer.fit(true_labs, pred_prob)

    # Check that binary mode was detected
    assert not optimizer.is_multiclass_
    assert isinstance(optimizer.threshold_, float)

    # Test prediction
    predictions = optimizer.predict(pred_prob)
    assert len(predictions) == len(true_labs)
    assert predictions.dtype == bool


def test_multiclass_metrics_averaging_methods():
    """Test different averaging methods for multiclass metrics."""
    # Create realistic confusion matrices
    cms = [
        (10, 80, 5, 5),  # Class 0: good performance, high support
        (3, 87, 3, 7),  # Class 1: medium performance, medium support
        (1, 95, 1, 3),  # Class 2: poor performance, low support
    ]

    f1_macro = multiclass_metric(cms, "f1", "macro")
    f1_micro = multiclass_metric(cms, "f1", "micro")
    f1_weighted = multiclass_metric(cms, "f1", "weighted")

    # All should be valid probabilities
    assert 0 <= f1_macro <= 1
    assert 0 <= f1_micro <= 1
    assert 0 <= f1_weighted <= 1

    # Weighted should be closer to macro than micro in this case
    # (since class 0 has high support and good performance)
    assert abs(f1_weighted - f1_macro) < abs(f1_micro - f1_macro)


def test_multiclass_edge_cases():
    """Test edge cases for multiclass classification."""
    # Test with perfect predictions
    true_labs = np.array([0, 1, 2])
    pred_prob = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    thresholds = get_optimal_multiclass_thresholds(true_labs, pred_prob)
    assert len(thresholds) == 3

    # Test with single class
    true_labs_single = np.array([0, 0, 0])
    pred_prob_single = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ]
    )

    thresholds_single = get_optimal_multiclass_thresholds(
        true_labs_single, pred_prob_single
    )
    assert len(thresholds_single) == 2


def test_new_metrics_registration():
    """Test that new metrics (precision, recall) are properly registered."""
    from optimal_cutoffs import METRIC_REGISTRY

    assert "precision" in METRIC_REGISTRY
    assert "recall" in METRIC_REGISTRY

    # Test the metrics
    tp, tn, fp, fn = 10, 80, 5, 5

    precision = METRIC_REGISTRY["precision"](tp, tn, fp, fn)
    recall = METRIC_REGISTRY["recall"](tp, tn, fp, fn)

    assert precision == 10 / (10 + 5)  # tp / (tp + fp)
    assert recall == 10 / (10 + 5)  # tp / (tp + fn)


if __name__ == "__main__":
    pytest.main([__file__])
