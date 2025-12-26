import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics import get, register, list_available


def confusion_matrix_at_threshold(y_true, y_prob, threshold):
    """Simple confusion matrix calculation for tests."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return tp, tn, fp, fn

# Local tolerance for test precision
TOLERANCE = 1e-10


def test_confusion_matrix_and_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.6, 0.7, 0.3, 0.4])
    threshold = 0.5
    tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)
    assert (tp, tn, fp, fn) == (2, 2, 0, 1)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)


def test_metric_registry_and_custom_registration():
    # Test that we can get built-in metrics
    f1_metric = get("f1")
    accuracy_metric = get("accuracy")
    
    assert f1_metric is not None
    assert accuracy_metric is not None
    
    # Test custom metric registration
    def sum_tp_tn(tp, tn, fp, fn):
        return tp + tn

    register("sum_tp_tn", sum_tp_tn)
    
    # Test that custom metric was registered
    sum_metric = get("sum_tp_tn")
    assert sum_metric is not None
    assert sum_metric(1, 1, 0, 0) == 2

    def tpr(tp, tn, fp, fn):
        return np.where(tp + fn > 0, tp / (tp + fn), 0.0)

    register("tpr", tpr)

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    result = optimize_thresholds(y_true, y_prob, metric="tpr")
    threshold = result.threshold
    assert -TOLERANCE <= threshold <= 1.0
