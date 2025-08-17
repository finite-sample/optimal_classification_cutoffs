import numpy as np
import pytest

from optimal_cutoffs import get_confusion_matrix, get_optimal_threshold
from optimal_cutoffs.metrics import METRIC_REGISTRY, register_metric, register_metrics


def test_confusion_matrix_and_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.6, 0.7, 0.3, 0.4])
    threshold = 0.5
    tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, threshold)
    assert (tp, tn, fp, fn) == (2, 2, 0, 1)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)


def test_metric_registry_and_custom_registration():
    assert "f1" in METRIC_REGISTRY
    assert "accuracy" in METRIC_REGISTRY

    @register_metric("sum_tp_tn")
    def sum_tp_tn(tp, tn, fp, fn):
        return tp + tn

    assert METRIC_REGISTRY["sum_tp_tn"](1, 1, 0, 0) == 2

    def tpr(tp, tn, fp, fn):
        return tp / (tp + fn) if tp + fn > 0 else 0.0

    register_metrics({"tpr": tpr})

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    thr = get_optimal_threshold(y_true, y_prob, metric="tpr")
    assert 0.0 <= thr <= 1.0
