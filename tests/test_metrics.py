import numpy as np
import pytest

from optimal_cut_offs import get_confusion_matrix


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
