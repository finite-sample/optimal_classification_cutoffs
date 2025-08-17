import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from optimal_cut_offs import get_optimal_threshold, cross_validate_thresholds


def sample_data():
    true_labs = [1, 0, 1, 1, 1]
    pred_prob = [0.22, 0.205, 0.49, 0.9, 0.24]
    return true_labs, pred_prob


def test_get_optimal_threshold_methods():
    true_labs, pred_prob = sample_data()
    expected = 0.21
    for method in ('smart_brute', 'minimize', 'gradient'):
        threshold = get_optimal_threshold(true_labs, pred_prob, method)
        assert abs(threshold - expected) < 0.01


def test_cross_validation_ranges():
    true_labs = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    pred_prob = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]
    thresholds, scores = cross_validate_thresholds(true_labs, pred_prob, k=5)
    assert len(thresholds) == 5
    assert len(scores) == 5
    for t in thresholds:
        assert 0 <= t <= 1
    for s in scores:
        assert 0 <= s <= 1
    assert sum(scores) / len(scores) > 0.8
