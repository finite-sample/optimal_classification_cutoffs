import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold, cv_threshold_optimization


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
