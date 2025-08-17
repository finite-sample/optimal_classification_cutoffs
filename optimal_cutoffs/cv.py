"""Cross-validation helpers for threshold optimization."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .optimizers import get_optimal_threshold, _metric_score


def cv_threshold_optimization(
    true_labs,
    pred_prob,
    metric="f1",
    method="smart_brute",
    cv=5,
    random_state=None,
):
    """Estimate thresholds via cross-validation and evaluate scores."""

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    thresholds = []
    scores = []
    for train_idx, test_idx in kf.split(true_labs):
        thr = get_optimal_threshold(
            true_labs[train_idx], pred_prob[train_idx], metric=metric, method=method
        )
        thresholds.append(thr)
        score = _metric_score(true_labs[test_idx], pred_prob[test_idx], thr, metric)
        scores.append(score)
    return np.array(thresholds), np.array(scores)


def nested_cv_threshold_optimization(
    true_labs,
    pred_prob,
    metric="f1",
    method="smart_brute",
    inner_cv=5,
    outer_cv=5,
    random_state=None,
):
    """Nested CV for threshold optimization and unbiased performance estimates."""

    outer = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    outer_thresholds = []
    outer_scores = []
    for train_idx, test_idx in outer.split(true_labs):
        inner_thresholds, _ = cv_threshold_optimization(
            true_labs[train_idx],
            pred_prob[train_idx],
            metric=metric,
            method=method,
            cv=inner_cv,
            random_state=random_state,
        )
        thr = float(np.mean(inner_thresholds))
        outer_thresholds.append(thr)
        score = _metric_score(true_labs[test_idx], pred_prob[test_idx], thr, metric)
        outer_scores.append(score)
    return np.array(outer_thresholds), np.array(outer_scores)
