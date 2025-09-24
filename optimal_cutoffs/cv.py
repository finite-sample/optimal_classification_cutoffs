"""Cross-validation helpers for threshold optimization."""

import numpy as np
from sklearn.model_selection import KFold

from .optimizers import _metric_score, get_optimal_threshold
from .types import ArrayLike, OptimizationMethod


def cv_threshold_optimization(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "smart_brute",
    cv: int = 5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an optimal threshold using cross-validation.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Predicted probabilities from a classifier.
    metric:
        Metric name to optimize; must exist in the metric registry.
    method:
        Optimization strategy passed to
        :func:`~optimal_cutoffs.optimizers.get_optimal_threshold`.
    cv:
        Number of folds for :class:`~sklearn.model_selection.KFold` cross-validation.
    random_state:
        Seed for the cross-validator shuffling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of per-fold thresholds and scores.
    """

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
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "smart_brute",
    inner_cv: int = 5,
    outer_cv: int = 5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested cross-validation for threshold optimization.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Predicted probabilities from a classifier.
    metric:
        Metric name to optimize.
    method:
        Optimization strategy passed to
        :func:`~optimal_cutoffs.optimizers.get_optimal_threshold`.
    inner_cv:
        Number of folds in the inner loop used to estimate thresholds.
    outer_cv:
        Number of outer folds for unbiased performance assessment.
    random_state:
        Seed for the cross-validators.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of outer-fold thresholds and scores.
    """

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
