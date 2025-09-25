"""Cross-validation helpers for threshold optimization."""

from typing import Any

import numpy as np
from sklearn.model_selection import KFold  # type: ignore[import-untyped]

from .optimizers import _metric_score, get_optimal_threshold
from .types import ArrayLike, OptimizationMethod, SampleWeightLike


def cv_threshold_optimization(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "smart_brute",
    cv: int = 5,
    random_state: int | None = None,
    sample_weight: SampleWeightLike = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    sample_weight:
        Optional array of sample weights for handling imbalanced datasets.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        Arrays of per-fold thresholds and scores.
    """

    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    thresholds = []
    scores = []
    for train_idx, test_idx in kf.split(true_labs):
        # Extract training data and weights
        train_weights = None if sample_weight is None else sample_weight[train_idx]
        test_weights = None if sample_weight is None else sample_weight[test_idx]

        thr = get_optimal_threshold(
            true_labs[train_idx],
            pred_prob[train_idx],
            metric=metric,
            method=method,
            sample_weight=train_weights,
        )
        thresholds.append(thr)
        # Convert threshold to float if needed for binary classification
        thr_float = float(thr) if isinstance(thr, np.ndarray) and thr.size == 1 else thr
        if not isinstance(thr_float, float):
            # For multiclass, we need to adjust this
            thr_float = 0.5  # fallback
        score = _metric_score(
            true_labs[test_idx], pred_prob[test_idx], thr_float, metric, test_weights
        )
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
    sample_weight: SampleWeightLike = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    sample_weight:
        Optional array of sample weights for handling imbalanced datasets.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        Arrays of outer-fold thresholds and scores.
    """

    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    outer = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    outer_thresholds = []
    outer_scores = []
    for train_idx, test_idx in outer.split(true_labs):
        # Extract training and test data with weights
        train_weights = None if sample_weight is None else sample_weight[train_idx]
        test_weights = None if sample_weight is None else sample_weight[test_idx]

        inner_thresholds, inner_scores = cv_threshold_optimization(
            true_labs[train_idx],
            pred_prob[train_idx],
            metric=metric,
            method=method,
            cv=inner_cv,
            random_state=random_state,
            sample_weight=train_weights,
        )
        # Select best threshold from inner CV instead of averaging
        best_idx = int(np.argmax(inner_scores))
        thr = float(inner_thresholds[best_idx])
        outer_thresholds.append(thr)
        score = _metric_score(
            true_labs[test_idx], pred_prob[test_idx], thr, metric, test_weights
        )
        outer_scores.append(score)
    return np.array(outer_thresholds), np.array(outer_scores)
