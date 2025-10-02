"""Cross-validation helpers for threshold optimization."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import (  # type: ignore[import-untyped]
    KFold,
    StratifiedKFold,
)

from .metrics import (
    METRIC_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    multiclass_metric,
    multiclass_metric_exclusive,
)
from .optimizers import get_optimal_threshold
from .types import (
    ComparisonOperatorLiteral,
    OptimizationMethodLiteral,
    SampleWeightLike,
)


def cv_threshold_optimization(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethodLiteral = "auto",
    cv: int | Any = 5,
    random_state: int | None = None,
    sample_weight: SampleWeightLike = None,
    *,
    comparison: ComparisonOperatorLiteral = ">",
    average: str = "macro",
    **opt_kwargs: Any,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Estimate optimal threshold(s) using cross-validation.

    Supports both binary and multiclass classification with proper handling
    of all threshold return formats (scalar, array, dict from expected mode).
    Uses StratifiedKFold by default for better class balance preservation.

    Parameters
    ----------
    true_labs : ArrayLike
        Array of true labels (binary or multiclass).
    pred_prob : ArrayLike
        Predicted probabilities. For binary: 1D array. For multiclass: 2D array.
    metric : str, default="f1"
        Metric name to optimize; must exist in the metric registry.
    method : OptimizationMethod, default="auto"
        Optimization strategy passed to get_optimal_threshold.
    cv : int or cross-validator, default=5
        Number of folds or custom cross-validator object.
    random_state : int, optional
        Seed for the cross-validator shuffling.
    sample_weight : ArrayLike, optional
        Sample weights for handling imbalanced datasets.
    comparison : ComparisonOperator, default=">"
        Comparison operator for threshold application.
    average : str, default="macro"
        Averaging strategy for multiclass metrics.
    **opt_kwargs : Any
        Additional arguments passed to get_optimal_threshold.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        Arrays of per-fold thresholds and scores.
    """

    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    # Choose splitter: stratify by default for classification when possible
    if hasattr(cv, "split"):
        splitter = cv  # custom splitter provided
    else:
        n_splits = int(cv)
        if true_labs.ndim == 1 and np.unique(true_labs).size > 1:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    thresholds = []
    scores = []
    for train_idx, test_idx in splitter.split(true_labs, true_labs):
        # Extract training data and weights
        train_weights = None if sample_weight is None else sample_weight[train_idx]
        test_weights = None if sample_weight is None else sample_weight[test_idx]

        result = get_optimal_threshold(
            true_labs[train_idx],
            pred_prob[train_idx],
            metric=metric,
            method=method,
            sample_weight=train_weights,
            comparison=comparison,
            average=average,
            **opt_kwargs,
        )
        thr = _extract_thresholds(result)
        thresholds.append(thr)
        scores.append(
            _evaluate_threshold_on_fold(
                true_labs[test_idx],
                pred_prob[test_idx],
                thr,
                metric=metric,
                average=average,
                sample_weight=test_weights,
                comparison=comparison,
            )
        )
    return np.array(thresholds, dtype=object), np.array(scores, dtype=float)


def nested_cv_threshold_optimization(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethodLiteral = "auto",
    inner_cv: int = 5,
    outer_cv: int = 5,
    random_state: int | None = None,
    sample_weight: SampleWeightLike = None,
    *,
    comparison: ComparisonOperatorLiteral = ">",
    average: str = "macro",
    **opt_kwargs: Any,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Nested cross-validation for unbiased threshold optimization.

    Inner CV selects best threshold, outer CV evaluates performance.
    Uses StratifiedKFold by default for better class balance.

    Parameters
    ----------
    true_labs : ArrayLike
        Array of true labels (binary or multiclass).
    pred_prob : ArrayLike
        Predicted probabilities. For binary: 1D array. For multiclass: 2D array.
    metric : str, default="f1"
        Metric name to optimize.
    method : OptimizationMethod, default="auto"
        Optimization strategy passed to get_optimal_threshold.
    inner_cv : int, default=5
        Number of folds in the inner loop used to estimate thresholds.
    outer_cv : int, default=5
        Number of outer folds for unbiased performance assessment.
    random_state : int, optional
        Seed for the cross-validators.
    sample_weight : ArrayLike, optional
        Sample weights for handling imbalanced datasets.
    comparison : ComparisonOperator, default=">"
        Comparison operator for threshold application.
    average : str, default="macro"
        Averaging strategy for multiclass metrics.
    **opt_kwargs : Any
        Additional arguments passed to get_optimal_threshold.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        Arrays of outer-fold thresholds and scores.
    """

    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    # stratify in outer loop when possible
    if true_labs.ndim == 1 and np.unique(true_labs).size > 1:
        outer = StratifiedKFold(
            n_splits=outer_cv, shuffle=True, random_state=random_state
        )
    else:
        outer = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    outer_thresholds = []
    outer_scores = []
    for train_idx, test_idx in outer.split(true_labs, true_labs):
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
            comparison=comparison,
            average=average,
            **opt_kwargs,
        )
        # Select best threshold from inner CV instead of averaging
        best_idx = int(np.argmax(inner_scores))
        thr = inner_thresholds[best_idx]
        outer_thresholds.append(thr)
        score = _evaluate_threshold_on_fold(
            true_labs[test_idx],
            pred_prob[test_idx],
            thr,
            metric=metric,
            average=average,
            sample_weight=test_weights,
            comparison=comparison,
        )
        outer_scores.append(score)
    return np.array(outer_thresholds, dtype=object), np.array(outer_scores, dtype=float)


# -------------------- helpers --------------------


def _extract_thresholds(thr_result: Any) -> Any:
    """Normalize outputs from get_optimal_threshold into usable thresholds.

    Handles float, ndarray, (thr, score), and dict shapes from 'expected' mode.
    """
    # (thr, score)
    if isinstance(thr_result, tuple) and len(thr_result) == 2:
        return thr_result[0]
    # dict from expected/micro or macro/weighted
    if isinstance(thr_result, dict):
        if "thresholds" in thr_result:
            return thr_result["thresholds"]
        if "threshold" in thr_result:
            return thr_result["threshold"]
        # Bayes with decisions has no thresholds; raise clearly
        if "decisions" in thr_result:
            raise ValueError("Bayes decisions cannot be used for threshold CV scoring.")
    return thr_result


def _evaluate_threshold_on_fold(
    y_true: ArrayLike,
    pred_prob: ArrayLike,
    thr: Any,
    *,
    metric: str,
    average: str,
    sample_weight: ArrayLike | None,
    comparison: ComparisonOperatorLiteral,
) -> float:
    """Compute the chosen metric on the test fold for a given threshold object."""
    y_true = np.asarray(y_true)
    pred_prob = np.asarray(pred_prob)
    sw = None if sample_weight is None else np.asarray(sample_weight)

    if pred_prob.ndim == 1:
        # scalar threshold required
        t = (
            float(thr)
            if not isinstance(thr, dict)
            else float(thr.get("threshold", thr))
        )
        tp, tn, fp, fn = get_confusion_matrix(
            y_true, pred_prob, t, sample_weight=sw, comparison=comparison
        )
        try:
            metric_fn = METRIC_REGISTRY[metric]
        except KeyError as e:
            raise ValueError(f"Unknown metric '{metric}'.") from e
        return float(metric_fn(tp, tn, fp, fn))

    # Multiclass / multilabel (n, K)
    K = pred_prob.shape[1]
    if isinstance(thr, dict):
        if "thresholds" in thr:
            thresholds = np.asarray(thr["thresholds"], dtype=float)
        elif "threshold" in thr:
            # micro: single global threshold – broadcast per class
            thresholds = np.full(K, float(thr["threshold"]), dtype=float)
        else:
            raise ValueError("Unexpected threshold dict shape for multiclass.")
    elif np.isscalar(thr):
        thresholds = np.full(K, float(thr), dtype=float)  # type: ignore[arg-type]
    else:
        thresholds = np.asarray(thr, dtype=float)
        if thresholds.shape != (K,):
            raise ValueError(
                f"Per-class thresholds must have shape ({K},), got {thresholds.shape}."
            )

    if metric == "accuracy":
        # Exclusive accuracy uses the margin-based single-label decision rule
        return float(
            multiclass_metric_exclusive(
                y_true, pred_prob, thresholds, "accuracy", comparison, sw
            )
        )
    cms = get_multiclass_confusion_matrix(
        y_true, pred_prob, thresholds, sample_weight=sw, comparison=comparison
    )
    return float(multiclass_metric(cms, metric, average))
