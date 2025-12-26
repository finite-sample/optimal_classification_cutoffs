"""Cross-validation helpers for threshold optimization."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import (
    KFold,  # type: ignore[import-untyped]
    StratifiedKFold,
)

from .api import optimize_thresholds
from .metrics_core import (
    METRICS,
    confusion_matrix_at_threshold,
    multiclass_confusion_matrices_at_thresholds,
    multiclass_metric_ovr,
    multiclass_metric_single_label,
)
from .validation import (
    _validate_averaging_method,
    _validate_comparison_operator,
    _validate_metric_name,
    _validate_optimization_method,
)


def cv_threshold_optimization(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: str = "auto",
    cv: int | Any = 5,
    random_state: int | None = None,
    sample_weight: ArrayLike | None = None,
    *,
    comparison: str = ">",
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
        Optimization strategy passed to optimize_thresholds.
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
        Additional arguments passed to optimize_thresholds.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        Arrays of per-fold thresholds and scores.
    """
    # Validate parameters early for better user experience
    _validate_metric_name(metric)
    _validate_comparison_operator(comparison)
    _validate_averaging_method(average)
    _validate_optimization_method(method)

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

        result = optimize_thresholds(
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
    method: str = "auto",
    inner_cv: int = 5,
    outer_cv: int = 5,
    random_state: int | None = None,
    sample_weight: ArrayLike | None = None,
    *,
    comparison: str = ">",
    average: str = "macro",
    **opt_kwargs: Any,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Nested cross-validation for unbiased threshold optimization.

    Inner CV estimates robust thresholds by averaging across folds, outer CV evaluates
    performance. Uses StratifiedKFold by default for better class balance. The threshold
    selection uses statistically sound averaging rather than cherry-picking the
    best-performing fold.

    Parameters
    ----------
    true_labs : ArrayLike
        Array of true labels (binary or multiclass).
    pred_prob : ArrayLike
        Predicted probabilities. For binary: 1D array. For multiclass: 2D array.
    metric : str, default="f1"
        Metric name to optimize.
    method : OptimizationMethod, default="auto"
        Optimization strategy passed to optimize_thresholds.
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
        Additional arguments passed to optimize_thresholds.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        Arrays of outer-fold thresholds and scores.
    """
    # Validate parameters early for better user experience
    _validate_metric_name(metric)
    _validate_comparison_operator(comparison)
    _validate_averaging_method(average)
    _validate_optimization_method(method)

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
        # Average thresholds across inner folds for robust estimate
        # This is statistically sound - considers all folds rather than cherry-picking
        if isinstance(inner_thresholds[0], float | np.floating):
            # Binary case: simple averaging
            thr = float(np.mean(inner_thresholds))
        elif isinstance(inner_thresholds[0], np.ndarray):
            # Multiclass: average each class threshold
            thr = np.mean(np.vstack(inner_thresholds), axis=0)
        elif isinstance(inner_thresholds[0], dict):
            # Dict format (e.g., from expected mode)
            thr = _average_threshold_dicts(inner_thresholds)
        else:
            # Fallback: try converting to array and averaging
            try:
                thr = np.mean(np.array(inner_thresholds))
            except (ValueError, TypeError):
                # If averaging fails, use mean score to select representative threshold
                mean_score = np.mean(inner_scores)
                closest_idx = np.argmin(np.abs(inner_scores - mean_score))
                thr = inner_thresholds[closest_idx]
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
    """Extract thresholds from OptimizationResult objects.

    Now primarily handles OptimizationResult objects since optimize_thresholds
    returns unified OptimizationResult. Maintains backward compatibility for legacy formats.
    """
    from .core import OptimizationResult

    # OptimizationResult (new unified format)
    if isinstance(thr_result, OptimizationResult):
        return thr_result.thresholds

    # Legacy formats for backward compatibility
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


def _average_threshold_dicts(threshold_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Average dictionary-based thresholds from multiple CV folds.

    Parameters
    ----------
    threshold_dicts : list[dict[str, Any]]
        List of threshold dictionaries from inner CV folds

    Returns
    -------
    dict[str, Any]
        Averaged threshold dictionary with same structure as input
    """
    if not threshold_dicts:
        raise ValueError("Cannot average empty list of threshold dictionaries")

    # Check for consistent structure
    first_dict = threshold_dicts[0]
    for i, d in enumerate(threshold_dicts[1:], 1):
        if set(d.keys()) != set(first_dict.keys()):
            raise ValueError(f"Inconsistent dict keys between folds 0 and {i}")

    result = {}

    # Average numerical values
    for key in first_dict:
        values = [d[key] for d in threshold_dicts]

        if key in ("threshold", "thresholds"):
            # These are the actual threshold values to average
            if isinstance(values[0], float | np.floating):
                result[key] = float(np.mean(values))
            elif isinstance(values[0], np.ndarray):
                result[key] = np.mean(np.vstack(values), axis=0)
            else:
                # Try to convert to array and average
                result[key] = np.mean(np.array(values))
        elif key == "score" or key.endswith("_score"):
            # Don't average scores - they're fold-specific performance
            # Use mean as representative value
            result[key] = float(np.mean(values))
        else:
            # For other keys (like per_class arrays), average if numeric
            try:
                if isinstance(values[0], np.ndarray):
                    result[key] = np.mean(np.vstack(values), axis=0)
                elif isinstance(values[0], int | float | np.number):
                    result[key] = float(np.mean(values))
                else:
                    # Non-numeric data - keep first fold's value
                    result[key] = values[0]
            except (TypeError, ValueError):
                # If averaging fails, keep first fold's value
                result[key] = values[0]

    return result


def _evaluate_threshold_on_fold(
    y_true: ArrayLike,
    pred_prob: ArrayLike,
    thr: Any,
    *,
    metric: str,
    average: str,
    sample_weight: ArrayLike | None,
    comparison: str,
) -> float:
    """Compute the chosen metric on the test fold for a given threshold object."""
    y_true = np.asarray(y_true)
    pred_prob = np.asarray(pred_prob)
    sw = None if sample_weight is None else np.asarray(sample_weight)

    if pred_prob.ndim == 1:
        # scalar threshold required
        if isinstance(thr, dict):
            t = float(thr.get("threshold", thr))
        else:
            # Handle both scalar and array cases
            thr_array = np.asarray(thr)
            t = (
                float(thr_array.item())
                if thr_array.ndim == 0
                else float(thr_array.flat[0])
            )
        tp, tn, fp, fn = confusion_matrix_at_threshold(
            y_true, pred_prob, t, sample_weight=sw, comparison=comparison
        )
        # Metric validation happens early in CV functions - no need to validate again
        metric_fn = METRICS[metric].fn
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
            multiclass_metric_single_label(
                y_true, pred_prob, thresholds, "accuracy", comparison, sw
            )
        )
    cms = multiclass_confusion_matrices_at_thresholds(
        y_true, pred_prob, thresholds, sample_weight=sw, comparison=comparison
    )
    return float(multiclass_metric_ovr(cms, metric, average))
