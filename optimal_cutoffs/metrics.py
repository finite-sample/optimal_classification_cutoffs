"""Metric registry, confusion matrix utilities, and built-in metrics."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import (
    _validate_comparison_operator,
    validate_binary_classification,
    validate_multiclass_classification,
    validate_threshold,
)

# ============================================================================
# Metric Registry
# ============================================================================


@dataclass
class MetricInfo:
    """Complete information about a metric."""

    scalar_fn: Callable[[float, float, float, float], float]
    vectorized_fn: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None
    ) = None
    is_piecewise: bool = True
    maximize: bool = True
    needs_proba: bool = False


# Unified metrics registry
METRICS: dict[str, MetricInfo] = {}


def register_metric(
    name: str | None = None,
    func: Callable[[float, float, float, float], float] | None = None,
    vectorized_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    | None = None,
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> (
    Callable[[float, float, float, float], float]
    | Callable[
        [Callable[[float, float, float, float], float]],
        Callable[[float, float, float, float], float],
    ]
):
    """Register a metric function with optional vectorized version.

    Parameters
    ----------
    name : str, optional
        Key under which to store the metric. If not provided, uses function's __name__.
    func : callable, optional
        Metric callable accepting (tp, tn, fp, fn) scalars and returning a float.
    vectorized_func : callable, optional
        Vectorized version accepting arrays. Used for O(n log n) optimization.
    is_piecewise : bool, default=True
        Whether metric is piecewise-constant w.r.t. threshold changes.
    maximize : bool, default=True
        Whether to maximize (True) or minimize (False) the metric.
    needs_proba : bool, default=False
        Whether metric requires probability scores (e.g., log-loss, Brier score).

    Returns
    -------
    callable or decorator
        The registered function or a decorator if func is None.
    """
    if func is not None:
        metric_name = name or func.__name__
        METRICS[metric_name] = MetricInfo(
            scalar_fn=func,
            vectorized_fn=vectorized_func,
            is_piecewise=is_piecewise,
            maximize=maximize,
            needs_proba=needs_proba,
        )
        # Clear cache when registry changes
        get_metric_function.cache_clear()
        return func

    def decorator(
        f: Callable[[float, float, float, float], float],
    ) -> Callable[[float, float, float, float], float]:
        metric_name = name or f.__name__
        METRICS[metric_name] = MetricInfo(
            scalar_fn=f,
            vectorized_fn=vectorized_func,
            is_piecewise=is_piecewise,
            maximize=maximize,
            needs_proba=needs_proba,
        )
        get_metric_function.cache_clear()
        return f

    return decorator


def register_alias(alias_name: str, target_name: str) -> None:
    """Register an alias for an existing metric.

    Parameters
    ----------
    alias_name : str
        The alias name to register.
    target_name : str
        The name of the existing metric to point to.

    Raises
    ------
    ValueError
        If the target metric doesn't exist.
    """
    if target_name not in METRICS:
        available = sorted(METRICS.keys())
        preview = ", ".join(available[:10])
        raise ValueError(
            f"Target metric '{target_name}' not found. Available: [{preview}]"
        )
    METRICS[alias_name] = METRICS[target_name]
    get_metric_function.cache_clear()


def register_metrics(
    metrics: dict[str, Callable[[float, float, float, float], float]],
    vectorized_metrics: dict[
        str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ]
    | None = None,
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> None:
    """Register multiple metric functions at once.

    Parameters
    ----------
    metrics : dict
        Mapping of metric names to scalar functions.
    vectorized_metrics : dict, optional
        Mapping of metric names to vectorized implementations.
    is_piecewise : bool, default=True
        Whether metrics are piecewise-constant.
    maximize : bool, default=True
        Whether metrics should be maximized.
    needs_proba : bool, default=False
        Whether metrics require probability scores.
    """
    for name, scalar_fn in metrics.items():
        vectorized_fn = vectorized_metrics.get(name) if vectorized_metrics else None
        METRICS[name] = MetricInfo(
            scalar_fn=scalar_fn,
            vectorized_fn=vectorized_fn,
            is_piecewise=is_piecewise,
            maximize=maximize,
            needs_proba=needs_proba,
        )
    get_metric_function.cache_clear()


@lru_cache(maxsize=128)
def get_metric_function(
    metric_name: str, vectorized: bool = False
) -> Callable[..., Any]:
    """Get metric function with caching for hot paths.

    Parameters
    ----------
    metric_name : str
        Name of the metric.
    vectorized : bool, default=False
        If True, return vectorized version.

    Returns
    -------
    callable
        The requested metric function.

    Raises
    ------
    ValueError
        If metric doesn't exist or vectorized version not available.
    """
    if metric_name not in METRICS:
        available = sorted(METRICS.keys())
        preview = ", ".join(available[:10])
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available (first 10): [{preview}] ... ({len(available)} total)"
        )

    info = METRICS[metric_name]

    if vectorized:
        if info.vectorized_fn is None:
            raise ValueError(
                f"Vectorized implementation not available for '{metric_name}'"
            )
        return info.vectorized_fn

    return info.scalar_fn


def is_piecewise_metric(metric_name: str) -> bool:
    """Check if a metric is piecewise-constant."""
    return METRICS.get(metric_name, MetricInfo(lambda *_: 0.0)).is_piecewise


def should_maximize_metric(metric_name: str) -> bool:
    """Check if a metric should be maximized."""
    return METRICS.get(metric_name, MetricInfo(lambda *_: 0.0)).maximize


def needs_probability_scores(metric_name: str) -> bool:
    """Check if a metric needs probability scores."""
    return METRICS.get(metric_name, MetricInfo(lambda *_: 0.0)).needs_proba


def has_vectorized_implementation(metric_name: str) -> bool:
    """Check if a metric has a vectorized implementation."""
    if metric_name not in METRICS:
        return False
    return METRICS[metric_name].vectorized_fn is not None


# ============================================================================
# Metric Computation Helpers
# ============================================================================


def _safe_div(
    numerator: np.ndarray | float, denominator: np.ndarray | float
) -> np.ndarray | float:
    """Safe division that returns 0 when denominator is 0, handles inf/nan cases.
    
    This function provides safe division for metric computation where:
    - Division by zero returns 0.0 (common convention for precision/recall)
    - Division by negative numbers works normally
    - Handles inf/nan cases appropriately
    """
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        num = np.asarray(numerator, dtype=float)
        den = np.asarray(denominator, dtype=float)
        
        # Use numpy's divide with proper handling of zero denominators
        result = np.zeros_like(num, dtype=float)
        valid_mask = den != 0
        
        # Perform division only where denominator is non-zero
        result = np.divide(num, den, out=result, where=valid_mask)
        
        # Handle any remaining inf/nan values (e.g., from inf/inf)
        result = np.where(np.isfinite(result), result, 0.0)
        
        return result
    else:
        # Scalar case
        if denominator == 0:
            return 0.0
        result = numerator / denominator
        # Handle inf/nan cases in scalar arithmetic
        return result if np.isfinite(result) else 0.0


def _scalarize(
    vectorized_func: Callable[..., np.ndarray],
) -> Callable[..., float]:
    """Convert a vectorized metric function to a scalar version."""

    def scalar_wrapper(
        tp: int | float, tn: int | float, fp: int | float, fn: int | float
    ) -> float:
        tp_arr = np.array([tp], dtype=float)
        tn_arr = np.array([tn], dtype=float)
        fp_arr = np.array([fp], dtype=float)
        fn_arr = np.array([fn], dtype=float)

        result_arr = vectorized_func(tp_arr, tn_arr, fp_arr, fn_arr)
        return float(result_arr[0])

    return scalar_wrapper


# ============================================================================
# Vectorized Metric Implementations
# ============================================================================


def _f1_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Vectorized F1 score: 2*TP / (2*TP + FP + FN)."""
    return cast(np.ndarray, _safe_div(2 * tp, 2 * tp + fp + fn))


def _accuracy_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Vectorized accuracy: (TP + TN) / (TP + TN + FP + FN)."""
    return cast(np.ndarray, _safe_div(tp + tn, tp + tn + fp + fn))


def _precision_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Vectorized precision: TP / (TP + FP)."""
    return cast(np.ndarray, _safe_div(tp, tp + fp))


def _recall_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Vectorized recall: TP / (TP + FN)."""
    return cast(np.ndarray, _safe_div(tp, tp + fn))


def _iou_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Vectorized IoU/Jaccard: TP / (TP + FP + FN)."""
    return cast(np.ndarray, _safe_div(tp, tp + fp + fn))


def _specificity_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Vectorized specificity: TN / (TN + FP)."""
    return cast(np.ndarray, _safe_div(tn, tn + fp))


# Scalar metric functions derived from vectorized implementations
f1_score = _scalarize(_f1_vectorized)
f1_score.__doc__ = "F1 score (harmonic mean of precision and recall)."

accuracy_score = _scalarize(_accuracy_vectorized)
accuracy_score.__doc__ = "Classification accuracy."

precision_score = _scalarize(_precision_vectorized)
precision_score.__doc__ = "Precision (positive predictive value)."

recall_score = _scalarize(_recall_vectorized)
recall_score.__doc__ = "Recall (sensitivity, true positive rate)."

jaccard_score = _scalarize(_iou_vectorized)
jaccard_score.__doc__ = "Jaccard index (IoU score)."

specificity_score = _scalarize(_specificity_vectorized)
specificity_score.__doc__ = "Specificity (true negative rate)."


# ============================================================================
# Confusion Matrix Computation
# ============================================================================


def confusion_matrix_from_predictions(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix from binary predictions (no thresholding).

    This is the canonical single-pass implementation used throughout the codebase.
    Uses optimized bincount approach (4x faster than boolean masking).

    Parameters
    ----------
    true_labels : array-like
        True binary labels (0 or 1)
    pred_labels : array-like
        Predicted binary labels (0 or 1)
    sample_weight : array-like, optional
        Sample weights. If None, uniform weights are used.

    Returns
    -------
    tuple[float, float, float, float]
        (tp, tn, fp, fn) - Always returns floats for consistency.

    Examples
    --------
    >>> true = [0, 1, 0, 1, 1]
    >>> pred = [0, 1, 1, 1, 0]
    >>> tp, tn, fp, fn = confusion_matrix_from_predictions(true, pred)
    >>> (tp, tn, fp, fn)
    (2.0, 1.0, 1.0, 1.0)
    """
    true_labels = np.asarray(true_labels, dtype=np.int8)
    pred_labels = np.asarray(pred_labels, dtype=np.int8)

    weights = (
        np.ones_like(true_labels, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    # Single-pass optimization: 2-bit encoding (true*2 + pred)
    # 0=TN, 1=FP, 2=FN, 3=TP
    combined = true_labels * 2 + pred_labels
    counts = np.bincount(combined, weights=weights, minlength=4)

    return float(counts[3]), float(counts[0]), float(counts[1]), float(counts[2])


def confusion_matrix_at_threshold(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    threshold: float,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    require_proba: bool = True,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix by applying threshold to probabilities.

    Parameters
    ----------
    true_labels : array-like
        True binary labels in {0, 1}.
    pred_proba : array-like
        Predicted probabilities in [0, 1] (if require_proba=True) or scores.
    threshold : float
        Decision threshold.
    sample_weight : array-like, optional
        Sample weights.
    comparison : {">" or ">="}, default=">"
        Comparison operator for thresholding.
    require_proba : bool, default=True
        If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns
    -------
    tuple[float, float, float, float]
        (tp, tn, fp, fn) - Always returns floats for consistency.
    """
    # Validate inputs
    true_labels, pred_proba, sample_weight = validate_binary_classification(
        true_labels, pred_proba, sample_weight, require_proba=require_proba
    )

    # Validate threshold
    if require_proba:
        validate_threshold(threshold, allow_epsilon_outside=False)

    _validate_comparison_operator(comparison)

    # Apply threshold
    if comparison == ">":
        pred_labels = (pred_proba > threshold).astype(np.int8)
    else:  # ">="
        pred_labels = (pred_proba >= threshold).astype(np.int8)

    # Use optimized computation
    return confusion_matrix_from_predictions(true_labels, pred_labels, sample_weight)


def compute_vectorized_confusion_matrices(
    y_sorted: NDArray[np.int8], weights_sorted: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Compute confusion matrices for all possible thresholds using cumulative sums.

    Given labels and weights sorted by descending probabilities, returns
    (tp, tn, fp, fn) as vectors for every possible cut k.

    Convention:
    - Index 0: Predict nothing as positive (all negative predictions)
    - Index k (k > 0): Predict first k items as positive

    Parameters
    ----------
    y_sorted : NDArray[np.int8]
        Binary labels sorted by descending probability.
    weights_sorted : NDArray[np.float64]
        Sample weights sorted by descending probability.

    Returns
    -------
    tuple[NDArray[np.float64], ...]
        Arrays of (tp, tn, fp, fn) for each threshold. Length is n+1.
    """
    # Total positive and negative weights
    P = float(np.sum(weights_sorted * y_sorted))
    N = float(np.sum(weights_sorted * (1 - y_sorted)))

    # Cumulative sums
    tp_cumsum = np.cumsum(weights_sorted * y_sorted)
    fp_cumsum = np.cumsum(weights_sorted * (1 - y_sorted))

    # Include "predict nothing" at index 0
    tp = np.concatenate([[0.0], tp_cumsum])
    fp = np.concatenate([[0.0], fp_cumsum])

    # Complement counts
    fn = P - tp
    tn = N - fp

    return tp, tn, fp, fn


def apply_metric_to_confusion_counts(
    metric_fn: Callable[[NDArray, NDArray, NDArray, NDArray], NDArray],
    tp: NDArray[np.float64],
    tn: NDArray[np.float64],
    fp: NDArray[np.float64],
    fn: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply vectorized metric function to confusion matrix counts.

    Parameters
    ----------
    metric_fn : callable
        Vectorized metric accepting (tp, tn, fp, fn) arrays.
    tp, tn, fp, fn : NDArray[np.float64]
        Confusion matrix count arrays.

    Returns
    -------
    NDArray[np.float64]
        Array of metric scores.

    Raises
    ------
    ValueError
        If metric function returns wrong shape.
    """
    scores = metric_fn(tp, tn, fp, fn)
    scores = np.asarray(scores)

    if scores.shape != tp.shape:
        raise ValueError(
            f"metric_fn must return array with shape {tp.shape}, got {scores.shape}"
        )

    return scores


# ============================================================================
# Multiclass Helper Functions
# ============================================================================


def compute_exclusive_predictions(
    pred_prob: np.ndarray,
    thresholds: np.ndarray,
    comparison: str = ">",
) -> np.ndarray:
    """Predict class with highest margin: argmax(p_j - tau_j).

    Falls back to argmax(p_j) when no class exceeds its threshold.

    Note: Margin-based decisions can select lower-probability classes with
    better margins. See documentation for details on decision rules.

    Parameters
    ----------
    pred_prob : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    thresholds : np.ndarray
        Per-class thresholds (n_classes,)
    comparison : str
        Comparison operator (">" or ">=")

    Returns
    -------
    np.ndarray
        Predicted class labels (n_samples,)
    """
    margins = pred_prob - thresholds  # broadcast
    mask = margins > 0 if comparison == ">" else margins >= 0

    # Argmax of margins where valid; -inf elsewhere
    masked_margins = np.where(mask, margins, -np.inf)
    best_by_margin = np.argmax(masked_margins, axis=1)
    any_above = np.any(mask, axis=1)
    best_by_prob = np.argmax(pred_prob, axis=1)

    return np.where(any_above, best_by_margin, best_by_prob)


def ovr_confusion_counts(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_label: int,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute TP, FP, FN for one class vs rest (skip TN for efficiency).

    Parameters
    ----------
    true_labels : np.ndarray
        True class labels
    pred_labels : np.ndarray
        Predicted class labels
    class_label : int
        Class to compute metrics for
    weights : np.ndarray, optional
        Sample weights

    Returns
    -------
    tuple[float, float, float]
        (tp, fp, fn) - TN not computed as it's often unused in OvR metrics
    """
    if weights is None:
        weights = np.ones_like(true_labels, dtype=float)

    true_positive_mask = true_labels == class_label
    pred_positive_mask = pred_labels == class_label

    tp = float(np.sum(weights[true_positive_mask & pred_positive_mask]))
    fp = float(np.sum(weights[~true_positive_mask & pred_positive_mask]))
    fn = float(np.sum(weights[true_positive_mask & ~pred_positive_mask]))

    return tp, fp, fn


# ============================================================================
# High-Level Metric Computation
# ============================================================================


def compute_metric_at_threshold(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    threshold: float,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
) -> float:
    """Compute metric score at a given threshold.

    Parameters
    ----------
    true_labels : array-like
        True binary labels
    pred_proba : array-like
        Predicted probabilities
    threshold : float
        Decision threshold
    metric : str, default="f1"
        Metric name (must be registered)
    sample_weight : array-like, optional
        Sample weights
    comparison : str, default=">"
        Comparison operator

    Returns
    -------
    float
        Metric score at the threshold
    """
    tp, tn, fp, fn = confusion_matrix_at_threshold(
        true_labels, pred_proba, threshold, sample_weight, comparison
    )

    metric_func = get_metric_function(metric, vectorized=False)
    return float(metric_func(tp, tn, fp, fn))


def multiclass_metric_single_label(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    thresholds: ArrayLike,
    metric_name: str,
    comparison: str = ">",
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute exclusive single-label multiclass metrics.

    Uses margin-based decision rule: predict class with highest margin (p_j - tau_j).
    Computes sample-level accuracy or macro-averaged precision/recall/F1.

    Parameters
    ----------
    true_labels : array-like
        True class labels (n_samples,)
    pred_proba : array-like
        Predicted probabilities (n_samples, n_classes)
    thresholds : array-like
        Per-class thresholds (n_classes,)
    metric_name : str
        Metric to compute ("accuracy", "f1", "precision", "recall")
    comparison : str, default=">"
        Comparison operator
    sample_weight : array-like, optional
        Sample weights

    Returns
    -------
    float
        Computed metric value
    """
    true_labels = np.asarray(true_labels)
    pred_proba = np.asarray(pred_proba)
    thresholds = np.asarray(thresholds)

    # Get exclusive predictions
    pred_labels = compute_exclusive_predictions(pred_proba, thresholds, comparison)

    if metric_name == "accuracy":
        # Sample-level accuracy
        correct = true_labels == pred_labels
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            return float(np.average(correct, weights=sample_weight))
        else:
            return float(np.mean(correct))

    # Macro-averaged metrics
    labels = np.unique(true_labels.astype(int))
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    metric_func = get_metric_function(metric_name, vectorized=False)
    per_class = []

    for c in labels:
        true_binary = (true_labels == c).astype(int)
        pred_binary = (pred_labels == c).astype(int)

        tp, tn, fp, fn = confusion_matrix_from_predictions(true_binary, pred_binary, sw)
        # TN not meaningful in macro-averaged OvR, pass 0
        per_class.append(metric_func(tp, 0, fp, fn))

    return float(np.mean(per_class) if per_class else 0.0)


def multiclass_metric_ovr(
    confusion_matrices: list[tuple[float, float, float, float]],
    metric_name: str,
    average: str = "macro",
) -> float | np.ndarray:
    """Compute multiclass metrics from per-class confusion matrices (OvR).

    Parameters
    ----------
    confusion_matrices : list of tuple
        List of per-class (tp, tn, fp, fn) tuples
    metric_name : str
        Metric name (must be registered)
    average : {"macro", "micro", "weighted", "none"}, default="macro"
        Averaging strategy

    Returns
    -------
    float or np.ndarray
        Aggregated metric (float) or per-class scores (array if average="none")

    Raises
    ------
    ValueError
        If metric doesn't support requested averaging or is unknown
    """
    metric_func = get_metric_function(metric_name, vectorized=False)

    if average == "macro":
        scores = [metric_func(*cm) for cm in confusion_matrices]
        return float(np.mean(scores))

    elif average == "micro":
        # Sum only TP, FP, FN (TN inflated in OvR)
        total_tp = sum(cm[0] for cm in confusion_matrices)
        total_fp = sum(cm[2] for cm in confusion_matrices)
        total_fn = sum(cm[3] for cm in confusion_matrices)

        # Compute micro metrics directly
        if metric_name == "precision":
            return float(
                total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
            )
        elif metric_name in ("recall", "sensitivity", "tpr"):
            return float(
                total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
            )
        elif metric_name == "f1":
            precision = (
                total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
            )
            recall = (
                total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
            )
            return float(
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        elif metric_name == "accuracy":
            raise ValueError(
                "Micro-averaged accuracy requires exclusive single-label predictions. "
                "Use multiclass_metric_single_label() instead."
            )
        else:
            raise ValueError(
                f"Micro-averaged '{metric_name}' is not defined in OvR. "
                "Supported: 'precision', 'recall', 'f1'."
            )

    elif average == "weighted":
        scores = []
        supports = []
        for cm in confusion_matrices:
            tp, tn, fp, fn = cm
            scores.append(metric_func(*cm))
            supports.append(tp + fn)  # True positives for this class

        total_support = sum(supports)
        if total_support == 0:
            return 0.0

        weighted_score = (
            sum(
                score * support
                for score, support in zip(scores, supports, strict=False)
            )
            / total_support
        )
        return float(weighted_score)

    elif average == "none":
        scores = [metric_func(*cm) for cm in confusion_matrices]
        return np.array(scores)

    else:
        raise ValueError(
            f"Unknown averaging method: {average}. "
            f"Must be one of: 'macro', 'micro', 'weighted', 'none'."
        )


def compute_multiclass_metrics_from_labels(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    metric: str = "f1",
    average: str = "macro",
    sample_weight: ArrayLike | None = None,
    n_classes: int | None = None,
) -> float | np.ndarray:
    """Compute multiclass metrics from true and predicted labels.

    Parameters
    ----------
    true_labels : array-like
        True class labels
    pred_labels : array-like
        Predicted class labels
    metric : str, default="f1"
        Metric to compute
    average : str, default="macro"
        Averaging strategy
    sample_weight : array-like, optional
        Sample weights
    n_classes : int, optional
        Number of classes (inferred if None)

    Returns
    -------
    float or np.ndarray
        Computed metric score
    """
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    if true_labels.shape != pred_labels.shape:
        raise ValueError("true_labels and pred_labels must have same shape")

    weights = (
        np.ones_like(true_labels, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    if weights.shape[0] != true_labels.shape[0]:
        raise ValueError("sample_weight must have same length as labels")

    if n_classes is None:
        n_classes = (
            int(max(true_labels.max(initial=-1), pred_labels.max(initial=-1))) + 1
        )

    # Special case: accuracy (computed directly)
    if metric == "accuracy":
        correct = (true_labels == pred_labels).astype(float)
        return float(np.average(correct, weights=weights))

    # Build OvR confusion matrices
    cms: list[tuple[float, float, float, float]] = []
    for k in range(n_classes):
        true_bin = (true_labels == k).astype(int)
        pred_bin = (pred_labels == k).astype(int)
        tp, tn, fp, fn = confusion_matrix_from_predictions(true_bin, pred_bin, weights)
        cms.append((tp, tn, fp, fn))

    return multiclass_metric_ovr(cms, metric_name=metric, average=average)


def multiclass_confusion_matrices_at_thresholds(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    thresholds: ArrayLike,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    require_proba: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Compute per-class confusion matrices for multiclass (OvR).

    Parameters
    ----------
    true_labels : array-like
        True class labels
    pred_proba : array-like
        Predicted probabilities (n_samples, n_classes) or scores
    thresholds : array-like
        Per-class thresholds
    sample_weight : array-like, optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    require_proba : bool, default=False
        If True, enforce [0,1] range

    Returns
    -------
    list[tuple[float, float, float, float]]
        List of per-class (tp, tn, fp, fn) tuples
    """
    # Validate inputs
    true_labels, pred_proba, sample_weight = validate_multiclass_classification(
        true_labels, pred_proba, sample_weight, require_proba=require_proba
    )
    _validate_comparison_operator(comparison)

    if pred_proba.ndim == 1:
        # Binary case
        thr_arr = np.asarray(thresholds)
        thr_scalar = float(thr_arr.reshape(-1)[0])
        return [
            confusion_matrix_at_threshold(
                true_labels,
                pred_proba,
                thr_scalar,
                sample_weight,
                comparison,
                require_proba=require_proba,
            )
        ]

    # Multiclass case
    n_classes = pred_proba.shape[1]
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.shape != (n_classes,):
        raise ValueError(
            f"thresholds must have shape ({n_classes},), got {thresholds.shape}"
        )

    if require_proba:
        validate_threshold(thresholds, n_classes)

    confusion_matrices = []

    for class_idx in range(n_classes):
        true_binary = (true_labels == class_idx).astype(int)
        pred_binary_proba = pred_proba[:, class_idx]
        threshold = thresholds[class_idx]

        cm = confusion_matrix_at_threshold(
            true_binary,
            pred_binary_proba,
            threshold,
            sample_weight,
            comparison,
            require_proba=require_proba,
        )
        confusion_matrices.append(cm)

    return confusion_matrices


# ============================================================================
# Metric Factory Functions
# ============================================================================


def make_linear_counts_metric(
    w_tp: float = 0.0,
    w_tn: float = 0.0,
    w_fp: float = 0.0,
    w_fn: float = 0.0,
    name: str | None = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Create a vectorized linear utility metric from confusion matrix.

    Returns: metric(tp, tn, fp, fn) = w_tp*tp + w_tn*tn + w_fp*fp + w_fn*fn

    Parameters
    ----------
    w_tp, w_tn, w_fp, w_fn : float
        Weights for each confusion matrix component
    name : str, optional
        If provided, automatically registers the metric

    Returns
    -------
    callable
        Vectorized metric function

    Examples
    --------
    >>> # Cost-sensitive: FN costs 5x more than FP
    >>> metric = make_linear_counts_metric(w_fp=-1.0, w_fn=-5.0, name="cost_5to1")
    >>> # Now can use: optimize_threshold(y, y_pred, metric="cost_5to1")
    """

    def _metric(
        tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
    ) -> np.ndarray:
        """Vectorized linear combination of confusion matrix counts."""
        return (
            w_tp * np.asarray(tp, dtype=float)
            + w_tn * np.asarray(tn, dtype=float)
            + w_fp * np.asarray(fp, dtype=float)
            + w_fn * np.asarray(fn, dtype=float)
        )

    # Auto-register if name provided
    if name is not None:
        scalar_fn = _scalarize(_metric)
        register_metric(
            name=name,
            func=scalar_fn,
            vectorized_func=_metric,
            is_piecewise=True,
            maximize=True,
            needs_proba=False,
        )

    return _metric


def make_cost_metric(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    name: str | None = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Create a vectorized cost-sensitive metric.

    Returns: tp_benefit*TP + tn_benefit*TN - fp_cost*FP - fn_cost*FN

    Parameters
    ----------
    fp_cost : float
        Cost of false positives (positive value)
    fn_cost : float
        Cost of false negatives (positive value)
    tp_benefit : float, default=0.0
        Benefit for true positives
    tn_benefit : float, default=0.0
        Benefit for true negatives
    name : str, optional
        If provided, automatically registers the metric

    Returns
    -------
    callable
        Vectorized metric function

    Examples
    --------
    >>> # Classic cost-sensitive
    >>> metric = make_cost_metric(fp_cost=1.0, fn_cost=5.0, name="cost_sensitive")
    >>> # Now can use: optimize_threshold(y, y_pred, metric="cost_sensitive")
    """
    return make_linear_counts_metric(
        w_tp=tp_benefit,
        w_tn=tn_benefit,
        w_fp=-fp_cost,
        w_fn=-fn_cost,
        name=name,
    )


# ============================================================================
# Register Built-in Metrics
# ============================================================================

register_metric("f1", f1_score, _f1_vectorized)
register_metric("accuracy", accuracy_score, _accuracy_vectorized)
register_metric("precision", precision_score, _precision_vectorized)
register_metric("recall", recall_score, _recall_vectorized)
register_metric("iou", jaccard_score, _iou_vectorized)
register_metric("specificity", specificity_score, _specificity_vectorized)

# Register aliases
register_alias("jaccard", "iou")
register_alias("tnr", "specificity")
register_alias("ppv", "precision")
register_alias("tpr", "recall")
register_alias("sensitivity", "recall")
