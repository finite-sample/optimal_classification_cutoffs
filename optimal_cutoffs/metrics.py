"""Metric registry, confusion matrix utilities, and built-in metrics."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import (
    _validate_comparison_operator,
    _validate_threshold,
    validate_inputs,
)


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
    name:
        Optional key under which to store the metric. If not provided the
        function's ``__name__`` is used.
    func:
        Metric callable accepting ``tp, tn, fp, fn`` scalars and returning a float.
        When supplied the function is registered immediately. If omitted, the
        returned decorator can be used to annotate a metric function.
    vectorized_func:
        Optional vectorized version of the metric that accepts ``tp, tn, fp, fn``
        as arrays and returns an array of scores. Used for O(n log n) optimization.
    is_piecewise:
        Whether the metric is piecewise-constant with respect to threshold changes.
        Piecewise metrics can be optimized using O(n log n) algorithms.
    maximize:
        Whether the metric should be maximized (True) or minimized (False).
    needs_proba:
        Whether the metric requires probability scores rather than just thresholds.
        Used for metrics like log-loss or Brier score.

    Returns
    -------
    Callable[[float, float, float, float], float] | Callable[[Callable[[float, float, float, float], float]], Callable[[float, float, float, float], float]]
        The registered function or decorator.
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
        return f

    return decorator


def register_alias(alias_name: str, target_name: str) -> None:
    """Register an alias for an existing metric.

    Parameters
    ----------
    alias_name:
        The alias name to register.
    target_name:
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
    """Register multiple metric functions.

    Parameters
    ----------
    metrics:
        Mapping of metric names to callables that accept ``tp, tn, fp, fn``.
    vectorized_metrics:
        Optional mapping of metric names to vectorized implementations.
    is_piecewise:
        Whether the metrics are piecewise-constant with respect to threshold changes.
    maximize:
        Whether the metrics should be maximized (True) or minimized (False).
    needs_proba:
        Whether the metrics require probability scores rather than just thresholds.

    Returns
    -------
    None
        This function mutates the global registry in-place.
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


def is_piecewise_metric(metric_name: str) -> bool:
    """Check if a metric is piecewise-constant.

    Parameters
    ----------
    metric_name:
        Name of the metric to check.

    Returns
    -------
    bool
        True if the metric is piecewise-constant, False otherwise.
        Defaults to True for unknown metrics.
    """
    if metric_name not in METRICS:
        return True  # Default for unknown metrics
    return METRICS[metric_name].is_piecewise


def should_maximize_metric(metric_name: str) -> bool:
    """Check if a metric should be maximized.

    Parameters
    ----------
    metric_name:
        Name of the metric to check.

    Returns
    -------
    bool
        True if the metric should be maximized, False if minimized.
        Defaults to True for unknown metrics.
    """
    if metric_name not in METRICS:
        return True  # Default for unknown metrics
    return METRICS[metric_name].maximize


def needs_probability_scores(metric_name: str) -> bool:
    """Check if a metric needs probability scores rather than just thresholds.

    Parameters
    ----------
    metric_name:
        Name of the metric to check.

    Returns
    -------
    bool
        True if the metric needs probability scores, False otherwise.
        Defaults to False for unknown metrics.
    """
    if metric_name not in METRICS:
        return False  # Default for unknown metrics
    return METRICS[metric_name].needs_proba


def has_vectorized_implementation(metric_name: str) -> bool:
    """Check if a metric has a vectorized implementation available.

    Parameters
    ----------
    metric_name:
        Name of the metric to check.

    Returns
    -------
    bool
        True if the metric has a vectorized implementation, False otherwise.
    """
    if metric_name not in METRICS:
        return False
    return METRICS[metric_name].vectorized_fn is not None


def get_vectorized_metric(metric_name: str) -> Callable[..., Any]:
    """Get vectorized version of a metric function.

    Parameters
    ----------
    metric_name:
        Name of the metric.

    Returns
    -------
    Callable[..., Any]
        Vectorized metric function that accepts arrays.

    Raises
    ------
    ValueError
        If metric is not available in vectorized form.
    """
    if metric_name not in METRICS:
        available = sorted(METRICS.keys())
        preview = ", ".join(available[:10])
        raise ValueError(f"Unknown metric '{metric_name}'. Available: [{preview}]")

    vectorized_fn = METRICS[metric_name].vectorized_fn
    if vectorized_fn is None:
        available_vectorized = [
            name for name, info in METRICS.items() if info.vectorized_fn is not None
        ]
        preview = ", ".join(available_vectorized[:10])
        raise ValueError(
            f"Vectorized implementation not available for metric '{metric_name}'. "
            f"Available vectorized (first 10): [{preview}] ... (total={len(available_vectorized)})"
        )

    return vectorized_fn


def _safe_div(
    numerator: np.ndarray[Any, Any] | float, denominator: np.ndarray[Any, Any] | float
) -> np.ndarray[Any, Any] | float:
    """Safe division that returns 0 when denominator is 0."""
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        # Ensure both are arrays for vectorized operations
        num = np.asarray(numerator, dtype=float)
        den = np.asarray(denominator, dtype=float)
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    else:
        # Scalar case
        return numerator / denominator if denominator > 0 else 0.0


def _scalarize(
    vectorized_func: Callable[..., np.ndarray[Any, Any]],
) -> Callable[..., float]:
    """Convert a vectorized metric function to a scalar version.

    Parameters
    ----------
    vectorized_func:
        A function that accepts (tp, tn, fp, fn) as arrays and returns array of scores.

    Returns
    -------
    Callable[..., float]
        A function that accepts (tp, tn, fp, fn) as scalars and returns a scalar score.
    """

    def scalar_wrapper(
        tp: int | float, tn: int | float, fp: int | float, fn: int | float
    ) -> float:
        # Convert scalars to single-element arrays
        tp_arr = np.array([tp], dtype=float)
        tn_arr = np.array([tn], dtype=float)
        fp_arr = np.array([fp], dtype=float)
        fn_arr = np.array([fn], dtype=float)

        # Call vectorized function and extract scalar result
        result_arr = vectorized_func(tp_arr, tn_arr, fp_arr, fn_arr)
        return float(result_arr[0])

    return scalar_wrapper


# Vectorized metric implementations for O(n log n) optimization
def _f1_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized F1 score computation using cleaner formula: 2*TP / (2*TP + FP + FN)."""
    return cast(np.ndarray[Any, Any], _safe_div(2 * tp, 2 * tp + fp + fn))


def _accuracy_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized accuracy computation."""
    return cast(np.ndarray[Any, Any], _safe_div(tp + tn, tp + tn + fp + fn))


def _precision_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized precision computation."""
    return cast(np.ndarray[Any, Any], _safe_div(tp, tp + fp))


def _recall_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized recall computation."""
    return cast(np.ndarray[Any, Any], _safe_div(tp, tp + fn))


def _iou_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized IoU/Jaccard score computation."""
    return cast(np.ndarray[Any, Any], _safe_div(tp, tp + fp + fn))


def _specificity_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized specificity computation."""
    return cast(np.ndarray[Any, Any], _safe_div(tn, tn + fp))


# Scalar metric functions derived from vectorized implementations
f1_score = _scalarize(_f1_vectorized)
f1_score.__doc__ = r"""Compute the F\ :sub:`1` score.

    Derived from vectorized implementation for consistency.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        The harmonic mean of precision and recall.
    """


accuracy_score = _scalarize(_accuracy_vectorized)
accuracy_score.__doc__ = """Compute classification accuracy.

    Derived from vectorized implementation for consistency.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Ratio of correct predictions to total samples.
    """


precision_score = _scalarize(_precision_vectorized)
precision_score.__doc__ = """Compute precision (positive predictive value).

    Derived from vectorized implementation for consistency.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Ratio of true positives to predicted positives.
    """


recall_score = _scalarize(_recall_vectorized)
recall_score.__doc__ = """Compute recall (sensitivity, true positive rate).

    Derived from vectorized implementation for consistency.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Ratio of true positives to actual positives.
    """


jaccard_score = _scalarize(_iou_vectorized)
jaccard_score.__doc__ = """Compute the Jaccard index (IoU score).

    Derived from vectorized implementation for consistency.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Intersection over Union: tp / (tp + fp + fn).
    """


specificity_score = _scalarize(_specificity_vectorized)
specificity_score.__doc__ = """Compute the specificity (true negative rate).

    Derived from vectorized implementation for consistency.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        True negative rate: tn / (tn + fp).
    """


def _compute_exclusive_predictions(
    pred_prob: np.ndarray[Any, Any],
    thresholds: np.ndarray[Any, Any],
    comparison: str = ">",
) -> np.ndarray[Any, Any]:
    """Compute exclusive single-label predictions using margin-based decision rule.

    **Decision Rule**:
    1. Compute margins: margin_j = p_j - tau_j for each class j
    2. Among classes with margin > 0 (or >= 0), predict the one with highest margin
    3. If no class has positive margin, predict the class with highest probability

    **Important**: This margin-based rule can sometimes select a class with lower
    absolute probability but higher margin. For example, if p_1=0.49, tau_1=0.3
    (margin=0.19) and p_3=0.41, tau_3=0.2 (margin=0.21), it will predict class 3
    despite class 1 having higher probability. This behavior is intentional for
    threshold-optimized classification but differs from standard argmax predictions.

    **When This Matters**:
    - Accuracy computations using this rule may differ from standard multiclass accuracy
    - Users comparing with argmax-based predictions may see different results
    - This is the correct behavior for optimized per-class thresholds

    **Performance**: This vectorized implementation is significantly faster than the
    previous Python loop version, especially for large datasets.

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
    n_samples, _ = pred_prob.shape
    margins = pred_prob - thresholds  # broadcast: (n_samples, n_classes)

    if comparison == ">":
        mask = margins > 0
    else:
        mask = margins >= 0

    # Argmax of margins where mask is True; set others to -inf
    masked_margins = np.where(mask, margins, -np.inf)
    best_by_margin = np.argmax(masked_margins, axis=1)
    any_above = np.any(mask, axis=1)
    best_by_prob = np.argmax(pred_prob, axis=1)

    return np.where(any_above, best_by_margin, best_by_prob)


def multiclass_metric_single_label(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
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
    true_labs : ArrayLike
        True class labels (n_samples,)
    pred_prob : ArrayLike
        Predicted probabilities (n_samples, n_classes)
    thresholds : ArrayLike
        Per-class thresholds (n_classes,)
    metric_name : str
        Metric to compute ("accuracy", "f1", "precision", "recall")
    comparison : str
        Comparison operator (">" or ">=")
    sample_weight : ArrayLike | None
        Optional sample weights

    Returns
    -------
    float
        Computed metric value
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    thresholds = np.asarray(thresholds)

    # Get exclusive predictions
    pred_labels = _compute_exclusive_predictions(pred_prob, thresholds, comparison)

    if metric_name == "accuracy":
        # Sample-level accuracy
        correct = true_labs == pred_labels
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            return float(np.average(correct, weights=sample_weight))
        else:
            return float(np.mean(correct))
    else:
        # Macro-averaged precision/recall/F1 without external deps
        # Use labels present in y_true (to mirror sklearn's default behavior)
        labels = np.unique(true_labs.astype(int))
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        per_class = []
        for c in labels:
            # Create binary labels for this class vs all others
            true_binary = (true_labs == c).astype(int)
            pred_binary = (pred_labels == c).astype(int)
            
            # Use centralized confusion matrix calculation
            tp, tn, fp, fn = _confusion_matrix_from_labels(true_binary, pred_binary, sw)
            
            # Get metric function from registry
            if metric_name not in METRICS:
                raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {list(METRICS.keys())}")
            
            metric_func = METRICS[metric_name].scalar_fn
            # For single-label metrics, we pass tn=0 since it's not meaningful in this context
            per_class.append(metric_func(tp, 0, fp, fn))
        return float(np.mean(per_class) if per_class else 0.0)


def multiclass_metric_ovr(
    confusion_matrices: list[tuple[int | float, int | float, int | float, int | float]],
    metric_name: str,
    average: str = "macro",
) -> float | np.ndarray[Any, Any]:
    """Compute multiclass metrics from per-class confusion matrices.

    Parameters
    ----------
    confusion_matrices:
        List of per-class confusion matrix tuples ``(tp, tn, fp, fn)``.
    metric_name:
        Name of the metric to compute (must be in METRIC_REGISTRY).
    average:
        Averaging strategy: "macro", "micro", "weighted", or "none".
        - "macro": Unweighted mean of per-class metrics (treats all classes equally)
        - "micro": Global metric computed on pooled confusion matrix
          (treats all samples equally, OvR multilabel)
        - "weighted": Weighted mean by support (number of true instances per class)
        - "none": No averaging, returns array of per-class metrics

        Note: For exclusive single-label accuracy, use multiclass_metric_single_label().

    Returns
    -------
    float | np.ndarray
        Aggregated metric score (float) or per-class scores (array) if average="none".
    """
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric: {metric_name}")

    metric_func = METRICS[metric_name].scalar_fn

    if average == "macro":
        # Unweighted mean of per-class scores
        scores = [metric_func(*cm) for cm in confusion_matrices]
        return float(np.mean(scores))

    elif average == "micro":
        # For micro averaging, sum only TP, FP, FN
        # (not TN which is inflated in One-vs-Rest)
        total_tp = sum(cm[0] for cm in confusion_matrices)
        total_fp = sum(cm[2] for cm in confusion_matrices)
        total_fn = sum(cm[3] for cm in confusion_matrices)

        # Compute micro metrics directly
        if metric_name == "precision":
            return float(
                total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
            )
        elif metric_name == "recall":
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
            # For accuracy in multiclass, we need exclusive single-label predictions
            # The OvR aggregation gives Jaccard/IoU, not accuracy
            # We should compute accuracy using exclusive predictions instead
            raise ValueError(
                "Micro-averaged accuracy requires exclusive single-label predictions. "
                "Use multiclass_metric_single_label() instead, or use 'macro' "
                "averaging "
                "which computes per-class accuracies independently."
            )
        else:
            # No generic micro reduction for other metrics
            raise ValueError(
                f"Micro-averaged '{metric_name}' is not defined in OvR counts. "
                "Supported micro metrics: 'precision', 'recall', 'f1'."
            )

    elif average == "weighted":
        # Weighted by support (number of true instances for each class)
        scores = []
        supports = []
        for cm in confusion_matrices:
            tp, tn, fp, fn = cm
            scores.append(metric_func(*cm))
            supports.append(tp + fn)  # actual positives for this class

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
        # No averaging: return per-class scores
        scores = [metric_func(*cm) for cm in confusion_matrices]
        return np.array(scores)

    else:
        raise ValueError(
            f"Unknown averaging method: {average}. "
            f"Must be one of: 'macro', 'micro', 'weighted', 'none'."
        )


def get_confusion_matrix(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    prob: float,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    require_proba: bool = True,
) -> tuple[int | float, int | float, int | float, int | float]:
    """Compute confusion-matrix counts for a given threshold.

    Parameters
    ----------
    true_labs:
        Array of true binary labels in {0, 1}.
    pred_prob:
        Array of predicted probabilities in [0, 1] (if require_proba=True)
        or scores (if require_proba=False).
    prob:
        Decision threshold applied to ``pred_prob``.
    sample_weight:
        Optional array of sample weights. If None, all samples have equal weight.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).
        - ">": pred_prob > threshold (default, excludes ties)
        - ">=": pred_prob >= threshold (includes ties)
    require_proba:
        Whether to enforce [0,1] range for pred_prob and threshold.
        Set to False for score-based empirical workflows.

    Returns
    -------
    tuple[int | float, int | float, int | float, int | float]
        Counts ``(tp, tn, fp, fn)``. Returns int when sample_weight is None,
        float when sample_weight is provided to preserve fractional weighted counts.
    """
    # Validate inputs
    true_labs, pred_prob, sample_weight = validate_inputs(
        true_labs,
        pred_prob,
        sample_weight,
        require_binary=True,
        allow_multiclass=False,
    )

    # Validate threshold bounds for public API by default
    if require_proba:
        _validate_threshold(float(prob))
    # For score-based workflows, allow thresholds outside [0,1]

    _validate_comparison_operator(comparison)

    # Apply threshold with specified comparison operator
    if comparison == ">":
        pred_labs = (pred_prob > prob).astype(int)
    else:  # ">="
        pred_labs = (pred_prob >= prob).astype(int)

    # Use the centralized confusion matrix function
    tp, tn, fp, fn = _confusion_matrix_from_labels(
        true_labs, pred_labs, sample_weight=sample_weight
    )

    # Return appropriate types for backward compatibility
    if sample_weight is None:
        return int(tp), int(tn), int(fp), int(fn)
    else:
        return float(tp), float(tn), float(fp), float(fn)


def _confusion_matrix_from_labels(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix components from predicted labels.

    This is the canonical function for computing confusion matrix components
    used throughout the codebase. All other confusion matrix calculations
    should use this function to ensure consistency.

    Parameters
    ----------
    true_labels : ArrayLike
        True binary labels (0 or 1)
    pred_labels : ArrayLike  
        Predicted binary labels (0 or 1)
    sample_weight : ArrayLike, optional
        Sample weights. If None, uniform weights are used.

    Returns
    -------
    tuple[float, float, float, float]
        True positives, true negatives, false positives, false negatives

    Examples
    --------
    >>> true_labs = [0, 1, 0, 1, 1]
    >>> pred_labs = [0, 1, 1, 1, 0]
    >>> tp, tn, fp, fn = _confusion_matrix_from_labels(true_labs, pred_labs)
    >>> (tp, tn, fp, fn)
    (2.0, 1.0, 1.0, 1.0)
    
    With sample weights:
    >>> weights = [1.0, 2.0, 1.0, 1.0, 0.5]
    >>> tp, tn, fp, fn = _confusion_matrix_from_labels(true_labs, pred_labs, weights)
    >>> (tp, tn, fp, fn)
    (3.0, 1.0, 1.0, 0.5)
    """
    import numpy as np

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
    else:
        weights = np.ones_like(true_labels, dtype=float)

    # Compute confusion matrix components
    tp = float(np.sum(weights[(true_labels == 1) & (pred_labels == 1)]))
    tn = float(np.sum(weights[(true_labels == 0) & (pred_labels == 0)]))
    fp = float(np.sum(weights[(true_labels == 0) & (pred_labels == 1)]))
    fn = float(np.sum(weights[(true_labels == 1) & (pred_labels == 0)]))

    return tp, tn, fp, fn


def compute_vectorized_confusion_matrices(
    y_sorted: NDArray[np.int8], weights_sorted: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute confusion matrix counts for all possible cuts using cumulative sums.

    Given labels and weights sorted in the same order as descending probabilities,
    returns (tp, tn, fp, fn) as vectors for every cut k.

    The indexing convention is:
    - Index 0: "predict nothing as positive" (all negative predictions)
    - Index k (k > 0): predict first k items as positive, rest as negative

    At cut index k:
      tp[k] = sum of weights for positive labels in first k items
      fp[k] = sum of weights for negative labels in first k items
      fn[k] = P - tp[k] (remaining positive weight)
      tn[k] = N - fp[k] (remaining negative weight)

    Where P = total positive weight, N = total negative weight.

    Parameters
    ----------
    y_sorted : NDArray[np.int8]
        Binary labels sorted by descending probability.
    weights_sorted : NDArray[np.float64]
        Sample weights sorted by descending probability.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Arrays of (tp, tn, fp, fn) counts for each cut position. Length is n+1
        where n is the number of samples, with index 0 being "predict nothing".
    """
    # Compute total positive and negative weights
    P = float(np.sum(weights_sorted * y_sorted))
    N = float(np.sum(weights_sorted * (1 - y_sorted)))

    # Cumulative weighted counts for cuts after each item
    tp_cumsum = np.cumsum(weights_sorted * y_sorted)
    fp_cumsum = np.cumsum(weights_sorted * (1 - y_sorted))

    # Include "predict nothing" case at the beginning
    tp = np.concatenate([[0.0], tp_cumsum])
    fp = np.concatenate([[0.0], fp_cumsum])

    # Complement counts
    fn = P - tp
    tn = N - fp

    return tp, tn, fp, fn


def apply_metric_to_confusion_counts(
    metric_fn: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    tp: NDArray[np.float64],
    tn: NDArray[np.float64],
    fp: NDArray[np.float64],
    fn: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply metric function to vectorized confusion matrix counts.

    Parameters
    ----------
    metric_fn : Callable
        Metric function that accepts (tp, tn, fp, fn) as arrays and returns
        array of scores.
    tp, tn, fp, fn : NDArray[np.float64]
        Confusion matrix count arrays.

    Returns
    -------
    NDArray[np.float64]
        Array of metric scores for each threshold.

    Raises
    ------
    ValueError
        If metric function doesn't return array with correct shape.
    """
    scores = metric_fn(tp, tn, fp, fn)

    # Ensure scores is a numpy array
    scores = np.asarray(scores)

    if scores.shape != tp.shape:
        raise ValueError(
            f"metric_fn must return array with shape {tp.shape}, got {scores.shape}."
        )

    return scores


def compute_metric_at_threshold(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    threshold: float,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
) -> float:
    """Compute metric score at a given threshold (compatibility wrapper).

    This function provides compatibility for existing code while using
    the new simplified API internally.
    """
    tp, tn, fp, fn = get_confusion_matrix(
        true_labs, pred_prob, threshold, sample_weight, comparison
    )
    if metric not in METRICS:
        raise ValueError(
            f"Metric '{metric}' not supported. Available: {list(METRICS.keys())}"
        )
    metric_func = METRICS[metric].scalar_fn
    return float(metric_func(tp, tn, fp, fn))


def compute_multiclass_metrics_from_labels(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    metric: str = "f1",
    average: str = "macro",
    sample_weight: ArrayLike | None = None,
    n_classes: int | None = None,
) -> float | np.ndarray[Any, Any]:
    """Compute multiclass metrics from true and predicted class labels.

    This is a centralized function that computes various multiclass metrics
    without external dependencies, supporting all averaging methods.

    Parameters
    ----------
    true_labels : ArrayLike
        True class labels (integers 0, 1, ..., n_classes-1)
    pred_labels : ArrayLike
        Predicted class labels (integers 0, 1, ..., n_classes-1)
    metric : str, default="f1"
        Metric to compute ("f1", "precision", "recall", "accuracy")
    average : str, default="macro"
        Averaging strategy: "macro", "micro", "weighted", "none"
    sample_weight : ArrayLike, optional
        Sample weights
    n_classes : int, optional
        Number of classes. If None, inferred from labels.

    Returns
    -------
    float or np.ndarray
        Computed metric score (float) or per-class scores (array if average="none")
    """
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    if true_labels.shape != pred_labels.shape:
        raise ValueError("true_labels and pred_labels must have same shape")

    if sample_weight is None:
        weights = np.ones_like(true_labels, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape[0] != true_labels.shape[0]:
            raise ValueError("sample_weight must have same length as labels")

    if n_classes is None:
        n_classes = (
            int(max(true_labels.max(initial=-1), pred_labels.max(initial=-1))) + 1
        )

    # Special case: accuracy (computed directly from labels)
    if metric == "accuracy":
        correct = (true_labels == pred_labels).astype(float)
        return float(np.average(correct, weights=weights))

    # Build OvR confusion matrices once using centralized function
    cms: list[tuple[float, float, float, float]] = []
    for k in range(n_classes):
        true_bin = (true_labels == k).astype(int)
        pred_bin = (pred_labels == k).astype(int)
        tp, tn, fp, fn = _confusion_matrix_from_labels(true_bin, pred_bin, weights)
        cms.append((tp, tn, fp, fn))

    # Route through multiclass_metric_ovr for all other metrics
    return multiclass_metric_ovr(cms, metric_name=metric, average=average)


def get_multiclass_confusion_matrix(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    thresholds: ArrayLike,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    require_proba: bool = False,
) -> list[tuple[int | float, int | float, int | float, int | float]]:
    """Compute per-class confusion-matrix counts for multiclass classification
    using One-vs-Rest.

    Parameters
    ----------
    true_labs:
        Array of true class labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Array of predicted probabilities with shape (n_samples, n_classes)
        (if require_proba=True) or scores (if require_proba=False).
    thresholds:
        Array of decision thresholds, one per class (or scalar for binary).
    sample_weight:
        Optional array of sample weights. If None, all samples have equal weight.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).
    require_proba:
        Whether to enforce [0,1] range for pred_prob and thresholds.
        Set to False for score-based empirical workflows.

    Returns
    -------
    list[tuple[int | float, int | float, int | float, int | float]]
        List of per-class counts ``(tp, tn, fp, fn)`` for each class.
        Returns int when sample_weight is None, float when sample_weight is provided.
    """
    # Validate inputs
    true_labs, pred_prob, sample_weight = validate_inputs(
        true_labs, pred_prob, sample_weight, require_binary=False
    )
    _validate_comparison_operator(comparison)

    if pred_prob.ndim == 1:
        # Binary case - backward compatibility
        thr_arr = np.asarray(thresholds)
        thr_scalar = float(thr_arr.reshape(-1)[0])  # accept scalars or length-1 arrays
        # For empirical score workflows, do not force [0,1]
        return [
            get_confusion_matrix(
                true_labs,
                pred_prob,
                thr_scalar,
                sample_weight,
                comparison,
                require_proba=require_proba,
            )
        ]

    # Multiclass case
    n_classes = pred_prob.shape[1]
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (n_classes,):
        raise ValueError(
            f"thresholds must have shape ({n_classes},), got {thresholds.shape}"
        )
    # Allow score thresholds by default
    # (If you truly need [0,1], pass require_proba=True)
    if require_proba:
        _validate_threshold(thresholds, n_classes)

    confusion_matrices = []

    for class_idx in range(n_classes):
        # One-vs-Rest: current class vs all others
        true_binary = (true_labs == class_idx).astype(int)
        pred_binary_prob = pred_prob[:, class_idx]
        threshold = thresholds[class_idx]

        cm = get_confusion_matrix(
            true_binary,
            pred_binary_prob,
            threshold,
            sample_weight,
            comparison,
            require_proba=require_proba,
        )
        confusion_matrices.append(cm)

    return confusion_matrices


# Linear utility/cost metric factories for economic optimization
def make_linear_counts_metric(
    w_tp: float = 0.0,
    w_tn: float = 0.0,
    w_fp: float = 0.0,
    w_fn: float = 0.0,
    name: str = "linear_utility",
) -> Callable[
    [
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
    np.ndarray[Any, Any],
]:
    """
    Create a vectorized metric that computes linear utility from confusion matrix.

    Returns metric(tp, tn, fp, fn) = w_tp*tp + w_tn*tn + w_fp*fp + w_fn*fn.
    Intended for expected utility maximization (benefits positive) or expected cost
    minimization (costs negative).

    Parameters
    ----------
    w_tp : float, default=0.0
        Weight/utility for true positives
    w_tn : float, default=0.0
        Weight/utility for true negatives
    w_fp : float, default=0.0
        Weight/utility for false positives (typically negative for costs)
    w_fn : float, default=0.0
        Weight/utility for false negatives (typically negative for costs)
    name : str, default="linear_utility"
        Name for the metric function

    Returns
    -------
    Callable
        Vectorized metric function compatible with sort-and-scan optimization

    Examples
    --------
    >>> # Cost-sensitive: penalize FN more than FP
    >>> metric = make_linear_counts_metric(w_fp=-1.0, w_fn=-5.0)
    >>>
    >>> # With benefits for correct predictions
    >>> metric = make_linear_counts_metric(w_tp=2.0, w_tn=0.5, w_fp=-1.0, w_fn=-5.0)
    """

    def _metric(tp: Any, tn: Any, fp: Any, fn: Any) -> Any:
        """Vectorized linear combination of confusion matrix counts."""
        return (
            w_tp * np.asarray(tp, dtype=float)
            + w_tn * np.asarray(tn, dtype=float)
            + w_fp * np.asarray(fp, dtype=float)
            + w_fn * np.asarray(fn, dtype=float)
        )

    _metric.__name__ = name
    return _metric


def make_cost_metric(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    name: str = "expected_utility",
) -> Callable[
    [
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ],
    np.ndarray[Any, Any],
]:
    """
    Create a vectorized cost-sensitive metric for utility maximization.

    Returns metric = tp_benefit*TP + tn_benefit*TN - fp_cost*FP - fn_cost*FN.
    This is a convenience wrapper around make_linear_counts_metric that handles
    the sign conversion from costs to utilities.

    Parameters
    ----------
    fp_cost : float
        Cost of false positive errors (positive value)
    fn_cost : float
        Cost of false negative errors (positive value)
    tp_benefit : float, default=0.0
        Benefit/reward for true positives (positive value)
    tn_benefit : float, default=0.0
        Benefit/reward for true negatives (positive value)
    name : str, default="expected_utility"
        Name for the metric function

    Returns
    -------
    Callable
        Vectorized metric function for expected utility maximization

    Examples
    --------
    >>> # Classic cost-sensitive: FN costs 5x more than FP
    >>> metric = make_cost_metric(fp_cost=1.0, fn_cost=5.0)
    >>>
    >>> # Include rewards for correct predictions
    >>> metric = make_cost_metric(
    ...     fp_cost=1.0, fn_cost=5.0, tp_benefit=2.0, tn_benefit=0.5
    ... )
    """
    return make_linear_counts_metric(
        w_tp=tp_benefit, w_tn=tn_benefit, w_fp=-fp_cost, w_fn=-fn_cost, name=name
    )


# Register built-in metrics (primary names only)
register_metric("f1", f1_score, _f1_vectorized)
register_metric("accuracy", accuracy_score, _accuracy_vectorized)
register_metric("precision", precision_score, _precision_vectorized)
register_metric("recall", recall_score, _recall_vectorized)
register_metric("iou", jaccard_score, _iou_vectorized)
register_metric("specificity", specificity_score, _specificity_vectorized)

# Register aliases to avoid duplication
register_alias("jaccard", "iou")
register_alias("tnr", "specificity")
register_alias("ppv", "precision")
register_alias("tpr", "recall")
register_alias("sensitivity", "recall")
