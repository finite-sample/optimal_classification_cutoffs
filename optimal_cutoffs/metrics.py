"""Metric registry, confusion matrix utilities, and built-in metrics."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from .types import ArrayLike, ComparisonOperator, MetricFunc
from .validation import (
    _validate_comparison_operator,
    _validate_inputs,
    _validate_threshold,
)

METRIC_REGISTRY: dict[str, MetricFunc] = {}
VECTORIZED_REGISTRY: dict[str, Callable[..., Any]] = {}  # Vectorized metrics
METRIC_PROPERTIES: dict[str, dict[str, bool | float]] = {}


def register_metric(
    name: str | None = None,
    func: MetricFunc | None = None,
    vectorized_func: Callable[..., Any] | None = None,
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> MetricFunc | Callable[[MetricFunc], MetricFunc]:
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
    MetricFunc | Callable[[MetricFunc], MetricFunc]
        The registered function or decorator.
    """
    if func is not None:
        metric_name = name or func.__name__
        METRIC_REGISTRY[metric_name] = func
        if vectorized_func is not None:
            VECTORIZED_REGISTRY[metric_name] = vectorized_func
        METRIC_PROPERTIES[metric_name] = {
            "is_piecewise": is_piecewise,
            "maximize": maximize,
            "needs_proba": needs_proba,
        }
        return func

    def decorator(f: MetricFunc) -> MetricFunc:
        metric_name = name or f.__name__
        METRIC_REGISTRY[metric_name] = f
        if vectorized_func is not None:
            VECTORIZED_REGISTRY[metric_name] = vectorized_func
        METRIC_PROPERTIES[metric_name] = {
            "is_piecewise": is_piecewise,
            "maximize": maximize,
            "needs_proba": needs_proba,
        }
        return f

    return decorator


def register_metrics(
    metrics: dict[str, MetricFunc],
    vectorized_metrics: dict[str, Callable[..., Any]] | None = None,
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
        This function mutates the global registries in-place.
    """
    METRIC_REGISTRY.update(metrics)
    if vectorized_metrics is not None:
        VECTORIZED_REGISTRY.update(vectorized_metrics)
    for name in metrics:
        METRIC_PROPERTIES[name] = {
            "is_piecewise": is_piecewise,
            "maximize": maximize,
            "needs_proba": needs_proba,
        }


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
    properties = METRIC_PROPERTIES.get(metric_name, {"is_piecewise": True})
    return bool(properties["is_piecewise"])


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
    return bool(METRIC_PROPERTIES.get(metric_name, {"maximize": True})["maximize"])


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
    properties = METRIC_PROPERTIES.get(metric_name, {"needs_proba": False})
    return bool(properties["needs_proba"])


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
    return metric_name in VECTORIZED_REGISTRY


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
    if metric_name not in VECTORIZED_REGISTRY:
        available = sorted(VECTORIZED_REGISTRY.keys())
        preview = ", ".join(available[:10])
        raise ValueError(
            f"Vectorized implementation not available for metric '{metric_name}'. "
            f"Available (first 10): [{preview}] ... (total={len(available)})"
        )
    return VECTORIZED_REGISTRY[metric_name]


# Vectorized metric implementations for O(n log n) optimization
def _f1_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized F1 score computation."""
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
    )
    f1_numerator = 2 * precision * recall
    f1_denominator = precision + recall
    return cast(
        np.ndarray[Any, Any],
        np.divide(
            f1_numerator,
            f1_denominator,
            out=np.zeros_like(tp, dtype=float),
            where=f1_denominator > 0,
        ),
    )


def _accuracy_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized accuracy computation."""
    total = tp + tn + fp + fn
    return cast(
        np.ndarray[Any, Any],
        np.divide(tp + tn, total, out=np.zeros_like(tp, dtype=float), where=total > 0),
    )


def _precision_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized precision computation."""
    return cast(
        np.ndarray[Any, Any],
        np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0),
    )


def _recall_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized recall computation."""
    return cast(
        np.ndarray[Any, Any],
        np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0),
    )


def _iou_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized IoU/Jaccard score computation."""
    denom = tp + fp + fn
    return cast(
        np.ndarray[Any, Any],
        np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0),
    )


def _specificity_vectorized(
    tp: np.ndarray[Any, Any],
    tn: np.ndarray[Any, Any],
    fp: np.ndarray[Any, Any],
    fn: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Vectorized specificity computation."""
    denom = tn + fp
    return cast(
        np.ndarray[Any, Any],
        np.divide(tn, denom, out=np.zeros_like(tn, dtype=float), where=denom > 0),
    )


def f1_score(
    tp: int | float, tn: int | float, fp: int | float, fn: int | float
) -> float:
    r"""Compute the F\ :sub:`1` score.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        The harmonic mean of precision and recall.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def accuracy_score(
    tp: int | float, tn: int | float, fp: int | float, fn: int | float
) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Ratio of correct predictions to total samples.
    """
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision_score(
    tp: int | float, tn: int | float, fp: int | float, fn: int | float
) -> float:
    """Compute precision (positive predictive value).

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Ratio of true positives to predicted positives.
    """
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall_score(
    tp: int | float, tn: int | float, fp: int | float, fn: int | float
) -> float:
    """Compute recall (sensitivity, true positive rate).

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Ratio of true positives to actual positives.
    """
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def jaccard_score(
    tp: int | float, tn: int | float, fp: int | float, fn: int | float
) -> float:
    """Compute the Jaccard index (IoU score).

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        Intersection over Union: tp / (tp + fp + fn).
    """
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def specificity_score(
    tp: int | float, tn: int | float, fp: int | float, fn: int | float
) -> float:
    """Compute the specificity (true negative rate).

    Parameters
    ----------
    tp, tn, fp, fn:
        Elements of the confusion matrix.

    Returns
    -------
    float
        True negative rate: tn / (tn + fp).
    """
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


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


def multiclass_metric_exclusive(
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
            true_pos = true_labs == c
            pred_pos = pred_labels == c
            tp = (
                float(np.sum(sw[true_pos & pred_pos]))
                if sw is not None
                else int(np.sum(true_pos & pred_pos))
            )
            fp = (
                float(np.sum(sw[~true_pos & pred_pos]))
                if sw is not None
                else int(np.sum(~true_pos & pred_pos))
            )
            fn = (
                float(np.sum(sw[true_pos & ~pred_pos]))
                if sw is not None
                else int(np.sum(true_pos & ~pred_pos))
            )
            # tn not needed for these macro metrics
            if metric_name == "precision":
                per_class.append(precision_score(tp, 0, fp, 0))
            elif metric_name == "recall":
                per_class.append(recall_score(tp, 0, 0, fn))
            elif metric_name == "f1":
                per_class.append(f1_score(tp, 0, fp, fn))
            else:
                raise ValueError(
                    f"Metric '{metric_name}' not supported for exclusive mode"
                )
        return float(np.mean(per_class) if per_class else 0.0)


def multiclass_metric(
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

        Note: For exclusive single-label accuracy, use multiclass_metric_exclusive().

    Returns
    -------
    float | np.ndarray
        Aggregated metric score (float) or per-class scores (array) if average="none".
    """
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}")

    metric_func = METRIC_REGISTRY[metric_name]

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
                "Use multiclass_metric_exclusive() instead, or use 'macro' averaging "
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
    comparison: ComparisonOperator = ">",
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
    true_labs, pred_prob, sample_weight = _validate_inputs(
        true_labs,
        pred_prob,
        require_binary=True,
        require_proba=require_proba,
        sample_weight=sample_weight,
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
    tp, tn, fp, fn = compute_confusion_matrix_from_labels(
        true_labs, pred_labs, sample_weight=sample_weight
    )

    # Return appropriate types for backward compatibility
    if sample_weight is None:
        return int(tp), int(tn), int(fp), int(fn)
    else:
        return float(tp), float(tn), float(fp), float(fn)


def compute_confusion_matrix_from_labels(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix components from predicted labels.

    This is a generic function that computes (tp, tn, fp, fn) from true and
    predicted labels, supporting sample weights.

    Parameters
    ----------
    true_labels : ArrayLike
        True binary labels (0 or 1)
    pred_labels : ArrayLike
        Predicted binary labels (0 or 1)
    sample_weight : ArrayLike, optional
        Sample weights

    Returns
    -------
    tuple[float, float, float, float]
        True positives, true negatives, false positives, false negatives
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


def compute_metric_at_threshold(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    threshold: float,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
) -> float:
    """Compute metric score at a given threshold for binary classification.

    This is a generic function that computes a metric score by applying a threshold
    to probabilities, converting to labels, and computing the specified metric.

    Parameters
    ----------
    true_labs : ArrayLike
        True binary labels (0 or 1)
    pred_prob : ArrayLike
        Predicted probabilities
    threshold : float
        Classification threshold
    metric : str, default="f1"
        Metric to compute (e.g., "f1", "accuracy", "precision", "recall")
    sample_weight : ArrayLike, optional
        Sample weights
    comparison : str, default=">"
        Comparison operator for threshold application (">" or ">=")

    Returns
    -------
    float
        Metric score at the given threshold

    Raises
    ------
    ValueError
        If metric is not in METRIC_REGISTRY or comparison operator is invalid
    """

    # Validate inputs
    true_labs, pred_prob, sample_weight = _validate_inputs(
        true_labs, pred_prob, sample_weight=sample_weight
    )
    _validate_comparison_operator(comparison)

    # Apply threshold to get predictions
    if comparison == ">":
        pred_labels = (pred_prob > threshold).astype(int)
    else:  # ">="
        pred_labels = (pred_prob >= threshold).astype(int)

    # Get confusion matrix components
    tp, tn, fp, fn = compute_confusion_matrix_from_labels(
        true_labs, pred_labels, sample_weight=sample_weight
    )

    # Compute metric using registry
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"Metric '{metric}' not supported. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )

    metric_func = METRIC_REGISTRY[metric]
    return float(metric_func(tp, tn, fp, fn))


def get_multiclass_confusion_matrix(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    thresholds: ArrayLike,
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
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
    true_labs, pred_prob, sample_weight = _validate_inputs(
        true_labs, pred_prob, sample_weight=sample_weight, require_proba=require_proba
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


# Register built-in metrics manually to avoid decorator type issues
register_metric("f1", f1_score, _f1_vectorized)
register_metric("accuracy", accuracy_score, _accuracy_vectorized)
register_metric("precision", precision_score, _precision_vectorized)
register_metric("recall", recall_score, _recall_vectorized)
register_metric("iou", jaccard_score, _iou_vectorized)
register_metric("jaccard", jaccard_score, _iou_vectorized)
register_metric("specificity", specificity_score, _specificity_vectorized)
register_metric("tnr", specificity_score, _specificity_vectorized)
register_metric("ppv", precision_score, _precision_vectorized)
register_metric("tpr", recall_score, _recall_vectorized)
register_metric("sensitivity", recall_score, _recall_vectorized)
