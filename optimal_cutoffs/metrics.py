"""Metric registry, confusion matrix utilities, and built-in metrics."""

from collections.abc import Callable

import numpy as np

from .types import ArrayLike, MetricFunc, ComparisonOperator
from .validation import (
    _validate_inputs,
    _validate_threshold,
    _validate_metric_name,
    _validate_averaging_method,
    _validate_comparison_operator,
)

METRIC_REGISTRY: dict[str, MetricFunc] = {}
METRIC_PROPERTIES: dict[str, dict[str, bool | float]] = {}


def register_metric(
    name: str | None = None,
    func: MetricFunc | None = None,
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> MetricFunc | Callable[[MetricFunc], MetricFunc]:
    """Register a metric function.

    Parameters
    ----------
    name:
        Optional key under which to store the metric. If not provided the
        function's ``__name__`` is used.
    func:
        Metric callable accepting ``tp, tn, fp, fn``. When supplied the
        function is registered immediately. If omitted, the returned decorator
        can be used to annotate a metric function.
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
        METRIC_PROPERTIES[metric_name] = {
            "is_piecewise": is_piecewise,
            "maximize": maximize,
            "needs_proba": needs_proba,
        }
        return func

    def decorator(f: MetricFunc) -> MetricFunc:
        metric_name = name or f.__name__
        METRIC_REGISTRY[metric_name] = f
        METRIC_PROPERTIES[metric_name] = {
            "is_piecewise": is_piecewise,
            "maximize": maximize,
            "needs_proba": needs_proba,
        }
        return f

    return decorator


def register_metrics(
    metrics: dict[str, MetricFunc],
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> None:
    """Register multiple metric functions.

    Parameters
    ----------
    metrics:
        Mapping of metric names to callables that accept ``tp, tn, fp, fn``.
    is_piecewise:
        Whether the metrics are piecewise-constant with respect to threshold changes.
    maximize:
        Whether the metrics should be maximized (True) or minimized (False).
    needs_proba:
        Whether the metrics require probability scores rather than just thresholds.

    Returns
    -------
    None
        This function mutates the global :data:`METRIC_REGISTRY` in-place.
    """
    METRIC_REGISTRY.update(metrics)
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
    return METRIC_PROPERTIES.get(metric_name, {"is_piecewise": True})["is_piecewise"]


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
    return METRIC_PROPERTIES.get(metric_name, {"maximize": True})["maximize"]


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
    return METRIC_PROPERTIES.get(metric_name, {"needs_proba": False})["needs_proba"]


@register_metric("f1")
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


@register_metric("accuracy")
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


@register_metric("precision")
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


@register_metric("recall")
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


def multiclass_metric(
    confusion_matrices: list[tuple[int | float, int | float, int | float, int | float]],
    metric_name: str,
    average: str = "macro",
) -> float | np.ndarray:
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
        - "micro": Global metric computed on pooled confusion matrix (treats all samples equally)
        - "weighted": Weighted mean by support (number of true instances per class)
        - "none": No averaging, returns array of per-class metrics

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
        # For micro averaging, sum only TP, FP, FN (not TN which is inflated in One-vs-Rest)
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
            # For accuracy in One-vs-Rest, compute as correct predictions / total predictions
            # This is equivalent to (total_tp) / (total_tp + total_fp + total_fn)
            total_predictions = total_tp + total_fp + total_fn
            return float(total_tp / total_predictions if total_predictions > 0 else 0.0)
        else:
            # Fallback: try using the metric function with computed values
            # Note: TN is not meaningful in One-vs-Rest micro averaging
            return float(metric_func(total_tp, 0, total_fp, total_fn))

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
            f"Unknown averaging method: {average}. Must be one of: 'macro', 'micro', 'weighted', 'none'."
        )


def get_confusion_matrix(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    prob: float,
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
) -> tuple[int | float, int | float, int | float, int | float]:
    """Compute confusion-matrix counts for a given threshold.

    Parameters
    ----------
    true_labs:
        Array of true binary labels in {0, 1}.
    pred_prob:
        Array of predicted probabilities in [0, 1].
    prob:
        Decision threshold applied to ``pred_prob``.
    sample_weight:
        Optional array of sample weights. If None, all samples have equal weight.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).
        - ">": pred_prob > threshold (default, excludes ties)
        - ">=": pred_prob >= threshold (includes ties)

    Returns
    -------
    tuple[int | float, int | float, int | float, int | float]
        Counts ``(tp, tn, fp, fn)``. Returns int when sample_weight is None,
        float when sample_weight is provided to preserve fractional weighted counts.
    """
    # Validate inputs
    true_labs, pred_prob, sample_weight = _validate_inputs(
        true_labs, pred_prob, require_binary=True, sample_weight=sample_weight, allow_multiclass=False
    )
    _validate_threshold(float(prob))
    _validate_comparison_operator(comparison)
    
    # Apply threshold with specified comparison operator
    if comparison == ">":
        pred_labs = pred_prob > prob
    else:  # ">="
        pred_labs = pred_prob >= prob

    if sample_weight is None:
        tp = np.sum(np.logical_and(pred_labs == 1, true_labs == 1))
        tn = np.sum(np.logical_and(pred_labs == 0, true_labs == 0))
        fp = np.sum(np.logical_and(pred_labs == 1, true_labs == 0))
        fn = np.sum(np.logical_and(pred_labs == 0, true_labs == 1))
        return int(tp), int(tn), int(fp), int(fn)
    else:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(true_labs):
            raise ValueError(
                f"Length mismatch: sample_weight ({len(sample_weight)}) vs true_labs ({len(true_labs)})"
            )
        tp = np.sum(sample_weight * np.logical_and(pred_labs == 1, true_labs == 1))
        tn = np.sum(sample_weight * np.logical_and(pred_labs == 0, true_labs == 0))
        fp = np.sum(sample_weight * np.logical_and(pred_labs == 1, true_labs == 0))
        fn = np.sum(sample_weight * np.logical_and(pred_labs == 0, true_labs == 1))
        # Return float values when using sample weights to preserve fractional counts
        return float(tp), float(tn), float(fp), float(fn)


def get_multiclass_confusion_matrix(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    thresholds: ArrayLike,
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
) -> list[tuple[int | float, int | float, int | float, int | float]]:
    """Compute per-class confusion-matrix counts for multiclass classification using One-vs-Rest.

    Parameters
    ----------
    true_labs:
        Array of true class labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Array of predicted probabilities with shape (n_samples, n_classes).
    thresholds:
        Array of decision thresholds, one per class.
    sample_weight:
        Optional array of sample weights. If None, all samples have equal weight.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    list[tuple[int | float, int | float, int | float, int | float]]
        List of per-class counts ``(tp, tn, fp, fn)`` for each class.
        Returns int when sample_weight is None, float when sample_weight is provided.
    """
    # Validate inputs
    true_labs, pred_prob, sample_weight = _validate_inputs(
        true_labs, pred_prob, sample_weight=sample_weight
    )
    _validate_comparison_operator(comparison)
    
    if pred_prob.ndim == 1:
        # Binary case - backward compatibility
        thresholds = np.asarray(thresholds)
        _validate_threshold(thresholds[0])
        return [
            get_confusion_matrix(true_labs, pred_prob, thresholds[0], sample_weight, comparison)
        ]
    
    # Multiclass case
    n_classes = pred_prob.shape[1]
    thresholds = np.asarray(thresholds)
    _validate_threshold(thresholds, n_classes)

    confusion_matrices = []

    for class_idx in range(n_classes):
        # One-vs-Rest: current class vs all others
        true_binary = (true_labs == class_idx).astype(int)
        pred_binary_prob = pred_prob[:, class_idx]
        threshold = thresholds[class_idx]

        cm = get_confusion_matrix(
            true_binary, pred_binary_prob, threshold, sample_weight, comparison
        )
        confusion_matrices.append(cm)

    return confusion_matrices
