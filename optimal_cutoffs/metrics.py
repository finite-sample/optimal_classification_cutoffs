"""Metric registry, confusion matrix utilities, and built-in metrics."""

from collections.abc import Callable

import numpy as np

from .types import ArrayLike, MetricFunc

METRIC_REGISTRY: dict[str, MetricFunc] = {}
METRIC_PROPERTIES: dict[str, dict[str, bool]] = {}


def register_metric(
    name: str | None = None,
    func: MetricFunc | None = None,
    is_piecewise: bool = True,
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

    Returns
    -------
    MetricFunc | Callable[[MetricFunc], MetricFunc]
        The registered function or decorator.
    """
    if func is not None:
        metric_name = name or func.__name__
        METRIC_REGISTRY[metric_name] = func
        METRIC_PROPERTIES[metric_name] = {"is_piecewise": is_piecewise}
        return func

    def decorator(f: MetricFunc) -> MetricFunc:
        metric_name = name or f.__name__
        METRIC_REGISTRY[metric_name] = f
        METRIC_PROPERTIES[metric_name] = {"is_piecewise": is_piecewise}
        return f

    return decorator


def register_metrics(
    metrics: dict[str, MetricFunc], is_piecewise: bool = True
) -> None:
    """Register multiple metric functions.

    Parameters
    ----------
    metrics:
        Mapping of metric names to callables that accept ``tp, tn, fp, fn``.
    is_piecewise:
        Whether the metrics are piecewise-constant with respect to threshold changes.

    Returns
    -------
    None
        This function mutates the global :data:`METRIC_REGISTRY` in-place.
    """
    METRIC_REGISTRY.update(metrics)
    for name in metrics:
        METRIC_PROPERTIES[name] = {"is_piecewise": is_piecewise}


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


@register_metric("f1")
def f1_score(tp: int, tn: int, fp: int, fn: int) -> float:
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
def accuracy_score(tp: int, tn: int, fp: int, fn: int) -> float:
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
def precision_score(tp: int, tn: int, fp: int, fn: int) -> float:
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
def recall_score(tp: int, tn: int, fp: int, fn: int) -> float:
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
    confusion_matrices: list[tuple[int, int, int, int]],
    metric_name: str,
    average: str = "macro"
) -> float:
    """Compute multiclass metrics from per-class confusion matrices.

    Parameters
    ----------
    confusion_matrices:
        List of per-class confusion matrix tuples ``(tp, tn, fp, fn)``.
    metric_name:
        Name of the metric to compute (must be in METRIC_REGISTRY).
    average:
        Averaging strategy: "macro" (unweighted mean), "micro" (global), or "weighted".

    Returns
    -------
    float
        Aggregated metric score.
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
            sum(score * support for score, support in zip(scores, supports, strict=False))
            / total_support
        )
        return float(weighted_score)

    else:
        raise ValueError(f"Unknown averaging method: {average}")


def get_confusion_matrix(
    true_labs: ArrayLike, pred_prob: ArrayLike, prob: float
) -> tuple[int, int, int, int]:
    """Compute confusion-matrix counts for a given threshold.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Array of predicted probabilities in ``[0, 1]``.
    prob:
        Decision threshold applied to ``pred_prob``.

    Returns
    -------
    tuple[int, int, int, int]
        Counts ``(tp, tn, fp, fn)``.
    """
    pred_labs = pred_prob > prob
    tp = np.sum(np.logical_and(pred_labs == 1, true_labs == 1))
    tn = np.sum(np.logical_and(pred_labs == 0, true_labs == 0))
    fp = np.sum(np.logical_and(pred_labs == 1, true_labs == 0))
    fn = np.sum(np.logical_and(pred_labs == 0, true_labs == 1))
    return tp, tn, fp, fn


def get_multiclass_confusion_matrix(
    true_labs: ArrayLike, pred_prob: ArrayLike, thresholds: ArrayLike
) -> list[tuple[int, int, int, int]]:
    """Compute per-class confusion-matrix counts for multiclass classification using One-vs-Rest.

    Parameters
    ----------
    true_labs:
        Array of true class labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Array of predicted probabilities with shape (n_samples, n_classes).
    thresholds:
        Array of decision thresholds, one per class.

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of per-class counts ``(tp, tn, fp, fn)`` for each class.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    thresholds = np.asarray(thresholds)

    # Input validation
    if len(true_labs) == 0:
        raise ValueError("true_labs cannot be empty")

    if pred_prob.ndim == 1:
        # Binary case - backward compatibility
        if len(true_labs) != len(pred_prob):
            raise ValueError(
                f"Length mismatch: true_labs ({len(true_labs)}) vs pred_prob ({len(pred_prob)})"
            )
        return [get_confusion_matrix(true_labs, pred_prob, thresholds[0])]

    if pred_prob.ndim != 2:
        raise ValueError(f"pred_prob must be 1D or 2D, got {pred_prob.ndim}D")

    if len(true_labs) != pred_prob.shape[0]:
        raise ValueError(
            f"Length mismatch: true_labs ({len(true_labs)}) vs pred_prob ({pred_prob.shape[0]})"
        )

    if np.any(np.isnan(pred_prob)) or np.any(np.isinf(pred_prob)):
        raise ValueError("pred_prob contains NaN or infinite values")

    # Check class labels are valid for One-vs-Rest
    unique_labels = np.unique(true_labs)
    expected_labels = np.arange(len(unique_labels))
    if not np.array_equal(np.sort(unique_labels), expected_labels):
        raise ValueError(
            f"Class labels must be consecutive integers starting from 0. "
            f"Got {unique_labels}, expected {expected_labels}"
        )

    n_classes = pred_prob.shape[1]
    if len(thresholds) != n_classes:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) must match number of classes ({n_classes})"
        )

    confusion_matrices = []

    for class_idx in range(n_classes):
        # One-vs-Rest: current class vs all others
        true_binary = (true_labs == class_idx).astype(int)
        pred_binary_prob = pred_prob[:, class_idx]
        threshold = thresholds[class_idx]

        cm = get_confusion_matrix(true_binary, pred_binary_prob, threshold)
        confusion_matrices.append(cm)

    return confusion_matrices
