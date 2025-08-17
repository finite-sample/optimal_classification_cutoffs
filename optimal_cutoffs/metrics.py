"""Metric registry, confusion matrix utilities, and built-in metrics."""

from typing import Callable, Dict
import numpy as np

METRIC_REGISTRY: Dict[str, Callable[[int, int, int, int], float]] = {}


def register_metric(name: str = None, func: Callable[[int, int, int, int], float] = None):
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

    Returns
    -------
    Callable
        The registered function or decorator.
    """
    if func is not None:
        METRIC_REGISTRY[name or func.__name__] = func
        return func

    def decorator(f: Callable[[int, int, int, int], float]):
        METRIC_REGISTRY[name or f.__name__] = f
        return f

    return decorator


def register_metrics(metrics: Dict[str, Callable[[int, int, int, int], float]]):
    """Register multiple metric functions.

    Parameters
    ----------
    metrics:
        Mapping of metric names to callables that accept ``tp, tn, fp, fn``.

    Returns
    -------
    None
        This function mutates the global :data:`METRIC_REGISTRY` in-place.
    """
    METRIC_REGISTRY.update(metrics)


@register_metric("f1")
def f1_score(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute the F\ :sub:`1` score.

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
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


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


def multiclass_metric(confusion_matrices, metric_name, average="macro"):
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
        # Global metric computed from summed confusion matrices
        total_tp = sum(cm[0] for cm in confusion_matrices)
        total_tn = sum(cm[1] for cm in confusion_matrices)
        total_fp = sum(cm[2] for cm in confusion_matrices)
        total_fn = sum(cm[3] for cm in confusion_matrices)
        return float(metric_func(total_tp, total_tn, total_fp, total_fn))
    
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
        
        weighted_score = sum(score * support for score, support in zip(scores, supports)) / total_support
        return float(weighted_score)
    
    else:
        raise ValueError(f"Unknown averaging method: {average}")


def get_confusion_matrix(true_labs, pred_prob, prob):
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


def get_multiclass_confusion_matrix(true_labs, pred_prob, thresholds):
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
    
    if pred_prob.ndim == 1:
        # Binary case - backward compatibility
        return [get_confusion_matrix(true_labs, pred_prob, thresholds[0])]
    
    n_classes = pred_prob.shape[1]
    confusion_matrices = []
    
    for class_idx in range(n_classes):
        # One-vs-Rest: current class vs all others
        true_binary = (true_labs == class_idx).astype(int)
        pred_binary_prob = pred_prob[:, class_idx]
        threshold = thresholds[class_idx]
        
        cm = get_confusion_matrix(true_binary, pred_binary_prob, threshold)
        confusion_matrices.append(cm)
    
    return confusion_matrices
