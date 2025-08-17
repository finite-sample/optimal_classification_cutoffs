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
