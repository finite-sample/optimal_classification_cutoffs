import numpy as np
from typing import Callable, Dict

# Registry mapping metric names to their corresponding functions
METRICS: Dict[str, Callable] = {}

def register_metric(name: str | None = None):
    """Decorator to register a metric function.

    Parameters
    ----------
    name: str | None
        Optional name for the metric. If omitted the function's ``__name__`` is
        used.
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        METRICS[metric_name] = func
        return func

    return decorator


def get_confusion_matrix(true_labs, pred_prob, prob):
    """Compute the elements of the confusion matrix."""
    pred_labs = pred_prob > prob
    tp = np.sum(np.logical_and(pred_labs == 1, true_labs == 1))
    tn = np.sum(np.logical_and(pred_labs == 0, true_labs == 0))
    fp = np.sum(np.logical_and(pred_labs == 1, true_labs == 0))
    fn = np.sum(np.logical_and(pred_labs == 0, true_labs == 1))
    return tp, tn, fp, fn


@register_metric("accuracy")
def accuracy(prob, true_labs, pred_prob, verbose: bool = False):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    acc = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print(f"Probability: {prob[0]:0.4f} Accuracy: {acc:0.4f}")
    return 1 - acc


@register_metric("f1")
def f1(prob, true_labs, pred_prob, verbose: bool = False):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    score = 2 * precision * recall / (precision + recall)
    if verbose:
        print(f"Probability: {prob[0]:0.4f} F1 score: {score:0.4f}")
    return 1 - score
