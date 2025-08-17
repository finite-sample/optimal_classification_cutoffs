"""Threshold search strategies for optimizing classification metrics."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .metrics import METRIC_REGISTRY, get_confusion_matrix


def _accuracy(prob, true_labs, pred_prob, verbose=False):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print(f"Probability: {prob[0]:0.4f} Accuracy: {accuracy:0.4f}")
    return 1 - accuracy


def _f1(prob, true_labs, pred_prob, verbose=False):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if verbose:
        print(f"Probability: {prob[0]:0.4f} F1 score: {f1:0.4f}")
    return 1 - f1


def get_probability(true_labs, pred_prob, objective="accuracy", verbose=False):
    """Brute-force search for a simple metric's best threshold.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Predicted probabilities from a classifier.
    objective:
        Metric to optimize. Supported values are ``"accuracy"`` and ``"f1"``.
    verbose:
        If ``True``, print intermediate metric values during the search.

    Returns
    -------
    float
        Threshold that maximizes the specified metric.
    """
    if objective == "accuracy":
        prob = optimize.brute(
            _accuracy, (slice(0.1, 0.9, 0.1),), args=(true_labs, pred_prob, verbose), disp=verbose
        )
    elif objective == "f1":
        prob = optimize.brute(
            _f1, (slice(0.1, 0.9, 0.1),), args=(true_labs, pred_prob, verbose), disp=verbose
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return float(prob[0] if isinstance(prob, np.ndarray) else prob)


def _metric_score(true_labs, pred_prob, threshold, metric="f1"):
    """Compute a metric score for a given threshold using registry metrics."""
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, threshold)
    try:
        metric_func = METRIC_REGISTRY[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {metric}") from exc
    return float(metric_func(tp, tn, fp, fn))


def get_optimal_threshold(true_labs, pred_prob, metric="f1", method="smart_brute"):
    """Find the threshold that optimizes a metric.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Predicted probabilities from a classifier.
    metric:
        Name of a metric registered in :data:`~optimal_cutoffs.metrics.METRIC_REGISTRY`.
    method:
        Strategy used for optimization: ``"smart_brute"`` evaluates all unique
        probabilities, ``"minimize"`` uses ``scipy.optimize.minimize_scalar``,
        and ``"gradient"`` performs a simple gradient ascent.

    Returns
    -------
    float
        The threshold that maximizes the chosen metric.
    """
    if method == "smart_brute":
        thresholds = np.unique(pred_prob)
        scores = [_metric_score(true_labs, pred_prob, t, metric) for t in thresholds]
        return float(thresholds[int(np.argmax(scores))])

    if method == "minimize":
        res = optimize.minimize_scalar(
            lambda t: -_metric_score(true_labs, pred_prob, t, metric),
            bounds=(0, 1),
            method="bounded",
        )
        # ``minimize_scalar`` may return a threshold that is suboptimal for
        # piecewise-constant metrics like F1. To provide a more robust
        # solution, also evaluate all unique predicted probabilities and pick
        # whichever threshold yields the highest score.
        candidates = np.unique(np.append(pred_prob, res.x))
        scores = [_metric_score(true_labs, pred_prob, t, metric) for t in candidates]
        return float(candidates[int(np.argmax(scores))])

    if method == "gradient":
        threshold = 0.5
        lr = 0.1
        eps = 1e-5
        for _ in range(100):
            grad = (
                _metric_score(true_labs, pred_prob, threshold + eps, metric)
                - _metric_score(true_labs, pred_prob, threshold - eps, metric)
            ) / (2 * eps)
            threshold = np.clip(threshold + lr * grad, 0.0, 1.0)
        return float(threshold)

    raise ValueError(f"Unknown method: {method}")


__all__ = ["get_probability", "get_optimal_threshold"]
