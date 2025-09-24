"""Threshold search strategies for optimizing classification metrics."""

from typing import Literal

import numpy as np
from scipy import optimize

from .metrics import (
    METRIC_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    is_piecewise_metric,
    multiclass_metric,
)
from .types import ArrayLike, AveragingMethod, OptimizationMethod


def _accuracy(
    prob: np.ndarray, true_labs: ArrayLike, pred_prob: ArrayLike, verbose: bool = False
) -> float:
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print(f"Probability: {prob[0]:0.4f} Accuracy: {accuracy:0.4f}")
    return 1 - accuracy


def _f1(
    prob: np.ndarray, true_labs: ArrayLike, pred_prob: ArrayLike, verbose: bool = False
) -> float:
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    if verbose:
        print(f"Probability: {prob[0]:0.4f} F1 score: {f1:0.4f}")
    return 1 - f1


def get_probability(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    objective: Literal["accuracy", "f1"] = "accuracy",
    verbose: bool = False
) -> float:
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
            _accuracy,
            (slice(0.1, 0.9, 0.1),),
            args=(true_labs, pred_prob, verbose),
            disp=verbose,
        )
    elif objective == "f1":
        prob = optimize.brute(
            _f1,
            (slice(0.1, 0.9, 0.1),),
            args=(true_labs, pred_prob, verbose),
            disp=verbose,
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return float(prob[0] if isinstance(prob, np.ndarray) else prob)


def _metric_score(
    true_labs: ArrayLike, pred_prob: ArrayLike, threshold: float, metric: str = "f1"
) -> float:
    """Compute a metric score for a given threshold using registry metrics."""
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, threshold)
    try:
        metric_func = METRIC_REGISTRY[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {metric}") from exc
    return float(metric_func(tp, tn, fp, fn))


def _multiclass_metric_score(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    thresholds: ArrayLike,
    metric: str = "f1",
    average: AveragingMethod = "macro"
) -> float:
    """Compute a multiclass metric score for given per-class thresholds."""
    confusion_matrices = get_multiclass_confusion_matrix(
        true_labs, pred_prob, thresholds
    )
    return multiclass_metric(confusion_matrices, metric, average)


def _optimal_threshold_piecewise(
    true_labs: ArrayLike, pred_prob: ArrayLike, metric: str = "f1"
) -> float:
    """Find optimal threshold using O(n log n) algorithm for piecewise metrics.

    This algorithm sorts predictions once and uses cumulative sums to compute
    confusion matrix elements in O(1) for each candidate threshold, resulting
    in O(n log n) total time complexity.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Array of predicted probabilities.
    metric:
        Name of metric to optimize from METRIC_REGISTRY.

    Returns
    -------
    float
        Optimal threshold that maximizes the metric.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)

    if len(true_labs) == 0:
        raise ValueError("true_labs cannot be empty")

    if len(true_labs) != len(pred_prob):
        raise ValueError(
            f"Length mismatch: true_labs ({len(true_labs)}) vs pred_prob ({len(pred_prob)})"
        )

    # Get metric function
    try:
        metric_func = METRIC_REGISTRY[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {metric}") from exc

    # Handle edge case: single prediction
    if len(pred_prob) == 1:
        return float(pred_prob[0])

    # Sort by predicted probability in descending order for efficiency
    sort_idx = np.argsort(-pred_prob)
    sorted_probs = pred_prob[sort_idx]
    sorted_labels = true_labs[sort_idx]

    # Compute total positives and negatives
    P = int(np.sum(true_labs))
    N = len(true_labs) - P

    # Handle edge case: all same class
    if P == 0 or N == 0:
        return 0.5

    # Find unique probabilities to use as threshold candidates
    unique_probs = np.unique(pred_prob)

    best_score = -np.inf
    best_threshold = 0.5

    # For O(n log n) optimization, we can use cumulative sums
    # Sort examples by probability (descending)
    sort_indices = np.argsort(-pred_prob)
    sorted_probs = pred_prob[sort_indices]
    sorted_labels = true_labs[sort_indices]

    # Cumulative sums for TP and total positives at each position
    cum_tp = np.cumsum(sorted_labels)
    cum_total = np.arange(1, len(sorted_labels) + 1)
    cum_fp = cum_total - cum_tp

    # Evaluate at each unique threshold
    for threshold in unique_probs:
        # Find position where we switch from positive to negative predictions
        # All samples with prob > threshold are predicted positive
        pos_mask = sorted_probs > threshold

        if np.any(pos_mask):
            # Find last position where prob > threshold
            last_pos = np.where(pos_mask)[0][-1]
            tp = int(cum_tp[last_pos])
            fp = int(cum_fp[last_pos])
        else:
            # No predictions above threshold -> all negative
            tp = fp = 0

        fn = P - tp
        tn = N - fp

        # Compute metric score
        score = float(metric_func(tp, tn, fp, fn))

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def get_optimal_threshold(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "smart_brute"
) -> float | np.ndarray:
    """Find the threshold that optimizes a metric.

    Parameters
    ----------
    true_labs:
        Array of true binary labels or multiclass labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Predicted probabilities from a classifier. For binary: 1D array (n_samples,).
        For multiclass: 2D array (n_samples, n_classes).
    metric:
        Name of a metric registered in :data:`~optimal_cutoffs.metrics.METRIC_REGISTRY`.
    method:
        Strategy used for optimization: ``"smart_brute"`` evaluates all unique
        probabilities, ``"minimize"`` uses ``scipy.optimize.minimize_scalar``,
        and ``"gradient"`` performs a simple gradient ascent.

    Returns
    -------
    float | np.ndarray
        For binary: The threshold that maximizes the chosen metric.
        For multiclass: Array of per-class thresholds.
    """
    pred_prob = np.asarray(pred_prob)

    # Check if this is multiclass
    if pred_prob.ndim == 2:
        return get_optimal_multiclass_thresholds(true_labs, pred_prob, metric, method)

    # Binary case (existing logic)
    if method == "smart_brute":
        # Use fast piecewise optimization for piecewise-constant metrics
        if is_piecewise_metric(metric):
            return _optimal_threshold_piecewise(true_labs, pred_prob, metric)
        else:
            # Fall back to original brute force for non-piecewise metrics
            thresholds = np.unique(pred_prob)
            scores = [
                _metric_score(true_labs, pred_prob, t, metric) for t in thresholds
            ]
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


def get_optimal_multiclass_thresholds(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "smart_brute",
    average: AveragingMethod = "macro"
) -> np.ndarray:
    """Find optimal per-class thresholds for multiclass classification using One-vs-Rest.

    Parameters
    ----------
    true_labs:
        Array of true class labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Array of predicted probabilities with shape (n_samples, n_classes).
    metric:
        Name of a metric registered in :data:`~optimal_cutoffs.metrics.METRIC_REGISTRY`.
    method:
        Strategy used for optimization: ``"smart_brute"``, ``"minimize"``, or ``"gradient"``.
    average:
        Averaging strategy for multiclass metrics: "macro", "micro", or "weighted".

    Returns
    -------
    np.ndarray
        Array of optimal thresholds, one per class.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)

    # Input validation
    if len(true_labs) == 0:
        raise ValueError("true_labs cannot be empty")

    if pred_prob.ndim != 2:
        raise ValueError(f"pred_prob must be 2D for multiclass, got {pred_prob.ndim}D")

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
    optimal_thresholds = np.zeros(n_classes)

    for class_idx in range(n_classes):
        # One-vs-Rest: current class vs all others
        true_binary = (true_labs == class_idx).astype(int)
        pred_binary_prob = pred_prob[:, class_idx]

        # Optimize threshold for this class
        optimal_thresholds[class_idx] = get_optimal_threshold(
            true_binary, pred_binary_prob, metric, method
        )

    return optimal_thresholds


__all__ = [
    "get_probability",
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
]
