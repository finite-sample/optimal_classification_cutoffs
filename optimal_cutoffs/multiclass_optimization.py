"""Core multiclass threshold optimization algorithms.

This module contains algorithms for multiclass threshold optimization,
using One-vs-Rest (OvR) strategy by default.
"""

from typing import Any

import numpy as np

from .binary_optimization import optimal_threshold_piecewise
from .types import (
    ArrayLike,
    AveragingMethod,
    ComparisonOperator,
    OptimizationMethod,
    SampleWeightLike,
)


def validate_multiclass_input(
    true_labs: ArrayLike, pred_prob: ArrayLike
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Simple validation for multiclass inputs."""
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)

    if pred_prob.ndim != 2:
        raise ValueError("pred_prob must be 2D for multiclass")
    if true_labs.shape[0] != pred_prob.shape[0]:
        raise ValueError("true_labs and pred_prob must have same number of samples")

    return true_labs, pred_prob


def multiclass_metric_score(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    thresholds: ArrayLike,
    metric: str = "f1",
    average: AveragingMethod = "macro",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperator = ">",
) -> float:
    """Compute multiclass metric score using One-vs-Rest strategy.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True class labels (0, 1, ..., n_classes-1)
    pred_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities
    thresholds : array-like of shape (n_classes,)
        Per-class thresholds
    metric : str, default="f1"
        Metric to compute
    average : {"macro", "micro", "weighted", "none"}, default="macro"
        Averaging strategy for multiclass metric
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application

    Returns
    -------
    float
        Multiclass metric score
    """
    true_labs, pred_prob = validate_multiclass_input(true_labs, pred_prob)
    thresholds = np.asarray(thresholds, dtype=float)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    n_samples, n_classes = pred_prob.shape

    if thresholds.shape[0] != n_classes:
        raise ValueError(
            f"thresholds must have shape ({n_classes},), got {thresholds.shape}"
        )

    # Apply thresholds to get binary predictions for each class
    if comparison == ">":
        binary_predictions = pred_prob > thresholds[np.newaxis, :]
    else:  # ">="
        binary_predictions = pred_prob >= thresholds[np.newaxis, :]

    # Convert to multiclass predictions using One-vs-Rest strategy
    # For each sample, predict the class with highest probability
    # among those above threshold
    # If no classes above threshold, predict the class with highest probability
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        above_threshold = np.where(binary_predictions[i])[0]
        if len(above_threshold) > 0:
            # Among classes above threshold, pick the one with highest probability
            predictions[i] = above_threshold[np.argmax(pred_prob[i, above_threshold])]
        else:
            # No class above threshold, pick highest probability class
            predictions[i] = np.argmax(pred_prob[i])

    # Compute multiclass metric - need to implement this properly
    # For now, use a simple approach
    from sklearn.metrics import (  # type: ignore[import-untyped]
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if metric == "f1":
        return float(
            f1_score(
                true_labs,
                predictions,
                average=average,
                sample_weight=sample_weight,
                zero_division=0,
            )
        )
    elif metric == "accuracy":
        return float(
            accuracy_score(true_labs, predictions, sample_weight=sample_weight)
        )
    elif metric == "precision":
        return float(
            precision_score(
                true_labs,
                predictions,
                average=average,
                sample_weight=sample_weight,
                zero_division=0,
            )
        )
    elif metric == "recall":
        return float(
            recall_score(
                true_labs,
                predictions,
                average=average,
                sample_weight=sample_weight,
                zero_division=0,
            )
        )
    else:
        raise ValueError(f"Metric '{metric}' not supported for multiclass")


def get_optimal_multiclass_thresholds(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "auto",
    average: AveragingMethod = "macro",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperator = ">",
) -> np.ndarray[Any, Any]:
    """Find optimal per-class thresholds for multiclass classification.

    Uses One-vs-Rest (OvR) strategy where each class is treated as a separate
    binary classification problem.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True class labels (0, 1, ..., n_classes-1)
    pred_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities
    metric : str, default="f1"
        Metric to optimize
    method : OptimizationMethod, default="auto"
        Optimization method to use
    average : {"macro", "micro", "weighted", "none"}, default="macro"
        Averaging strategy (affects optimization for "micro")
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application

    Returns
    -------
    np.ndarray of shape (n_classes,)
        Optimal threshold for each class

    Notes
    -----
    - For average="micro", uses a single global threshold across all classes
    - For other averaging strategies, optimizes per-class thresholds independently
    """
    true_labs, pred_prob = validate_multiclass_input(true_labs, pred_prob)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    n_samples, n_classes = pred_prob.shape

    if average == "micro":
        # Micro averaging: use single global threshold
        # Flatten all class probabilities and create binary labels
        true_binary_flat = np.zeros((n_samples * n_classes,), dtype=int)
        pred_prob_flat = pred_prob.ravel()

        # Create binary labels for micro averaging
        for i in range(n_samples):
            for j in range(n_classes):
                idx = i * n_classes + j
                true_binary_flat[idx] = 1 if true_labs[i] == j else 0

        # Sample weights for micro averaging
        if sample_weight is not None:
            sample_weight_flat = np.repeat(sample_weight, n_classes)
        else:
            sample_weight_flat = None

        # Find single optimal threshold
        optimal_threshold = optimal_threshold_piecewise(
            true_binary_flat, pred_prob_flat, metric, sample_weight_flat, comparison
        )

        # Return same threshold for all classes
        return np.full(n_classes, optimal_threshold, dtype=float)

    else:
        # Macro/weighted/none averaging: optimize per-class thresholds independently
        optimal_thresholds = np.zeros(n_classes, dtype=float)

        # Create binary labels for each class (One-vs-Rest)
        true_binary_all = np.zeros((n_samples, n_classes), dtype=int)
        for class_idx in range(n_classes):
            true_binary_all[:, class_idx] = (true_labs == class_idx).astype(int)

        # TODO: Implement fully vectorized version
        for class_idx in range(n_classes):
            optimal_thresholds[class_idx] = optimal_threshold_piecewise(
                true_binary_all[:, class_idx],
                pred_prob[:, class_idx],
                metric,
                sample_weight,
                comparison,
            )

        return optimal_thresholds
