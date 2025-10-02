"""Core multiclass threshold optimization algorithms.

This module contains algorithms for multiclass threshold optimization,
using One-vs-Rest (OvR) strategy by default.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .binary_optimization import optimal_threshold_piecewise
from .types import (
    AveragingMethodLiteral,
    ComparisonOperatorLiteral,
    OptimizationMethodLiteral,
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



def get_optimal_multiclass_thresholds(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethodLiteral = "auto",
    average: AveragingMethodLiteral = "macro",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperatorLiteral = ">",
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

    # Handle coordinate ascent method
    if method == "coord_ascent":
        # Coordinate ascent has specific requirements and limitations
        if sample_weight is not None:
            raise NotImplementedError(
                "Coordinate ascent does not support sample weights. "
                "This limitation could be lifted in future versions."
            )
        if comparison != ">":
            raise NotImplementedError(
                "Coordinate ascent only supports '>' comparison. "
                "Support for '>=' could be added in future versions."
            )
        if metric != "f1":
            raise NotImplementedError(
                "Coordinate ascent only supports 'f1' metric. "
                "Support for other piecewise metrics could be added in future versions."
            )

        # Import required functions
        from .metrics import get_vectorized_metric
        from .multiclass_coord import optimal_multiclass_thresholds_coord_ascent
        from .piecewise import optimal_threshold_sortscan

        # Get vectorized F1 metric
        f1_metric = get_vectorized_metric("f1")

        # Call coordinate ascent algorithm
        thresholds, _, _ = optimal_multiclass_thresholds_coord_ascent(
            true_labs,
            pred_prob,
            sortscan_metric_fn=f1_metric,
            sortscan_kernel=optimal_threshold_sortscan,
            max_iter=20,
            init="ovr_sortscan",
            tol_stops=1,
        )

        return thresholds

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
