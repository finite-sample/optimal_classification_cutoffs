"""Core multiclass threshold optimization algorithms.

This module contains algorithms for multiclass threshold optimization,
using One-vs-Rest (OvR) strategy by default.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .binary_optimization import find_optimal_threshold
from .types import (
    AveragingMethodLiteral,
    ComparisonOperatorLiteral,
    OptimizationMethodLiteral,
    SampleWeightLike,
)
from .validation import validate_multiclass_input


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
    true_labs, pred_prob = validate_multiclass_input(
        true_labs, pred_prob, require_consecutive=False, require_proba=False
    )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    n_samples, n_classes = pred_prob.shape

    # Handle coordinate ascent method
    if method == "coord_ascent":
        # Additional validation for coordinate ascent - requires consecutive labels
        from .validation import validate_multiclass_labels

        validation_result = validate_multiclass_labels(
            true_labs, n_classes=n_classes, require_consecutive=True
        )
        validation_result.raise_if_invalid()

        # New implementation with fewer limitations
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

        # Use the new high-performance implementation
        from .multiclass_coord import coordinate_ascent_kernel

        # Convert inputs to proper types for Numba
        true_labs_int32 = np.asarray(true_labs, dtype=np.int32)
        pred_prob_float64 = np.asarray(pred_prob, dtype=np.float64, order="C")

        # Call the optimized kernel directly
        thresholds, _, _ = coordinate_ascent_kernel(
            true_labs_int32, pred_prob_float64, max_iter=20, tol=1e-12
        )

        return thresholds

    # Map method to strategy name for binary optimization calls
    method_mapping = {
        "sort_scan": "sort_scan",
        "unique_scan": "sort_scan",
        "minimize": "scipy",
        "gradient": "gradient",
        "coord_ascent": "sort_scan",  # Fallback for non-coord_ascent cases
    }
    strategy = method_mapping.get(method, "sort_scan")
    operator = ">=" if comparison == ">=" else ">"

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

        # Find single optimal threshold using new API
        optimal_threshold, _ = find_optimal_threshold(
            true_binary_flat,
            pred_prob_flat,
            metric,
            sample_weight_flat,
            strategy,
            operator,
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
            threshold, _ = find_optimal_threshold(
                true_binary_all[:, class_idx],
                pred_prob[:, class_idx],
                metric,
                sample_weight,
                strategy,
                operator,
            )
            optimal_thresholds[class_idx] = threshold

        return optimal_thresholds
