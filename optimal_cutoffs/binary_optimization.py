"""Core binary threshold optimization algorithms.

This module contains the fundamental algorithms for binary threshold optimization,
extracted to break circular dependencies and provide a clean separation of concerns.
"""

import numpy as np
from numpy.typing import ArrayLike

from .metrics import (
    compute_metric_at_threshold,
    get_vectorized_metric,
)
from .piecewise import optimal_threshold_sortscan
from .types import ComparisonOperatorLiteral, SampleWeightLike
from .validation import _validate_inputs

# Note: confusion matrix and metric computation functions are now centralized
# in metrics.py. Use compute_confusion_matrix_from_labels() and
# compute_metric_at_threshold() instead.


def optimal_threshold_piecewise(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperatorLiteral = ">",
    require_proba: bool = True,
) -> float:
    """Find optimal threshold using O(n log n) piecewise-constant optimization.

    Uses the sort-and-scan algorithm for metrics that are piecewise-constant
    as a function of threshold.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    metric : str, default="f1"
        Metric to optimize (must be piecewise-constant)
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application
    require_proba : bool, default=True
        If True, validate that pred_prob is in [0, 1]. If False, allow
        arbitrary score ranges (e.g., logits).

    Returns
    -------
    float
        Optimal threshold value

    Raises
    ------
    ValueError
        If the metric is not supported or inputs are invalid
    """
    true_labs, pred_prob, _ = _validate_inputs(
        true_labs, pred_prob, require_proba=require_proba
    )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    # Try the fast piecewise optimization
    try:
        vectorized_metric = get_vectorized_metric(metric)
        threshold, _, _ = optimal_threshold_sortscan(
            true_labs,
            pred_prob,
            vectorized_metric,
            sample_weight=sample_weight,
            inclusive=(comparison == ">="),
            require_proba=require_proba,
        )
        # Allow slight tolerance outside [0,1] for edge cases
        # (e.g., all-negative with >=)
        # Use machine epsilon for floating point precision
        if require_proba:
            eps = np.finfo(float).eps  # approximately 2.22e-16
            tolerance = eps * 10  # small buffer for floating point comparisons
            threshold = max(
                float(-tolerance), min(float(1.0 + tolerance), float(threshold))
            )
        return float(threshold)
    except (ValueError, NotImplementedError, KeyError):
        pass

    # Fall back to brute force optimization
    return _optimal_threshold_piecewise_fallback(
        true_labs, pred_prob, metric, sample_weight, comparison
    )


def _optimal_threshold_piecewise_fallback(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperatorLiteral = ">",
) -> float:
    """Fallback implementation using brute force over unique probabilities."""
    true_labs, pred_prob, _ = _validate_inputs(true_labs, pred_prob)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    # Get unique thresholds to test
    unique_probs = np.unique(pred_prob)

    # Add boundary values
    if comparison == ">":
        candidates = np.concatenate([[0.0], unique_probs, [1.0]])
    else:  # ">="
        candidates = np.concatenate([[0.0], unique_probs])

    best_threshold = 0.0
    best_score = -np.inf

    for threshold in candidates:
        try:
            score = compute_metric_at_threshold(
                true_labs, pred_prob, threshold, metric, sample_weight, comparison
            )
            if score > best_score:
                best_score = score
                best_threshold = threshold
        except (ValueError, ZeroDivisionError):
            # Skip invalid thresholds
            continue

    return float(best_threshold)


def optimal_threshold_minimize(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperatorLiteral = ">",
) -> float:
    """Find optimal threshold using scipy.optimize.minimize_scalar.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    metric : str, default="f1"
        Metric to optimize
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application

    Returns
    -------
    float
        Optimal threshold value
    """
    from scipy import optimize  # type: ignore[import-untyped]

    true_labs, pred_prob, _ = _validate_inputs(true_labs, pred_prob)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    def objective(threshold: float) -> float:
        """Objective function to minimize (negative metric score)."""
        try:
            score = compute_metric_at_threshold(
                true_labs, pred_prob, threshold, metric, sample_weight, comparison
            )
            return -score  # Minimize negative score = maximize score
        except (ValueError, ZeroDivisionError):
            return np.inf  # Return large value for invalid thresholds

    # Optimize over [0, 1] interval
    scipy_threshold = None
    scipy_score = -np.inf

    try:
        result = optimize.minimize_scalar(
            objective, bounds=(0.0, 1.0), method="bounded"
        )
        if result.success:
            scipy_threshold = float(result.x)
            scipy_score = -result.fun  # Convert back from negative
    except Exception:
        pass

    # Always try a few key candidate thresholds to ensure we don't miss obvious optima
    unique_probs = np.unique(pred_prob)
    candidates = [0.0, 1.0]

    # Add unique probabilities and their neighbors for discrete optimization
    for prob in unique_probs:
        candidates.extend([prob - 1e-10, prob, prob + 1e-10])

    # Remove duplicates and sort
    candidates = sorted(set(candidates))
    candidates = [c for c in candidates if 0.0 <= c <= 1.0]

    best_threshold = scipy_threshold if scipy_threshold is not None else 0.5
    best_score = scipy_score

    # Test candidate thresholds
    for threshold in candidates:
        try:
            score = compute_metric_at_threshold(
                true_labs, pred_prob, threshold, metric, sample_weight, comparison
            )
            if score > best_score:
                best_score = score
                best_threshold = threshold
        except (ValueError, ZeroDivisionError):
            continue

    # If we still don't have a good result, fallback to piecewise method
    if best_score == -np.inf or best_threshold is None:
        return optimal_threshold_piecewise(
            true_labs, pred_prob, metric, sample_weight, comparison
        )

    return float(best_threshold)


def optimal_threshold_gradient(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperatorLiteral = ">",
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> float:
    """Find optimal threshold using gradient ascent.

    Simple gradient-based optimization for threshold selection.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    metric : str, default="f1"
        Metric to optimize
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application
    learning_rate : float, default=0.01
        Learning rate for gradient ascent
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-6
        Tolerance for convergence

    Returns
    -------
    float
        Optimal threshold value
    """
    true_labs, pred_prob, _ = _validate_inputs(true_labs, pred_prob)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    # Start from middle of probability range
    threshold = 0.5

    for _ in range(max_iter):
        # Compute gradient using finite differences
        eps = 1e-6

        # Clamp to valid range
        thresh_plus = min(threshold + eps, 1.0)
        thresh_minus = max(threshold - eps, 0.0)

        try:
            score_plus = compute_metric_at_threshold(
                true_labs, pred_prob, thresh_plus, metric, sample_weight, comparison
            )
            score_minus = compute_metric_at_threshold(
                true_labs, pred_prob, thresh_minus, metric, sample_weight, comparison
            )

            # Finite difference gradient
            gradient = (score_plus - score_minus) / (thresh_plus - thresh_minus)

            # Update threshold
            new_threshold = threshold + learning_rate * gradient
            new_threshold = np.clip(new_threshold, 0.0, 1.0)

            # Check convergence
            if abs(new_threshold - threshold) < tol:
                break

            threshold = new_threshold

        except (ValueError, ZeroDivisionError):
            # Fallback to piecewise method
            return optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )

    return float(threshold)
