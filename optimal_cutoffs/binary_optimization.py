"""Core binary threshold optimization algorithms.

This module contains the fundamental algorithms for binary threshold optimization,
extracted to break circular dependencies and provide a clean separation of concerns.
"""

import numpy as np

from .metrics import METRIC_REGISTRY
from .piecewise import get_vectorized_metric, optimal_threshold_sortscan
from .types import ArrayLike, ComparisonOperator, SampleWeightLike
from .validation import _validate_inputs


def compute_confusion_matrix(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    sample_weight: SampleWeightLike = None,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix components (tp, tn, fp, fn).

    Parameters
    ----------
    true_labels : ArrayLike
        True binary labels (0 or 1)
    pred_labels : ArrayLike
        Predicted binary labels (0 or 1)
    sample_weight : SampleWeightLike, optional
        Sample weights

    Returns
    -------
    tuple[float, float, float, float]
        True positives, true negatives, false positives, false negatives
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
    else:
        weights = np.ones_like(true_labels, dtype=float)

    # Compute confusion matrix components
    tp = float(np.sum(weights[(true_labels == 1) & (pred_labels == 1)]))
    tn = float(np.sum(weights[(true_labels == 0) & (pred_labels == 0)]))
    fp = float(np.sum(weights[(true_labels == 0) & (pred_labels == 1)]))
    fn = float(np.sum(weights[(true_labels == 1) & (pred_labels == 0)]))

    return tp, tn, fp, fn


def metric_score(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    threshold: float,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperator = ">",
) -> float:
    """Compute metric score at a given threshold for binary classification.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    pred_prob : array-like of shape (n_samples,)
        Predicted probabilities
    threshold : float
        Classification threshold
    metric : str, default="f1"
        Metric to compute (e.g., "f1", "accuracy", "precision", "recall")
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application

    Returns
    -------
    float
        Metric score at the given threshold
    """
    true_labs, pred_prob, _ = _validate_inputs(true_labs, pred_prob)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    # Apply threshold to get predictions
    if comparison == ">":
        pred_labels = (pred_prob > threshold).astype(int)
    else:  # ">="
        pred_labels = (pred_prob >= threshold).astype(int)

    # Get confusion matrix components
    tp, tn, fp, fn = compute_confusion_matrix(
        true_labs, pred_labels, sample_weight=sample_weight
    )

    # Compute metric using registry
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"Metric '{metric}' not supported. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )

    metric_func = METRIC_REGISTRY[metric]
    return float(metric_func(tp, tn, fp, fn))


def optimal_threshold_piecewise(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperator = ">",
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
        # Ensure threshold is in valid range when using probabilities
        if require_proba:
            threshold = max(0.0, min(1.0, threshold))
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
    comparison: ComparisonOperator = ">",
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
            score = metric_score(
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
    comparison: ComparisonOperator = ">",
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
            score = metric_score(
                true_labs, pred_prob, threshold, metric, sample_weight, comparison
            )
            return -score  # Minimize negative score = maximize score
        except (ValueError, ZeroDivisionError):
            return np.inf  # Return large value for invalid thresholds

    # Optimize over [0, 1] interval
    try:
        result = optimize.minimize_scalar(
            objective, bounds=(0.0, 1.0), method="bounded"
        )
        if result.success:
            return float(result.x)
    except Exception:
        pass

    # Fallback to piecewise method
    return optimal_threshold_piecewise(
        true_labs, pred_prob, metric, sample_weight, comparison
    )


def optimal_threshold_gradient(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperator = ">",
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
            score_plus = metric_score(
                true_labs, pred_prob, thresh_plus, metric, sample_weight, comparison
            )
            score_minus = metric_score(
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
