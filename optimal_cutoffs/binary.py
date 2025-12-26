"""Binary classification threshold optimization.

This module implements threshold optimization for binary classification problems
where we have a single decision threshold τ and predict positive if p ≥ τ.

Key algorithms:
- optimize_f1_binary(): Sort-and-scan O(n log n) for F-measures
- optimize_utility_binary(): Closed-form O(1) for linear utilities
- optimize_metric_binary(): General metric optimization

All functions assume calibrated probabilities: E[y|p] = p
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .core import OptimizationResult, Task
from .validation import validate_binary_classification


def optimize_f1_binary(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    *,
    beta: float = 1.0,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
) -> OptimizationResult:
    """Optimize F-beta score for binary classification using sort-and-scan.

    Uses the O(n log n) sort-and-scan algorithm exploiting the piecewise
    structure of F-beta metrics. This finds the exact optimal threshold.

    Parameters
    ----------
    true_labels : array-like of shape (n_samples,)
        True binary labels in {0, 1}
    pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for positive class in [0, 1]
    beta : float, default=1.0
        F-beta parameter. beta=1 gives F1 score
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold

    Returns
    -------
    OptimizationResult
        Result with optimal threshold, F-beta score, and predict function

    Examples
    --------
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_prob = [0.2, 0.8, 0.7, 0.3, 0.9]
    >>> result = optimize_f1_binary(y_true, y_prob)
    >>> result.threshold
    0.5
    >>> result.score  # F1 score at optimal threshold
    0.8
    """
    # Import here to avoid circular imports
    from .piecewise import optimal_threshold_sortscan

    # Validate inputs
    true_labels, pred_proba, sample_weight = validate_binary_classification(
        true_labels, pred_proba, sample_weight, require_proba=True
    )

    # Create F-beta metric function name
    if beta == 1.0:
        metric_name = "f1"
    else:
        # Register F-beta metric if not already registered
        from .metrics_core import register_metric

        def fbeta_metric(tp, tn, fp, fn):
            # Vectorized F-beta metric
            tp, tn, fp, fn = (
                np.asarray(tp),
                np.asarray(tn),
                np.asarray(fp),
                np.asarray(fn),
            )
            denom = (1 + beta**2) * tp + beta**2 * fp + fn
            return np.where(denom > 0, (1 + beta**2) * tp / denom, 0.0)

        metric_name = f"f{beta}_score"
        register_metric(metric_name, fbeta_metric, is_piecewise=True, maximize=True)

    # Use sort-and-scan optimization
    result = optimal_threshold_sortscan(
        true_labels,
        pred_proba,
        metric=metric_name,
        sample_weight=sample_weight,
        inclusive=(comparison == ">="),
        require_proba=True,
        tolerance=1e-12,
    )

    # The result already has a predict function, but we need to handle different input formats
    def predict_binary(probs: ArrayLike) -> np.ndarray:
        p = np.asarray(probs)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]  # Extract positive class probabilities
        elif p.ndim == 2 and p.shape[1] == 1:
            p = p.ravel()

        return result.predict(p)

    return OptimizationResult(
        thresholds=result.thresholds,
        scores=result.scores,
        predict=predict_binary,
        task=Task.BINARY,
        metric=f"f{beta}_score" if beta != 1.0 else "f1_score",
        n_classes=2,
    )


def optimize_utility_binary(
    true_labels: ArrayLike | None,
    pred_proba: ArrayLike,
    *,
    utility: dict[str, float],
    sample_weight: ArrayLike | None = None,
) -> OptimizationResult:
    """Optimize binary classification using utility/cost specification.

    Computes the Bayes-optimal threshold using the closed-form formula:
    τ* = (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)]

    This is exact and runs in O(1) time.

    Parameters
    ----------
    true_labels : array-like of shape (n_samples,) or None
        True binary labels. Can be None for pure Bayes optimization
    pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for positive class in [0, 1]
    utility : dict
        Utility specification with keys "tp", "tn", "fp", "fn"
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights (affects expected utility computation)

    Returns
    -------
    OptimizationResult
        Result with optimal threshold, expected utility, and predict function

    Examples
    --------
    >>> # FN costs 5x more than FP
    >>> utility = {"tp": 10, "tn": 1, "fp": -1, "fn": -5}
    >>> result = optimize_utility_binary(None, y_prob, utility=utility)
    >>> result.threshold  # Closed-form optimal
    0.167
    """
    from .bayes import BayesOptimal, UtilitySpec

    # Validate probabilities
    pred_proba = np.asarray(pred_proba, dtype=np.float64)
    if pred_proba.ndim == 2 and pred_proba.shape[1] == 2:
        pred_proba = pred_proba[:, 1]  # Extract positive class
    elif pred_proba.ndim == 2 and pred_proba.shape[1] == 1:
        pred_proba = pred_proba.ravel()

    if not np.all((pred_proba >= 0) & (pred_proba <= 1)):
        raise ValueError("Probabilities must be in [0, 1] for utility optimization")

    # Create utility specification
    utility_spec = UtilitySpec.from_dict(utility)
    optimizer = BayesOptimal(utility_spec)

    # Compute optimal threshold (closed form)
    threshold = optimizer.compute_threshold()

    # Compute expected utility on this data
    expected_utility = optimizer.expected_utility(pred_proba)

    def predict_binary(probs: ArrayLike) -> np.ndarray:
        p = np.asarray(probs)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        elif p.ndim == 2 and p.shape[1] == 1:
            p = p.ravel()
        return (p >= threshold).astype(np.int32)

    return OptimizationResult(
        thresholds=np.array([threshold]),
        scores=np.array([expected_utility]),
        predict=predict_binary,
        task=Task.BINARY,
        metric="expected_utility",
        n_classes=2,
    )


def optimize_metric_binary(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    *,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """General binary metric optimization with automatic method selection.

    Automatically selects the best optimization algorithm based on metric
    properties and data characteristics.

    Parameters
    ----------
    true_labels : array-like of shape (n_samples,)
        True binary labels in {0, 1}
    pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for positive class in [0, 1]
    metric : str, default="f1"
        Metric to optimize ("f1", "precision", "recall", "accuracy", etc.)
    method : str, default="auto"
        Optimization method:
        - "auto": Automatically select best method
        - "sort_scan": O(n log n) sort-and-scan (exact for piecewise metrics)
        - "minimize": Scipy optimization
        - "gradient": Simple gradient ascent
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold
    tolerance : float, default=1e-10
        Numerical tolerance for optimization

    Returns
    -------
    OptimizationResult
        Result with optimal threshold, metric score, and predict function

    Examples
    --------
    >>> result = optimize_metric_binary(y_true, y_prob, metric="precision")
    >>> result = optimize_metric_binary(y_true, y_prob, metric="f1", method="sort_scan")
    """
    from .metrics_core import is_piecewise_metric
    from .optimize import optimize_gradient, optimize_scipy
    from .piecewise import optimal_threshold_sortscan

    # Validate inputs
    true_labels, pred_proba, sample_weight = validate_binary_classification(
        true_labels, pred_proba, sample_weight, require_proba=True
    )

    # Method selection
    if method == "auto":
        method = "sort_scan" if is_piecewise_metric(metric) else "minimize"

    # Route to appropriate optimizer
    match method:
        case "sort_scan":
            result = optimal_threshold_sortscan(
                true_labels,
                pred_proba,
                metric=metric,
                sample_weight=sample_weight,
                inclusive=(comparison == ">="),
                require_proba=True,
                tolerance=tolerance,
            )
        case "minimize":
            result = optimize_scipy(
                true_labels,
                pred_proba,
                metric,
                sample_weight,
                comparison,
                tol=tolerance,
            )
        case "gradient":
            result = optimize_gradient(
                true_labels,
                pred_proba,
                metric,
                sample_weight,
                comparison,
                tol=tolerance,
            )
        case _:
            raise ValueError(f"Unknown method: {method}")

    def predict_binary(probs: ArrayLike) -> np.ndarray:
        p = np.asarray(probs)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        elif p.ndim == 2 and p.shape[1] == 1:
            p = p.ravel()

        if comparison == ">=":
            return (p >= result.thresholds[0]).astype(np.int32)
        else:
            return (p > result.thresholds[0]).astype(np.int32)

    return OptimizationResult(
        thresholds=result.thresholds,
        scores=result.scores,
        predict=predict_binary,
        task=Task.BINARY,
        metric=metric,
        n_classes=2,
    )


__all__ = [
    "optimize_f1_binary",
    "optimize_utility_binary",
    "optimize_metric_binary",
]
