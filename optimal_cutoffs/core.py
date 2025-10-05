"""Core threshold optimization functionality.

This module contains the main get_optimal_threshold function and its supporting
infrastructure, serving as the primary entry point for threshold optimization.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .metrics import is_piecewise_metric
from .types_minimal import OptimizationResult
from .validation import (
    _validate_comparison_operator,
    _validate_metric_name,
    _validate_optimization_method,
    validate_inputs,
)


def get_optimal_threshold(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    mode: str = "empirical",
    utility: dict[str, float] | None = None,
    utility_matrix: np.ndarray[Any, Any] | None = None,
    minimize_cost: bool | None = None,
    beta: float = 1.0,
    class_weight: ArrayLike | None = None,
    average: str = "macro",
) -> OptimizationResult:
    """Find the optimal classification threshold(s) for a given metric.

    This is the main entry point for threshold optimization, supporting both
    binary and multiclass classification across multiple optimization modes.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,), optional
        True class labels. For binary: values in {0, 1}. For multiclass:
        values in {0, 1, ..., n_classes-1}. Can be None for mode='bayes'
        with utility_matrix.
    pred_prob : array-like
        Predicted probabilities. For binary: 1D array of shape (n_samples,)
        with probabilities for the positive class. For multiclass: 2D array
        of shape (n_samples, n_classes) with class probabilities.
    metric : str, default="f1"
        Metric to optimize. Supported metrics include "accuracy", "f1",
        "precision", "recall", etc. See metrics.METRICS for full list.
    method : {
        "auto", "sort_scan", "unique_scan", "minimize", "gradient", "coord_ascent"
    }
        default="auto"
        Optimization method:
        - "auto": Automatically selects best method based on metric and data
        - "sort_scan": O(n log n) algorithm for piecewise metrics with
          vectorized implementation
        - "unique_scan": Evaluates all unique probabilities
        - "minimize": Uses scipy.optimize.minimize_scalar
        - "gradient": Simple gradient ascent
        - "coord_ascent": Coordinate ascent for coupled multiclass optimization
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights for handling class imbalance.
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application.
    mode : {"empirical", "expected", "bayes"}, default="empirical"
        Optimization mode:
        - "empirical": Standard threshold optimization on observed data
        - "expected": Expected metric optimization using Dinkelbach method
        - "bayes": Bayes-optimal decisions under calibrated probabilities
    utility : dict, optional
        Utility specification for cost/benefit-aware optimization.
        Dict with keys "tp", "tn", "fp", "fn" specifying utilities/costs.
    utility_matrix : array-like of shape (D, K), optional
        Utility matrix for multiclass Bayes decisions where D=decisions, K=classes.
    minimize_cost : bool, optional
        If True, interpret utility values as costs to minimize.
    beta : float, default=1.0
        F-beta parameter for expected mode (beta >= 0).
    class_weight : array-like of shape (K,), optional
        Per-class weights for weighted averaging in expected mode.
    average : {"macro", "micro", "weighted", "none"}, default="macro"
        Averaging strategy for multiclass metrics.

    Returns
    -------
    OptimizationResult
        Unified optimization result with:
        - thresholds: array of optimal thresholds
        - scores: array of metric scores at thresholds
        - predict: function for making predictions
        - diagnostics: optional computation details
        - Works consistently across all modes and methods

    Examples
    --------
    >>> # Binary classification
    >>> y_true = [0, 1, 0, 1, 1]
    >>> y_prob = [0.1, 0.8, 0.3, 0.9, 0.7]
    >>> result = get_optimal_threshold(y_true, y_prob, metric="f1")
    >>> result.threshold  # Access single threshold
    >>> result.predict(y_prob)  # Make predictions

    >>> # Multiclass classification
    >>> y_true = [0, 1, 2, 1, 0]
    >>> y_prob = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], ...]
    >>> result = get_optimal_threshold(y_true, y_prob, metric="f1")
    >>> result.thresholds  # Access per-class thresholds
    >>> result.predict(y_prob)  # Make predictions
    """
    # Validate comparison operator early
    _validate_comparison_operator(comparison)

    # Validate metric name
    _validate_metric_name(metric)

    # Validate optimization method
    _validate_optimization_method(method)

    # Validate inputs if we have true labels
    if true_labs is not None:
        validate_inputs(true_labs, pred_prob, allow_multiclass=True)

    # Route to mode-specific optimizers (simplified from router pattern)
    result: Any
    if mode == "empirical":
        result = _optimize_empirical(
            true_labs,
            pred_prob,
            metric,
            method,
            sample_weight,
            comparison,
            utility,
            minimize_cost,
            average,
        )
    elif mode == "expected":
        result = _optimize_expected(
            true_labs,
            pred_prob,
            metric,
            method,
            sample_weight,
            comparison,
            beta,
            class_weight,
            average,
        )
    elif mode == "bayes":
        result = _optimize_bayes(
            true_labs, pred_prob, utility, utility_matrix, minimize_cost, comparison
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Return result directly - no more conversion layers
    return result  # type: ignore[no-any-return]


# Removed _convert_to_result - no longer needed with direct return types


def _optimal_threshold_unique_scan(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
) -> float:
    """Find optimal threshold using brute force over unique probabilities."""
    from .optimize import find_optimal_threshold

    operator = ">=" if comparison == ">=" else ">"
    result = find_optimal_threshold(
        true_labs,
        pred_prob,
        metric,
        sample_weight,
        "sort_scan",
        operator,
        require_probability=True,
    )
    return result.threshold


def _optimize_empirical(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str,
    method: str,
    sample_weight: ArrayLike | None,
    comparison: str,
    utility: dict[str, float] | None,
    minimize_cost: bool | None,
    average: str,
) -> OptimizationResult:
    """Empirical threshold optimization."""
    from .optimize import find_optimal_threshold, find_optimal_threshold_multiclass

    if true_labs is None:
        raise ValueError("true_labs is required for empirical utility optimization")

    pred_prob = np.asarray(pred_prob)

    # Detect binary vs multiclass
    is_multiclass = pred_prob.ndim == 2 and pred_prob.shape[1] > 1

    if is_multiclass:
        # Multiclass optimization
        return find_optimal_threshold_multiclass(
            true_labs, pred_prob, metric, method, average, sample_weight, comparison
        )
    else:
        # Binary optimization
        if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
            pred_prob = pred_prob.ravel()

        # Handle utility-based optimization
        if utility is not None:
            # Convert utility to cost-sensitive metric
            # This would need implementation
            pass

        # Handle specific method cases
        if method == "unique_scan":
            threshold = _optimal_threshold_unique_scan(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
            # Create a simple OptimizationResult for unique_scan
            def predict_binary(probs):
                p = np.asarray(probs)
                if p.ndim == 2 and p.shape[1] == 2:
                    p = p[:, 1]
                elif p.ndim == 2 and p.shape[1] == 1:
                    p = p.ravel()
                if comparison == ">=":
                    return (p >= threshold).astype(np.int32)
                else:
                    return (p > threshold).astype(np.int32)

            return OptimizationResult(
                thresholds=np.array([threshold]),
                scores=np.array([0.0]),  # Score not computed in unique_scan
                predict=predict_binary,
                metric=metric,
                n_classes=2,
            )

        # Select optimization method
        if method == "auto":
            method = "sort_scan" if is_piecewise_metric(metric) else "minimize"

        # Map remaining method names to strategy names
        method_mapping = {
            "sort_scan": "sort_scan",
            "minimize": "scipy",
            "gradient": "gradient",
            "coord_ascent": "sort_scan",  # Fallback to sort_scan for binary
        }

        if method not in method_mapping:
            raise ValueError(f"Invalid optimization method: {method}")

        strategy = method_mapping[method]
        operator = ">=" if comparison == ">=" else ">"

        # Use new API
        return find_optimal_threshold(
            true_labs,
            pred_prob,
            metric,
            sample_weight,
            strategy,
            operator,
            require_probability=True,  # Default to requiring probabilities
        )


def _optimize_expected(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str,
    method: str,
    sample_weight: ArrayLike | None,
    comparison: str,
    beta: float,
    class_weight: ArrayLike | None,
    average: str,
) -> OptimizationResult:
    """Expected metric optimization using Dinkelbach method."""
    from .expected import (
        dinkelbach_expected_fbeta_binary,
        dinkelbach_expected_fbeta_multilabel,
    )

    # Expected mode doesn't actually need true_labs since it works with
    # calibrated probabilities

    pred_prob = np.asarray(pred_prob)
    is_multiclass = pred_prob.ndim == 2 and pred_prob.shape[1] > 1

    if is_multiclass:
        # Ensure average is compatible with expected function
        avg_method = average if average in {"macro", "weighted", "micro"} else "macro"
        sw = np.asarray(sample_weight) if sample_weight is not None else None
        return dinkelbach_expected_fbeta_multilabel(
            pred_prob,
            beta,
            sw,  # sample_weight
            avg_method,  # average
            true_labels=true_labs,  # true_labels
            comparison=comparison,
        )
    else:
        if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
            pred_prob = pred_prob.ravel()
        sw_array = np.asarray(sample_weight) if sample_weight is not None else None
        return dinkelbach_expected_fbeta_binary(pred_prob, beta, sw_array, comparison)


def _optimize_bayes(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    utility: dict[str, float] | None,
    utility_matrix: np.ndarray[Any, Any] | None,
    minimize_cost: bool | None,
    comparison: str,
) -> OptimizationResult:
    """Bayes-optimal threshold optimization."""
    from .bayes import (
        bayes_optimal_decisions,
        bayes_optimal_threshold,
        bayes_thresholds_from_costs,
    )

    pred_prob = np.asarray(pred_prob)
    is_multiclass = pred_prob.ndim == 2 and pred_prob.shape[1] > 1

    if utility_matrix is not None:
        # Use utility matrix for Bayes decisions
        if not is_multiclass:
            raise ValueError("utility_matrix requires multiclass probabilities")
        return bayes_optimal_decisions(pred_prob, utility_matrix)

    elif utility is not None:
        # Use utility dict for threshold computation

        # For multiclass, require both fp and fn to be specified
        if is_multiclass:
            if "fp" not in utility or "fn" not in utility:
                raise ValueError("Multiclass Bayes requires 'fp' and 'fn' as arrays")

        # Extract costs and benefits
        fp_cost = utility.get("fp", 0)
        fn_cost = utility.get("fn", 0)
        tp_benefit = utility.get("tp", 0)
        tn_benefit = utility.get("tn", 0)

        # Handle minimize_cost flag
        if minimize_cost:
            # Negate costs to convert to utilities
            fp_cost = -abs(fp_cost) if fp_cost >= 0 else fp_cost
            fn_cost = -abs(fn_cost) if fn_cost >= 0 else fn_cost

        if is_multiclass:
            # Per-class thresholds using vectorized function
            fp_costs = np.asarray(fp_cost) if not np.isscalar(fp_cost) else [fp_cost]
            fn_costs = np.asarray(fn_cost) if not np.isscalar(fn_cost) else [fn_cost]

            return bayes_thresholds_from_costs(fp_costs, fn_costs)
        else:
            # Single threshold
            return bayes_optimal_threshold(
                float(fp_cost), float(fn_cost), float(tp_benefit), float(tn_benefit)
            )

    else:
        raise ValueError("mode='bayes' requires utility parameter or utility_matrix")


__all__ = [
    "get_optimal_threshold",
]
