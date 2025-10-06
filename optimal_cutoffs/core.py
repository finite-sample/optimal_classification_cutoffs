"""Core threshold optimization functionality.

This module contains the main get_optimal_threshold function and its supporting
infrastructure, serving as the primary entry point for threshold optimization.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .expected import (
    dinkelbach_expected_fbeta_binary,
    dinkelbach_expected_fbeta_multilabel,
)
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
    tolerance: float = 1e-10,
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
    tolerance : float, default=1e-10
        Numerical tolerance for floating-point comparisons in optimization
        algorithms. Affects boundary conditions and tie-breaking in sort-scan
        and scipy optimization methods.

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
            tolerance,
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
            tolerance,
        )
    elif mode == "bayes":
        from .bayes import optimize_bayes_thresholds
        result = optimize_bayes_thresholds(
            true_labs,
            pred_prob,
            utility,
            sample_weight,
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
    tolerance: float,
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
            true_labs, pred_prob, metric, method, average, sample_weight, comparison, tolerance
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
            tolerance=tolerance,
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
    tolerance: float,
):
    P = np.asarray(pred_prob, dtype=np.float64)
    is_multiclass = (P.ndim == 2 and P.shape[1] > 1)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    # We currently support expected F-beta only in this façade
    if metric.lower() not in {"f1", "fbeta"}:
        raise ValueError("mode='expected' currently supports F-beta only")

    if not is_multiclass:
        if P.ndim == 2 and P.shape[1] == 1:
            P = P.ravel()
        result = dinkelbach_expected_fbeta_binary(P, beta=beta, sample_weight=sw, comparison=comparison)
        thr, score = result.threshold, result.score

        def predict_binary(probs: ArrayLike) -> np.ndarray:
            q = np.asarray(probs, dtype=np.float64)
            if q.ndim == 2:
                if q.shape[1] == 2:
                    q = q[:, 1]
                elif q.shape[1] == 1:
                    q = q.ravel()
                else:
                    raise ValueError("binary probabilities must be (n,) or (n,2)")
            return (q > thr if comparison == ">" else q >= thr).astype(np.int32)

        return OptimizationResult(
            thresholds=np.array([thr], dtype=np.float64),
            scores=np.array([score], dtype=np.float64),
            predict=predict_binary,
            metric=f"expected_fbeta(beta={beta})",
            n_classes=2,
        )

    # Multilabel/multiclass: micro returns a single threshold; macro/weighted return per-class thresholds
    avg = average if average in {"macro", "micro", "weighted"} else "macro"
    out = dinkelbach_expected_fbeta_multilabel(
        P,
        beta=beta,
        sample_weight=sw,
        average=avg,
        true_labels=(None if true_labs is None else np.asarray(true_labs, dtype=int)),
        comparison=comparison,
    )

    # Check if this is micro averaging (single threshold) or macro/weighted (per-class thresholds)
    if out.thresholds.size == 1:  # micro averaging
        thr = float(out.thresholds[0])
        score = float(out.scores[0])

        def predict_micro(probs: ArrayLike) -> np.ndarray:
            q = np.asarray(probs, dtype=np.float64).ravel()
            return (q > thr if comparison == ">" else q >= thr).astype(np.int32)

        return OptimizationResult(
            thresholds=np.array([thr], dtype=np.float64),
            scores=np.array([score], dtype=np.float64),
            predict=predict_micro,
            metric=f"expected_fbeta(beta={beta},average=micro)",
            n_classes=P.shape[1],
        )

    # macro / weighted
    thrs = np.asarray(out.thresholds, dtype=np.float64)
    score = float(out.scores[0]) if out.scores.size == 1 else float(np.mean(out.scores))

    def predict_perclass(probs: ArrayLike) -> np.ndarray:
        q = np.asarray(probs, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != thrs.shape[0]:
            raise ValueError("Multiclass probabilities must be (n_samples, n_classes)")
        mask = (q > thrs if comparison == ">" else q >= thrs)
        masked = np.where(mask, q, -np.inf)
        pred = np.argmax(masked, axis=1)
        none = ~np.any(mask, axis=1)
        if np.any(none):
            pred[none] = np.argmax(q[none], axis=1)
        return pred.astype(np.int32)

    return OptimizationResult(
        thresholds=thrs,
        scores=np.array([score], dtype=np.float64),
        predict=predict_perclass,
        metric=f"expected_fbeta(beta={beta},average={avg})",
        n_classes=P.shape[1],
    )


__all__ = [
    "get_optimal_threshold",
]
