"""Core threshold optimization functionality.

This module contains the main get_optimal_threshold function and its supporting
infrastructure, serving as the primary entry point for threshold optimization.
"""

from collections.abc import Callable

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
    utility_matrix: np.ndarray | None = None,
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

    >>> # Cost-sensitive optimization
    >>> utility = {"tp": 10, "tn": 1, "fp": -5, "fn": -20}
    >>> result = get_optimal_threshold(y_true, y_prob, utility=utility)
    >>> result.predict(y_prob)

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
    
    # Validate beta for expected mode
    if mode == "expected" and beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

    # Validate inputs if we have true labels
    if true_labs is not None:
        validate_inputs(true_labs, pred_prob, allow_multiclass=True)

    # Route to mode-specific optimizers
    result: OptimizationResult
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

    return result


def _compute_utility_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    utility: dict[str, float],
    sample_weight: np.ndarray | None = None,
    minimize_cost: bool = False,
) -> float:
    """Compute utility-based metric for binary classification.
    
    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array of shape (n_samples,)
        Predicted binary labels (0 or 1)
    utility : dict
        Dict with keys "tp", "tn", "fp", "fn" specifying utilities/costs
    sample_weight : array of shape (n_samples,), optional
        Sample weights
    minimize_cost : bool, default=False
        If True, negate the utility (for cost minimization)
    
    Returns
    -------
    float
        Total utility or cost
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Compute confusion matrix elements
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        tp_count = np.sum(sample_weight[tp_mask])
        tn_count = np.sum(sample_weight[tn_mask])
        fp_count = np.sum(sample_weight[fp_mask])
        fn_count = np.sum(sample_weight[fn_mask])
    else:
        tp_count = np.sum(tp_mask)
        tn_count = np.sum(tn_mask)
        fp_count = np.sum(fp_mask)
        fn_count = np.sum(fn_mask)
    
    # Calculate total utility
    total_utility = (
        utility.get("tp", 0.0) * tp_count +
        utility.get("tn", 0.0) * tn_count +
        utility.get("fp", 0.0) * fp_count +
        utility.get("fn", 0.0) * fn_count
    )
    
    # Negate if minimizing cost
    if minimize_cost:
        total_utility = -total_utility
    
    return float(total_utility)


def _create_utility_metric_fn(
    utility: dict[str, float],
    minimize_cost: bool = False,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]:
    """Create a metric function for utility-based optimization.
    
    Parameters
    ----------
    utility : dict
        Dict with keys "tp", "tn", "fp", "fn"
    minimize_cost : bool, default=False
        If True, negate utilities for cost minimization
    
    Returns
    -------
    callable
        Metric function with signature (y_true, y_pred, sample_weight) -> float
    """
    def utility_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        return _compute_utility_metric(y_true, y_pred, utility, sample_weight, minimize_cost)
    
    return utility_metric


def _optimal_threshold_unique_scan(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    utility: dict[str, float] | None = None,
    minimize_cost: bool = False,
) -> tuple[float, float]:
    """Find optimal threshold using brute force over unique probabilities.
    
    Returns
    -------
    tuple[float, float]
        (optimal_threshold, best_score)
    """
    from .optimize import find_optimal_threshold

    # Handle utility-based metrics
    if utility is not None:
        pred_prob_arr = np.asarray(pred_prob)
        true_labs_arr = np.asarray(true_labs)
        
        # Get unique thresholds to try
        unique_probs = np.unique(pred_prob_arr)
        thresholds_to_try = np.concatenate([unique_probs, [0.0, 1.0]])
        thresholds_to_try = np.unique(thresholds_to_try)
        
        best_score = -np.inf
        best_threshold = 0.5
        
        utility_fn = _create_utility_metric_fn(utility, minimize_cost)
        
        for threshold in thresholds_to_try:
            if comparison == ">=":
                y_pred = (pred_prob_arr >= threshold).astype(int)
            else:
                y_pred = (pred_prob_arr > threshold).astype(int)
            
            score = utility_fn(true_labs_arr, y_pred, sample_weight)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    # Standard metric optimization
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
    return result.threshold, result.score


def _create_binary_predict_fn(
    threshold: float,
    comparison: str,
) -> Callable[[ArrayLike], np.ndarray]:
    """Create a prediction function for binary classification.
    
    Parameters
    ----------
    threshold : float
        Classification threshold
    comparison : str
        Comparison operator (">" or ">=")
    
    Returns
    -------
    callable
        Prediction function
    """
    def predict_binary(probs: ArrayLike) -> np.ndarray:
        p = np.asarray(probs)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        elif p.ndim == 2 and p.shape[1] == 1:
            p = p.ravel()
        if comparison == ">=":
            return (p >= threshold).astype(np.int32)
        else:
            return (p > threshold).astype(np.int32)
    
    return predict_binary


def _create_multiclass_predict_fn(
    thresholds: np.ndarray,
    comparison: str,
) -> Callable[[ArrayLike], np.ndarray]:
    """Create a prediction function for multiclass classification.
    
    Parameters
    ----------
    thresholds : array of shape (n_classes,)
        Per-class thresholds
    comparison : str
        Comparison operator (">" or ">=")
    
    Returns
    -------
    callable
        Prediction function
    """
    def predict_perclass(probs: ArrayLike) -> np.ndarray:
        q = np.asarray(probs, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != thresholds.shape[0]:
            raise ValueError(
                f"Multiclass probabilities must be (n_samples, {thresholds.shape[0]}), "
                f"got shape {q.shape}"
            )
        mask = (q > thresholds if comparison == ">" else q >= thresholds)
        masked = np.where(mask, q, -np.inf)
        pred = np.argmax(masked, axis=1)
        none = ~np.any(mask, axis=1)
        if np.any(none):
            pred[none] = np.argmax(q[none], axis=1)
        return pred.astype(np.int32)
    
    return predict_perclass


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
        raise ValueError("true_labs is required for empirical optimization")

    pred_prob_arr = np.asarray(pred_prob)

    # Detect binary vs multiclass
    is_multiclass = pred_prob_arr.ndim == 2 and pred_prob_arr.shape[1] > 1

    if is_multiclass:
        # Utility-based optimization not yet supported for multiclass
        if utility is not None:
            raise NotImplementedError(
                "Utility-based optimization is not yet supported for multiclass classification"
            )
        
        # Multiclass optimization  
        return find_optimal_threshold_multiclass(
            true_labs, pred_prob_arr, metric, method, average, sample_weight, comparison, tolerance
        )
    
    # Binary classification - reshape if needed
    if pred_prob_arr.ndim == 2 and pred_prob_arr.shape[1] == 1:
        pred_prob_arr = pred_prob_arr.ravel()

    # Handle utility-based optimization
    if utility is not None:
        # Validate utility dict
        required_keys = {"tp", "tn", "fp", "fn"}
        if not required_keys.issubset(utility.keys()):
            raise ValueError(f"utility dict must contain keys: {required_keys}")
        
        # Default minimize_cost to False if not specified
        if minimize_cost is None:
            minimize_cost = False
        
        # For utility-based optimization, use unique_scan or scipy minimize
        if method == "auto" or method == "sort_scan":
            method = "unique_scan"  # Use brute force for utility metrics
        
        if method == "unique_scan":
            threshold, score = _optimal_threshold_unique_scan(
                true_labs, pred_prob_arr, metric, sample_weight, comparison,
                utility, minimize_cost
            )
            
            return OptimizationResult(
                thresholds=np.array([threshold]),
                scores=np.array([score]),
                predict=_create_binary_predict_fn(threshold, comparison),
                metric="utility_based",
                n_classes=2,
            )
        else:
            raise ValueError(
                f"Method '{method}' not supported for utility-based optimization. "
                f"Use 'auto' or 'unique_scan'."
            )

    # Standard metric optimization
    if method == "unique_scan":
        threshold, score = _optimal_threshold_unique_scan(
            true_labs, pred_prob_arr, metric, sample_weight, comparison
        )
        
        return OptimizationResult(
            thresholds=np.array([threshold]),
            scores=np.array([score]),
            predict=_create_binary_predict_fn(threshold, comparison),
            metric=metric,
            n_classes=2,
        )

    # Select optimization method
    if method == "auto":
        method = "sort_scan" if is_piecewise_metric(metric) else "minimize"

    # Map method names to strategy names
    method_mapping = {
        "sort_scan": "sort_scan",
        "minimize": "scipy",
        "gradient": "gradient",
        "coord_ascent": "sort_scan",  # Binary fallback
    }

    if method not in method_mapping:
        raise ValueError(f"Invalid optimization method: {method}")

    strategy = method_mapping[method]
    operator = ">=" if comparison == ">=" else ">"

    # Use standard optimization API
    return find_optimal_threshold(
        true_labs,
        pred_prob_arr,
        metric,
        sample_weight,
        strategy,
        operator,
        require_probability=True,
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
) -> OptimizationResult:
    """Expected metric optimization using Dinkelbach method."""
    P = np.asarray(pred_prob, dtype=np.float64)
    is_multiclass = (P.ndim == 2 and P.shape[1] > 1)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    # We currently support expected F-beta only
    if metric.lower() not in {"f1", "fbeta"}:
        raise ValueError(
            f"mode='expected' currently supports F-beta only, got metric='{metric}'"
        )

    if not is_multiclass:
        # Binary classification
        if P.ndim == 2 and P.shape[1] == 1:
            P = P.ravel()
        
        result = dinkelbach_expected_fbeta_binary(
            P, beta=beta, sample_weight=sw, comparison=comparison
        )
        thr = result.thresholds[0]  # Extract scalar from array
        score = result.scores[0]    # Extract actual score, not mean

        return OptimizationResult(
            thresholds=np.array([thr], dtype=np.float64),
            scores=np.array([score], dtype=np.float64),
            predict=_create_binary_predict_fn(thr, comparison),
            metric=f"expected_fbeta(beta={beta})",
            n_classes=2,
        )

    # Multilabel/multiclass: micro returns a single threshold; macro/weighted return per-class
    avg = average if average in {"macro", "micro", "weighted"} else "macro"
    out = dinkelbach_expected_fbeta_multilabel(
        P,
        beta=beta,
        sample_weight=sw,
        average=avg,
        true_labels=(None if true_labs is None else np.asarray(true_labs, dtype=int)),
        comparison=comparison,
    )

    # Check if this is micro averaging (single threshold) or macro/weighted (per-class)
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

    return OptimizationResult(
        thresholds=thrs,
        scores=np.array([score], dtype=np.float64),
        predict=_create_multiclass_predict_fn(thrs, comparison),
        metric=f"expected_fbeta(beta={beta},average={avg})",
        n_classes=P.shape[1],
    )


__all__ = [
    "get_optimal_threshold",
]