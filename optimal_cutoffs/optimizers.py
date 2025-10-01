"""Threshold search strategies for optimizing classification metrics."""

from typing import Any

import numpy as np

from .metrics import (
    is_piecewise_metric,
)
from .results import ThresholdResult, create_result
from .types import (
    ArrayLike,
    AveragingMethod,
    ComparisonOperator,
    EstimationMode,
    ExpectedResult,
    OptimizationMethod,
    SampleWeightLike,
    UtilityDict,
    UtilityMatrix,
)


def get_optimal_threshold(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
    *,
    mode: EstimationMode = "empirical",
    utility: UtilityDict | None = None,
    utility_matrix: UtilityMatrix | None = None,
    minimize_cost: bool | None = None,
    beta: float = 1.0,
    class_weight: ArrayLike | None = None,
    average: AveragingMethod = "macro",
    return_result: bool = False,
) -> (
    float
    | np.ndarray[Any, Any]
    | ExpectedResult
    | tuple[float, float]
    | ThresholdResult
):
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
        "precision", "recall", etc. See metrics.METRIC_REGISTRY for full list.
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
    return_result : bool, default=False
        If True, return unified ThresholdResult object. If False, return
        legacy format for backward compatibility.

    Returns
    -------
    threshold : float or np.ndarray or ExpectedResult or tuple or ThresholdResult
        Optimal threshold(s). Return type depends on return_result parameter:

        If return_result=True: Always returns ThresholdResult with consistent interface

        If return_result=False (default): Legacy format depends on mode and input:
        - Binary empirical: float (single threshold)
        - Multiclass empirical: np.ndarray (per-class thresholds)
        - Expected mode: ExpectedResult dict or tuple[float, float]
        - Bayes mode: varies based on utility specification

    Examples
    --------
    >>> # Binary classification
    >>> y_true = [0, 1, 0, 1, 1]
    >>> y_prob = [0.1, 0.8, 0.3, 0.9, 0.7]
    >>> threshold = get_optimal_threshold(y_true, y_prob, metric="f1")

    >>> # Multiclass classification
    >>> y_true = [0, 1, 2, 1, 0]
    >>> y_prob = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], ...]
    >>> thresholds = get_optimal_threshold(y_true, y_prob, metric="f1")
    """
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

    # Handle return format
    if return_result:
        # Convert to ThresholdResult if not already
        if isinstance(result, ThresholdResult):
            return result
        else:
            # Convert legacy format to ThresholdResult
            return _convert_to_result(result, mode, method, average)
    else:
        # Return legacy format
        if isinstance(result, ThresholdResult):
            return result.to_legacy_format()
        else:
            return result  # type: ignore[no-any-return]


def _convert_to_result(
    legacy_result: Any, mode: str, method: str, average: str
) -> ThresholdResult:
    """Convert legacy result format to ThresholdResult."""
    if isinstance(legacy_result, (int, float)):
        # Binary empirical
        return create_result(
            threshold=float(legacy_result),
            method=method,
            mode=mode,
        )
    elif isinstance(legacy_result, np.ndarray):
        # Multiclass empirical or Bayes
        return create_result(
            threshold=legacy_result,
            method=method,
            mode=mode,
            averaging_method=average,
        )
    elif isinstance(legacy_result, tuple) and len(legacy_result) == 2:
        # Expected binary
        threshold, score = legacy_result
        return create_result(
            threshold=float(threshold),
            score=float(score),
            method=method,
            mode=mode,
        )
    elif isinstance(legacy_result, dict):
        # Expected multiclass
        if "threshold" in legacy_result:
            # Micro averaging
            return create_result(
                threshold=legacy_result["threshold"],
                score=legacy_result.get("f_beta"),
                method=method,
                mode=mode,
                averaging_method="micro",
            )
        else:
            # Macro/weighted/none averaging
            return create_result(
                threshold=legacy_result["thresholds"],
                score=legacy_result.get("f_beta"),
                per_class_scores=legacy_result.get("f_beta_per_class"),
                method=method,
                mode=mode,
                averaging_method=average,
            )
    else:
        # Fallback - just wrap the result
        return create_result(
            threshold=legacy_result,
            method=method,
            mode=mode,
        )


def _optimal_threshold_unique_scan(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: SampleWeightLike = None,
    comparison: ComparisonOperator = ">",
) -> float:
    """Find optimal threshold using brute force over unique probabilities."""
    from .binary_optimization import _optimal_threshold_piecewise_fallback

    return _optimal_threshold_piecewise_fallback(
        true_labs, pred_prob, metric, sample_weight, comparison
    )


def _optimize_empirical(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str,
    method: OptimizationMethod,
    sample_weight: ArrayLike | None,
    comparison: ComparisonOperator,
    utility: UtilityDict | None,
    minimize_cost: bool | None,
    average: AveragingMethod,
) -> float | np.ndarray[Any, Any]:
    """Empirical threshold optimization."""
    from .binary_optimization import (
        optimal_threshold_gradient,
        optimal_threshold_minimize,
        optimal_threshold_piecewise,
    )
    from .multiclass_optimization import get_optimal_multiclass_thresholds

    if true_labs is None:
        raise ValueError("true_labs is required for empirical mode")

    pred_prob = np.asarray(pred_prob)

    # Detect binary vs multiclass
    is_multiclass = pred_prob.ndim == 2 and pred_prob.shape[1] > 1

    if is_multiclass:
        # Multiclass optimization
        return get_optimal_multiclass_thresholds(
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

        # Select optimization method
        if method == "auto":
            method = "sort_scan" if is_piecewise_metric(metric) else "minimize"

        if method == "sort_scan":
            return optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        elif method == "unique_scan":
            # Use brute force over unique probabilities
            return _optimal_threshold_unique_scan(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        elif method == "minimize":
            return optimal_threshold_minimize(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        elif method == "gradient":
            return optimal_threshold_gradient(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        elif method == "coord_ascent":
            # Coord ascent only for multiclass, fallback to piecewise for binary
            return optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        else:
            raise ValueError(f"Invalid optimization method: {method}")


def _optimize_expected(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str,
    method: OptimizationMethod,
    sample_weight: ArrayLike | None,
    comparison: ComparisonOperator,
    beta: float,
    class_weight: ArrayLike | None,
    average: AveragingMethod,
) -> ExpectedResult | tuple[float, float]:
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
        cw = np.asarray(class_weight) if class_weight is not None else None
        return dinkelbach_expected_fbeta_multilabel(
            pred_prob, beta, avg_method, sw, cw, comparison  # type: ignore[arg-type]
        )
    else:
        if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
            pred_prob = pred_prob.ravel()
        sw_array = np.asarray(sample_weight) if sample_weight is not None else None
        return dinkelbach_expected_fbeta_binary(pred_prob, beta, sw_array, comparison)


def _optimize_bayes(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    utility: UtilityDict | None,
    utility_matrix: UtilityMatrix | None,
    minimize_cost: bool | None,
    comparison: ComparisonOperator,
) -> float | np.ndarray[Any, Any]:
    """Bayes-optimal threshold optimization."""
    from .bayes import (
        bayes_decision_from_utility_matrix,
        bayes_threshold_from_costs_scalar,
        bayes_thresholds_from_costs_vector,
    )

    pred_prob = np.asarray(pred_prob)
    is_multiclass = pred_prob.ndim == 2 and pred_prob.shape[1] > 1

    if utility_matrix is not None:
        # Use utility matrix for Bayes decisions
        if not is_multiclass:
            raise ValueError("utility_matrix requires multiclass probabilities")
        return bayes_decision_from_utility_matrix(pred_prob, utility_matrix)  # type: ignore[return-value]

    elif utility is not None:
        # Use utility dict for threshold computation

        # For multiclass, require both fp and fn to be specified
        if is_multiclass:
            if "fp" not in utility or "fn" not in utility:
                raise ValueError("Multiclass Bayes requires 'fp' and 'fn' as arrays")

        if minimize_cost:
            # Negate costs to convert to utilities
            fp_cost = -utility.get("fp", 0)
            fn_cost = -utility.get("fn", 0)
            tp_benefit = -utility.get("tp", 0)
            tn_benefit = -utility.get("tn", 0)
        else:
            fp_cost = utility.get("fp", 0)
            fn_cost = utility.get("fn", 0)
            tp_benefit = utility.get("tp", 0)
            tn_benefit = utility.get("tn", 0)

        if is_multiclass:
            # Per-class thresholds
            n_classes = pred_prob.shape[1]

            # Handle both scalar and vector costs
            def _expand_cost(cost: Any, n_classes: int) -> list[float]:
                if np.isscalar(cost):
                    return [float(cost)] * n_classes  # type: ignore[arg-type]
                else:
                    cost_array = np.asarray(cost)
                    if cost_array.size == 1:
                        return [float(cost_array.item())] * n_classes
                    elif cost_array.size == n_classes:
                        return [float(x) for x in cost_array.tolist()]
                    else:
                        raise ValueError(
                            f"Cost array size {cost_array.size} doesn't match "
                            f"n_classes {n_classes}"
                        )

            fp_costs = _expand_cost(fp_cost, n_classes)
            fn_costs = _expand_cost(fn_cost, n_classes)
            tp_benefits = _expand_cost(tp_benefit, n_classes)
            tn_benefits = _expand_cost(tn_benefit, n_classes)

            return bayes_thresholds_from_costs_vector(
                fp_costs, fn_costs, tp_benefits, tn_benefits, comparison
            )  # type: ignore[return-value]
        else:
            # Single threshold
            return bayes_threshold_from_costs_scalar(
                fp_cost, fn_cost, tp_benefit, tn_benefit, comparison
            )

    else:
        raise ValueError("mode='bayes' requires utility parameter or utility_matrix")


# Legacy function removal - these are now in separate modules:
# - get_optimal_multiclass_thresholds -> multiclass_optimization.py
# Functions that are no longer needed due to router/handler elimination

__all__ = [
    "get_optimal_threshold",
]
