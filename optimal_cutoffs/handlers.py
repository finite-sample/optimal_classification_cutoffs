"""Mode-specific handlers for threshold optimization.

This module provides a clean separation between different estimation modes:
- EmpiricalHandler: Traditional optimization on observed data
- ExpectedHandler: Expected metric optimization under calibration
- BayesHandler: Bayes-optimal decisions under known utilities
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import (
    ArrayLike,
    ComparisonOperator,
    OptimizationMethod,
    SampleWeightLike,
    UtilityDict,
    UtilityMatrix,
)


class BaseHandler(ABC):
    """Base class for mode-specific optimization handlers."""

    @abstractmethod
    def optimize_threshold(
        self,
        true_labs: ArrayLike,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod = "auto",
        sample_weight: SampleWeightLike = None,
        comparison: ComparisonOperator = ">",
        **kwargs: Any,
    ) -> Any:
        """Optimize threshold for the specific mode.

        Parameters
        ----------
        true_labs : ArrayLike
            True labels
        pred_prob : ArrayLike
            Predicted probabilities
        metric : str
            Metric to optimize
        method : OptimizationMethod
            Optimization method
        sample_weight : SampleWeightLike
            Sample weights
        comparison : ComparisonOperator
            Comparison operator
        **kwargs : Any
            Additional mode-specific parameters

        Returns
        -------
        Any
            Optimization result (format depends on mode)
        """
        pass


class EmpiricalHandler(BaseHandler):
    """Handler for empirical threshold optimization on observed data."""

    def optimize_threshold(
        self,
        true_labs: ArrayLike,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod = "auto",
        sample_weight: SampleWeightLike = None,
        comparison: ComparisonOperator = ">",
        *,
        utility: UtilityDict | None = None,
        minimize_cost: bool | None = None,
        **kwargs: Any,
    ) -> float | np.ndarray[Any, Any]:
        """Optimize threshold using empirical optimization methods.

        Handles both standard metric optimization and utility-based optimization.
        """

        from .metrics import (
            has_vectorized_implementation,
            is_piecewise_metric,
            make_linear_counts_metric,
        )
        from .optimizers import get_optimal_multiclass_thresholds
        from .piecewise import optimal_threshold_sortscan
        from .validation import (
            _validate_comparison_operator,
            _validate_inputs,
            _validate_metric_name,
            _validate_optimization_method,
        )

        # Handle utility/cost-based optimization first
        if utility is not None or minimize_cost:
            pred_prob = np.asarray(pred_prob, dtype=float)

            # Parse utility dict
            u = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
            if utility is not None:
                u.update({k: float(v) for k, v in utility.items()})
            if minimize_cost:
                u["fp"] = -abs(u["fp"])
                u["fn"] = -abs(u["fn"])

            # Handle multiclass case
            if pred_prob.ndim == 2:
                raise NotImplementedError(
                    "Utility/cost-based optimization not yet implemented for "
                    "multiclass. Binary classification only for now."
                )

            # Empirical optimum via sort-scan on linear counts objective
            if true_labs is None:
                raise ValueError(
                    "true_labs is required for empirical utility optimization"
                )

            true_labs = np.asarray(true_labs)
            sample_weight_array = (
                np.asarray(sample_weight) if sample_weight is not None else None
            )

            metric_fn = make_linear_counts_metric(
                u["tp"], u["tn"], u["fp"], u["fn"], name="user_utility"
            )

            thr, best, _ = optimal_threshold_sortscan(
                true_labs,
                pred_prob,
                metric_fn,
                inclusive=(comparison == ">="),
                sample_weight=sample_weight_array,
            )
            return float(thr)

        # Validate inputs for standard metric-based optimization
        if true_labs is None:
            raise ValueError("true_labs is required for empirical optimization")

        true_labs, pred_prob, sample_weight = _validate_inputs(
            true_labs, pred_prob, sample_weight=sample_weight
        )
        _validate_metric_name(metric)
        _validate_optimization_method(method)
        _validate_comparison_operator(comparison)

        # Check if this is multiclass
        if pred_prob.ndim == 2:
            return get_optimal_multiclass_thresholds(
                true_labs,
                pred_prob,
                metric,
                method,
                average="macro",
                sample_weight=sample_weight,
                comparison=comparison,
            )

        # Binary case - implement method routing with auto detection
        if method == "auto":
            if is_piecewise_metric(metric) and has_vectorized_implementation(metric):
                method = "sort_scan"
            else:
                method = "unique_scan"

        return self._optimize_binary_threshold(
            true_labs, pred_prob, metric, method, sample_weight, comparison
        )

    def _optimize_binary_threshold(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        method: OptimizationMethod,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize threshold for binary classification using strategy pattern."""
        from .strategies import get_strategy

        strategy = get_strategy(method)
        return strategy.optimize(
            true_labs, pred_prob, metric, sample_weight, comparison
        )


class ExpectedHandler(BaseHandler):
    """Handler for expected metric optimization under calibration assumption."""

    def optimize_threshold(
        self,
        true_labs: ArrayLike,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod = "auto",
        sample_weight: SampleWeightLike = None,
        comparison: ComparisonOperator = ">",
        *,
        beta: float = 1.0,
        average: str = "macro",
        class_weight: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[str, float | np.ndarray[Any, Any]] | tuple[float, float]:
        """Optimize threshold using expected metric optimization.

        Uses the generalized Dinkelbach framework for fractional-linear metrics.
        """
        from typing import cast

        from .expected import (
            dinkelbach_expected_fbeta_binary,
            dinkelbach_expected_fbeta_multilabel,
        )
        from .expected_fractional import (
            coeffs_for_metric,
            dinkelbach_expected_fractional_binary,
            dinkelbach_expected_fractional_ovr,
        )

        # Convert probabilities to numpy array
        pred_prob = np.asarray(pred_prob, dtype=float)
        is_binary = (pred_prob.ndim == 1) or (
            pred_prob.ndim == 2 and pred_prob.shape[1] == 1
        )

        if is_binary:
            # Binary case (1D array or 2D with 1 column)
            if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
                pred_prob = pred_prob[:, 0]  # Flatten single column

            # Convert sample_weight to array if needed
            sw = np.asarray(sample_weight) if sample_weight is not None else None

            # Get coefficients for the requested metric
            try:
                coeffs = coeffs_for_metric(
                    metric,
                    beta=beta,
                    tversky_alpha=0.5,  # Default values
                    tversky_beta=0.5,
                )

                # Use generalized Dinkelbach framework
                threshold, expected_score, direction = (
                    dinkelbach_expected_fractional_binary(
                        pred_prob,
                        coeffs,
                        sample_weight=sw,
                        comparison=comparison,
                    )
                )

                # Verify direction matches comparison (should be ">")
                if direction != ">":
                    # This is rare but possible for exotic metrics
                    pass  # Still return the result

                return (float(threshold), float(expected_score))

            except ValueError as e:
                # Only fallback for F-beta metrics, re-raise for unsupported metrics
                if metric.lower() in {"f1", "f2", "fbeta", "f_beta"}:
                    # Fallback to F-beta specific implementation
                    binary_result: tuple[float, float] = (
                        dinkelbach_expected_fbeta_binary(
                            pred_prob,
                            beta=beta,
                            sample_weight=sw,
                            comparison=comparison,
                        )
                    )
                    return binary_result
                else:
                    # Re-raise the error for truly unsupported metrics
                    raise e
        else:
            # Multiclass/multilabel case (including 2-class as multiclass)
            # Convert "none" to "macro" for expected mode
            avg = "macro" if average == "none" else average

            # Convert sample_weight and class_weight to arrays if needed
            sw = np.asarray(sample_weight) if sample_weight is not None else None
            cw = np.asarray(class_weight) if class_weight is not None else class_weight

            # Try generalized framework first
            try:
                result_dict = dinkelbach_expected_fractional_ovr(
                    pred_prob,
                    metric,
                    beta=beta,
                    tversky_alpha=0.5,
                    tversky_beta=0.5,
                    average=avg,
                    sample_weight=sw,
                    class_weight=cw,
                    comparison=comparison,
                )

                if avg == "micro":
                    # Return dictionary for micro averaging (backward compatibility)
                    return {
                        "threshold": float(result_dict["threshold"]),
                        "f_beta": float(result_dict["score"]),
                    }
                else:
                    # Return dict for macro/weighted averaging (backward compatibility)
                    return {
                        "thresholds": cast(
                            np.ndarray[Any, Any], result_dict["thresholds"]
                        ),
                        "f_beta_per_class": cast(
                            np.ndarray[Any, Any], result_dict["per_class"]
                        ),
                        "f_beta": float(result_dict["score"]),
                    }

            except ValueError as e:
                # Only fallback for F-beta metrics, re-raise for unsupported metrics
                if metric.lower() in {"f1", "f2", "fbeta", "f_beta"}:
                    # Fallback to F-beta specific implementation
                    return dinkelbach_expected_fbeta_multilabel(
                        pred_prob,
                        beta=beta,
                        average=avg,
                        sample_weight=sw,
                        class_weight=cw,
                        comparison=comparison,
                    )
                else:
                    # Re-raise the error for truly unsupported metrics
                    raise e


class BayesHandler(BaseHandler):
    """Handler for Bayes-optimal threshold decisions under known utilities."""

    def optimize_threshold(
        self,
        true_labs: ArrayLike,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod = "auto",
        sample_weight: SampleWeightLike = None,
        comparison: ComparisonOperator = ">",
        *,
        utility: UtilityDict | None = None,
        utility_matrix: UtilityMatrix | None = None,
        minimize_cost: bool | None = None,
        **kwargs: Any,
    ) -> float | np.ndarray[Any, Any]:
        """Optimize threshold using Bayes-optimal decision theory.

        Uses utility matrices or cost-benefit analysis for optimal decisions.
        """
        from .bayes import (
            bayes_decision_from_utility_matrix,
            bayes_threshold_from_costs_scalar,
            bayes_thresholds_from_costs_vector,
        )

        # Convert probabilities to numpy array
        pred_prob = np.asarray(pred_prob, dtype=float)
        is_binary = (pred_prob.ndim == 1) or (
            pred_prob.ndim == 2 and pred_prob.shape[1] == 1
        )

        # Handle utility matrix case (multiclass decisions)
        if utility_matrix is not None:
            result = bayes_decision_from_utility_matrix(pred_prob, utility_matrix)
            return result  # type: ignore[return-value]

        # Handle utility dict case
        if utility is None:
            raise ValueError(
                "mode='bayes' requires utility parameter or utility_matrix to be "
                "specified"
            )

        if is_binary:
            # Binary case - use scalar closed-form
            u = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
            u.update({k: float(v) for k, v in utility.items()})
            if minimize_cost:
                u["fp"] = -abs(u["fp"])
                u["fn"] = -abs(u["fn"])

            # Handle single-column probability case
            if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
                pred_prob = pred_prob[:, 0]  # Flatten single column

            return bayes_threshold_from_costs_scalar(
                u["fp"], u["fn"], u["tp"], u["tn"], comparison=comparison
            )
        else:
            # Multiclass OvR (including 2-class as multiclass) - expect per-class
            # vectors in utility
            fp_costs = utility.get("fp", None)
            fn_costs = utility.get("fn", None)
            tp_benefits = utility.get("tp", None)
            tn_benefits = utility.get("tn", None)

            if fp_costs is None or fn_costs is None:
                raise ValueError(
                    "Multiclass Bayes requires 'fp' and 'fn' as arrays in utility dict"
                )

            # Convert to arrays if needed
            fp_array = np.asarray(fp_costs) if fp_costs is not None else fp_costs
            fn_array = np.asarray(fn_costs) if fn_costs is not None else fn_costs
            tp_array = (
                np.asarray(tp_benefits) if tp_benefits is not None else tp_benefits
            )
            tn_array = (
                np.asarray(tn_benefits) if tn_benefits is not None else tn_benefits
            )

            return bayes_thresholds_from_costs_vector(
                fp_array, fn_array, tp_array, tn_array
            )


# Factory function for handler creation
def get_handler(mode: str) -> BaseHandler:
    """Get the appropriate handler for the specified mode.

    Parameters
    ----------
    mode : str
        Estimation mode: 'empirical', 'expected', or 'bayes'

    Returns
    -------
    BaseHandler
        Handler instance for the specified mode

    Raises
    ------
    ValueError
        If mode is not recognized
    """
    handlers = {
        "empirical": EmpiricalHandler,
        "expected": ExpectedHandler,
        "bayes": BayesHandler,
    }

    if mode not in handlers:
        raise ValueError(
            f"Unknown mode '{mode}'. Available modes: {list(handlers.keys())}"
        )

    return handlers[mode]()
