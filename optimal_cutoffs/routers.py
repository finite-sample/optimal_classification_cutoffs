"""Mode-specific routers for threshold optimization.

This module provides clean, focused interfaces for each optimization mode,
reducing the complexity of the main router function and improving maintainability.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import (
    ArrayLike,
    AveragingMethod,
    ComparisonOperator,
    OptimizationMethod,
    SampleWeightLike,
    UtilityDict,
    UtilityMatrix,
)


class BaseRouter(ABC):
    """Base router for mode-specific optimization."""

    @abstractmethod
    def route(
        self,
        true_labs: ArrayLike | None,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod,
        sample_weight: SampleWeightLike,
        comparison: ComparisonOperator,
        **kwargs: Any,
    ) -> Any:
        """Route optimization request to appropriate handler."""
        pass


class EmpiricalRouter(BaseRouter):
    """Router for empirical threshold optimization."""

    def route(
        self,
        true_labs: ArrayLike | None,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod,
        sample_weight: SampleWeightLike,
        comparison: ComparisonOperator,
        *,
        utility: UtilityDict | None = None,
        minimize_cost: bool | None = None,
        **kwargs: Any,
    ) -> float | np.ndarray[Any, Any]:
        """Route empirical optimization requests."""
        from .handlers import EmpiricalHandler

        handler = EmpiricalHandler()
        return handler.optimize_threshold(
            true_labs=true_labs,
            pred_prob=pred_prob,
            metric=metric,
            method=method,
            sample_weight=sample_weight,
            comparison=comparison,
            utility=utility,
            minimize_cost=minimize_cost,
        )


class ExpectedRouter(BaseRouter):
    """Router for expected metric optimization under calibration."""

    def route(
        self,
        true_labs: ArrayLike | None,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod,
        sample_weight: SampleWeightLike,
        comparison: ComparisonOperator,
        *,
        beta: float = 1.0,
        average: AveragingMethod = "macro",
        class_weight: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[str, float | np.ndarray[Any, Any]] | tuple[float, float]:
        """Route expected optimization requests."""
        from .handlers import ExpectedHandler

        handler = ExpectedHandler()
        return handler.optimize_threshold(
            true_labs=true_labs,
            pred_prob=pred_prob,
            metric=metric,
            method=method,
            sample_weight=sample_weight,
            comparison=comparison,
            beta=beta,
            average=average,
            class_weight=class_weight,
        )


class BayesRouter(BaseRouter):
    """Router for Bayes-optimal decisions under known utilities."""

    def route(
        self,
        true_labs: ArrayLike | None,
        pred_prob: ArrayLike,
        metric: str,
        method: OptimizationMethod,
        sample_weight: SampleWeightLike,
        comparison: ComparisonOperator,
        *,
        utility: UtilityDict | None = None,
        utility_matrix: UtilityMatrix | None = None,
        minimize_cost: bool | None = None,
        **kwargs: Any,
    ) -> float | np.ndarray[Any, Any]:
        """Route Bayes optimization requests."""
        from .handlers import BayesHandler

        handler = BayesHandler()
        return handler.optimize_threshold(
            true_labs=true_labs,
            pred_prob=pred_prob,
            metric=metric,
            method=method,
            sample_weight=sample_weight,
            comparison=comparison,
            utility=utility,
            utility_matrix=utility_matrix,
            minimize_cost=minimize_cost,
        )


def get_router(mode: str) -> BaseRouter:
    """Get the appropriate router for the specified mode.

    Parameters
    ----------
    mode : str
        Optimization mode: 'empirical', 'expected', or 'bayes'

    Returns
    -------
    BaseRouter
        Router instance for the specified mode

    Raises
    ------
    ValueError
        If mode is not recognized
    """
    routers = {
        "empirical": EmpiricalRouter,
        "expected": ExpectedRouter,
        "bayes": BayesRouter,
    }

    if mode not in routers:
        raise ValueError(
            f"Unknown mode '{mode}'. Available modes: {list(routers.keys())}"
        )

    return routers[mode]()
