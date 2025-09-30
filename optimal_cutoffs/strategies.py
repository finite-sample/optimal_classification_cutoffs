"""Strategy pattern for optimization methods.

This module implements the Strategy pattern to encapsulate different
optimization algorithms for threshold selection. Each strategy handles
a specific optimization method (sort_scan, unique_scan, minimize, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import optimize

from .types import (
    ComparisonOperator,
    OptimizationMethod,
)


class OptimizationStrategy(ABC):
    """Base strategy for threshold optimization methods."""

    @abstractmethod
    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize threshold using this strategy.

        Parameters
        ----------
        true_labs : np.ndarray
            True binary labels
        pred_prob : np.ndarray
            Predicted probabilities
        metric : str
            Metric to optimize
        sample_weight : np.ndarray | None
            Sample weights
        comparison : ComparisonOperator
            Comparison operator

        Returns
        -------
        float
            Optimal threshold
        """
        pass

    def _metric_score(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        threshold: float,
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None = None,
        comparison: ComparisonOperator = ">",
    ) -> float:
        """Compute a metric score for a given threshold."""
        from .metrics import METRIC_REGISTRY, get_confusion_matrix

        tp, tn, fp, fn = get_confusion_matrix(
            true_labs, pred_prob, threshold, sample_weight, comparison
        )
        try:
            metric_func = METRIC_REGISTRY[metric]
        except KeyError as exc:
            raise ValueError(f"Unknown metric: {metric}") from exc
        return float(metric_func(tp, tn, fp, fn))


class SortScanStrategy(OptimizationStrategy):
    """O(n log n) sort-and-scan optimization for vectorized piecewise metrics."""

    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize using sort-and-scan algorithm."""
        from .metrics import get_vectorized_metric, has_vectorized_implementation
        from .piecewise import optimal_threshold_sortscan

        if not has_vectorized_implementation(metric):
            raise ValueError(
                f"sort_scan method requires vectorized implementation for "
                f"metric '{metric}'"
            )

        vectorized_metric = get_vectorized_metric(metric)
        threshold, _, _ = optimal_threshold_sortscan(
            true_labs,
            pred_prob,
            vectorized_metric,
            sample_weight=sample_weight,
            inclusive=(comparison == ">="),
        )
        return threshold


class UniqueScanStrategy(OptimizationStrategy):
    """Optimization by evaluating all unique probability values."""

    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize by scanning unique probabilities."""
        from .metrics import is_piecewise_metric
        from .optimizers import _optimal_threshold_piecewise

        if is_piecewise_metric(metric):
            return _optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        else:
            # Fall back to original brute force for non-piecewise metrics
            thresholds = np.unique(pred_prob)
            scores = [
                self._metric_score(
                    true_labs, pred_prob, t, metric, sample_weight, comparison
                )
                for t in thresholds
            ]
            return float(thresholds[int(np.argmax(scores))])


class MinimizeStrategy(OptimizationStrategy):
    """Scipy-based optimization with robust fallback for piecewise metrics."""

    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize using scipy.optimize.minimize_scalar."""
        from .metrics import is_piecewise_metric
        from .optimizers import _optimal_threshold_piecewise

        res = optimize.minimize_scalar(
            lambda t: -self._metric_score(
                true_labs, pred_prob, t, metric, sample_weight, comparison
            ),
            bounds=(0, 1),
            method="bounded",
        )

        # Robust fallback for piecewise metrics
        if is_piecewise_metric(metric):
            scipy_threshold = float(res.x)
            piecewise_threshold = _optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )

            scipy_score = self._metric_score(
                true_labs, pred_prob, scipy_threshold, metric, sample_weight, comparison
            )
            piecewise_score = self._metric_score(
                true_labs,
                pred_prob,
                piecewise_threshold,
                metric,
                sample_weight,
                comparison,
            )

            return (
                piecewise_threshold
                if piecewise_score >= scipy_score
                else scipy_threshold
            )

        return float(res.x)


class GradientStrategy(OptimizationStrategy):
    """Simple gradient ascent optimization."""

    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize using gradient ascent."""
        threshold = 0.5
        lr = 0.1
        eps = 1e-5
        for _ in range(100):
            # Ensure evaluation points are within bounds
            thresh_plus = np.clip(threshold + eps, 0.0, 1.0)
            thresh_minus = np.clip(threshold - eps, 0.0, 1.0)

            grad = (
                self._metric_score(
                    true_labs, pred_prob, thresh_plus, metric, sample_weight, comparison
                )
                - self._metric_score(
                    true_labs,
                    pred_prob,
                    thresh_minus,
                    metric,
                    sample_weight,
                    comparison,
                )
            ) / (2 * eps)
            threshold = np.clip(threshold + lr * grad, 0.0, 1.0)
        # Final safety clip to ensure numerical precision doesn't cause issues
        return float(np.clip(threshold, 0.0, 1.0))


class CoordAscentStrategy(OptimizationStrategy):
    """Coordinate ascent for coupled multiclass optimization."""

    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Optimize using coordinate ascent (only supports multiclass)."""
        raise NotImplementedError(
            "CoordAscentStrategy should only be used through multiclass handlers"
        )


class AutoStrategy(OptimizationStrategy):
    """Automatic strategy selection based on metric properties."""

    def optimize(
        self,
        true_labs: np.ndarray[Any, Any],
        pred_prob: np.ndarray[Any, Any],
        metric: str,
        sample_weight: np.ndarray[Any, Any] | None,
        comparison: ComparisonOperator,
    ) -> float:
        """Automatically select and apply the best optimization strategy."""
        from .metrics import has_vectorized_implementation, is_piecewise_metric

        # Auto routing: prefer sort_scan for piecewise metrics with vectorized
        # implementation
        if is_piecewise_metric(metric) and has_vectorized_implementation(metric):
            strategy = SortScanStrategy()
        else:
            strategy = UniqueScanStrategy()

        return strategy.optimize(
            true_labs, pred_prob, metric, sample_weight, comparison
        )


# Strategy factory
def get_strategy(method: OptimizationMethod) -> OptimizationStrategy:
    """Get optimization strategy for the specified method.

    Parameters
    ----------
    method : OptimizationMethod
        Optimization method name

    Returns
    -------
    OptimizationStrategy
        Strategy instance for the specified method

    Raises
    ------
    ValueError
        If method is not recognized
    """
    strategies = {
        "auto": AutoStrategy,
        "sort_scan": SortScanStrategy,
        "unique_scan": UniqueScanStrategy,
        "minimize": MinimizeStrategy,
        "gradient": GradientStrategy,
        "coord_ascent": CoordAscentStrategy,
    }

    if method not in strategies:
        raise ValueError(
            f"Unknown optimization method '{method}'. Available methods: "
            f"{list(strategies.keys())}"
        )

    return strategies[method]()
