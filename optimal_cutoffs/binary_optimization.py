"""Modern binary threshold optimization with pluggable strategies."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, final

import numpy as np
from scipy import optimize

# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class BinaryData:
    """Validated binary classification data."""

    labels: np.ndarray  # int8, shape (n,)
    scores: np.ndarray  # float64, shape (n,)
    weights: np.ndarray | None = None  # float64, shape (n,) or None

    def __post_init__(self):
        """Validate and normalize data."""
        # Convert and validate
        labels = np.asarray(self.labels, dtype=np.int8)
        scores = np.asarray(self.scores, dtype=np.float64)

        if labels.ndim != 1 or scores.ndim != 1:
            raise ValueError("Labels and scores must be 1D")
        if len(labels) != len(scores):
            raise ValueError(f"Length mismatch: {len(labels)} vs {len(scores)}")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("Labels must be binary (0 or 1)")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Scores must be finite")

        # Handle weights
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.ndim != 1 or len(weights) != len(labels):
                raise ValueError("Invalid weights shape")
            if not np.all(weights >= 0):
                raise ValueError("Weights must be non-negative")
            if np.sum(weights) == 0:
                raise ValueError("Weights sum to zero")
            object.__setattr__(self, "weights", weights)

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "scores", scores)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.labels)

    @property
    def score_range(self) -> tuple[float, float]:
        """Min and max scores."""
        return float(np.min(self.scores)), float(np.max(self.scores))

    @property
    def unique_scores(self) -> np.ndarray:
        """Unique score values sorted."""
        return np.unique(self.scores)


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Result from threshold optimization."""

    threshold: float
    score: float
    metadata: dict = field(default_factory=dict)

    def apply(self, scores: np.ndarray, operator: str = ">=") -> np.ndarray:
        """Apply threshold to get predictions."""
        if operator == ">=":
            return scores >= self.threshold
        elif operator == ">":
            return scores > self.threshold
        else:
            raise ValueError(f"Invalid operator: {operator}")


# ============================================================================
# Metric Protocol
# ============================================================================


class BinaryMetric(Protocol):
    """Protocol for binary classification metrics."""

    def __call__(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Compute metric from confusion matrix."""
        ...

    @property
    def is_piecewise_constant(self) -> bool:
        """Whether metric is piecewise constant in threshold."""
        ...

    @property
    def requires_probability(self) -> bool:
        """Whether metric requires probabilities (vs arbitrary scores)."""
        ...


# ============================================================================
# Fast Numba Kernels
# ============================================================================

try:
    from numba import float64, int32, jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy decorators for when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

    float64 = float
    int32 = int


if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, cache=True)
    def compute_confusion_matrix_weighted(
        labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None
    ) -> tuple[float64, float64, float64, float64]:
        """Compute weighted confusion matrix elements."""
        tp = tn = fp = fn = 0.0

        if weights is None:
            for i in prange(len(labels)):
                if labels[i] == 1:
                    if predictions[i]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predictions[i]:
                        fp += 1
                    else:
                        tn += 1
        else:
            for i in prange(len(labels)):
                w = weights[i]
                if labels[i] == 1:
                    if predictions[i]:
                        tp += w
                    else:
                        fn += w
                else:
                    if predictions[i]:
                        fp += w
                    else:
                        tn += w

        return tp, tn, fp, fn

    @jit(nopython=True, fastmath=True, cache=True)
    def fast_f1_score(tp: float64, tn: float64, fp: float64, fn: float64) -> float64:
        """Compute F1 score from confusion matrix."""
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    @jit(nopython=True, parallel=True, cache=True)
    def sort_scan_kernel(
        labels: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray | None,
        inclusive: bool,
    ) -> tuple[float64, float64]:
        """Pure Numba sort-and-scan implementation."""
        n = len(labels)

        # Sort by scores (descending for easier logic)
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        sorted_weights = weights[order] if weights is not None else None

        # Initial state: all negative (threshold > max score)
        tp = 0.0
        fn = 0.0
        fp = 0.0
        tn = 0.0

        # Count initial state
        for i in range(n):
            if sorted_labels[i] == 1:
                fn += sorted_weights[i] if sorted_weights is not None else 1.0
            else:
                tn += sorted_weights[i] if sorted_weights is not None else 1.0

        best_threshold = sorted_scores[0] + 1e-10
        best_score = fast_f1_score(tp, tn, fp, fn)

        # Scan through thresholds (decreasing scores)
        for i in range(n):
            # Update confusion matrix by moving threshold to include this sample
            if sorted_labels[i] == 1:
                tp += sorted_weights[i] if sorted_weights is not None else 1.0
                fn -= sorted_weights[i] if sorted_weights is not None else 1.0
            else:
                fp += sorted_weights[i] if sorted_weights is not None else 1.0
                tn -= sorted_weights[i] if sorted_weights is not None else 1.0

            # Compute score
            score = fast_f1_score(tp, tn, fp, fn)

            # Update best
            if score > best_score:
                best_score = score
                if i < n - 1:
                    # Midpoint between current and next
                    best_threshold = 0.5 * (sorted_scores[i] + sorted_scores[i + 1])
                else:
                    # After last score
                    best_threshold = sorted_scores[i] - 1e-10

        return best_threshold, best_score

else:
    # Pure Python fallbacks when Numba is not available
    def compute_confusion_matrix_weighted(
        labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None
    ) -> tuple[float, float, float, float]:
        """Compute weighted confusion matrix elements (Python fallback)."""
        tp = tn = fp = fn = 0.0

        if weights is None:
            for i in range(len(labels)):
                if labels[i] == 1:
                    if predictions[i]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predictions[i]:
                        fp += 1
                    else:
                        tn += 1
        else:
            for i in range(len(labels)):
                w = weights[i]
                if labels[i] == 1:
                    if predictions[i]:
                        tp += w
                    else:
                        fn += w
                else:
                    if predictions[i]:
                        fp += w
                    else:
                        tn += w

        return tp, tn, fp, fn

    def fast_f1_score(tp: float, tn: float, fp: float, fn: float) -> float:
        """Compute F1 score from confusion matrix (Python fallback)."""
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    def sort_scan_kernel(
        labels: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray | None,
        inclusive: bool,
    ) -> tuple[float, float]:
        """Pure Python sort-and-scan implementation."""
        n = len(labels)

        # Sort by scores (descending for easier logic)
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        sorted_weights = weights[order] if weights is not None else None

        # Initial state: all negative (threshold > max score)
        tp = 0.0
        fn = float(
            np.sum(sorted_weights[sorted_labels == 1])
            if sorted_weights is not None
            else np.sum(sorted_labels)
        )
        fp = 0.0
        tn = float(
            np.sum(sorted_weights[sorted_labels == 0])
            if sorted_weights is not None
            else np.sum(1 - sorted_labels)
        )

        best_threshold = float(sorted_scores[0] + 1e-10)
        best_score = fast_f1_score(tp, tn, fp, fn)

        # Scan through thresholds (decreasing scores)
        for i in range(n):
            # Update confusion matrix by moving threshold to include this sample
            w = sorted_weights[i] if sorted_weights is not None else 1.0
            if sorted_labels[i] == 1:
                tp += w
                fn -= w
            else:
                fp += w
                tn -= w

            # Compute score
            score = fast_f1_score(tp, tn, fp, fn)

            # Update best
            if score > best_score:
                best_score = score
                if i < n - 1:
                    # Midpoint between current and next
                    best_threshold = 0.5 * (sorted_scores[i] + sorted_scores[i + 1])
                else:
                    # After last score
                    best_threshold = float(sorted_scores[i] - 1e-10)

        return best_threshold, best_score


# ============================================================================
# Optimization Strategies
# ============================================================================


class OptimizationStrategy(ABC):
    """Abstract base for optimization strategies."""

    @abstractmethod
    def optimize(
        self,
        data: BinaryData,
        metric: Callable[[int, int, int, int], float],
        operator: str = ">=",
    ) -> ThresholdResult:
        """Find optimal threshold."""
        ...


@final
class SortScanStrategy(OptimizationStrategy):
    """O(n log n) sort-and-scan for piecewise constant metrics."""

    def optimize(
        self,
        data: BinaryData,
        metric: Callable[[int, int, int, int], float],
        operator: str = ">=",
    ) -> ThresholdResult:
        """Optimize using sort-and-scan algorithm."""
        # Use fast Numba kernel for F1
        if hasattr(metric, "__name__") and metric.__name__ in ("f1_score", "f1"):
            threshold, score = sort_scan_kernel(
                data.labels, data.scores, data.weights, inclusive=(operator == ">=")
            )
            return ThresholdResult(
                threshold=float(threshold),
                score=float(score),
                metadata={
                    "method": "sort_scan_numba",
                    "numba_available": NUMBA_AVAILABLE,
                },
            )

        # Generic implementation for other metrics
        return self._generic_sort_scan(data, metric, operator)

    def _generic_sort_scan(
        self, data: BinaryData, metric: Callable, operator: str
    ) -> ThresholdResult:
        """Generic sort-and-scan implementation."""
        # Sort by scores
        order = np.argsort(data.scores)
        sorted_labels = data.labels[order]
        sorted_scores = data.scores[order]

        best_threshold = float(sorted_scores[0] - 1e-10)
        best_score = -np.inf

        # Scan through unique thresholds
        unique_scores = np.unique(sorted_scores)

        for i, score_val in enumerate(unique_scores):
            # Find threshold position
            if operator == ">=":
                predictions = sorted_scores >= score_val
            else:
                predictions = sorted_scores > score_val

            # Compute metric
            tp, tn, fp, fn = compute_confusion_matrix_weighted(
                sorted_labels, predictions, data.weights
            )
            score = metric(int(tp), int(tn), int(fp), int(fn))

            if score > best_score:
                best_score = score
                if i > 0:
                    best_threshold = 0.5 * (unique_scores[i - 1] + score_val)
                else:
                    best_threshold = float(score_val - 1e-10)

        return ThresholdResult(
            threshold=best_threshold,
            score=best_score,
            metadata={"method": "sort_scan_generic", "n_unique": len(unique_scores)},
        )


@final
class ScipyOptimizeStrategy(OptimizationStrategy):
    """Scipy-based optimization for smooth metrics."""

    def __init__(self, method: str = "bounded", tol: float = 1e-6):
        self.method = method
        self.tol = tol

    def optimize(
        self,
        data: BinaryData,
        metric: Callable[[int, int, int, int], float],
        operator: str = ">=",
    ) -> ThresholdResult:
        """Optimize using scipy.optimize."""

        def objective(threshold: float) -> float:
            """Objective to minimize (negative metric)."""
            predictions = (
                (data.scores >= threshold)
                if operator == ">="
                else (data.scores > threshold)
            )
            tp, tn, fp, fn = compute_confusion_matrix_weighted(
                data.labels, predictions, data.weights
            )
            try:
                return -metric(int(tp), int(tn), int(fp), int(fn))
            except (ValueError, ZeroDivisionError):
                return np.inf

        # Optimize within score range
        min_score, max_score = data.score_range

        result = optimize.minimize_scalar(
            objective,
            bounds=(min_score - 1e-10, max_score + 1e-10),
            method=self.method,
            options={"xatol": self.tol},
        )

        if result.success:
            return ThresholdResult(
                threshold=float(result.x),
                score=-result.fun,
                metadata={"method": "scipy", "iterations": result.nfev},
            )

        # Fallback to sort-scan
        return SortScanStrategy().optimize(data, metric, operator)


@final
class GradientStrategy(OptimizationStrategy):
    """Gradient-based optimization (warning: ineffective for piecewise metrics)."""

    def __init__(
        self, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-6
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        data: BinaryData,
        metric: Callable[[int, int, int, int], float],
        operator: str = ">=",
    ) -> ThresholdResult:
        """Optimize using gradient ascent."""
        warnings.warn(
            "Gradient optimization is ineffective for piecewise-constant metrics. "
            "Use SortScanStrategy instead.",
            UserWarning,
            stacklevel=2,
        )

        # Initialize at median score
        threshold = float(np.median(data.scores))
        min_score, max_score = data.score_range

        iteration = 0
        for iteration in range(self.max_iter):  # noqa: B007 (used in metadata)
            # Finite difference gradient
            eps = 1e-6
            t_plus = min(threshold + eps, max_score + 1e-10)
            t_minus = max(threshold - eps, min_score - 1e-10)

            # Compute scores
            pred_plus = (
                (data.scores >= t_plus) if operator == ">=" else (data.scores > t_plus)
            )
            pred_minus = (
                (data.scores >= t_minus)
                if operator == ">="
                else (data.scores > t_minus)
            )

            tp_p, tn_p, fp_p, fn_p = compute_confusion_matrix_weighted(
                data.labels, pred_plus, data.weights
            )
            tp_m, tn_m, fp_m, fn_m = compute_confusion_matrix_weighted(
                data.labels, pred_minus, data.weights
            )

            score_plus = metric(int(tp_p), int(tn_p), int(fp_p), int(fn_p))
            score_minus = metric(int(tp_m), int(tn_m), int(fp_m), int(fn_m))

            # Gradient and update
            gradient = (score_plus - score_minus) / (t_plus - t_minus)
            new_threshold = threshold + self.learning_rate * gradient
            new_threshold = np.clip(new_threshold, min_score - 1e-10, max_score + 1e-10)

            if abs(new_threshold - threshold) < self.tol:
                break

            threshold = new_threshold

        # Final evaluation
        predictions = (
            (data.scores >= threshold)
            if operator == ">="
            else (data.scores > threshold)
        )
        tp, tn, fp, fn = compute_confusion_matrix_weighted(
            data.labels, predictions, data.weights
        )
        score = metric(int(tp), int(tn), int(fp), int(fn))

        return ThresholdResult(
            threshold=float(threshold),
            score=score,
            metadata={"method": "gradient", "iterations": iteration + 1},
        )


# ============================================================================
# Strategy Selection
# ============================================================================


class StrategySelector:
    """Automatically select best strategy based on metric properties."""

    PIECEWISE_METRICS = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "specificity",
        "npv",
        "balanced_accuracy",
    }

    SMOOTH_METRICS = {"auc", "log_loss", "brier_score", "cross_entropy"}

    @classmethod
    def select(
        cls, metric_name: str | None = None, force_strategy: str | None = None
    ) -> OptimizationStrategy:
        """Select appropriate optimization strategy."""
        if force_strategy:
            strategies = {
                "sort_scan": SortScanStrategy(),
                "scipy": ScipyOptimizeStrategy(),
                "gradient": GradientStrategy(),
            }
            if force_strategy not in strategies:
                raise ValueError(f"Unknown strategy: {force_strategy}")
            return strategies[force_strategy]

        if metric_name:
            if metric_name.lower() in cls.PIECEWISE_METRICS:
                return SortScanStrategy()
            elif metric_name.lower() in cls.SMOOTH_METRICS:
                return ScipyOptimizeStrategy()

        # Default to sort-scan (works for everything, optimal for piecewise)
        return SortScanStrategy()


# ============================================================================
# Main API
# ============================================================================


class ThresholdOptimizer:
    """High-level threshold optimizer with automatic strategy selection."""

    def __init__(
        self,
        strategy: OptimizationStrategy | str | None = None,
        operator: str = ">=",
        require_probability: bool = True,
    ):
        """Initialize optimizer.

        Parameters
        ----------
        strategy : OptimizationStrategy, str, or None
            Optimization strategy or name ('sort_scan', 'scipy', 'gradient')
        operator : str, default=">="
            Comparison operator (">=" or ">")
        require_probability : bool, default=True
            Whether to require scores in [0, 1]
        """
        if isinstance(strategy, str):
            self.strategy = StrategySelector.select(force_strategy=strategy)
        elif strategy is None:
            self.strategy = None  # Will auto-select based on metric
        else:
            self.strategy = strategy

        self.operator = operator
        self.require_probability = require_probability

    def optimize(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        metric: str | Callable = "f1",
        weights: np.ndarray | None = None,
    ) -> ThresholdResult:
        """Find optimal threshold.

        Parameters
        ----------
        labels : array-like
            Binary labels (0 or 1)
        scores : array-like
            Prediction scores
        metric : str or callable
            Metric to optimize
        weights : array-like, optional
            Sample weights

        Returns
        -------
        ThresholdResult
            Optimal threshold and score
        """
        # Validate data
        data = BinaryData(labels, scores, weights)

        # Check probability requirement
        if self.require_probability:
            min_score, max_score = data.score_range
            if min_score < 0 or max_score > 1:
                raise ValueError(
                    f"Scores must be in [0, 1], got [{min_score}, {max_score}]"
                )

        # Get metric function
        if isinstance(metric, str):
            metric_fn = self._get_metric_function(metric)
            # Auto-select strategy based on metric if not specified
            if self.strategy is None:
                self.strategy = StrategySelector.select(metric_name=metric)
        else:
            metric_fn = metric
            # Auto-select default strategy if not specified
            if self.strategy is None:
                self.strategy = StrategySelector.select()

        # Optimize
        result = self.strategy.optimize(data, metric_fn, self.operator)

        # Clamp to [0, 1] if required
        if self.require_probability:
            result = ThresholdResult(
                threshold=np.clip(result.threshold, 0.0, 1.0),
                score=result.score,
                metadata=result.metadata,
            )

        return result

    @staticmethod
    def _get_metric_function(name: str) -> Callable:
        """Get metric function by name."""
        metrics = {
            "f1": lambda tp, tn, fp, fn: (
                2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            ),
            "precision": lambda tp, tn, fp, fn: tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": lambda tp, tn, fp, fn: tp / (tp + fn) if (tp + fn) > 0 else 0,
            "accuracy": lambda tp, tn, fp, fn: (tp + tn) / (tp + tn + fp + fn),
            "specificity": lambda tp, tn, fp, fn: (
                tn / (tn + fp) if (tn + fp) > 0 else 0
            ),
        }

        if name not in metrics:
            raise ValueError(f"Unknown metric: {name}")

        return metrics[name]


# ============================================================================
# Convenience Functions
# ============================================================================


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
    weights: np.ndarray | None = None,
    strategy: str = "auto",
    operator: str = ">=",
    require_probability: bool = True,
) -> tuple[float, float]:
    """Simple functional interface for threshold optimization.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    metric : str, default="f1"
        Metric to optimize
    weights : array-like, optional
        Sample weights
    strategy : str, default="auto"
        Optimization strategy ("auto", "sort_scan", "scipy", "gradient")
    operator : str, default=">="
        Comparison operator (">=" or ">")
    require_probability : bool, default=True
        Whether to require scores in [0, 1]

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_score)
    """
    if strategy == "auto":
        strategy_obj = StrategySelector.select(metric_name=metric)
    else:
        strategy_obj = StrategySelector.select(force_strategy=strategy)

    optimizer = ThresholdOptimizer(
        strategy=strategy_obj,
        operator=operator,
        require_probability=require_probability,
    )
    result = optimizer.optimize(labels, scores, metric, weights)

    return result.threshold, result.score


# ============================================================================
# Performance Information
# ============================================================================


def get_performance_info() -> dict:
    """Get information about performance optimizations available."""
    return {
        "numba_available": NUMBA_AVAILABLE,
        "numba_version": (
            None
            if not NUMBA_AVAILABLE
            else getattr(__import__("numba"), "__version__", "unknown")
        ),
        "expected_speedup": "10-100x" if NUMBA_AVAILABLE else "1x (Python fallback)",
        "parallel_processing": NUMBA_AVAILABLE,
        "fastmath_enabled": NUMBA_AVAILABLE,
    }
