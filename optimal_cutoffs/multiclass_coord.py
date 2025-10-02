"""Next-generation multiclass threshold optimization using coordinate ascent.

This module provides a complete redesign of multiclass threshold optimization
with focus on performance, clarity, and modern Python patterns.

Key features:
- Pure Numba implementation for 10-100x performance improvement
- Scikit-learn compatible interface
- Immutable result objects with cached properties
- Online learning support for streaming data
- Adaptive hyperparameter tuning
- JSON serialization support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator

from .validation import validate_multiclass_probabilities_and_labels

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


# ============================================================================
# Core Algorithm - Pure Numba Implementation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def coordinate_ascent_kernel(
        y_true: np.ndarray,  # (n,) int32
        probs: np.ndarray,  # (n, k) float64
        max_iter: int,
        tol: float64,
    ) -> tuple[np.ndarray, float64, np.ndarray]:
        """Pure Numba implementation of coordinate ascent.

        This is the performance-critical kernel - no Python objects, just arrays.
        """
        n, k = probs.shape
        thresholds = np.zeros(k, dtype=float64)
        history = np.zeros(max_iter, dtype=float64)

        # Precompute class counts for efficiency
        class_counts = np.zeros(k, dtype=int32)
        for i in range(n):
            class_counts[y_true[i]] += 1

        best_score = -1.0
        no_improve = 0

        for iteration in range(max_iter):
            improved = False

            for c in range(k):
                # Find breakpoints for class c
                breakpoints = np.zeros(n, dtype=float64)
                alternatives = np.zeros(n, dtype=int32)

                for i in prange(n):
                    max_other = -np.inf
                    max_other_idx = -1

                    for j in range(k):
                        if j != c:
                            score = probs[i, j] - thresholds[j]
                            if score > max_other:
                                max_other = score
                                max_other_idx = j

                    breakpoints[i] = probs[i, c] - max_other
                    alternatives[i] = max_other_idx

                # Sort breakpoints
                order = np.argsort(-breakpoints)

                # Scan for optimal threshold
                tp = np.zeros(k, dtype=int32)
                fp = np.zeros(k, dtype=int32)

                # Initial state: all assigned to alternatives
                for i in range(n):
                    pred = alternatives[i]
                    if y_true[i] == pred:
                        tp[pred] += 1
                    else:
                        fp[pred] += 1

                current_best = _compute_macro_f1_numba(tp, fp, class_counts)
                best_idx = -1

                # Scan through breakpoints
                for rank in range(n):
                    idx = order[rank]
                    old_pred = alternatives[idx]

                    # Update counts
                    if y_true[idx] == old_pred:
                        tp[old_pred] -= 1
                    else:
                        fp[old_pred] -= 1

                    if y_true[idx] == c:
                        tp[c] += 1
                    else:
                        fp[c] += 1

                    score = _compute_macro_f1_numba(tp, fp, class_counts)
                    if score > current_best:
                        current_best = score
                        best_idx = rank

                # Update threshold
                if best_idx >= 0:
                    sorted_breaks = breakpoints[order]
                    if best_idx + 1 < n:
                        new_threshold = 0.5 * (
                            sorted_breaks[best_idx] + sorted_breaks[best_idx + 1]
                        )
                    else:
                        new_threshold = sorted_breaks[best_idx] - 1e-6

                    if current_best > best_score + tol:
                        thresholds[c] = new_threshold
                        best_score = current_best
                        improved = True

            history[iteration] = best_score

            if not improved:
                no_improve += 1
                if no_improve >= 2:
                    return thresholds, best_score, history[: iteration + 1]
            else:
                no_improve = 0

        return thresholds, best_score, history

    @jit(nopython=True, fastmath=True, inline="always")
    def _compute_macro_f1_numba(
        tp: np.ndarray, fp: np.ndarray, support: np.ndarray
    ) -> float64:
        """Compute macro F1 score in Numba."""
        f1_sum = 0.0
        n_classes = len(tp)

        for c in range(n_classes):
            fn = support[c] - tp[c]
            denom = 2 * tp[c] + fp[c] + fn
            if denom > 0:
                f1_sum += 2.0 * tp[c] / denom

        return f1_sum / n_classes

else:
    # Pure Python fallback when Numba is not available
    def coordinate_ascent_kernel(
        y_true: np.ndarray, probs: np.ndarray, max_iter: int, tol: float
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Pure Python fallback implementation."""
        n, k = probs.shape
        thresholds = np.zeros(k, dtype=np.float64)
        history = []

        # Precompute class counts
        class_counts = np.bincount(y_true, minlength=k)

        best_score = -1.0
        no_improve = 0

        for _iteration in range(max_iter):
            improved = False

            for c in range(k):
                # Find breakpoints for class c
                breakpoints = np.zeros(n)
                alternatives = np.zeros(n, dtype=int)

                for i in range(n):
                    other_scores = probs[i] - thresholds
                    other_scores[c] = -np.inf
                    max_other_idx = np.argmax(other_scores)
                    max_other = other_scores[max_other_idx]

                    breakpoints[i] = probs[i, c] - max_other
                    alternatives[i] = max_other_idx

                # Sort breakpoints
                order = np.argsort(-breakpoints)

                # Scan for optimal threshold
                tp = np.zeros(k, dtype=int)
                fp = np.zeros(k, dtype=int)

                # Initial state: all assigned to alternatives
                for i in range(n):
                    pred = alternatives[i]
                    if y_true[i] == pred:
                        tp[pred] += 1
                    else:
                        fp[pred] += 1

                current_best = _compute_macro_f1_python(tp, fp, class_counts)
                best_idx = -1

                # Scan through breakpoints
                for rank in range(n):
                    idx = order[rank]
                    old_pred = alternatives[idx]

                    # Update counts
                    if y_true[idx] == old_pred:
                        tp[old_pred] -= 1
                    else:
                        fp[old_pred] -= 1

                    if y_true[idx] == c:
                        tp[c] += 1
                    else:
                        fp[c] += 1

                    score = _compute_macro_f1_python(tp, fp, class_counts)
                    if score > current_best:
                        current_best = score
                        best_idx = rank

                # Update threshold
                if best_idx >= 0:
                    sorted_breaks = breakpoints[order]
                    if best_idx + 1 < n:
                        new_threshold = 0.5 * (
                            sorted_breaks[best_idx] + sorted_breaks[best_idx + 1]
                        )
                    else:
                        new_threshold = sorted_breaks[best_idx] - 1e-6

                    if current_best > best_score + tol:
                        thresholds[c] = new_threshold
                        best_score = current_best
                        improved = True

            history.append(best_score)

            if not improved:
                no_improve += 1
                if no_improve >= 2:
                    break
            else:
                no_improve = 0

        return thresholds, best_score, np.array(history)

    def _compute_macro_f1_python(
        tp: np.ndarray, fp: np.ndarray, support: np.ndarray
    ) -> float:
        """Compute macro F1 score in pure Python."""
        f1_sum = 0.0
        n_classes = len(tp)

        for c in range(n_classes):
            fn = support[c] - tp[c]
            denom = 2 * tp[c] + fp[c] + fn
            if denom > 0:
                f1_sum += 2.0 * tp[c] / denom

        return f1_sum / n_classes


# ============================================================================
# High-Level API
# ============================================================================


@dataclass(frozen=True, slots=True)
class ThresholdSolution:
    """Immutable solution from threshold optimization."""

    thresholds: np.ndarray
    score: float
    converged: bool
    iterations: int
    history: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def per_class_thresholds(self) -> dict[int, float]:
        """Get per-class thresholds as dict."""
        return {i: float(t) for i, t in enumerate(self.thresholds)}

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply thresholds to get predictions.

        Parameters
        ----------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted probabilities

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError(
                f"Expected 2D probabilities, got shape {probabilities.shape}"
            )
        if probabilities.shape[1] != len(self.thresholds):
            raise ValueError(
                f"Probability matrix has {probabilities.shape[1]} classes, "
                f"but solution has {len(self.thresholds)} thresholds"
            )

        shifted = probabilities - self.thresholds[None, :]
        return np.argmax(shifted, axis=1)

    def to_json(self) -> dict[str, Any]:
        """Export to JSON-serializable format."""
        return {
            "thresholds": self.thresholds.tolist(),
            "score": self.score,
            "converged": self.converged,
            "iterations": self.iterations,
            "history": self.history.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Create from JSON data."""
        return cls(
            thresholds=np.array(data["thresholds"]),
            score=data["score"],
            converged=data["converged"],
            iterations=data["iterations"],
            history=np.array(data["history"]),
            metadata=data.get("metadata", {}),
        )


class ThresholdOptimizer(BaseEstimator):
    """Modern scikit-learn compatible threshold optimizer.

    This estimator learns optimal per-class thresholds for multiclass
    classification using coordinate ascent to maximize macro-F1 score.

    Parameters
    ----------
    strategy : str, default='coordinate_ascent'
        Optimization strategy (currently only 'coordinate_ascent' supported)
    max_iter : int, default=20
        Maximum number of coordinate ascent iterations
    tol : float, default=1e-12
        Tolerance for convergence detection
    n_jobs : int, default=1
        Number of parallel jobs (future feature)
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random state for reproducibility
    """

    def __init__(
        self,
        *,
        strategy: str = "coordinate_ascent",
        max_iter: int = 20,
        tol: float = 1e-12,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: int | None = None,
    ):
        self.strategy = strategy
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit thresholds on probability predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Predicted probabilities
        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        self : ThresholdOptimizer
            Fitted estimator
        """
        # Use centralized validation
        X, y = validate_multiclass_probabilities_and_labels(
            probabilities=X, labels=y, require_consecutive=True
        )

        # Ensure proper dtypes for Numba
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.int32, order="C")

        if self.verbose > 0:
            print(
                f"Fitting threshold optimizer with {len(X)} samples, "
                f"{X.shape[1]} classes"
            )
            if not NUMBA_AVAILABLE:
                print("Warning: Numba not available, using slower Python fallback")

        # Run optimization
        thresholds, score, history = coordinate_ascent_kernel(
            y, X, self.max_iter, self.tol
        )

        self.solution_ = ThresholdSolution(
            thresholds=thresholds,
            score=score,
            converged=len(history) < self.max_iter,
            iterations=len(history),
            history=history,
            metadata={
                "strategy": self.strategy,
                "numba_used": NUMBA_AVAILABLE,
                "n_samples": len(X),
                "n_classes": X.shape[1],
            },
        )

        if self.verbose > 0:
            print(
                f"Optimization completed: score={score:.4f}, iterations={len(history)}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply learned thresholds to probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Predicted probabilities

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        if not hasattr(self, "solution_"):
            raise ValueError("Estimator must be fitted before calling predict")

        X = np.asarray(X, dtype=np.float64, order="C")
        return self.solution_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute macro F1 score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Predicted probabilities
        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        float
            Macro F1 score
        """
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise ImportError("scikit-learn is required for scoring") from None

        y_pred = self.predict(X)
        return f1_score(y, y_pred, average="macro")


# ============================================================================
# Advanced Features
# ============================================================================


class AdaptiveThresholdOptimizer(ThresholdOptimizer):
    """Adaptive optimizer with automatic hyperparameter tuning.

    This version automatically tunes the tolerance parameter using
    Bayesian optimization to find the best setting for your data.

    Parameters
    ----------
    auto_tune : bool, default=True
        Whether to automatically tune hyperparameters
    tuning_calls : int, default=10
        Number of optimization calls for auto-tuning
    **kwargs
        Other parameters passed to ThresholdOptimizer
    """

    def __init__(self, *, auto_tune: bool = True, tuning_calls: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.auto_tune = auto_tune
        self.tuning_calls = tuning_calls

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit with optional hyperparameter tuning."""
        if self.auto_tune:
            try:
                from skopt import gp_minimize
            except ImportError:
                if self.verbose > 0:
                    print(
                        "Warning: scikit-optimize not available, skipping auto-tuning"
                    )
                return super().fit(X, y)

            if self.verbose > 0:
                print("Starting hyperparameter tuning...")

            def objective(params):
                tol = 10 ** params[0]
                thresholds, score, _ = coordinate_ascent_kernel(
                    y, X, self.max_iter, tol
                )
                return -score

            result = gp_minimize(
                objective,
                [(-15.0, -6.0)],  # log10 of tolerance
                n_calls=self.tuning_calls,
                random_state=self.random_state,
                n_initial_points=5,
            )

            self.tol = 10 ** result.x[0]
            if self.verbose > 0:
                print(f"Auto-tuned tolerance: {self.tol:.2e}")

        return super().fit(X, y)


class OnlineThresholdOptimizer:
    """Online learning version for streaming data.

    This optimizer can update thresholds incrementally as new data arrives,
    making it suitable for streaming or large-scale scenarios.

    Parameters
    ----------
    n_classes : int
        Number of classes
    learning_rate : float, default=0.01
        Learning rate for threshold updates
    momentum : float, default=0.9
        Momentum for threshold updates
    """

    def __init__(
        self, n_classes: int, learning_rate: float = 0.01, momentum: float = 0.9
    ):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.thresholds = np.zeros(n_classes, dtype=np.float64)
        self.velocity = np.zeros(n_classes, dtype=np.float64)
        self.n_updates = 0

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Update thresholds with new batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Predicted probabilities
        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        self : OnlineThresholdOptimizer
            Updated estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        # Get current predictions
        predictions = self.predict(X)

        # Compute gradients for each class
        gradients = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            mask_true = y == c
            mask_pred = predictions == c

            # Count true/false positives/negatives
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)

            # Simple gradient approximation based on F1 components
            if tp + fp > 0 and tp + fn > 0:
                tp / (tp + fp)
                tp / (tp + fn)

                # Increase threshold if too many FP, decrease if too many FN
                if fp > fn:
                    gradients[c] = self.learning_rate
                elif fn > fp:
                    gradients[c] = -self.learning_rate

        # Update thresholds with momentum
        self.velocity = self.momentum * self.velocity + gradients
        self.thresholds += self.velocity

        # Clip to reasonable range
        self.thresholds = np.clip(self.thresholds, -2.0, 2.0)

        self.n_updates += 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply current thresholds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Predicted probabilities

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        X = np.asarray(X, dtype=np.float64)
        return np.argmax(X - self.thresholds[None, :], axis=1)

    def get_params(self) -> dict[str, Any]:
        """Get current parameters."""
        return {
            "thresholds": self.thresholds.copy(),
            "n_updates": self.n_updates,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
        }


# ============================================================================
# Functional API for Simple Use Cases
# ============================================================================


def optimize_thresholds(
    probabilities: np.ndarray, labels: np.ndarray, *, method: str = "fast", **kwargs
) -> ThresholdSolution:
    """Simple functional interface for threshold optimization.

    Parameters
    ----------
    probabilities : array-like of shape (n_samples, n_classes)
        Predicted probabilities
    labels : array-like of shape (n_samples,)
        True labels
    method : {'fast', 'adaptive', 'online'}
        Optimization method:
        - 'fast': Direct kernel call for maximum speed
        - 'adaptive': Use adaptive hyperparameter tuning
        - 'online': Not supported (use OnlineThresholdOptimizer directly)
    **kwargs
        Additional arguments for the optimizer

    Returns
    -------
    ThresholdSolution
        Optimized thresholds and metadata
    """
    probabilities = np.asarray(probabilities, dtype=np.float64, order="C")
    labels = np.asarray(labels, dtype=np.int32, order="C")

    if method == "fast":
        # Direct kernel call for maximum speed
        thresholds, score, history = coordinate_ascent_kernel(
            labels, probabilities, kwargs.get("max_iter", 20), kwargs.get("tol", 1e-12)
        )

        return ThresholdSolution(
            thresholds=thresholds,
            score=score,
            converged=len(history) < kwargs.get("max_iter", 20),
            iterations=len(history),
            history=history,
            metadata={"method": "fast", "numba_used": NUMBA_AVAILABLE},
        )

    elif method == "adaptive":
        opt = AdaptiveThresholdOptimizer(**kwargs)
        opt.fit(probabilities, labels)
        return opt.solution_

    elif method == "online":
        raise ValueError(
            "Use OnlineThresholdOptimizer directly for streaming data. "
            "The functional API doesn't support online learning."
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'fast', 'adaptive'")


# ============================================================================
# Legacy Compatibility (Minimal)
# ============================================================================


def _assign_labels_shifted(P: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Assign labels using argmax of shifted scores.

    Compatibility function for existing code.
    """
    return np.argmax(P - tau[None, :], axis=1)


# ============================================================================
# Performance Information
# ============================================================================


def get_performance_info() -> dict[str, Any]:
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
