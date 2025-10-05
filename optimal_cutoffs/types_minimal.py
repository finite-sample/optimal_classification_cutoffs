"""Minimal type definitions for optimal_cutoffs package.

This module contains only the essential type aliases needed for the public API.
All complex validated classes have been removed in favor of simple, direct validation.
"""

# ============================================================================
# Core Type Aliases and Classes
# ============================================================================
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    """Result from threshold optimization with clear separation of concerns."""

    # ============== CORE RESULT (Finding thresholds) ==============
    thresholds: NDArray[np.float64]  # Shape (1,) for binary, (n_classes,) for multi
    scores: NDArray[np.float64]  # Metric values at these thresholds

    # ============== APPLICATION (Making predictions) ==============
    predict: Callable[[NDArray], NDArray] = field(repr=False)
    """Function to apply thresholds: predict(probabilities) -> predictions"""

    # ============== DIAGNOSTICS (Tracking computation) ==============
    diagnostics: dict[str, Any] | None = field(default=None, repr=False)
    """Optional computation details for debugging (iterations, convergence, etc.)"""

    # ============== METADATA (Always useful) ==============
    metric: str = "f1"
    n_classes: int = 2

    @property
    def threshold(self) -> float:
        """Convenience for binary case."""
        if len(self.thresholds) == 1:
            return float(self.thresholds[0])
        raise ValueError("Use .thresholds for multiclass")

    @property
    def score(self) -> float:
        """Convenience for overall score."""
        return float(np.mean(self.scores))

    def __repr__(self) -> str:
        """Clean representation showing only what matters."""
        if self.n_classes == 2:
            return f"OptimizationResult(threshold={self.threshold:.3f}, {self.metric}={self.score:.3f})"
        else:
            return f"OptimizationResult(n_classes={self.n_classes}, mean_{self.metric}={self.score:.3f})"

