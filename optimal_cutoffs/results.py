"""Unified result types for threshold optimization.

This module provides a single unified result class for consistent handling
across all optimization modes, improving type safety and API clarity.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ThresholdResult:
    """Unified result for all threshold optimization modes.
    
    This class provides a consistent interface for all threshold optimization
    results, eliminating the confusion of different return types based on
    parameters.
    
    Attributes
    ----------
    threshold : float | NDArray[np.float64]
        Single threshold (binary) or per-class thresholds (multiclass)
    score : float | None
        Overall metric score achieved (if available)
    per_class_scores : NDArray[np.float64] | None
        Per-class scores for multiclass problems (if available)
    confidence_interval : tuple[float, float] | None
        Confidence interval for the score (if available)
    metadata : dict[str, Any]
        Additional optimization metadata (method, mode, averaging, etc.)
    """
    
    threshold: float | NDArray[np.float64]
    score: float | None = None
    per_class_scores: NDArray[np.float64] | None = None
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_multiclass(self) -> bool:
        """Check if this is a multiclass result."""
        return isinstance(self.threshold, np.ndarray)
    
    @property
    def method(self) -> str:
        """Get the optimization method used."""
        return self.metadata.get("method", "unknown")
    
    @property
    def mode(self) -> str:
        """Get the optimization mode used."""
        return self.metadata.get("mode", "empirical")
    
    def to_legacy_format(self) -> float | NDArray[np.float64] | tuple[float, float] | dict[str, Any]:
        """Convert to legacy format for backward compatibility.
        
        Returns the appropriate legacy format based on mode and type:
        - Binary empirical: float
        - Multiclass empirical: NDArray
        - Expected binary: tuple[float, float]
        - Expected multiclass: dict with keys based on averaging
        - Bayes: depends on utility specification
        """
        mode = self.mode
        
        if mode == "expected":
            if self.is_multiclass:
                # Expected multiclass returns dict
                averaging = self.metadata.get("averaging", "macro")
                if averaging == "micro":
                    return {
                        "threshold": float(self.threshold[0]) if self.threshold.size == 1 else self.threshold,
                        "f_beta": self.score or 0.0,
                    }
                else:
                    return {
                        "thresholds": self.threshold,
                        "f_beta_per_class": self.per_class_scores,
                        "f_beta": self.score or 0.0,
                    }
            else:
                # Expected binary returns tuple
                return (float(self.threshold), self.score or 0.0)
        
        elif mode == "bayes":
            # Bayes mode can return various types based on utility specification
            if "decisions" in self.metadata:
                return self.metadata["decisions"]
            else:
                return self.threshold
        
        else:
            # Empirical mode returns threshold(s) directly
            return self.threshold


# Backward compatibility aliases
BinaryResult = ThresholdResult
MulticlassResult = ThresholdResult
ExpectedBinaryResult = ThresholdResult
ExpectedMulticlassResult = ThresholdResult
BayesDecisionResult = ThresholdResult

# Union type for backward compatibility
OptimizationResult = ThresholdResult


def create_result(
    *,
    threshold: float | NDArray[np.float64] | None = None,
    score: float | NDArray[np.float64] | None = None,
    method: str = "unknown",
    mode: str = "empirical",
    averaging_method: str | None = None,
    per_class_scores: NDArray[np.float64] | None = None,
    confidence_interval: tuple[float, float] | None = None,
    **kwargs: Any,
) -> ThresholdResult:
    """Factory function to create ThresholdResult.

    Parameters
    ----------
    threshold : float | NDArray[np.float64] | None
        Threshold value(s)
    score : float | NDArray[np.float64] | None
        Overall score value
    method : str
        Optimization method used
    mode : str
        Optimization mode ("empirical", "expected", "bayes")
    averaging_method : str | None
        Averaging method for multiclass
    per_class_scores : NDArray[np.float64] | None
        Per-class scores for multiclass
    confidence_interval : tuple[float, float] | None
        Confidence interval for the score
    **kwargs : Any
        Additional metadata

    Returns
    -------
    ThresholdResult
        Unified result object

    Raises
    ------
    ValueError
        If inputs are inconsistent or invalid
    """
    if threshold is None:
        raise ValueError("threshold is required")

    # Prepare metadata
    metadata = {
        "method": method,
        "mode": mode,
        **kwargs,
    }
    
    if averaging_method is not None:
        metadata["averaging"] = averaging_method

    # Handle per-class scores
    if per_class_scores is None and isinstance(score, np.ndarray):
        per_class_scores = score
        # Use overall score if provided, otherwise compute mean
        if not isinstance(score, np.ndarray) or score.ndim == 0:
            pass  # score is already overall
        else:
            score = float(np.mean(score))

    # Convert score to float if it's a scalar array
    if isinstance(score, np.ndarray) and score.ndim == 0:
        score = float(score)

    return ThresholdResult(
        threshold=threshold,
        score=score,
        per_class_scores=per_class_scores,
        confidence_interval=confidence_interval,
        metadata=metadata,
    )
