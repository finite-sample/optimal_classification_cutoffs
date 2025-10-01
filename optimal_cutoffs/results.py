"""Unified result types for threshold optimization.

This module provides a single unified result class for consistent handling
across all optimization modes, improving type safety and API clarity.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ThresholdResult:
    """Unified result for all threshold optimization modes.

    This class provides a consistent interface for all threshold optimization
    results, eliminating the confusion of different return types based on
    parameters.

    Attributes
    ----------
    threshold : float | NDArray[np.float64] | None
        Single threshold (binary) or per-class thresholds (multiclass)
    score : float | None
        Overall metric score achieved (if available)
    per_class_scores : NDArray[np.float64] | None
        Per-class scores for multiclass problems (if available)
    confidence_interval : tuple[float, float] | None
        Confidence interval for the score (if available)
    decisions : NDArray[Any] | None
        Bayes-optimal decisions from utility matrix (if available)
    directions : str | NDArray[np.str_] | None
        Decision directions per class (e.g., '>' or '<') for exotic cases
    metadata : dict[str, Any]
        Additional optimization metadata (method, mode, averaging, etc.)
    """

    threshold: float | NDArray[np.float64] | None
    score: float | None = None
    per_class_scores: NDArray[np.float64] | None = None
    confidence_interval: tuple[float, float] | None = None
    # Optional fields for modes that don't produce thresholds (e.g., Bayes decisions)
    decisions: NDArray[Any] | None = None
    # Optional decision directions (e.g., per-class '>' / '<' in exotic Bayes/expected)
    directions: str | NDArray[np.str_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize threshold: allow float or 1D float64 array;
        # convert 0-D arrays to float
        if isinstance(self.threshold, np.ndarray):
            if self.threshold.ndim == 0:
                self.threshold = float(self.threshold)
            elif self.threshold.ndim != 1:
                raise ValueError("threshold array must be 1D if provided.")
            else:
                self.threshold = self.threshold.astype(np.float64, copy=False)
        # Normalize per-class scores to 1D float64 if provided
        if isinstance(self.per_class_scores, np.ndarray):
            if self.per_class_scores.ndim != 1:
                raise ValueError("per_class_scores must be 1D if provided.")
            self.per_class_scores = self.per_class_scores.astype(np.float64, copy=False)
        # Normalize decisions to a 1D array if provided
        if self.decisions is not None:
            self.decisions = np.asarray(self.decisions)
            if self.decisions.ndim != 1:
                raise ValueError("decisions must be 1D if provided.")

    @property
    def is_multiclass(self) -> bool:
        """Check if this is a multiclass result."""
        return isinstance(self.threshold, np.ndarray) and self.threshold.size > 1

    @property
    def method(self) -> str:
        """Get the optimization method used."""
        return str(self.metadata.get("method", "unknown"))

    @property
    def mode(self) -> str:
        """Get the optimization mode used."""
        return str(self.metadata.get("mode", "empirical"))

    def to_legacy_format(
        self,
    ) -> float | NDArray[np.float64] | tuple[float, float] | dict[str, Any]:
        """Convert to legacy format for backward compatibility.

        Returns the appropriate legacy format based on mode and averaging:
        - Binary empirical: float
        - Multiclass empirical: NDArray
        - Expected binary: tuple[float, float]
        - Expected micro: dict with {"threshold", "score"}
          (and "f_beta" for F-beta family)
        - Expected macro/weighted: dict with {"thresholds", "per_class", "score"}
        - Bayes: decisions array or thresholds depending on specification
        """
        mode = self.mode
        average = self.metadata.get("average", self.metadata.get("averaging", "macro"))
        metric = (self.metadata.get("metric") or "").lower()

        def _score_value(x: float | None) -> float:
            return 0.0 if x is None else float(x)

        if mode == "expected":
            # Route by averaging strategy, not by array-ness of threshold
            if average == "micro":
                thr = self.threshold
                if isinstance(thr, np.ndarray):
                    thr = float(thr.reshape(-1)[0])  # accept length-1 arrays
                legacy_result = {
                    "threshold": float(thr) if thr is not None else 0.0,
                    "score": _score_value(self.score),
                }
                # legacy alias for F-beta family
                if metric in {"fbeta", "f1", "f2"}:
                    legacy_result["f_beta"] = legacy_result["score"]
                return legacy_result
            else:
                # Ensure thresholds is always an array for expected mode
                thresholds_array: NDArray[np.float64]
                if isinstance(self.threshold, np.ndarray):
                    thresholds_array = self.threshold
                else:
                    thresholds_array = np.asarray([self.threshold], dtype=np.float64)

                legacy_result_macro: dict[str, Any] = {
                    "thresholds": thresholds_array,
                    "per_class": self.per_class_scores,
                    "score": _score_value(self.score),
                }
                if metric in {"fbeta", "f1", "f2"}:
                    # legacy aliases
                    legacy_result_macro["f_beta"] = legacy_result_macro["score"]
                    legacy_result_macro["f_beta_per_class"] = self.per_class_scores
                return legacy_result_macro

        elif mode == "bayes":
            # Bayes can yield decisions (utility matrix) or thresholds (OvR)
            if self.decisions is not None:
                return self.decisions
            # For Bayes mode, threshold should not be None
            if self.threshold is None:
                raise ValueError(
                    "Bayes mode result missing both decisions and threshold"
                )
            return self.threshold  # thresholds array or float

        else:
            # Empirical mode returns threshold(s) directly
            if self.threshold is None:
                raise ValueError("Empirical mode result missing threshold")
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
    average: str | None = None,
    averaging_method: str | None = None,  # legacy alias; mapped to 'average'
    per_class_scores: NDArray[np.float64] | None = None,
    confidence_interval: tuple[float, float] | None = None,
    decisions: Any | None = None,
    directions: Any | None = None,
    metric: str | None = None,
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
    average : str | None
        Averaging method ('macro','micro','weighted','none').
    averaging_method : str | None
        Legacy alias for 'average' (kept for compatibility).
    per_class_scores : NDArray[np.float64] | None
        Per-class scores for multiclass
    confidence_interval : tuple[float, float] | None
        Confidence interval for the score
    decisions : Any | None
        Bayes-optimal decisions from utility matrix
    directions : Any | None
        Decision directions (e.g., per-class '>' / '<')
    metric : str | None
        Metric name for legacy aliasing
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
    if threshold is None and decisions is None:
        raise ValueError("either 'threshold' or 'decisions' must be provided")

    # Prepare metadata
    metadata = {
        "method": method,
        "mode": mode,
        **kwargs,
    }
    # Normalize average key; prefer 'average', but accept legacy 'averaging_method'
    meta_average = average if average is not None else averaging_method
    if meta_average is not None:
        metadata["average"] = meta_average
    if metric is not None:
        metadata["metric"] = metric.lower()

    # Handle per-class scores & overall score harmonization
    if per_class_scores is None and isinstance(score, np.ndarray):
        # If caller passed an array as 'score', treat it as per-class scores
        per_class_scores = np.asarray(score, dtype=np.float64)
        # Compute overall score if not provided elsewhere
        score = (
            float(np.nanmean(per_class_scores)) if per_class_scores.size > 0 else 0.0
        )
    elif per_class_scores is not None and score is None:
        # If per_class_scores provided but no overall score, compute it
        per_class_scores = np.asarray(per_class_scores, dtype=np.float64)
        score = (
            float(np.nanmean(per_class_scores)) if per_class_scores.size > 0 else 0.0
        )

    if isinstance(score, np.ndarray):
        if score.ndim == 0:
            score = float(score)
        else:
            # Unexpected 1D/2D score after handling per_class_scores: reduce
            score = float(np.nanmean(np.asarray(score, dtype=float)))

    # Normalize threshold dtype now; ThresholdResult.__post_init__ enforces shapes
    thr_norm = threshold
    if isinstance(thr_norm, np.ndarray):
        thr_norm = thr_norm.astype(np.float64, copy=False)

    return ThresholdResult(
        threshold=thr_norm,
        score=score if score is None or np.isfinite(score) else float(score),
        per_class_scores=per_class_scores,
        confidence_interval=confidence_interval,
        decisions=None if decisions is None else np.asarray(decisions),
        directions=directions,
        metadata=metadata,
    )
