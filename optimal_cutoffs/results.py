"""Unified result types for threshold optimization.

This module provides dataclasses for consistent result handling across
different optimization modes, improving type safety and API clarity.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BinaryResult:
    """Result for binary threshold optimization.

    Attributes
    ----------
    threshold : float
        Optimal threshold value
    score : float | None
        Achieved metric score (if available)
    method : str
        Optimization method used
    """

    threshold: float
    score: float | None = None
    method: str = "unknown"

    def to_legacy_format(self) -> float:
        """Convert to legacy float format for backward compatibility."""
        return self.threshold


@dataclass
class MulticlassResult:
    """Result for multiclass threshold optimization.

    Attributes
    ----------
    thresholds : np.ndarray
        Per-class optimal thresholds
    scores : np.ndarray | None
        Per-class metric scores (if available)
    overall_score : float | None
        Overall averaged score (if available)
    method : str
        Optimization method used
    """

    thresholds: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any] | None = None
    overall_score: float | None = None
    method: str = "unknown"

    def to_legacy_format(self) -> np.ndarray[Any, Any]:
        """Convert to legacy array format for backward compatibility."""
        return self.thresholds


@dataclass
class ExpectedBinaryResult:
    """Result for expected binary optimization.

    Attributes
    ----------
    threshold : float
        Optimal threshold value
    expected_score : float
        Expected metric score under calibration
    method : str
        Optimization method used
    """

    threshold: float
    expected_score: float
    method: str = "expected"

    def to_legacy_format(self) -> tuple[float, float]:
        """Convert to legacy tuple format for backward compatibility."""
        return (self.threshold, self.expected_score)


@dataclass
class ExpectedMulticlassResult:
    """Result for expected multiclass optimization.

    Attributes
    ----------
    thresholds : np.ndarray
        Per-class optimal thresholds
    per_class_scores : np.ndarray
        Per-class expected scores
    overall_score : float
        Overall averaged expected score
    averaging_method : str
        Averaging method used ("macro", "micro", "weighted")
    method : str
        Optimization method used
    """

    thresholds: np.ndarray[Any, Any]
    per_class_scores: np.ndarray[Any, Any]
    overall_score: float
    averaging_method: str = "macro"
    method: str = "expected"

    def to_legacy_format(self) -> dict[str, float | np.ndarray[Any, Any]]:
        """Convert to legacy dict format for backward compatibility."""
        if self.averaging_method == "micro":
            return {
                "threshold": float(self.thresholds[0])
                if self.thresholds.size == 1
                else self.thresholds,
                "f_beta": self.overall_score,
            }
        else:
            return {
                "thresholds": self.thresholds,
                "f_beta_per_class": self.per_class_scores,
                "f_beta": self.overall_score,
            }


@dataclass
class BayesDecisionResult:
    """Result for Bayes decision optimization.

    Attributes
    ----------
    decisions : np.ndarray
        Optimal class decisions for each sample
    scores : np.ndarray | None
        Decision scores (if available)
    method : str
        Optimization method used
    """

    decisions: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any] | None = None
    method: str = "bayes"

    def to_legacy_format(self) -> np.ndarray[Any, Any]:
        """Convert to legacy array format for backward compatibility."""
        return self.decisions


# Union type for all result types
OptimizationResult = (
    BinaryResult
    | MulticlassResult
    | ExpectedBinaryResult
    | ExpectedMulticlassResult
    | BayesDecisionResult
)


def create_result(
    *,
    threshold: float | np.ndarray[Any, Any] | None = None,
    score: float | np.ndarray[Any, Any] | None = None,
    method: str = "unknown",
    mode: str = "empirical",
    averaging_method: str | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Factory function to create appropriate result type.

    Parameters
    ----------
    threshold : float | np.ndarray | None
        Threshold value(s)
    score : float | np.ndarray | None
        Score value(s)
    method : str
        Optimization method used
    mode : str
        Optimization mode ("empirical", "expected", "bayes")
    averaging_method : str | None
        Averaging method for multiclass
    **kwargs : Any
        Additional arguments

    Returns
    -------
    OptimizationResult
        Appropriate result type based on inputs

    Raises
    ------
    ValueError
        If inputs are inconsistent or invalid
    """
    if threshold is None:
        raise ValueError("threshold is required")

    # Determine if binary or multiclass
    is_binary = isinstance(threshold, (int, float)) or (
        isinstance(threshold, np.ndarray) and threshold.ndim == 0
    )

    if mode == "bayes" and "decisions" in kwargs:
        return BayesDecisionResult(
            decisions=kwargs["decisions"],
            scores=kwargs.get("scores"),
            method=method,
        )

    if mode == "expected":
        if is_binary:
            expected_score = score if isinstance(score, (int, float)) else None
            if expected_score is None:
                raise ValueError("expected_score required for expected binary results")
            return ExpectedBinaryResult(
                threshold=float(threshold),
                expected_score=expected_score,
                method=method,
            )
        else:
            if not isinstance(threshold, np.ndarray):
                raise ValueError("multiclass threshold must be array")
            if not isinstance(score, (np.ndarray, float)):
                raise ValueError("multiclass score required")

            per_class_scores = kwargs.get("per_class_scores")
            if per_class_scores is None and isinstance(score, np.ndarray):
                per_class_scores = score
                overall_score = kwargs.get("overall_score", float(np.mean(score)))
            else:
                overall_score = float(score) if isinstance(score, (int, float)) else 0.0

            return ExpectedMulticlassResult(
                thresholds=threshold,
                per_class_scores=per_class_scores or np.array([]),
                overall_score=overall_score,
                averaging_method=averaging_method or "macro",
                method=method,
            )

    if is_binary:
        return BinaryResult(
            threshold=float(threshold),
            score=float(score) if isinstance(score, (int, float)) else None,
            method=method,
        )
    else:
        if not isinstance(threshold, np.ndarray):
            raise ValueError("multiclass threshold must be array")
        return MulticlassResult(
            thresholds=threshold,
            scores=score if isinstance(score, np.ndarray) else None,
            overall_score=float(score) if isinstance(score, (int, float)) else None,
            method=method,
        )
