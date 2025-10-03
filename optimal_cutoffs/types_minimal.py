"""Minimal type definitions for optimal_cutoffs package.

This module contains only the essential type aliases needed for the public API.
All complex validated classes have been removed in favor of simple, direct validation.
"""

# ============================================================================
# Core Type Aliases
# ============================================================================
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricFunc: TypeAlias = Callable[
    [int | float, int | float, int | float, int | float], float
]
"""Function signature for metrics: (tp, tn, fp, fn) -> score."""

MulticlassMetricReturn: TypeAlias = float | NDArray[np.float64]
"""Return type for multiclass metrics: float for averaged, array for average='none'."""

# ============================================================================
# Legacy Type Aliases for API Compatibility
# ============================================================================

UtilityDict: TypeAlias = dict[str, float]
"""Utility specification dict with keys 'tp', 'tn', 'fp', 'fn'."""

UtilityMatrix: TypeAlias = NDArray[np.float64]
"""Utility matrix for multiclass problems."""

CostVector: TypeAlias = NDArray[np.float64] | list[float]
"""Per-class costs/benefits."""

ExpectedResult: TypeAlias = dict[str, float | NDArray[np.float64]]
"""Results from expected value optimization."""

SampleWeightLike: TypeAlias = ArrayLike | None
"""Optional sample weights."""

# String literals for method/mode parameters
OptimizationMethodLiteral: TypeAlias = str
AveragingMethodLiteral: TypeAlias = str
ComparisonOperatorLiteral: TypeAlias = str
EstimationModeLiteral: TypeAlias = str

# ============================================================================
# Simple enum-like classes for public API compatibility
# ============================================================================


class OptimizationMethod:
    """Simple optimization method constants."""

    AUTO = "auto"
    SORT_SCAN = "sort_scan"
    UNIQUE_SCAN = "unique_scan"
    MINIMIZE = "minimize"
    GRADIENT = "gradient"
    COORD_ASCENT = "coord_ascent"


class AveragingMethod:
    """Simple averaging method constants."""

    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"
    NONE = "none"


class ComparisonOperator:
    """Simple comparison operator constants."""

    GT = ">"
    GTE = ">="


class EstimationMode:
    """Simple estimation mode constants."""

    EMPIRICAL = "empirical"
    EXPECTED = "expected"
    BAYES = "bayes"


# Validation sets
OPTIMIZATION_METHODS = {
    "auto",
    "sort_scan",
    "unique_scan",
    "minimize",
    "gradient",
    "coord_ascent",
}

AVERAGING_METHODS = {"macro", "micro", "weighted", "none"}

COMPARISON_OPERATORS = {">", ">="}

ESTIMATION_MODES = {"empirical", "expected", "bayes"}
