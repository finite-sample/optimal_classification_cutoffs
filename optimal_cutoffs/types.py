"""Type definitions and protocols for optimal_cutoffs package."""

from collections.abc import Callable, Iterator
from typing import (
    Generic,
    Literal,
    NewType,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

# Core type variables for generics
T = TypeVar("T", bound=np.generic)

# Base types using numpy.typing for better type safety
ArrayLike: TypeAlias = npt.ArrayLike
"""Type for array-like inputs that can be converted to numpy arrays."""

SampleWeightLike: TypeAlias = ArrayLike | None
"""Type for optional sample weights."""

MetricFunc: TypeAlias = Callable[
    [int | float, int | float, int | float, int | float], float
]
"""Function signature for metrics: (tp, tn, fp, fn) -> score."""
OptimizationMethod: TypeAlias = Literal[
    "auto",
    "unique_scan",
    "sort_scan",
    "minimize",
    "gradient",
    "coord_ascent",
]
"""Available optimization methods for finding optimal thresholds."""

AveragingMethod: TypeAlias = Literal["macro", "micro", "weighted", "none"]
"""Averaging strategies for multiclass metrics."""

ComparisonOperator: TypeAlias = Literal[">", ">="]
"""Comparison operators for threshold application."""

EstimationMode: TypeAlias = Literal["empirical", "bayes", "expected"]
"""Estimation regimes for threshold optimization."""
MulticlassMetricReturn: TypeAlias = float | NDArray[np.float64]
"""Return type for multiclass metrics: float for averaged, array for average='none'."""

# Type aliases for enhanced Bayes and expected functionality
UtilityMatrix: TypeAlias = NDArray[np.float64]
"""Utility matrix of shape (D, K) for D decisions, K classes."""

UtilityDict: TypeAlias = dict[str, float]
"""Binary utility dict: {'tp': ..., 'tn': ..., 'fp': ..., 'fn': ...}."""

CostVector: TypeAlias = NDArray[np.float64] | list[float]
"""Per-class costs/benefits as array or list."""

ExpectedResult: TypeAlias = dict[str, float | NDArray[np.float64]]
"""Expected mode results containing thresholds and scores."""

# Specific array types for better type safety
BinaryLabels: TypeAlias = NDArray[np.int8]
"""Binary labels array of shape (n_samples,) with values in {0, 1}."""

MulticlassLabels: TypeAlias = NDArray[np.int8]
"""Multiclass labels array of shape (n_samples,) with values in {0, 1, ..., K-1}.
"""

BinaryProbabilities: TypeAlias = NDArray[np.float64]
"""Binary probabilities array of shape (n_samples,) with values in [0, 1]."""

MulticlassProbabilities: TypeAlias = NDArray[np.float64]
"""Multiclass probabilities array of shape (n_samples, n_classes) with values in [0, 1].
"""

Thresholds: TypeAlias = float | NDArray[np.float64]
"""Single threshold or array of per-class thresholds."""

RandomState: TypeAlias = int | np.random.RandomState | np.random.Generator | None
"""Random state for reproducible randomness."""

# Semantic types using NewType for stronger type safety
Score = NewType("Score", float)
"""Semantic type for metric scores."""

Threshold = NewType("Threshold", float)
"""Semantic type for classification thresholds."""

ClassIndex = NewType("ClassIndex", int)
"""Semantic type for class indices (0, 1, 2, ...)."""

SampleIndex = NewType("SampleIndex", int)
"""Semantic type for sample indices."""

# Constants for common values
DEFAULT_BETA: float = 1.0
"""Default beta parameter for F-beta score (F1 when beta=1.0)."""

MIN_THRESHOLD: float = 0.0
"""Minimum valid threshold value."""

MAX_THRESHOLD: float = 1.0
"""Maximum valid threshold value."""

DEFAULT_TOLERANCE: float = 1e-8
"""Default numerical tolerance for comparisons."""


# Protocol for sklearn-compatible classifiers
class ProbabilisticClassifier(Protocol, Generic[T]):
    """Protocol for classifiers that can output prediction probabilities."""

    def predict_proba(self, X: NDArray[T]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Parameters
        ----------
        X : NDArray[T]
            Input samples.

        Returns
        -------
        NDArray[np.float64]
            Predicted class probabilities.
        """
        ...

    def fit(self, X: NDArray[T], y: ArrayLike) -> "ProbabilisticClassifier[T]":
        """Fit the classifier.

        Parameters
        ----------
        X : NDArray[T]
            Training samples.
        y : array-like
            Target values.

        Returns
        -------
        ProbabilisticClassifier[T]
            Fitted classifier.
        """
        ...


# Protocol for cross-validators
class CrossValidator(Protocol):
    """Protocol for sklearn-compatible cross-validators."""

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """Generate train/test splits.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target variable for supervised splits.

        Yields
        ------
        tuple[NDArray[np.int_], NDArray[np.int_]]
            Train and test indices.
        """
        ...


# Runtime validation helpers using TypeGuard
def is_binary_labels(arr: ArrayLike) -> TypeGuard[BinaryLabels]:
    """Check if array contains valid binary labels (0 or 1)."""
    arr_np = np.asarray(arr)
    return bool(
        arr_np.ndim == 1
        and np.issubdtype(arr_np.dtype, np.integer)
        and np.all(np.isin(arr_np, [0, 1]))
    )


def is_probability_array(arr: ArrayLike) -> TypeGuard[NDArray[np.float64]]:
    """Check if array contains valid probabilities in [0, 1]."""
    arr_np = np.asarray(arr, dtype=np.float64)
    return bool(
        np.all(np.isfinite(arr_np)) and np.all(arr_np >= 0.0) and np.all(arr_np <= 1.0)
    )


def is_multiclass_labels(arr: ArrayLike) -> TypeGuard[MulticlassLabels]:
    """Check if array contains valid multiclass labels (consecutive integers from 0)."""
    arr_np = np.asarray(arr)
    if arr_np.ndim != 1 or not np.issubdtype(arr_np.dtype, np.integer):
        return False

    unique_labels = np.unique(arr_np)
    return bool(
        len(unique_labels) > 0
        and unique_labels[0] == 0
        and np.array_equal(unique_labels, np.arange(len(unique_labels)))
    )


def is_multiclass_probabilities(arr: ArrayLike) -> TypeGuard[MulticlassProbabilities]:
    """Check if array is valid multiclass probabilities (2D, rows sum to ~1)."""
    arr_np = np.asarray(arr, dtype=np.float64)
    if arr_np.ndim != 2:
        return False

    return is_probability_array(arr_np) and np.allclose(
        np.sum(arr_np, axis=1), 1.0, rtol=1e-3, atol=1e-3
    )
