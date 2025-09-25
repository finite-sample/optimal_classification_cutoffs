"""Type definitions and protocols for optimal_cutoffs package."""

from collections.abc import Callable

# Type aliases for better readability and consistency
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

ArrayLike: TypeAlias = np.ndarray[Any, Any] | list[float] | list[int]
SampleWeightLike: TypeAlias = ArrayLike | None
MetricFunc: TypeAlias = Callable[
    [int | float, int | float, int | float, int | float], float
]
OptimizationMethod: TypeAlias = Literal[
    "auto",
    "smart_brute",
    "sort_scan",
    "minimize",
    "gradient",
    "coord_ascent",
    "dinkelbach",
]
AveragingMethod: TypeAlias = Literal["macro", "micro", "weighted", "none"]
ComparisonOperator: TypeAlias = Literal[">", ">="]
MulticlassMetricReturn: TypeAlias = (
    float | np.ndarray[Any, Any]
)  # float for averaged, array for average="none"

# Enhanced type aliases for validation
BinaryLabels: TypeAlias = np.ndarray[Any, Any]  # Shape (n_samples,), values {0, 1}
MulticlassLabels: TypeAlias = np.ndarray[
    Any, Any
]  # Shape (n_samples,) with values in {0, 1, ..., n_classes-1}
BinaryProbabilities: TypeAlias = np.ndarray[Any, Any]  # Shape (n_samples,), [0, 1]
MulticlassProbabilities: TypeAlias = np.ndarray[
    Any, Any
]  # Shape (n_samples, n_classes) with values in [0, 1]
Thresholds: TypeAlias = float | np.ndarray[Any, Any]  # Single or array
RandomState: TypeAlias = int | np.random.RandomState | np.random.Generator | None


# Protocol for sklearn-compatible classifiers
class ProbabilisticClassifier(Protocol):
    """Protocol for classifiers that can output prediction probabilities."""

    def predict_proba(self, X: ArrayLike) -> np.ndarray[Any, Any]:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Input samples.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        ...

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ProbabilisticClassifier":
        """Fit the classifier.

        Parameters
        ----------
        X : array-like
            Training samples.
        y : array-like
            Target values.

        Returns
        -------
        ProbabilisticClassifier
            Fitted classifier.
        """
        ...


# Protocol for cross-validators
class CrossValidator(Protocol):
    """Protocol for sklearn-compatible cross-validators."""

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Generate train/test splits.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target variable for supervised splits.

        Yields
        ------
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
            Train and test indices.
        """
        ...
