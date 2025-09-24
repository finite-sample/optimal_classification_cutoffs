"""Type definitions and protocols for optimal_cutoffs package."""

from collections.abc import Callable
from typing import Protocol, TypeAlias

import numpy as np

# Type aliases for better readability and consistency
from typing import Literal

ArrayLike: TypeAlias = np.ndarray | list[float] | list[int]
SampleWeightLike: TypeAlias = ArrayLike | None
MetricFunc: TypeAlias = Callable[
    [int | float, int | float, int | float, int | float], float
]
OptimizationMethod: TypeAlias = Literal["smart_brute", "minimize", "gradient"]
AveragingMethod: TypeAlias = Literal["macro", "micro", "weighted", "none"]
ComparisonOperator: TypeAlias = Literal[">", ">="]
MulticlassMetricReturn: TypeAlias = float | np.ndarray  # float for averaged, array for average="none"

# Enhanced type aliases for validation
BinaryLabels: TypeAlias = np.ndarray  # Shape (n_samples,) with values in {0, 1}
MulticlassLabels: TypeAlias = np.ndarray  # Shape (n_samples,) with values in {0, 1, ..., n_classes-1}
BinaryProbabilities: TypeAlias = np.ndarray  # Shape (n_samples,) with values in [0, 1]
MulticlassProbabilities: TypeAlias = np.ndarray  # Shape (n_samples, n_classes) with values in [0, 1]
Thresholds: TypeAlias = float | np.ndarray  # Single threshold or array of thresholds
RandomState: TypeAlias = int | np.random.RandomState | np.random.Generator | None


# Protocol for sklearn-compatible classifiers
class ProbabilisticClassifier(Protocol):
    """Protocol for classifiers that can output prediction probabilities."""

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
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
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target variable for supervised splits.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Train and test indices.
        """
        ...
