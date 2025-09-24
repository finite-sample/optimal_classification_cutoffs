"""Type definitions and protocols for optimal_cutoffs package."""

from collections.abc import Callable
from typing import Protocol, TypeAlias

import numpy as np

# Type aliases for better readability and consistency
ArrayLike: TypeAlias = np.ndarray | list[float] | list[int]
MetricFunc: TypeAlias = Callable[[int, int, int, int], float]
OptimizationMethod: TypeAlias = str  # Could be Literal["smart_brute", "minimize", "gradient"]
AveragingMethod: TypeAlias = str     # Could be Literal["macro", "micro", "weighted"]

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

    def fit(self, X: ArrayLike, y: ArrayLike) -> 'ProbabilisticClassifier':
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

    def split(self, X: ArrayLike, y: ArrayLike | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
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
