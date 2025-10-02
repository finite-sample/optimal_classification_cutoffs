"""Modern type system for optimal_cutoffs package.

This module provides a comprehensive, type-safe system for classification threshold
optimization. It uses validated data classes, rich enums with behavior, and modern
Python features to eliminate runtime errors and improve maintainability.

Key design principles:
- Validated types that cannot be constructed with invalid data
- Rich domain objects with behavior, not just data containers
- Immutable objects for thread safety and clarity
- Modern Python features (frozen dataclasses, pattern matching, cached_property)
- Zero tolerance for invalid states
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from typing import (
    Any,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    final,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ============================================================================
# Type Variables and Core Aliases
# ============================================================================

T = TypeVar("T", bound=np.generic)
Shape = TypeVar("Shape")  # For shape-aware types

# Define specific shape types
N = TypeVar("N")  # Number of samples dimension
K = TypeVar("K")  # Number of classes dimension

# Keep minimal compatibility with existing code
MetricFunc: TypeAlias = Callable[
    [int | float, int | float, int | float, int | float], float
]
"""Function signature for metrics: (tp, tn, fp, fn) -> score."""

MulticlassMetricReturn: TypeAlias = float | NDArray[np.float64]
"""Return type for multiclass metrics: float for averaged, array for average='none'."""

# ============================================================================
# Rich Enumerations with Behavior
# ============================================================================

class OptimizationMethod(Enum):
    """Optimization methods with associated metadata and behavior."""

    AUTO = auto()
    UNIQUE_SCAN = auto()
    SORT_SCAN = auto()
    MINIMIZE = auto()
    GRADIENT = auto()
    COORD_ASCENT = auto()

    @property
    def requires_gradient(self) -> bool:
        """Whether this method requires gradient computation."""
        return self in (OptimizationMethod.GRADIENT, OptimizationMethod.COORD_ASCENT)

    @property
    def is_exhaustive(self) -> bool:
        """Whether this method searches all possible thresholds."""
        return self in (OptimizationMethod.UNIQUE_SCAN, OptimizationMethod.SORT_SCAN)

    @property
    def computational_complexity(self) -> str:
        """Big-O complexity of the method."""
        if self == OptimizationMethod.UNIQUE_SCAN:
            return "O(n²)"
        elif self == OptimizationMethod.SORT_SCAN:
            return "O(n log n)"
        elif self in (OptimizationMethod.MINIMIZE, OptimizationMethod.GRADIENT):
            return "O(n × iterations)"
        elif self == OptimizationMethod.COORD_ASCENT:
            return "O(n × k × iterations)"
        elif self == OptimizationMethod.AUTO:
            return "Varies"
        else:
            # This should never happen with a proper enum
            raise ValueError(f"Unknown optimization method: {self}")

    @property
    def supports_multiclass(self) -> bool:
        """Whether this method supports multiclass optimization."""
        return self in (OptimizationMethod.AUTO, OptimizationMethod.COORD_ASCENT)


class AveragingMethod(Enum):
    """Averaging methods with computation logic."""

    MACRO = auto()
    MICRO = auto()
    WEIGHTED = auto()
    NONE = auto()

    def compute_average(
        self,
        scores: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None
    ) -> float | NDArray[np.float64]:
        """Apply averaging to per-class scores."""
        if self == AveragingMethod.NONE:
            return scores
        elif self == AveragingMethod.MACRO:
            return float(np.mean(scores))
        elif self == AveragingMethod.WEIGHTED:
            if weights is None:
                raise ValueError("Weights required for weighted averaging")
            return float(np.average(scores, weights=weights))
        elif self == AveragingMethod.MICRO:
            # Micro averaging handled differently (needs raw counts)
            raise NotImplementedError("Micro averaging requires confusion matrices")
        else:
            raise ValueError(f"Unknown averaging method: {self}")

    @property
    def needs_weights(self) -> bool:
        """Whether this averaging method requires class weights."""
        return self == AveragingMethod.WEIGHTED


class ComparisonOperator(Enum):
    """Comparison operators as callable objects with numpy integration."""

    GT = ">"
    GTE = ">="

    def __call__(self, a: float, b: float) -> bool:
        """Make enum members callable for scalar comparisons."""
        if self == ComparisonOperator.GT:
            return a > b
        elif self == ComparisonOperator.GTE:
            return a >= b
        else:
            raise ValueError(f"Unknown comparison operator: {self}")

    @property
    def numpy_ufunc(self) -> Callable[[Any, Any], Any]:
        """Get numpy ufunc for vectorized operations."""
        if self == ComparisonOperator.GT:
            return np.greater
        elif self == ComparisonOperator.GTE:
            return np.greater_equal
        else:
            raise ValueError(f"Unknown comparison operator: {self}")

    @property
    def symbol(self) -> str:
        """Get the string representation of the operator."""
        return self.value


class EstimationMode(Enum):
    """Estimation modes with validation requirements."""

    EMPIRICAL = auto()
    BAYES = auto()
    EXPECTED = auto()

    @property
    def requires_true_labels(self) -> bool:
        """Whether this mode requires true labels for optimization."""
        return self == EstimationMode.EMPIRICAL

    @property
    def requires_utilities(self) -> bool:
        """Whether this mode requires utility specification."""
        return self in (EstimationMode.BAYES, EstimationMode.EXPECTED)

    @property
    def supports_multiclass(self) -> bool:
        """Whether this mode supports multiclass problems."""
        return self in (EstimationMode.EMPIRICAL, EstimationMode.EXPECTED)


# ============================================================================
# Validated Array Types with Shape Awareness
# ============================================================================

@dataclass(frozen=True, slots=True)
class ValidatedArray(Generic[Shape]):
    """Base class for validated numpy arrays with shape tracking."""

    data: NDArray[Any]
    _original_input: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate array after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Override in subclasses for specific validation."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(self.data)}")

        if not np.all(np.isfinite(self.data)):
            raise ValueError("Array contains NaN or infinite values")

    @property
    def shape(self) -> tuple[int, ...]:
        """Get array shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.data.size

    def __array__(self) -> NDArray[Any]:
        """Support numpy operations."""
        return self.data

    def __len__(self) -> int:
        """Get length of first dimension."""
        return len(self.data)


@final
@dataclass(frozen=True, slots=True)
class BinaryLabels(ValidatedArray["N"]):
    """Binary classification labels in {0, 1} with validation and properties."""

    @classmethod
    def from_array(cls, labels: ArrayLike) -> Self:
        """Create from array-like input with validation."""
        arr = np.asarray(labels)
        return cls(data=arr, _original_input=labels)

    def _validate(self) -> None:
        """Validate binary labels."""
        super()._validate()

        if self.data.ndim != 1:
            raise ValueError(f"Binary labels must be 1D, got shape {self.data.shape}")

        unique = np.unique(self.data)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"Binary labels must be in {{0, 1}}, got {unique}")

        # Ensure integer type
        if not np.issubdtype(self.data.dtype, np.integer):
            object.__setattr__(self, 'data', self.data.astype(np.int8))

    @cached_property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.data)

    @cached_property
    def class_counts(self) -> dict[int, int]:
        """Count of samples per class."""
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=False))

    @cached_property
    def class_balance(self) -> float:
        """Proportion of positive class (class 1)."""
        return float(np.mean(self.data))

    @cached_property
    def is_balanced(self) -> bool:
        """Whether classes are reasonably balanced (30-70% split)."""
        return 0.3 <= self.class_balance <= 0.7


@final
@dataclass(frozen=True, slots=True)
class MulticlassLabels(ValidatedArray["N"]):
    """Multiclass labels in {0, 1, ..., K-1} with validation and properties."""

    @classmethod
    def from_array(cls, labels: ArrayLike) -> Self:
        """Create from array-like input with validation."""
        arr = np.asarray(labels)
        return cls(data=arr, _original_input=labels)

    def _validate(self) -> None:
        """Validate multiclass labels."""
        super()._validate()

        if self.data.ndim != 1:
            raise ValueError(f"Labels must be 1D, got shape {self.data.shape}")

        # Ensure integer type
        if not np.issubdtype(self.data.dtype, np.integer):
            object.__setattr__(self, 'data', self.data.astype(np.int32))

        unique = np.unique(self.data)
        if len(unique) == 0:
            raise ValueError("Labels array is empty")

        if unique[0] < 0:
            raise ValueError("Labels must be non-negative")

        # Check for consecutive integers from 0
        expected = np.arange(np.max(unique) + 1)
        missing = set(expected) - set(unique)
        if missing:
            raise ValueError(
                f"Labels must be consecutive integers from 0, missing: {missing}"
            )

    @cached_property
    def n_classes(self) -> int:
        """Number of classes."""
        return int(np.max(self.data) + 1)

    @cached_property
    def class_counts(self) -> NDArray[np.int64]:
        """Count of samples per class (length n_classes)."""
        return np.bincount(self.data, minlength=self.n_classes)

    @cached_property
    def class_weights(self) -> NDArray[np.float64]:
        """Normalized class weights based on frequency."""
        counts = self.class_counts
        return counts / len(self.data)

    @cached_property
    def is_balanced(self) -> bool:
        """Whether all classes have roughly equal representation."""
        weights = self.class_weights
        return float(np.std(weights)) < 0.1  # Low standard deviation


@final
@dataclass(frozen=True, slots=True)
class Probabilities(ValidatedArray[Any]):
    """Probability array with automatic binary/multiclass detection and validation."""

    @classmethod
    def from_array(cls, probs: ArrayLike) -> Self:
        """Create from array-like input with validation."""
        arr = np.asarray(probs)
        return cls(data=arr, _original_input=probs)

    def _validate(self) -> None:
        """Validate probabilities."""
        super()._validate()

        if self.data.ndim not in {1, 2}:
            raise ValueError(f"Probabilities must be 1D or 2D, got {self.data.ndim}D")

        if np.any(self.data < 0) or np.any(self.data > 1):
            min_val, max_val = np.min(self.data), np.max(self.data)
            raise ValueError(
                f"Probabilities must be in [0, 1], got range "
                f"[{min_val:.6f}, {max_val:.6f}]"
            )

        # Ensure float type
        if not np.issubdtype(self.data.dtype, np.floating):
            object.__setattr__(self, 'data', self.data.astype(np.float64))

        # For 2D (multiclass), check row sums
        if self.data.ndim == 2:
            row_sums = np.sum(self.data, axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-3):
                import warnings
                warnings.warn(
                    f"Multiclass probabilities don't sum to 1.0 "
                    f"(range: [{row_sums.min():.3f}, {row_sums.max():.3f}]). "
                    "This may indicate unnormalized scores.",
                    UserWarning,
                    stacklevel=4
                )

    @cached_property
    def is_binary(self) -> bool:
        """Whether this represents binary classification probabilities."""
        return self.data.ndim == 1

    @cached_property
    def is_multiclass(self) -> bool:
        """Whether this represents multiclass probabilities."""
        return self.data.ndim == 2

    @cached_property
    def n_classes(self) -> int:
        """Number of classes."""
        return 2 if self.is_binary else self.data.shape[1]

    @cached_property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.data.shape[0]

    def get_class_probabilities(self, class_idx: int) -> NDArray[np.float64]:
        """Get probabilities for a specific class."""
        if self.is_binary:
            if class_idx == 0:
                return 1.0 - self.data
            elif class_idx == 1:
                return self.data
            else:
                raise ValueError(
                    f"Binary probabilities only have classes 0 and 1, got {class_idx}"
                )

        if class_idx < 0 or class_idx >= self.n_classes:
            raise ValueError(
                f"Class index {class_idx} out of range [0, {self.n_classes-1}]"
            )

        return self.data[:, class_idx]


@final
@dataclass(frozen=True, slots=True)
class SampleWeights(ValidatedArray["N"]):
    """Sample weights with validation and normalization utilities."""

    @classmethod
    def from_array(cls, weights: ArrayLike) -> Self:
        """Create from array-like input with validation."""
        arr = np.asarray(weights)
        return cls(data=arr, _original_input=weights)

    def _validate(self) -> None:
        """Validate sample weights."""
        super()._validate()

        if self.data.ndim != 1:
            raise ValueError(f"Weights must be 1D, got shape {self.data.shape}")

        if np.any(self.data < 0):
            raise ValueError("Weights must be non-negative")

        if np.sum(self.data) == 0:
            raise ValueError("Weights cannot sum to zero")

        # Ensure float type
        if not np.issubdtype(self.data.dtype, np.floating):
            object.__setattr__(self, 'data', self.data.astype(np.float64))

    @cached_property
    def total_weight(self) -> float:
        """Total sum of all weights."""
        return float(np.sum(self.data))

    @cached_property
    def normalized(self) -> NDArray[np.float64]:
        """Get normalized weights that sum to 1."""
        return self.data / self.total_weight

    @cached_property
    def effective_sample_size(self) -> float:
        """Effective sample size accounting for weight variance."""
        weights = self.normalized
        return float(1.0 / np.sum(weights ** 2))


# ============================================================================
# Threshold Types
# ============================================================================

@final
@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    """Threshold specification with validation and application behavior."""

    values: float | NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if isinstance(self.values, (int, float)):
            values = float(self.values)
            if not (0.0 <= values <= 1.0):
                raise ValueError(f"Threshold must be in [0, 1], got {values}")
            if not np.isfinite(values):
                raise ValueError("Threshold cannot be NaN or infinite")
            # Keep as float - no need to reassign
        else:
            values_array = np.asarray(self.values, dtype=np.float64)
            if not np.all(np.isfinite(values_array)):
                raise ValueError("Thresholds contain NaN or infinite values")
            if np.any(values_array < 0) or np.any(values_array > 1):
                raise ValueError(
                    f"Thresholds must be in [0, 1], got range "
                    f"[{values_array.min()}, {values_array.max()}]"
                )
            object.__setattr__(self, 'values', values_array)

    @property
    def is_scalar(self) -> bool:
        """Whether this is a single threshold."""
        return isinstance(self.values, float)

    @property
    def is_multiclass(self) -> bool:
        """Whether this represents multiclass thresholds."""
        return not self.is_scalar

    @property
    def n_classes(self) -> int:
        """Number of classes these thresholds apply to."""
        if self.is_scalar:
            return 1
        else:
            # self.values is NDArray here
            assert isinstance(self.values, np.ndarray)
            return len(self.values)

    def apply(
        self,
        probabilities: Probabilities,
        operator: ComparisonOperator = ComparisonOperator.GTE
    ) -> NDArray[np.bool_]:
        """Apply threshold to probabilities to get binary predictions."""
        if self.is_scalar:
            if probabilities.is_binary:
                result = operator.numpy_ufunc(probabilities.data, self.values)
                return np.asarray(result, dtype=bool)
            else:
                # For multiclass with scalar threshold, apply to class 1 probabilities
                class_1_probs = probabilities.get_class_probabilities(1)
                result = operator.numpy_ufunc(class_1_probs, self.values)
                return np.asarray(result, dtype=bool)

        # Multiclass case with per-class thresholds
        if probabilities.is_binary:
            raise ValueError(
                "Cannot apply multiclass thresholds to binary probabilities"
            )

        if self.n_classes != probabilities.n_classes:
            raise ValueError(
                f"Threshold count ({self.n_classes}) must match class count "
                f"({probabilities.n_classes})"
            )

        # For multiclass, use One-vs-Rest approach by default
        predictions = np.zeros(probabilities.n_samples, dtype=bool)
        assert isinstance(self.values, np.ndarray)  # Should be array in multiclass case
        for i in range(self.n_classes):
            class_probs = probabilities.get_class_probabilities(i)
            result = operator.numpy_ufunc(class_probs, self.values[i])
            class_preds = np.asarray(result, dtype=bool)
            predictions |= class_preds

        return predictions


# ============================================================================
# Result Types
# ============================================================================

@final
@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Confusion matrix with derived metrics and validation."""

    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        """Validate confusion matrix values."""
        if any(val < 0 for val in [self.tp, self.tn, self.fp, self.fn]):
            raise ValueError("Confusion matrix values must be non-negative")

    @classmethod
    def from_predictions(
        cls,
        y_true: BinaryLabels[Any],
        y_pred: NDArray[np.bool_],
        sample_weight: SampleWeights[Any] | None = None
    ) -> Self:
        """Create from true labels and binary predictions."""
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between labels and predictions")

        weights = sample_weight.data if sample_weight else None

        if weights is not None:
            # Use np.average instead of np.sum with weights for type safety
            tp = float(np.sum(((y_true.data == 1) & y_pred).astype(float) * weights))
            tn = float(np.sum(((y_true.data == 0) & ~y_pred).astype(float) * weights))
            fp = float(np.sum(((y_true.data == 0) & y_pred).astype(float) * weights))
            fn = float(np.sum(((y_true.data == 1) & ~y_pred).astype(float) * weights))
        else:
            tp = float(np.sum((y_true.data == 1) & y_pred))
            tn = float(np.sum((y_true.data == 0) & ~y_pred))
            fp = float(np.sum((y_true.data == 0) & y_pred))
            fn = float(np.sum((y_true.data == 1) & ~y_pred))

        return cls(int(tp), int(tn), int(fp), int(fn))

    @cached_property
    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.tn + self.fp + self.fn

    @cached_property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @cached_property
    def precision(self) -> float:
        """Precision (Positive Predictive Value)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @cached_property
    def recall(self) -> float:
        """Recall (True Positive Rate, Sensitivity)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @cached_property
    def specificity(self) -> float:
        """Specificity (True Negative Rate)."""
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @cached_property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)."""
        return self.f_score(beta=1.0)

    def f_score(self, beta: float = 1.0) -> float:
        """F-beta score."""
        if self.precision == 0 and self.recall == 0:
            return 0.0
        beta_sq = beta ** 2
        return (
            (1 + beta_sq) * self.precision * self.recall
            / (beta_sq * self.precision + self.recall)
        )

    @cached_property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient."""
        denom = float(np.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn)
            * (self.tn + self.fp) * (self.tn + self.fn)
        ))
        if denom == 0:
            return 0.0
        return float((self.tp * self.tn - self.fp * self.fn) / denom)


@final
@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Complete result from threshold optimization with metadata."""

    threshold: ThresholdSpec
    score: float
    confusion_matrix: ConfusionMatrix | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Whether optimization converged (if applicable)."""
        return bool(self.metadata.get('converged', True))

    @property
    def n_iterations(self) -> int | None:
        """Number of iterations taken (if applicable)."""
        result = self.metadata.get('n_iterations')
        return int(result) if result is not None else None

    @property
    def method_used(self) -> str | None:
        """Optimization method that was used."""
        return self.metadata.get('method')


# ============================================================================
# Utility Specification
# ============================================================================

@final
@dataclass(frozen=True, slots=True)
class UtilitySpec:
    """Complete utility specification for decision theory approaches."""

    tp_utility: float = 1.0
    tn_utility: float = 1.0
    fp_utility: float = -1.0
    fn_utility: float = -1.0

    def compute_utility(self, cm: ConfusionMatrix) -> float:
        """Compute total utility from confusion matrix."""
        return (
            self.tp_utility * cm.tp +
            self.tn_utility * cm.tn +
            self.fp_utility * cm.fp +
            self.fn_utility * cm.fn
        )

    @classmethod
    def from_costs(cls, fp_cost: float, fn_cost: float) -> Self:
        """Create from misclassification costs (converted to negative utilities)."""
        return cls(
            tp_utility=0.0,
            tn_utility=0.0,
            fp_utility=-abs(fp_cost),
            fn_utility=-abs(fn_cost)
        )

    @classmethod
    def from_dict(cls, utility_dict: dict[str, float]) -> Self:
        """Create from dictionary with keys 'tp', 'tn', 'fp', 'fn'."""
        required_keys = {'tp', 'tn', 'fp', 'fn'}
        if not all(key in utility_dict for key in required_keys):
            raise ValueError(f"Utility dict must contain keys: {required_keys}")

        return cls(
            tp_utility=utility_dict['tp'],
            tn_utility=utility_dict['tn'],
            fp_utility=utility_dict['fp'],
            fn_utility=utility_dict['fn']
        )


# ============================================================================
# Legacy Type Aliases for Backward Compatibility (Minimal)
# ============================================================================

# Keep these for gradual migration
UtilityDict: TypeAlias = dict[str, float]
"""Legacy: Use UtilitySpec.from_dict() instead."""

UtilityMatrix: TypeAlias = NDArray[np.float64]
"""Legacy: Utility matrix for multiclass problems."""

CostVector: TypeAlias = NDArray[np.float64] | list[float]
"""Legacy: Per-class costs/benefits."""

ExpectedResult: TypeAlias = dict[str, float | NDArray[np.float64]]
"""Legacy: Results from expected value optimization."""

SampleWeightLike: TypeAlias = ArrayLike | None
"""Legacy: Optional sample weights (use SampleWeights.from_array() instead)."""

# String literals for enum compatibility
OptimizationMethodLiteral: TypeAlias = str  # Will be replaced by OptimizationMethod
AveragingMethodLiteral: TypeAlias = str     # Will be replaced by AveragingMethod
ComparisonOperatorLiteral: TypeAlias = str  # Will be replaced by ComparisonOperator
EstimationModeLiteral: TypeAlias = str      # Will be replaced by EstimationMode
