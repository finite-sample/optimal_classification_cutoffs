"""Comprehensive input validation utilities for robust API behavior."""

import warnings
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike

# Type alias for numpy arrays
NDArray = np.ndarray[Any, Any]

T = TypeVar('T')


# ============================================================================
# Base Validators - Reusable validation components
# ============================================================================

class ArrayValidator:
    """Encapsulates common array validation patterns."""

    @staticmethod
    def ensure_array(data: ArrayLike, name: str) -> NDArray:
        """Convert to array and check for emptiness."""
        arr = np.asarray(data)
        if len(arr) == 0:
            raise ValueError(f"{name} cannot be empty")
        return arr

    @staticmethod
    def check_finite(arr: NDArray, name: str) -> None:
        """Check array contains only finite values."""
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")

    @staticmethod
    def check_dimensionality(arr: NDArray, expected_dim: int, name: str) -> None:
        """Validate array dimensionality."""
        if arr.ndim != expected_dim:
            raise ValueError(f"{name} must be {expected_dim}D, got {arr.ndim}D")

    @staticmethod
    def check_length_match(arr1: NDArray, arr2: NDArray,
                          name1: str, name2: str) -> None:
        """Check two arrays have matching length."""
        if len(arr1) != len(arr2):
            raise ValueError(
                f"Length mismatch: {name1} ({len(arr1)}) vs {name2} ({len(arr2)})"
            )

    @staticmethod
    def check_range(arr: NDArray, min_val: float, max_val: float,
                   name: str) -> None:
        """Check array values are within specified range."""
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_min < min_val or arr_max > max_val:
            raise ValueError(
                f"{name} must be in [{min_val}, {max_val}], "
                f"got range [{arr_min:.6f}, {arr_max:.6f}]"
            )


class ChoiceValidator(Generic[T]):
    """Validates string choices against allowed values."""

    def __init__(self, allowed_values: set[T], value_type: str):
        self.allowed_values = allowed_values
        self.value_type = value_type

    def validate(self, value: T) -> None:
        """Validate value is in allowed set."""
        if value not in self.allowed_values:
            raise ValueError(
                f"Invalid {self.value_type} '{value}'. "
                f"Must be one of: {self.allowed_values}"
            )


# ============================================================================
# Specialized Validators - Domain-specific validation logic
# ============================================================================

class LabelValidator:
    """Validates classification labels."""

    def __init__(self, validator: ArrayValidator):
        self.validator = validator

    def validate_binary(self, labels: NDArray) -> None:
        """Validate binary classification labels."""
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            raise ValueError("true_labs contains no values")

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"Binary labels must be from {{0, 1}}, "
                f"got unique values: {unique_labels}"
            )

    def validate_multiclass(
        self, labels: NDArray, n_classes: int | None = None
    ) -> None:
        """Validate multiclass classification labels."""
        # Check non-negative integers
        if not np.all(labels >= 0):
            raise ValueError("Labels must be non-negative")
        if not np.all(labels == labels.astype(int)):
            raise ValueError("Labels must be integers")

        # Check valid range if n_classes specified
        if n_classes is not None:
            unique_labels = np.unique(labels)
            if np.any((unique_labels < 0) | (unique_labels >= n_classes)):
                raise ValueError(
                    f"Labels {unique_labels} must be within [0, {n_classes - 1}] "
                    f"to match pred_prob shape with {n_classes} classes"
                )


class ProbabilityValidator:
    """Validates probability arrays."""

    def __init__(self, validator: ArrayValidator):
        self.validator = validator

    def validate_range(self, probs: NDArray) -> None:
        """Check probabilities are in [0, 1]."""
        if np.any(probs < 0) or np.any(probs > 1):
            prob_min, prob_max = np.min(probs), np.max(probs)
            raise ValueError(
                f"Probabilities must be in [0, 1], got range "
                f"[{prob_min:.6f}, {prob_max:.6f}]"
            )

    def validate_multiclass_sum(
        self, probs: NDArray, tolerance: float = 1e-3
    ) -> None:
        """Check multiclass probabilities sum to approximately 1."""
        if probs.ndim != 2:
            return

        row_sums = np.sum(probs, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=tolerance, atol=tolerance):
            sum_min, sum_max = np.min(row_sums), np.max(row_sums)
            warnings.warn(
                f"Multiclass probabilities don't sum to 1.0 "
                f"(range: [{sum_min:.3f}, {sum_max:.3f}]). "
                "This may indicate unnormalized scores rather than probabilities.",
                UserWarning,
                stacklevel=4
            )


class WeightValidator:
    """Validates sample weights."""

    def __init__(self, validator: ArrayValidator):
        self.validator = validator

    def validate(self, weights: NDArray, n_samples: int) -> NDArray:
        """Validate sample weights."""
        self.validator.check_dimensionality(weights, 1, "sample_weight")

        if len(weights) != n_samples:
            raise ValueError(
                f"Length mismatch: sample_weight ({len(weights)}) vs "
                f"samples ({n_samples})"
            )

        self.validator.check_finite(weights, "sample_weight")

        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        if np.sum(weights) == 0:
            raise ValueError("sample_weight cannot sum to zero")

        return weights


# ============================================================================
# Configuration and Registry
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    require_binary: bool = False
    require_proba: bool = True
    allow_multiclass: bool = True


# Define choice validators as module-level constants
AVERAGING_VALIDATOR = ChoiceValidator(
    {"macro", "micro", "weighted", "none"},
    "averaging method"
)

OPTIMIZATION_VALIDATOR = ChoiceValidator(
    {"auto", "unique_scan", "sort_scan", "minimize", "gradient", "coord_ascent"},
    "optimization method"
)

COMPARISON_VALIDATOR = ChoiceValidator(
    {">", ">="},
    "comparison operator"
)


# ============================================================================
# Main Validation Functions - Public API
# ============================================================================


def _validate_inputs(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    require_binary: bool = False,
    require_proba: bool = True,
    sample_weight: ArrayLike | None = None,
    allow_multiclass: bool = True,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Validate and convert inputs with comprehensive checks.

    Parameters
    ----------
    true_labs:
        Array of true labels.
    pred_prob:
        Array of predicted probabilities or scores.
    require_binary:
        If True, require true_labs to be binary {0, 1}.
    require_proba:
        If True, require pred_prob to be probabilities in [0, 1].
    sample_weight:
        Optional array of sample weights.
    allow_multiclass:
        If True, allow 2D pred_prob for multiclass classification.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        Validated and converted (true_labs, pred_prob, sample_weight).

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    config = ValidationConfig(
        require_binary=require_binary,
        require_proba=require_proba,
        allow_multiclass=allow_multiclass
    )

    # Initialize validators
    array_val = ArrayValidator()
    label_val = LabelValidator(array_val)
    prob_val = ProbabilityValidator(array_val)
    weight_val = WeightValidator(array_val)

    # Convert and validate arrays
    true_labs = array_val.ensure_array(true_labs, "true_labs")
    pred_prob = array_val.ensure_array(pred_prob, "pred_prob")

    # Check dimensions
    array_val.check_dimensionality(true_labs, 1, "true_labs")

    # Validate prediction shape and get number of classes
    n_classes = None
    if pred_prob.ndim == 1:
        array_val.check_length_match(true_labs, pred_prob, "true_labs", "pred_prob")
    elif pred_prob.ndim == 2:
        if not config.allow_multiclass:
            raise ValueError("2D pred_prob not allowed, expected 1D array")
        if len(true_labs) != pred_prob.shape[0]:
            raise ValueError(
                f"Length mismatch: true_labs ({len(true_labs)}) vs "
                f"pred_prob rows ({pred_prob.shape[0]})"
            )
        n_classes = pred_prob.shape[1]
    else:
        raise ValueError(f"pred_prob must be 1D or 2D, got {pred_prob.ndim}D")

    # Check finite values
    array_val.check_finite(true_labs, "true_labs")
    array_val.check_finite(pred_prob, "pred_prob")

    # Validate labels
    if config.require_binary:
        label_val.validate_binary(true_labs)
    else:
        label_val.validate_multiclass(true_labs, n_classes)

    # Validate probabilities
    if config.require_proba:
        prob_val.validate_range(pred_prob)
        prob_val.validate_multiclass_sum(pred_prob)

    # Validate sample weights if provided
    validated_weight = None
    if sample_weight is not None:
        weight_array = array_val.ensure_array(sample_weight, "sample_weight")
        validated_weight = weight_val.validate(weight_array, len(true_labs))

    return true_labs, pred_prob, validated_weight


def _validate_threshold(
    threshold: float | np.ndarray[Any, Any],
    n_classes: int | None = None,
) -> np.ndarray[Any, Any]:
    """Validate threshold values.

    Parameters
    ----------
    threshold:
        Threshold value(s) to validate.
    n_classes:
        Expected number of classes for multiclass thresholds.

    Returns
    -------
    np.ndarray
        Validated threshold array.

    Raises
    ------
    ValueError
        If threshold validation fails.
    """
    validator = ArrayValidator()

    threshold = np.asarray(threshold)
    validator.check_finite(threshold, "threshold")

    # Check range [0, 1] with exact same error message as before
    if np.any(threshold < 0) or np.any(threshold > 1):
        thresh_min, thresh_max = np.min(threshold), np.max(threshold)
        raise ValueError(
            f"threshold must be in [0, 1], got range "
            f"[{thresh_min:.6f}, {thresh_max:.6f}]"
        )

    if n_classes is not None:
        validator.check_dimensionality(threshold, 1, "multiclass threshold")
        if len(threshold) != n_classes:
            raise ValueError(
                f"threshold length ({len(threshold)}) must match "
                f"number of classes ({n_classes})"
            )

    return threshold


def _validate_metric_name(metric_name: str) -> None:
    """Validate that a metric name is registered.

    Parameters
    ----------
    metric_name:
        Name of the metric to validate.

    Raises
    ------
    ValueError
        If metric is not registered.
    """
    from .metrics import METRIC_REGISTRY

    if not isinstance(metric_name, str):
        raise TypeError(f"metric must be a string, got {type(metric_name)}")
    if metric_name not in METRIC_REGISTRY:
        available_metrics = list(METRIC_REGISTRY.keys())
        raise ValueError(
            f"Unknown metric '{metric_name}'. Available metrics: {available_metrics}"
        )


def _validate_averaging_method(average: str) -> None:
    """Validate averaging method.

    Parameters
    ----------
    average:
        Averaging method to validate.

    Raises
    ------
    ValueError
        If averaging method is invalid.
    """
    AVERAGING_VALIDATOR.validate(average)


def _validate_optimization_method(method: str) -> None:
    """Validate optimization method.

    Parameters
    ----------
    method:
        Optimization method to validate.

    Raises
    ------
    ValueError
        If optimization method is invalid.
    """
    OPTIMIZATION_VALIDATOR.validate(method)


def _validate_comparison_operator(comparison: str) -> None:
    """Validate comparison operator.

    Parameters
    ----------
    comparison:
        Comparison operator to validate.

    Raises
    ------
    ValueError
        If comparison operator is invalid.
    """
    COMPARISON_VALIDATOR.validate(comparison)
