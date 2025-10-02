"""Input validation for classification tasks - simple, fast, and composable."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Type aliases for clarity
Float64Array: TypeAlias = NDArray[np.float64]
Int32Array: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]


# ============================================================================
# Validation Results - Collect errors without throwing immediately
# ============================================================================


class ValidationError(NamedTuple):
    """Single validation error."""

    field: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass
class ValidationResult:
    """Collects validation errors and warnings."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str) -> None:
        """Add an error."""
        self.errors.append(ValidationError(field, message, "error"))

    def add_warning(self, field: str, message: str) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(field, message, "warning"))

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def raise_if_invalid(self) -> None:
        """Raise ValueError with all errors if invalid."""
        if not self.is_valid:
            error_msgs = [f"{e.field}: {e.message}" for e in self.errors]
            raise ValueError("Validation failed:\n" + "\n".join(error_msgs))

    def emit_warnings(self) -> None:
        """Emit all warnings."""
        import warnings

        for w in self.warnings:
            warnings.warn(f"{w.field}: {w.message}", UserWarning, stacklevel=3)


# ============================================================================
# Core Validation Functions - Pure functions, no side effects
# ============================================================================


def validate_array_properties(
    arr: NDArray[Any],
    name: str,
    *,
    expected_ndim: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_length: int | None = None,
    allow_empty: bool = False,
    require_finite: bool = True,
    min_value: float | None = None,
    max_value: float | None = None,
) -> ValidationResult:
    """Validate general array properties."""
    result = ValidationResult()

    # Check emptiness
    if not allow_empty and arr.size == 0:
        result.add_error(name, "cannot be empty")

    # Check dimensions
    if expected_ndim is not None and arr.ndim != expected_ndim:
        result.add_error(name, f"must be {expected_ndim}D, got {arr.ndim}D")

    # Check shape
    if expected_shape is not None and arr.shape != expected_shape:
        result.add_error(name, f"expected shape {expected_shape}, got {arr.shape}")

    # Check length
    if expected_length is not None and len(arr) != expected_length:
        result.add_error(name, f"expected length {expected_length}, got {len(arr)}")

    # Check finite values
    if require_finite and not np.all(np.isfinite(arr)):
        result.add_error(name, "contains NaN or infinite values")

    # Check value range
    if arr.size > 0 and (min_value is not None or max_value is not None):
        arr_min, arr_max = np.min(arr), np.max(arr)
        if min_value is not None and arr_min < min_value:
            result.add_error(
                name, f"values below minimum {min_value}, got {arr_min:.6f}"
            )
        if max_value is not None and arr_max > max_value:
            result.add_error(
                name, f"values above maximum {max_value}, got {arr_max:.6f}"
            )

    return result


def validate_binary_labels(arr: NDArray[Any], name: str = "labels") -> ValidationResult:
    """Validate binary classification labels."""
    result = validate_array_properties(arr, name, expected_ndim=1, require_finite=True)

    if result.is_valid and arr.size > 0:
        unique = np.unique(arr)
        if not np.all(np.isin(unique, [0, 1])):
            result.add_error(
                name, f"must be binary (0 or 1), got unique values: {unique}"
            )

    return result


def validate_multiclass_labels(
    arr: NDArray[Any],
    n_classes: int | None = None,
    require_consecutive: bool = False,
    name: str = "labels",
) -> ValidationResult:
    """Validate multiclass labels."""
    result = validate_array_properties(arr, name, expected_ndim=1, require_finite=True)

    if not result.is_valid or arr.size == 0:
        return result

    # Check integer type
    if not np.all(arr == arr.astype(int)):
        result.add_error(name, "must be integers")

    # Check non-negative
    if np.any(arr < 0):
        result.add_error(name, "must be non-negative")

    unique = np.unique(arr)

    # Check consecutive if required
    if require_consecutive:
        expected = np.arange(len(unique))
        if unique[0] != 0 or not np.array_equal(unique, expected):
            result.add_error(name, f"must be consecutive integers from 0, got {unique}")

    # Check class count if specified
    if n_classes is not None:
        max_label = np.max(arr)
        if max_label >= n_classes:
            result.add_error(
                name, f"contains label {max_label} >= n_classes {n_classes}"
            )

    return result


def validate_probabilities(
    arr: NDArray[Any],
    multiclass: bool = False,
    check_sum: bool = True,
    tolerance: float = 1e-3,
    name: str = "probabilities",
) -> ValidationResult:
    """Validate probability array."""
    expected_ndim = 2 if multiclass else 1
    result = validate_array_properties(
        arr,
        name,
        expected_ndim=expected_ndim,
        require_finite=True,
        min_value=0.0,
        max_value=1.0,
    )

    # Check row sums for multiclass
    if multiclass and check_sum and result.is_valid and arr.shape[0] > 0:
        row_sums = np.sum(arr, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=tolerance, atol=tolerance):
            min_sum, max_sum = np.min(row_sums), np.max(row_sums)
            result.add_warning(
                name, f"rows don't sum to 1.0 (range: [{min_sum:.3f}, {max_sum:.3f}])"
            )

    return result


def validate_sample_weights(
    arr: NDArray[Any], n_samples: int, name: str = "sample_weight"
) -> ValidationResult:
    """Validate sample weights."""
    result = validate_array_properties(
        arr,
        name,
        expected_ndim=1,
        expected_length=n_samples,
        require_finite=True,
        min_value=0.0,
    )

    if result.is_valid and np.sum(arr) == 0:
        result.add_error(name, "cannot sum to zero")

    return result


# ============================================================================
# High-Level Validation Classes
# ============================================================================


class ProblemType(Enum):
    """Classification problem type."""

    BINARY = auto()
    MULTICLASS = auto()
    MULTILABEL = auto()

    @classmethod
    def infer(cls, labels: NDArray[Any], predictions: NDArray[Any]) -> ProblemType:
        """Infer problem type from data shapes."""
        if predictions.ndim == 1:
            return cls.BINARY
        elif predictions.ndim == 2:
            if predictions.shape[1] == 2:
                return cls.BINARY
            else:
                return cls.MULTICLASS
        else:
            raise ValueError(
                f"Cannot infer problem type from shape {predictions.shape}"
            )


@dataclass(frozen=True)
class ValidatedData:
    """Container for validated classification data."""

    labels: NDArray[Any]
    predictions: NDArray[Any]
    weights: NDArray[Any] | None
    problem_type: ProblemType
    n_classes: int
    n_samples: int

    @classmethod
    def create(
        cls,
        labels: ArrayLike,
        predictions: ArrayLike,
        weights: ArrayLike | None = None,
        problem_type: ProblemType | None = None,
        require_proba: bool = True,
        dtype_labels: type = np.int32,
        dtype_predictions: type = np.float64,
        dtype_weights: type = np.float64,
    ) -> ValidatedData:
        """Create validated data with automatic problem type detection."""
        # Convert to arrays
        labels_arr = np.asarray(labels, dtype=dtype_labels)
        pred_arr = np.asarray(predictions, dtype=dtype_predictions)
        weights_arr = (
            np.asarray(weights, dtype=dtype_weights) if weights is not None else None
        )

        # Infer problem type if not specified
        if problem_type is None:
            problem_type = ProblemType.infer(labels_arr, pred_arr)

        # Create validation result collector
        result = ValidationResult()

        # Validate based on problem type
        if problem_type == ProblemType.BINARY:
            # Validate binary labels
            result.errors.extend(validate_binary_labels(labels_arr).errors)

            # Validate predictions
            if pred_arr.ndim == 1:
                prob_result = validate_probabilities(
                    pred_arr, multiclass=False, name="predictions"
                )
            else:
                # Binary with 2D predictions (shape [n, 2])
                if pred_arr.shape[1] != 2:
                    result.add_error(
                        "predictions",
                        f"binary problem expects 2 classes, got {pred_arr.shape[1]}",
                    )
                prob_result = validate_probabilities(
                    pred_arr, multiclass=True, name="predictions"
                )

            result.errors.extend(prob_result.errors)
            result.warnings.extend(prob_result.warnings)
            n_classes = 2

        elif problem_type == ProblemType.MULTICLASS:
            # Get number of classes from predictions
            if pred_arr.ndim != 2:
                result.add_error(
                    "predictions", f"multiclass requires 2D array, got {pred_arr.ndim}D"
                )
                n_classes = 0
            else:
                n_classes = pred_arr.shape[1]

                # Validate labels
                label_result = validate_multiclass_labels(
                    labels_arr, n_classes=n_classes, require_consecutive=True
                )
                result.errors.extend(label_result.errors)

                # Validate predictions
                prob_result = validate_probabilities(
                    pred_arr, multiclass=True, name="predictions"
                )
                result.errors.extend(prob_result.errors)
                result.warnings.extend(prob_result.warnings)

        else:
            raise NotImplementedError(f"Problem type {problem_type} not yet supported")

        # Check shapes match
        if labels_arr.shape[0] != pred_arr.shape[0]:
            result.add_error(
                "shape",
                f"labels ({labels_arr.shape[0]}) and predictions "
                f"({pred_arr.shape[0]}) length mismatch",
            )

        # Validate weights if provided
        if weights_arr is not None:
            weight_result = validate_sample_weights(weights_arr, labels_arr.shape[0])
            result.errors.extend(weight_result.errors)

        # Emit warnings and raise if invalid
        result.emit_warnings()
        result.raise_if_invalid()

        return cls(
            labels=labels_arr,
            predictions=pred_arr,
            weights=weights_arr,
            problem_type=problem_type,
            n_classes=n_classes,
            n_samples=len(labels_arr),
        )


# ============================================================================
# Simple Convenience Functions
# ============================================================================


def validate_binary_inputs(
    labels: ArrayLike,
    scores: ArrayLike,
    weights: ArrayLike | None = None,
    require_proba: bool = True,
) -> tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64] | None]:
    """Simple validation for binary classification."""
    data = ValidatedData.create(
        labels,
        scores,
        weights,
        problem_type=ProblemType.BINARY,
        require_proba=require_proba,
        dtype_labels=np.int8,
        dtype_predictions=np.float64,
    )
    return data.labels, data.predictions, data.weights


def validate_multiclass_inputs(
    labels: ArrayLike, probabilities: ArrayLike, weights: ArrayLike | None = None
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64] | None]:
    """Simple validation for multiclass classification."""
    data = ValidatedData.create(
        labels,
        probabilities,
        weights,
        problem_type=ProblemType.MULTICLASS,
        require_proba=True,
    )
    return data.labels, data.predictions, data.weights


def validate_choice(value: str, choices: set[str], name: str) -> str:
    """Validate string choice."""
    if value not in choices:
        raise ValueError(f"Invalid {name} '{value}'. Must be one of: {choices}")
    return value


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================


def validate_binary_classification(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    *,
    require_proba: bool = True,
    sample_weight: ArrayLike | None = None,
    return_default_weights: bool = False,
    force_dtypes: bool = False,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]:
    """Legacy compatible binary classification validation."""
    # Use new validation system
    dtype_labels = np.int8 if force_dtypes else np.int32
    dtype_predictions = np.float64 if force_dtypes else np.float64

    data = ValidatedData.create(
        true_labs,
        pred_prob,
        sample_weight,
        problem_type=ProblemType.BINARY,
        require_proba=require_proba,
        dtype_labels=dtype_labels,
        dtype_predictions=dtype_predictions,
    )

    # Handle return_default_weights option
    weights = data.weights
    if return_default_weights and weights is None:
        weights = np.ones(data.n_samples, dtype=np.float64)

    return data.labels, data.predictions, weights


def validate_multiclass_input(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    require_consecutive: bool = False,
    require_proba: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Legacy compatible multiclass validation."""
    data = ValidatedData.create(
        true_labs,
        pred_prob,
        problem_type=ProblemType.MULTICLASS,
        require_proba=require_proba,
    )

    # Additional consecutive check if required
    if require_consecutive:
        result = validate_multiclass_labels(
            data.labels, data.n_classes, require_consecutive=True
        )
        result.raise_if_invalid()

    return data.labels, data.predictions


def validate_multiclass_probabilities_and_labels(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Legacy compatible multiclass validation with specific dtypes."""
    data = ValidatedData.create(
        true_labs,
        pred_prob,
        problem_type=ProblemType.MULTICLASS,
        require_proba=True,
        dtype_labels=np.int32,
        dtype_predictions=np.float64,
    )
    return data.labels, data.predictions


# Choice validators for backward compatibility
def _validate_metric_name(metric_name: str) -> None:
    """Validate metric name (placeholder for metric registry validation)."""
    # This is a placeholder - actual validation is done by the metric registry
    # We keep this for API compatibility
    pass


def _validate_averaging_method(average: str) -> None:
    """Validate averaging method."""
    validate_choice(average, {"macro", "micro", "weighted", "none"}, "averaging method")


def _validate_optimization_method(method: str) -> None:
    """Validate optimization method."""
    validate_choice(
        method,
        {"auto", "unique_scan", "sort_scan", "minimize", "gradient", "coord_ascent"},
        "optimization method",
    )


def _validate_comparison_operator(comparison: str) -> None:
    """Validate comparison operator."""
    validate_choice(comparison, {">", ">="}, "comparison operator")


def _validate_inputs(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    require_binary: bool = False,
    require_proba: bool = True,
    sample_weight: ArrayLike | None = None,
    allow_multiclass: bool = True,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]:
    """Legacy validation function for backward compatibility."""
    if require_binary:
        return validate_binary_classification(
            true_labs,
            pred_prob,
            require_proba=require_proba,
            sample_weight=sample_weight,
        )
    else:
        # Try to infer problem type
        pred_arr = np.asarray(pred_prob)
        if pred_arr.ndim == 1 or (pred_arr.ndim == 2 and pred_arr.shape[1] == 2):
            return validate_binary_classification(
                true_labs,
                pred_prob,
                require_proba=require_proba,
                sample_weight=sample_weight,
            )
        elif pred_arr.ndim == 2 and allow_multiclass:
            labels, predictions = validate_multiclass_input(
                true_labs, pred_prob, require_proba=require_proba
            )
            weights = None
            if sample_weight is not None:
                weight_result = validate_sample_weights(
                    np.asarray(sample_weight), len(labels)
                )
                weight_result.raise_if_invalid()
                weights = np.asarray(sample_weight, dtype=np.float64)
            return labels, predictions, weights
        else:
            raise ValueError(f"Invalid prediction array shape: {pred_arr.shape}")


def _validate_threshold(threshold: float) -> None:
    """Validate threshold value."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
