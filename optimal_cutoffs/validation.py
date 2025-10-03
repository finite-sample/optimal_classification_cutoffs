"""validation.py - Simple, direct validation with fail-fast semantics."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

# ============================================================================
# Core validation - Just simple functions that return clean arrays
# ============================================================================


def validate_binary_labels(labels: ArrayLike) -> np.ndarray[Any, np.dtype[np.int8]]:
    """Validate and return binary labels as int8 array.

    Parameters
    ----------
    labels : array-like
        Input labels

    Returns
    -------
    np.ndarray of int8
        Validated binary labels in {0, 1}

    Raises
    ------
    ValueError
        If labels are not binary
    """
    arr = np.asarray(labels, dtype=np.int8)

    if arr.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("Labels cannot be empty")

    unique = np.unique(arr)
    if (
        not np.array_equal(unique, [0, 1])
        and not np.array_equal(unique, [0])
        and not np.array_equal(unique, [1])
    ):
        raise ValueError(f"Labels must be binary (0 or 1), got unique values: {unique}")

    return arr


def validate_multiclass_labels(
    labels: ArrayLike, n_classes: int | None = None
) -> np.ndarray:
    """Validate and return multiclass labels as int32 array.

    Parameters
    ----------
    labels : array-like
        Input labels
    n_classes : int, optional
        If provided, validate that labels are in [0, n_classes)

    Returns
    -------
    np.ndarray of int32
        Validated labels starting from 0

    Raises
    ------
    ValueError
        If labels are invalid
    """
    arr = np.asarray(labels, dtype=np.int32)

    if arr.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("Labels cannot be empty")

    if np.any(arr < 0):
        raise ValueError(f"Labels must be non-negative, got min {arr.min()}")

    # Check consecutive from 0
    unique = np.unique(arr)
    if unique[0] != 0:
        raise ValueError(f"Labels must start from 0, got min {unique[0]}")

    max_label = unique[-1]
    expected = np.arange(max_label + 1)
    if not np.array_equal(unique, expected):
        missing = set(expected) - set(unique)
        raise ValueError(f"Labels must be consecutive from 0, missing: {missing}")

    # Check against n_classes if provided
    if n_classes is not None and max_label >= n_classes:
        raise ValueError(f"Found label {max_label} but n_classes={n_classes}")

    return arr


def validate_probabilities(probs: ArrayLike, binary: bool = False) -> np.ndarray:
    """Validate and return probabilities as float64 array.

    Parameters
    ----------
    probs : array-like
        Probabilities or scores
    binary : bool
        If True, expect 1D array. If False, infer from shape.

    Returns
    -------
    np.ndarray of float64
        Validated probabilities

    Raises
    ------
    ValueError
        If probabilities are invalid
    """
    arr = np.asarray(probs, dtype=np.float64)

    if arr.size == 0:
        raise ValueError("Probabilities cannot be empty")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Probabilities must be finite (no NaN/inf)")

    # Check shape
    if binary:
        if arr.ndim != 1:
            raise ValueError(f"Binary probabilities must be 1D, got shape {arr.shape}")
    else:
        if arr.ndim not in {1, 2}:
            raise ValueError(f"Probabilities must be 1D or 2D, got {arr.ndim}D")

    # Check range [0, 1]
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError(
            f"Probabilities must be in [0, 1], got range "
            f"[{arr.min():.3f}, {arr.max():.3f}]"
        )

    # For multiclass, warn if rows don't sum to 1 (but don't fail)
    if arr.ndim == 2 and arr.shape[1] > 1:
        row_sums = np.sum(arr, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-3):
            import warnings

            warnings.warn(
                f"Probability rows don't sum to 1 (range: "
                f"[{row_sums.min():.3f}, {row_sums.max():.3f}])",
                UserWarning,
                stacklevel=2,
            )

    return arr


def validate_weights(weights: ArrayLike, n_samples: int) -> np.ndarray:
    """Validate and return sample weights as float64 array.

    Parameters
    ----------
    weights : array-like
        Sample weights
    n_samples : int
        Expected number of samples

    Returns
    -------
    np.ndarray of float64
        Validated weights

    Raises
    ------
    ValueError
        If weights are invalid
    """
    arr = np.asarray(weights, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"Weights must be 1D, got shape {arr.shape}")

    if len(arr) != n_samples:
        raise ValueError(f"Expected {n_samples} weights, got {len(arr)}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Weights must be finite")

    if np.any(arr < 0):
        raise ValueError("Weights must be non-negative")

    if np.sum(arr) == 0:
        raise ValueError("Weights cannot sum to zero")

    return arr


# ============================================================================
# High-level validation - Combine multiple validations
# ============================================================================


def validate_multiclass_classification(
    labels: ArrayLike, probabilities: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate multiclass classification inputs.

    Returns
    -------
    tuple
        (labels as int32, probabilities as float64, weights as float64 or None)
    """
    # Convert to arrays first to check shape
    probs = np.asarray(probabilities, dtype=np.float64)

    # Validate probabilities
    probs = validate_probabilities(probs, binary=False)

    # Determine n_classes from probability matrix
    if probs.ndim == 2:
        n_classes = probs.shape[1]
    else:
        # 1D probabilities - treat as binary
        n_classes = 2

    # Validate labels with n_classes constraint
    labels = validate_multiclass_labels(labels, n_classes)

    # Check shapes match
    n_samples = len(labels)
    if probs.ndim == 2 and probs.shape[0] != n_samples:
        raise ValueError(
            f"Shape mismatch: {n_samples} labels vs {probs.shape[0]} probability rows"
        )
    elif probs.ndim == 1 and len(probs) != n_samples:
        raise ValueError(
            f"Length mismatch: {n_samples} labels vs {len(probs)} probabilities"
        )

    # Validate weights if provided
    if weights is not None:
        weights = validate_weights(weights, n_samples)

    return labels, probs, weights


def validate_threshold(
    threshold: float | ArrayLike, n_classes: int | None = None
) -> np.ndarray:
    """Validate threshold value(s).

    Parameters
    ----------
    threshold : float or array-like
        Threshold(s) to validate
    n_classes : int, optional
        For multiclass, expected number of thresholds

    Returns
    -------
    np.ndarray of float64
        Validated threshold(s)
    """
    arr = np.atleast_1d(threshold).astype(np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError("Thresholds must be finite")

    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError(
            f"Thresholds must be in [0, 1], got range "
            f"[{arr.min():.3f}, {arr.max():.3f}]"
        )

    if n_classes is not None and len(arr) != n_classes:
        raise ValueError(f"Expected {n_classes} thresholds, got {len(arr)}")

    return arr


# ============================================================================
# Convenience functions for common patterns
# ============================================================================


def ensure_binary(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Quick validation for binary classification without weights.

    Returns (labels, scores) validated and converted.
    """
    labels, scores, _ = validate_binary_classification_simple(labels, scores)
    return labels, scores


def ensure_multiclass(
    labels: ArrayLike, probs: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Quick validation for multiclass without weights.

    Returns (labels, probabilities) validated and converted.
    """
    labels, probs, _ = validate_multiclass_classification(labels, probs)
    return labels, probs


def infer_problem_type(predictions: ArrayLike) -> str:
    """Infer whether this is binary or multiclass from predictions shape.

    Returns
    -------
    str
        "binary" or "multiclass"
    """
    arr = np.asarray(predictions)

    if arr.ndim == 1:
        return "binary"
    elif arr.ndim == 2:
        return "binary" if arr.shape[1] <= 2 else "multiclass"
    else:
        raise ValueError(f"Cannot infer problem type from shape {arr.shape}")


def validate_classification(
    labels: ArrayLike, predictions: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """Validate any classification problem, inferring the type.

    Returns
    -------
    tuple
        (labels, predictions, weights, problem_type)
        where problem_type is "binary" or "multiclass"
    """
    problem_type = infer_problem_type(predictions)

    if problem_type == "binary":
        labels, predictions, weights = validate_binary_classification_simple(
            labels, predictions, weights
        )
    else:
        labels, predictions, weights = validate_multiclass_classification(
            labels, predictions, weights
        )

    return labels, predictions, weights, problem_type


def validate_choice(value: str, choices: set[str], name: str) -> str:
    """Validate string choice."""
    if value not in choices:
        raise ValueError(f"Invalid {name} '{value}'. Must be one of: {choices}")
    return value


# ============================================================================
# Legacy compatibility functions - Keep signatures compatible
# ============================================================================


def validate_binary_classification(
    true_labs: ArrayLike | None = None,
    pred_prob: ArrayLike | None = None,
    *,
    require_proba: bool = True,
    sample_weight: ArrayLike | None = None,
    return_default_weights: bool = False,
    force_dtypes: bool = False,
    labels: ArrayLike | None = None,
    scores: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Legacy compatible binary classification validation.

    Supports both old (true_labs, pred_prob) and new (labels, scores) signatures.
    """
    # Handle different calling patterns
    if labels is not None or scores is not None:
        # New style calling
        labels = labels if labels is not None else true_labs
        scores = scores if scores is not None else pred_prob
        weights = weights if weights is not None else sample_weight
    else:
        # Old style calling
        labels = true_labs
        scores = pred_prob
        weights = sample_weight

    # Validate using the new functions
    labels, scores, weights = validate_binary_classification_simple(
        labels, scores, weights
    )

    # Handle return_default_weights option
    if return_default_weights and weights is None:
        weights = np.ones(len(labels), dtype=np.float64)

    return labels, scores, weights


def validate_binary_classification_simple(
    labels: ArrayLike, scores: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Simple binary classification validation - renamed to avoid conflicts."""
    # Validate each component
    labels = validate_binary_labels(labels)
    scores = validate_probabilities(scores, binary=True)

    # Check shapes match
    if len(labels) != len(scores):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(scores)} scores"
        )

    # Validate weights if provided
    if weights is not None:
        weights = validate_weights(weights, len(labels))

    return labels, scores, weights


def _validate_inputs(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    require_binary: bool = False,
    require_proba: bool = True,
    sample_weight: ArrayLike | None = None,
    allow_multiclass: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Legacy validation function for backward compatibility."""
    if require_binary:
        return validate_binary_classification(
            true_labs,
            pred_prob,
            sample_weight=sample_weight,
        )
    else:
        # Try to infer problem type
        pred_arr = np.asarray(pred_prob)
        if pred_arr.ndim == 1 or (pred_arr.ndim == 2 and pred_arr.shape[1] == 2):
            return validate_binary_classification(
                true_labs,
                pred_prob,
                sample_weight=sample_weight,
            )
        elif pred_arr.ndim == 2 and allow_multiclass:
            return validate_multiclass_classification(
                true_labs, pred_prob, sample_weight
            )
        else:
            raise ValueError(f"Invalid prediction array shape: {pred_arr.shape}")


def validate_multiclass_input(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    require_consecutive: bool = False,
    require_proba: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy compatible multiclass validation."""
    labels, predictions, _ = validate_multiclass_classification(
        true_labs,
        pred_prob,
    )
    return labels, predictions


def validate_multiclass_probabilities_and_labels(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy compatible multiclass validation with specific dtypes."""
    labels, predictions, _ = validate_multiclass_classification(
        true_labs,
        pred_prob,
    )
    return labels, predictions


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


def _validate_threshold(threshold: float) -> None:
    """Validate threshold value."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
