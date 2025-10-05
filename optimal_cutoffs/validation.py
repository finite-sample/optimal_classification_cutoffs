"""validation.py - Simple, direct validation with fail-fast semantics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ============================================================================
# Core validation - Just simple functions that return clean arrays
# ============================================================================


def validate_binary_labels(labels: ArrayLike) -> NDArray[np.int8]:
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
    # Handle None labels (allowed in Bayes mode)
    if labels is None:
        return None
    
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
) -> NDArray[np.int32]:
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
    # Handle None labels (allowed in Bayes mode)
    if labels is None:
        if n_classes is None:
            raise ValueError("n_classes must be provided when labels is None")
        return None
    
    arr = np.asarray(labels, dtype=np.int32)

    if arr.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("Labels cannot be empty")

    if np.any(arr < 0):
        raise ValueError(f"Labels must be non-negative, got min {arr.min()}")

    # Check against n_classes if provided (labels must be valid indices)
    if n_classes is not None:
        max_label = np.max(arr)
        if max_label >= n_classes:
            raise ValueError(f"Found label {max_label} but n_classes={n_classes}")

    # Note: Labels don't need to be consecutive from 0 for One-vs-Rest optimization
    # Each class gets its own threshold regardless of which classes appear in data

    return arr


def validate_probabilities(
    probs: ArrayLike, binary: bool = False, require_proba: bool = True
) -> NDArray[np.float64]:
    """Validate and return probabilities or scores as float64 array.

    Parameters
    ----------
    probs : array-like
        Probabilities or scores
    binary : bool
        If True, expect 1D array. If False, infer from shape.
    require_proba : bool, default=True
        If True, enforce [0,1] range for probabilities. If False, allow arbitrary scores.

    Returns
    -------
    np.ndarray of float64
        Validated probabilities or scores

    Raises
    ------
    ValueError
        If probabilities/scores are invalid
    """
    arr = np.asarray(probs, dtype=np.float64)

    if arr.size == 0:
        raise ValueError("Probabilities cannot be empty")

    if not np.all(np.isfinite(arr)):
        if np.any(np.isnan(arr)):
            raise ValueError("Probabilities contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError("Probabilities contains infinite values")
        raise ValueError("Probabilities must be finite (no NaN/inf)")

    # Check shape
    if binary:
        if arr.ndim != 1:
            raise ValueError(f"Binary probabilities must be 1D, got shape {arr.shape}")
    else:
        if arr.ndim not in {1, 2}:
            raise ValueError(f"Probabilities must be 1D or 2D, got {arr.ndim}D")

    # Check range [0, 1] only if probabilities are required
    if require_proba:
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


def validate_weights(weights: ArrayLike, n_samples: int) -> NDArray[np.float64]:
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
        raise ValueError(f"Length mismatch: {n_samples} samples vs {len(arr)} weights")

    if not np.all(np.isfinite(arr)):
        if np.any(np.isnan(arr)):
            raise ValueError("Sample weights contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError("Sample weights contains infinite values")
        raise ValueError("Sample weights must be finite")

    if np.any(arr < 0):
        raise ValueError("Sample weights must be non-negative")

    if np.sum(arr) == 0:
        raise ValueError("Sample weights sum to zero")

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

    # Check shapes match (only if labels is not None)
    if labels is not None:
        n_samples = len(labels)
        if probs.ndim == 2 and probs.shape[0] != n_samples:
            raise ValueError(
                f"Shape mismatch: {n_samples} labels vs {probs.shape[0]} probability rows"
            )
        elif probs.ndim == 1 and len(probs) != n_samples:
            raise ValueError(
                f"Length mismatch: {n_samples} labels vs {len(probs)} probabilities"
        )

    # Validate weights if provided (only if labels is not None)
    if weights is not None:
        if labels is not None:
            weights = validate_weights(weights, n_samples)
        else:
            # For None labels, use probability shape for weight validation
            n_samples = probs.shape[0]
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
        labels, predictions, weights = validate_binary_classification(
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
# High-level validation functions
# ============================================================================


def validate_binary_classification(
    labels: ArrayLike,
    scores: ArrayLike,
    weights: ArrayLike | None = None,
    require_proba: bool = True,
    force_dtypes: bool = False,
) -> tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate binary classification inputs.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Predicted scores/probabilities
    weights : array-like, optional
        Sample weights
    require_proba : bool, default=True
        If True, enforce [0,1] range for probabilities. If False, allow arbitrary scores.
    force_dtypes : bool, default=False
        If True, ensure specific dtypes (int8 for labels, float64 for scores).

    Returns
    -------
    tuple
        (labels as int8, scores as float64, weights as float64 or None)
    """
    # Validate each component
    labels = validate_binary_labels(labels)
    scores = validate_probabilities(scores, binary=True, require_proba=require_proba)

    # Check shapes match (only if labels is not None)
    if labels is not None:
        if len(labels) != len(scores):
            raise ValueError(
                f"Length mismatch: {len(labels)} labels vs {len(scores)} scores"
            )

    # Validate weights if provided
    if weights is not None:
        if labels is not None:
            weights = validate_weights(weights, len(labels))
        else:
            # For None labels, use scores shape for weight validation
            weights = validate_weights(weights, len(scores))

    return labels, scores, weights


def validate_inputs(
    labels: ArrayLike,
    predictions: ArrayLike,
    weights: ArrayLike | None = None,
    require_binary: bool = False,
    allow_multiclass: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate classification inputs with automatic type detection.

    Parameters
    ----------
    labels : array-like
        True class labels
    predictions : array-like
        Predicted scores/probabilities
    weights : array-like, optional
        Sample weights
    require_binary : bool, default=False
        If True, force binary classification validation
    allow_multiclass : bool, default=True
        If False, raise error for multiclass inputs

    Returns
    -------
    tuple
        (labels, predictions, weights) validated and converted
    """
    if require_binary:
        return validate_binary_classification(labels, predictions, weights)
    else:
        # Try to infer problem type
        pred_arr = np.asarray(predictions)
        if pred_arr.ndim == 1:
            return validate_binary_classification(labels, predictions, weights)
        elif pred_arr.ndim == 2 and allow_multiclass:
            return validate_multiclass_classification(labels, predictions, weights)
        else:
            raise ValueError(f"Invalid prediction array shape: {pred_arr.shape}")


# Choice validators for backward compatibility
def _validate_metric_name(metric_name: str) -> None:
    """Validate that metric exists in the metric registry.

    Parameters
    ----------
    metric_name : str
        Name of the metric to validate

    Raises
    ------
    TypeError
        If metric_name is not a string
    ValueError
        If metric is not registered
    """
    if not isinstance(metric_name, str):
        raise TypeError("metric must be a string")
    
    from .metrics import METRICS

    if metric_name not in METRICS:
        available = sorted(METRICS.keys())
        raise ValueError(
            f"Unknown metric '{metric_name}'. Available metrics: {', '.join(available)}"
        )


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


def _validate_threshold(threshold: float | ArrayLike) -> None:
    """Validate threshold value(s)."""
    arr = np.atleast_1d(threshold)

    if not np.all(np.isfinite(arr)):
        if np.any(np.isnan(arr)):
            raise ValueError("Threshold contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError("Threshold contains infinite values")
        raise ValueError("Threshold must be finite")

    if not np.all((arr >= 0.0) & (arr <= 1.0)):
        raise ValueError(
            f"Threshold must be in [0, 1], got range [{arr.min():.3f}, {arr.max():.3f}]"
        )
