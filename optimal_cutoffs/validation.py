"""Comprehensive input validation utilities for robust API behavior."""

from typing import Any

import numpy as np

from .types import ArrayLike


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
    # Convert to arrays
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)

    # Check for empty inputs
    if len(true_labs) == 0:
        raise ValueError("true_labs cannot be empty")
    if len(pred_prob) == 0:
        raise ValueError("pred_prob cannot be empty")

    # Check dimensionality
    if true_labs.ndim != 1:
        raise ValueError(f"true_labs must be 1D, got {true_labs.ndim}D")

    if pred_prob.ndim == 1:
        # Binary case
        if len(true_labs) != len(pred_prob):
            raise ValueError(
                f"Length mismatch: true_labs ({len(true_labs)}) vs "
                f"pred_prob ({len(pred_prob)})"
            )
    elif pred_prob.ndim == 2:
        # Multiclass case
        if not allow_multiclass:
            raise ValueError("2D pred_prob not allowed, expected 1D array")
        if len(true_labs) != pred_prob.shape[0]:
            raise ValueError(
                f"Length mismatch: true_labs ({len(true_labs)}) vs "
                f"pred_prob rows ({pred_prob.shape[0]})"
            )
    else:
        raise ValueError(f"pred_prob must be 1D or 2D, got {pred_prob.ndim}D")

    # Check for finite values
    if not np.all(np.isfinite(true_labs)):
        raise ValueError("true_labs contains NaN or infinite values")
    if not np.all(np.isfinite(pred_prob)):
        raise ValueError("pred_prob contains NaN or infinite values")

    # Validate true labels
    if require_binary:
        unique_labels = np.unique(true_labs)
        if len(unique_labels) == 0:
            raise ValueError("true_labs contains no values")
        if not np.array_equal(np.sort(unique_labels), [0, 1]):
            # Check if it's at least in {0, 1} (could be just 0s or just 1s)
            if not np.all(np.isin(unique_labels, [0, 1])):
                raise ValueError(
                    f"Binary labels must be from {{0, 1}}, got unique values: "
                    f"{unique_labels}"
                )
    else:
        # For multiclass, check labels are non-negative integers
        if not np.all(true_labs >= 0):
            raise ValueError("Labels must be non-negative")
        if not np.all(true_labs == true_labs.astype(int)):
            raise ValueError("Labels must be integers")

        # Check labels are consecutive starting from 0 for multiclass
        if pred_prob.ndim == 2:
            unique_labels = np.unique(true_labs)
            expected_labels = np.arange(len(unique_labels))
            if not np.array_equal(np.sort(unique_labels), expected_labels):
                raise ValueError(
                    f"Multiclass labels must be consecutive integers starting from 0. "
                    f"Got {unique_labels}, expected {expected_labels}"
                )

            # Check number of classes matches probability matrix
            n_classes = len(unique_labels)
            if pred_prob.shape[1] != n_classes:
                raise ValueError(
                    f"Number of probability columns ({pred_prob.shape[1]}) must match "
                    f"number of classes ({n_classes})"
                )

    # Validate probabilities
    if require_proba:
        if np.any(pred_prob < 0) or np.any(pred_prob > 1):
            prob_min, prob_max = np.min(pred_prob), np.max(pred_prob)
            raise ValueError(
                f"Probabilities must be in [0, 1], got range "
                f"[{prob_min:.6f}, {prob_max:.6f}]"
            )

        # For multiclass probabilities, optionally check if they sum to ~1
        if pred_prob.ndim == 2:
            row_sums = np.sum(pred_prob, axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-3):
                sum_min, sum_max = np.min(row_sums), np.max(row_sums)
                # Issue warning but don't fail - not all use cases require
                # normalized probabilities
                import warnings

                warnings.warn(
                    f"Multiclass probabilities don't sum to 1.0 (range: "
                    f"[{sum_min:.3f}, {sum_max:.3f}]). "
                    "This may indicate unnormalized scores rather than probabilities.",
                    UserWarning,
                    stacklevel=3,
                )

    # Validate sample weights
    validated_sample_weight = None
    if sample_weight is not None:
        validated_sample_weight = np.asarray(sample_weight)

        # Check dimensionality
        if validated_sample_weight.ndim != 1:
            raise ValueError(
                f"sample_weight must be 1D, got {validated_sample_weight.ndim}D"
            )

        # Check length
        if len(validated_sample_weight) != len(true_labs):
            raise ValueError(
                f"Length mismatch: sample_weight ({len(validated_sample_weight)}) vs "
                f"true_labs ({len(true_labs)})"
            )

        # Check for finite positive values
        if not np.all(np.isfinite(validated_sample_weight)):
            raise ValueError("sample_weight contains NaN or infinite values")
        if np.any(validated_sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")
        if np.sum(validated_sample_weight) == 0:
            raise ValueError("sample_weight cannot sum to zero")

    return true_labs, pred_prob, validated_sample_weight


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
    threshold = np.asarray(threshold)

    # Check for finite values
    if not np.all(np.isfinite(threshold)):
        raise ValueError("threshold contains NaN or infinite values")

    # Check range [0, 1]
    if np.any(threshold < 0) or np.any(threshold > 1):
        thresh_min, thresh_max = np.min(threshold), np.max(threshold)
        raise ValueError(
            f"threshold must be in [0, 1], got range "
            f"[{thresh_min:.6f}, {thresh_max:.6f}]"
        )

    # Check dimensionality and length for multiclass
    if n_classes is not None:
        if threshold.ndim != 1:
            raise ValueError(f"multiclass threshold must be 1D, got {threshold.ndim}D")
        if len(threshold) != n_classes:
            raise ValueError(
                f"threshold length ({len(threshold)}) must match number of "
                f"classes ({n_classes})"
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
    valid_averages = {"macro", "micro", "weighted", "none"}
    if average not in valid_averages:
        raise ValueError(
            f"Invalid averaging method '{average}'. Must be one of: {valid_averages}"
        )


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
    valid_methods = {
        "auto",
        "smart_brute",
        "sort_scan",
        "minimize",
        "gradient",
        "coord_ascent",
        "dinkelbach",
    }
    if method not in valid_methods:
        raise ValueError(
            f"Invalid optimization method '{method}'. Must be one of: {valid_methods}"
        )


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
    valid_operators = {">", ">="}
    if comparison not in valid_operators:
        raise ValueError(
            f"Invalid comparison operator '{comparison}'. Must be one of: "
            f"{valid_operators}"
        )
