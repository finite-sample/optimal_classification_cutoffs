"""Optimized O(n log n) sort-and-scan kernel for piecewise-constant metrics.

This module provides an exact optimizer for binary classification metrics that are
piecewise-constant with respect to the decision threshold. The algorithm sorts
predictions once and scans all n cuts in a single pass, achieving true O(n log n)
complexity with vectorized operations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]

# Numerical tolerance for floating-point comparisons
NUMERICAL_TOLERANCE = 1e-12


def _validate_inputs(y_true: Array, pred_prob: Array) -> tuple[Array, Array]:
    """Validate and convert inputs for binary classification.

    Parameters
    ----------
    y_true : Array
        True binary labels.
    pred_prob : Array
        Predicted probabilities.

    Returns
    -------
    Tuple[Array, Array]
        Validated and converted (y_true, pred_prob) arrays.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    y = np.asarray(y_true)
    p = np.asarray(pred_prob)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y_true and pred_prob must be 1D arrays.")

    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and pred_prob must be the same length.")

    if y.shape[0] == 0:
        raise ValueError("y_true and pred_prob cannot be empty.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary in {0,1}.")

    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pred_prob must be finite and in [0, 1].")

    return y.astype(np.int8, copy=False), p.astype(np.float64, copy=False)


def _validate_sample_weights(sample_weight: Array | None, n_samples: int) -> Array:
    """Validate and convert sample weights.

    Parameters
    ----------
    sample_weight : Array or None
        Sample weights.
    n_samples : int
        Number of samples expected.

    Returns
    -------
    Array
        Validated sample weights array.

    Raises
    ------
    ValueError
        If sample weights are invalid.
    """
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)

    w = np.asarray(sample_weight, dtype=np.float64)

    if w.ndim != 1:
        raise ValueError("sample_weight must be 1D array.")

    if w.shape[0] != n_samples:
        raise ValueError(
            f"sample_weight length ({w.shape[0]}) must match number of "
            f"samples ({n_samples})."
        )

    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("sample_weight must be finite and non-negative.")

    return w


def _vectorized_counts(
    y_sorted: Array, weights_sorted: Array
) -> tuple[Array, Array, Array, Array]:
    """Compute confusion matrix counts for all possible cuts using cumulative sums.

    Given labels and weights sorted in the same order as descending probabilities,
    returns (tp, tn, fp, fn) as vectors for every cut k, including the
    "predict nothing" case.

    At cut k (include indices 0..k as predicted positive):
      tp[k] = sum_{j<=k} weights_sorted[j] * y_sorted[j]
      fp[k] = sum_{j<=k} weights_sorted[j] * (1 - y_sorted[j])
      fn[k] = P - tp[k]
      tn[k] = N - fp[k]

    The first element (index 0) represents the "predict nothing as positive" case.
    Subsequent elements represent cuts after items 0, 1, 2, ..., n-1.

    Where P = total positive weight, N = total negative weight.

    Parameters
    ----------
    y_sorted : Array
        Binary labels sorted by descending probability.
    weights_sorted : Array
        Sample weights sorted by descending probability.

    Returns
    -------
    Tuple[Array, Array, Array, Array]
        Arrays of (tp, tn, fp, fn) counts for each cut position, with the first
        element being the "predict nothing" case.
    """

    # Compute total positive and negative weights
    P = float(np.sum(weights_sorted * y_sorted))
    N = float(np.sum(weights_sorted * (1 - y_sorted)))

    # Cumulative weighted counts for cuts after each item
    tp_cumsum = np.cumsum(weights_sorted * y_sorted)
    fp_cumsum = np.cumsum(weights_sorted * (1 - y_sorted))

    # Include "predict nothing" case at the beginning
    tp = np.concatenate([[0.0], tp_cumsum])
    fp = np.concatenate([[0.0], fp_cumsum])

    # Complement counts
    fn = P - tp
    tn = N - fp

    return tp, tn, fp, fn


def _metric_from_counts(
    metric_fn: Callable[[Array, Array, Array, Array], Array],
    tp: Array,
    tn: Array,
    fp: Array,
    fn: Array,
) -> Array:
    """Apply metric function to vectorized confusion matrix counts.

    Parameters
    ----------
    metric_fn : Callable
        Metric function that accepts (tp, tn, fp, fn) as arrays and returns
        array of scores.
    tp, tn, fp, fn : Array
        Confusion matrix count arrays.

    Returns
    -------
    Array
        Array of metric scores for each threshold.

    Raises
    ------
    ValueError
        If metric function doesn't return array with correct shape.
    """
    scores = metric_fn(tp, tn, fp, fn)

    # Ensure scores is a numpy array
    scores = np.asarray(scores)

    if scores.shape != tp.shape:
        raise ValueError(
            f"metric_fn must return array with shape {tp.shape}, got {scores.shape}."
        )

    return scores


def _compute_threshold_midpoint(
    p_sorted: Array, k_star: int, inclusive: str = ">"
) -> float:
    """Compute threshold as midpoint between adjacent probabilities.

    With the new indexing where k=0 means "predict nothing as positive":
    - k=0: Predict nothing as positive (threshold > max probability)
    - k=1: Predict item 0 as positive, others negative
    - k=2: Predict items 0,1 as positive, others negative
    - k=n: Predict all items as positive (threshold <= min probability)

    Parameters
    ----------
    p_sorted : Array
        Probabilities sorted in descending order.
    k_star : int
        Optimal cut position with new indexing.
    inclusive : str
        Comparison operator: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    float
        Threshold value as midpoint or epsilon-adjusted value.
    """
    n = p_sorted.size

    # Special case: predict nothing as positive (k_star == 0)
    if k_star == 0:
        # Threshold should be set so NO probabilities pass the comparison
        max_prob = float(p_sorted[0])
        if inclusive == ">":
            # For '>', we need all p > threshold to be false, so threshold >= max_prob
            threshold = max_prob
        else:  # ">="
            # For '>=', we need all p >= threshold to be false, so threshold > max_prob
            threshold = float(np.nextafter(max_prob, np.inf))

    # Special case: predict all as positive (k_star == n)
    elif k_star == n:
        # Threshold should be set so ALL probabilities pass the comparison
        min_prob = float(p_sorted[-1])
        if inclusive == ">":
            # For '>', we need all p > threshold to be true, so threshold < min_prob
            threshold = float(np.nextafter(min_prob, -np.inf))
        else:  # ">="
            # For '>=', we need all p >= threshold to be true, so threshold <= min_prob
            threshold = min_prob
    else:
        # Normal case: k_star corresponds to including items 0..k_star-1 as positive
        # Find the probability range we need to separate
        included_prob = float(
            p_sorted[k_star - 1]
        )  # Last prob included in positive predictions
        excluded_prob = float(
            p_sorted[k_star]
        )  # First prob excluded from positive predictions

        if included_prob - excluded_prob > NUMERICAL_TOLERANCE:
            # Normal case: probabilities are sufficiently different
            # Use midpoint between them
            threshold = 0.5 * (included_prob + excluded_prob)

            # For inclusive comparison, nudge slightly lower to ensure proper
            # comparison behavior
            if inclusive == ">=":
                threshold = float(np.nextafter(threshold, -np.inf))
        else:
            # Edge case: adjacent probabilities are tied or very close
            # (abs(included_prob - excluded_prob) <= NUMERICAL_TOLERANCE)
            # When probabilities are tied, we cannot cleanly separate the included
            # vs excluded items with a single threshold. We use a heuristic based
            # on the comparison operator to decide whether to include or exclude
            # all tied values.
            tied_prob = excluded_prob

            if inclusive == ">":
                # For '>', set threshold slightly above tied_prob
                # This means tied_prob > threshold is false, excluding tied values
                threshold = float(np.nextafter(tied_prob, np.inf))
            else:  # inclusive == ">="
                # For '>=', set threshold slightly below tied_prob
                # This means tied_prob >= threshold is true, including tied values
                threshold = float(np.nextafter(tied_prob, -np.inf))

    # Ensure threshold stays within valid bounds [0, 1]
    return max(0.0, min(1.0, threshold))


def optimal_threshold_sortscan(
    y_true: Array,
    pred_prob: Array,
    metric_fn: Callable[[Array, Array, Array, Array], Array],
    *,
    sample_weight: Array | None = None,
    inclusive: str = ">",  # ">" or ">="
) -> tuple[float, float, int]:
    """Exact optimizer for piecewise-constant metrics using O(n log n) sort-and-scan.

    This algorithm sorts predictions by descending probability once, then uses
    cumulative sums to compute confusion matrix elements in O(1) for each of the n
    possible cuts, resulting in O(n log n) total time complexity.

    The threshold is returned as the midpoint between adjacent probabilities,
    which is more numerically stable than returning the exact probability values.

    Parameters
    ----------
    y_true : Array
        True binary labels (0 or 1).
    pred_prob : Array
        Predicted probabilities in [0, 1].
    metric_fn : Callable
        Metric function that accepts (tp, tn, fp, fn) arrays and returns score array.
    sample_weight : Array, optional
        Sample weights for imbalanced datasets.
    inclusive : str, default=">"
        Comparison operator: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    Tuple[float, float, int]
        - threshold: Optimal decision threshold
        - best_score: Best metric score achieved
        - k_star: Optimal cut position in sorted array

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> def f1_vectorized(tp, tn, fp, fn):
    ...     precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    ...     recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    ...     return np.where(precision + recall > 0,
    ...                    2 * precision * recall / (precision + recall), 0.0)
    >>> y_true = [0, 0, 1, 1]
    >>> pred_prob = [0.1, 0.4, 0.6, 0.9]
    >>> threshold, score, k = optimal_threshold_sortscan(
    ...     y_true, pred_prob, f1_vectorized
    ... )
    >>> print(f"Optimal threshold: {threshold:.3f}, F1 score: {score:.3f}")
    """
    # Validate inputs
    y, p = _validate_inputs(y_true, pred_prob)
    weights = _validate_sample_weights(sample_weight, y.shape[0])

    n = y.shape[0]

    # Handle edge case: single sample
    if n == 1:
        # For single sample, threshold doesn't matter much, use probability value
        score = float(
            metric_fn(
                np.array([y[0]]), np.array([1 - y[0]]), np.array([0]), np.array([0])
            )[0]
        )
        return float(p[0]), score, 0

    # Handle edge case: all same class
    if np.all(y == 0) or np.all(y == 1):
        # Return default threshold with whatever score we get
        score = float(
            metric_fn(
                np.array([y.sum()]),
                np.array([n - y.sum()]),
                np.array([0]),
                np.array([0]),
            )[0]
        )
        return 0.5, score, 0

    # Sort by descending probability (stable sort for reproducibility)
    sort_idx = np.argsort(-p, kind="mergesort")
    y_sorted = y[sort_idx]
    p_sorted = p[sort_idx]
    weights_sorted = weights[sort_idx]

    # Vectorized confusion matrix counts for all cuts
    tp, tn, fp, fn = _vectorized_counts(y_sorted, weights_sorted)

    # Vectorized metric computation over all cuts
    scores = _metric_from_counts(metric_fn, tp, tn, fp, fn)

    # Find optimal cut
    k_star = int(np.argmax(scores))
    best_score_theoretical = float(scores[k_star])

    # Compute stable threshold as midpoint
    threshold = _compute_threshold_midpoint(p_sorted, k_star, inclusive)

    # For cases with tied probabilities, verify the achievable score
    if inclusive == ">":
        pred_mask = p > threshold
    else:  # inclusive == ">="
        pred_mask = p >= threshold

    # Compute actual confusion matrix with this threshold
    if sample_weight is not None:
        tp_actual = float(np.sum(weights * (pred_mask & (y == 1))))
        tn_actual = float(np.sum(weights * (~pred_mask & (y == 0))))
        fp_actual = float(np.sum(weights * (pred_mask & (y == 0))))
        fn_actual = float(np.sum(weights * (~pred_mask & (y == 1))))
    else:
        tp_actual = float(np.sum(pred_mask & (y == 1)))
        tn_actual = float(np.sum(~pred_mask & (y == 0)))
        fp_actual = float(np.sum(pred_mask & (y == 0)))
        fn_actual = float(np.sum(~pred_mask & (y == 1)))

    # Compute actual achievable score
    actual_score = float(
        metric_fn(
            np.array([tp_actual]),
            np.array([tn_actual]),
            np.array([fp_actual]),
            np.array([fn_actual]),
        )[0]
    )

    # If there's a large discrepancy due to ties, adjust the threshold
    # Use a tolerance larger than numerical precision
    if abs(actual_score - best_score_theoretical) > max(
        1e-6, NUMERICAL_TOLERANCE * 100
    ):
        # Try alternative threshold strategies for better tie handling
        alternative_thresholds = []

        # Try thresholds that are slightly above and below tied probability values
        unique_probs = np.unique(p)
        for prob in unique_probs:
            if 0.0 < prob < 1.0:  # Valid threshold range
                alternative_thresholds.extend(
                    [
                        max(0.0, min(1.0, float(np.nextafter(prob, -np.inf)))),
                        max(0.0, min(1.0, float(np.nextafter(prob, np.inf)))),
                    ]
                )

        # Evaluate alternatives and pick the best
        best_alt_score = actual_score
        best_alt_threshold = threshold

        for alt_thresh in alternative_thresholds:
            if inclusive == ">":
                alt_pred_mask = p > alt_thresh
            else:
                alt_pred_mask = p >= alt_thresh

            if sample_weight is not None:
                alt_tp = float(np.sum(weights * (alt_pred_mask & (y == 1))))
                alt_tn = float(np.sum(weights * (~alt_pred_mask & (y == 0))))
                alt_fp = float(np.sum(weights * (alt_pred_mask & (y == 0))))
                alt_fn = float(np.sum(weights * (~alt_pred_mask & (y == 1))))
            else:
                alt_tp = float(np.sum(alt_pred_mask & (y == 1)))
                alt_tn = float(np.sum(~alt_pred_mask & (y == 0)))
                alt_fp = float(np.sum(alt_pred_mask & (y == 0)))
                alt_fn = float(np.sum(~alt_pred_mask & (y == 1)))

            alt_score = float(
                metric_fn(
                    np.array([alt_tp]),
                    np.array([alt_tn]),
                    np.array([alt_fp]),
                    np.array([alt_fn]),
                )[0]
            )

            if alt_score > best_alt_score:
                best_alt_score = alt_score
                best_alt_threshold = alt_thresh

        return best_alt_threshold, best_alt_score, k_star

    return threshold, actual_score, k_star


# Vectorized metric functions for common metrics
def f1_vectorized(tp: Array, tn: Array, fp: Array, fn: Array) -> Array:
    """Vectorized F1 score computation."""
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    return np.where(
        precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
    )


def accuracy_vectorized(tp: Array, tn: Array, fp: Array, fn: Array) -> Array:
    """Vectorized accuracy computation."""
    total = tp + tn + fp + fn
    return np.where(total > 0, (tp + tn) / total, 0.0)


def precision_vectorized(tp: Array, tn: Array, fp: Array, fn: Array) -> Array:
    """Vectorized precision computation."""
    return np.where(tp + fp > 0, tp / (tp + fp), 0.0)


def recall_vectorized(tp: Array, tn: Array, fp: Array, fn: Array) -> Array:
    """Vectorized recall computation."""
    return np.where(tp + fn > 0, tp / (tp + fn), 0.0)


# Mapping from metric names to vectorized functions
VECTORIZED_METRICS = {
    "f1": f1_vectorized,
    "accuracy": accuracy_vectorized,
    "precision": precision_vectorized,
    "recall": recall_vectorized,
}


def get_vectorized_metric(
    metric_name: str,
) -> Callable[[Array, Array, Array, Array], Array]:
    """Get vectorized version of a metric function.

    Parameters
    ----------
    metric_name : str
        Name of the metric.

    Returns
    -------
    Callable
        Vectorized metric function.

    Raises
    ------
    ValueError
        If metric is not available in vectorized form.
    """
    if metric_name not in VECTORIZED_METRICS:
        raise ValueError(
            f"Vectorized implementation not available for metric '{metric_name}'. "
            f"Available: {list(VECTORIZED_METRICS.keys())}"
        )
    return VECTORIZED_METRICS[metric_name]
