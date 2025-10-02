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

from .validation import validate_binary_classification

Array = np.ndarray[Any, Any]

# Numerical tolerance for floating-point comparisons
NUMERICAL_TOLERANCE = 1e-12


def _validate_inputs(
    y_true: Array, pred_prob: Array, require_proba: bool = True
) -> tuple[Array, Array]:
    """Validate and convert inputs for binary classification.

    This is a thin wrapper around the centralized validation for compatibility.
    """
    validated_labels, validated_probs, _ = validate_binary_classification(
        true_labs=y_true,
        pred_prob=pred_prob,
        require_proba=require_proba,
        force_dtypes=True,  # Maintain int8/float64 for performance
    )
    return validated_labels, validated_probs


def _validate_sample_weights(sample_weight: Array | None, n_samples: int) -> Array:
    """Validate and convert sample weights.

    This is a thin wrapper around the centralized validation for compatibility.
    """
    _, _, validated_weights = validate_binary_classification(
        true_labs=np.zeros(n_samples),  # Dummy labels for weight validation
        pred_prob=np.zeros(n_samples),  # Dummy probabilities for weight validation
        sample_weight=sample_weight,
        return_default_weights=True,  # Return ones array when None
        require_proba=False,  # Skip prob validation for dummy data
    )

    # validated_weights is guaranteed to be non-None due to return_default_weights=True
    return validated_weights


def _vectorized_counts(
    y_sorted: Array, weights_sorted: Array
) -> tuple[Array, Array, Array, Array]:
    """Compute confusion matrix counts for all possible cuts using cumulative sums.

    Given labels and weights sorted in the same order as descending probabilities,
    returns (tp, tn, fp, fn) as vectors for every cut k.

    The indexing convention is:
    - Index 0: "predict nothing as positive" (all negative predictions)
    - Index k (k > 0): predict first k items as positive, rest as negative

    At cut index k:
      tp[k] = sum of weights for positive labels in first k items
      fp[k] = sum of weights for negative labels in first k items
      fn[k] = P - tp[k] (remaining positive weight)
      tn[k] = N - fp[k] (remaining negative weight)

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
        Arrays of (tp, tn, fp, fn) counts for each cut position. Length is n+1
        where n is the number of samples, with index 0 being "predict nothing".
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
    p_sorted: Array, k_star: int, inclusive: bool = False, require_proba: bool = True
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
        Probabilities or scores sorted in descending order.
    k_star : int
        Optimal cut position with new indexing.
    inclusive : bool
        Comparison operator: False for ">" (exclusive), True for ">=" (inclusive).
    require_proba : bool, default=True
        If True, clamp threshold to [0, 1]. If False, allow arbitrary ranges.

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
        if not inclusive:  # exclusive ">"
            # For '>', we need all p > threshold to be false, so threshold >= max_prob
            threshold = max_prob
        else:  # inclusive ">="
            # For '>=', we need all p >= threshold to be false, so threshold > max_prob
            threshold = float(np.nextafter(max_prob, np.inf))

    # Special case: predict all as positive (k_star == n)
    elif k_star == n:
        # Threshold should be set so ALL probabilities pass the comparison
        min_prob = float(p_sorted[-1])
        if not inclusive:  # exclusive ">"
            # For '>', we need all p > threshold to be true, so threshold < min_prob
            threshold = float(np.nextafter(min_prob, -np.inf))
        else:  # inclusive ">="
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
            if inclusive:
                threshold = float(np.nextafter(threshold, -np.inf))
        else:
            # Edge case: adjacent probabilities are tied or very close
            # (abs(included_prob - excluded_prob) <= NUMERICAL_TOLERANCE)
            # When probabilities are tied, we cannot cleanly separate the included
            # vs excluded items with a single threshold. We use a heuristic based
            # on the comparison operator to decide whether to include or exclude
            # all tied values.
            tied_prob = excluded_prob

            if not inclusive:  # exclusive ">"
                # For '>', set threshold slightly above tied_prob
                # This means tied_prob > threshold is false, excluding tied values
                threshold = float(np.nextafter(tied_prob, np.inf))
            else:  # inclusive ">="
                # For '>=', set threshold slightly below tied_prob
                # This means tied_prob >= threshold is true, including tied values
                threshold = float(np.nextafter(tied_prob, -np.inf))

    # Don't clamp here - let the main algorithm handle clamping after checking
    # for discrepancies
    return threshold


def _realized_k(p_sorted: Array, threshold: float, inclusive: bool) -> int:
    """Compute the realized cut index from final threshold.

    Given a threshold and comparison mode, compute how many samples would actually
    be predicted positive based on the sorted probabilities.

    Parameters
    ----------
    p_sorted : Array
        Probabilities or scores sorted in descending order.
    threshold : float
        Final threshold value.
    inclusive : bool
        Whether comparison is inclusive (>=) or exclusive (>).

    Returns
    -------
    int
        Number of samples that would be predicted positive (realized k).
    """
    # p_sorted is descending; use -p_sorted to search as ascending
    if inclusive:
        return int(np.searchsorted(-p_sorted, -threshold, side="right"))
    else:
        return int(np.searchsorted(-p_sorted, -threshold, side="left"))


def optimal_threshold_sortscan(
    y_true: Array,
    pred_prob: Array,
    metric_fn: Callable[[Array, Array, Array, Array], Array],
    *,
    sample_weight: Array | None = None,
    inclusive: bool = False,  # True for ">=" (inclusive), False for ">" (exclusive)
    require_proba: bool = True,  # True for probabilities [0,1], False for scores
) -> tuple[float, float, int]:
    """Exact optimizer for piecewise-constant metrics using O(n log n) sort-and-scan.

    This algorithm sorts predictions by descending score once, then uses
    cumulative sums to compute confusion matrix elements in O(1) for each of the n
    possible cuts, resulting in O(n log n) total time complexity.

    The threshold is returned as the midpoint between adjacent scores,
    which is more numerically stable than returning the exact score values.

    Parameters
    ----------
    y_true : Array
        True binary labels (0 or 1).
    pred_prob : Array
        Predicted probabilities in [0, 1] or arbitrary scores if require_proba=False.
    metric_fn : Callable
        Metric function that accepts (tp, tn, fp, fn) arrays and returns score array.
    sample_weight : Array, optional
        Sample weights for imbalanced datasets.
    inclusive : bool, default=False
        Comparison operator: False for ">" (exclusive), True for ">=" (inclusive).
    require_proba : bool, default=True
        If True, validate that pred_prob is in [0, 1] and clamp thresholds.
        If False, allow arbitrary score ranges (e.g., logits).

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
    y, p = _validate_inputs(y_true, pred_prob, require_proba=require_proba)
    weights = _validate_sample_weights(sample_weight, y.shape[0])

    n = y.shape[0]

    # Handle edge case: all same class
    if np.all(y == 0):  # All negatives - optimal threshold should predict all negative
        max_score = float(np.max(p))
        if require_proba:
            threshold = (
                max_score if not inclusive else float(np.nextafter(max_score, np.inf))
            )
        else:
            threshold = (
                max_score if not inclusive else float(np.nextafter(max_score, np.inf))
            )
        # Use weighted counts
        total_weight = float(np.sum(weights))
        score = float(
            metric_fn(
                np.array([0.0]),
                np.array([total_weight]),
                np.array([0.0]),
                np.array([0.0]),
            )[0]
        )
        return threshold, score, 0
    elif np.all(y == 1):  # All positives - optimal threshold predicts all positive
        min_score = float(np.min(p))
        if not inclusive:  # exclusive ">"
            # For exclusive comparison, need threshold < min_score
            threshold = float(np.nextafter(min_score, -np.inf))
            if require_proba:
                threshold = max(0.0, threshold)
        else:  # inclusive ">="
            # For inclusive comparison, threshold = min_score works
            threshold = min_score
        # Use weighted counts
        total_weight = float(np.sum(weights))
        score = float(
            metric_fn(
                np.array([total_weight]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
            )[0]
        )
        return threshold, score, n

    # Sort by descending probability (stable sort for reproducibility)
    sort_idx = np.argsort(-p, kind="mergesort")
    y_sorted = y[sort_idx]
    p_sorted = p[sort_idx]
    weights_sorted = weights[sort_idx]

    # Vectorized confusion matrix counts for all cuts
    tp_vec, tn_vec, fp_vec, fn_vec = _vectorized_counts(
        y_sorted, weights_sorted
    )  # type: tuple[Array, Array, Array, Array]

    # Vectorized metric computation over all cuts
    scores = _metric_from_counts(metric_fn, tp_vec, tn_vec, fp_vec, fn_vec)

    # Find optimal cut
    k_star = int(np.argmax(scores))
    best_score_theoretical = float(scores[k_star])

    # Compute stable threshold as midpoint
    threshold = _compute_threshold_midpoint(p_sorted, k_star, inclusive, require_proba)

    # For cases with tied probabilities, verify the achievable score
    if not inclusive:  # exclusive ">"
        pred_mask = p > threshold
    else:  # inclusive ">="
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

    # If there's a large discrepancy due to ties, use a targeted local fallback
    # Use a tolerance larger than numerical precision
    if abs(actual_score - best_score_theoretical) > max(
        1e-6, NUMERICAL_TOLERANCE * 100
    ):
        best_alt_score = actual_score
        best_alt_threshold = threshold
        fallback_candidates: list[float] = []

        # Extremes for "none"/"all" predictions
        min_s = float(p_sorted[-1])  # smallest score (last in descending order)
        max_s = float(p_sorted[0])  # largest score (first in descending order)
        if inclusive:
            fallback_candidates += [min_s, float(np.nextafter(max_s, np.inf))]
        else:
            fallback_candidates += [float(np.nextafter(min_s, -np.inf)), max_s]

        # Local nudges around the k_star boundary
        if 0 < k_star < n:
            inc = float(p_sorted[k_star - 1])  # last included score
            exc = float(p_sorted[k_star])  # first excluded score
            fallback_candidates += [
                float(np.nextafter(inc, -np.inf)),  # just below last included
                float(np.nextafter(exc, np.inf)),  # just above first excluded
            ]

        # Test all fallback candidates
        for alt_thresh in fallback_candidates:
            # Clamp to valid range if needed
            if require_proba:
                alt_thresh = max(0.0, min(1.0, alt_thresh))

            if not inclusive:  # exclusive ">"
                alt_pred_mask = p > alt_thresh
            else:  # inclusive ">="
                alt_pred_mask = p >= alt_thresh

            # Compute weighted confusion matrix
            alt_tp = float(np.sum(weights * (alt_pred_mask & (y == 1))))
            alt_tn = float(np.sum(weights * (~alt_pred_mask & (y == 0))))
            alt_fp = float(np.sum(weights * (alt_pred_mask & (y == 0))))
            alt_fn = float(np.sum(weights * (~alt_pred_mask & (y == 1))))

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

        # Apply final clamping for probability constraints and recalculate score
        # if needed
        if require_proba:
            original_threshold = best_alt_threshold
            best_alt_threshold = max(0.0, min(1.0, best_alt_threshold))

            # If threshold changed due to clamping, recalculate the score
            # Use a very small tolerance to catch even tiny threshold changes
            # that can affect scoring
            if abs(best_alt_threshold - original_threshold) > 0:
                if not inclusive:
                    final_pred_mask = p > best_alt_threshold
                else:
                    final_pred_mask = p >= best_alt_threshold

                final_tp = float(np.sum(weights * (final_pred_mask & (y == 1))))
                final_tn = float(np.sum(weights * (~final_pred_mask & (y == 0))))
                final_fp = float(np.sum(weights * (final_pred_mask & (y == 0))))
                final_fn = float(np.sum(weights * (~final_pred_mask & (y == 1))))

                best_alt_score = float(
                    metric_fn(
                        np.array([final_tp]),
                        np.array([final_tn]),
                        np.array([final_fp]),
                        np.array([final_fn]),
                    )[0]
                )

        k_real = _realized_k(p_sorted, best_alt_threshold, inclusive)
        return best_alt_threshold, best_alt_score, k_real

    # Apply final clamping for probability constraints and recalculate score if needed
    if require_proba:
        original_threshold = threshold
        threshold = max(0.0, min(1.0, threshold))

        # If threshold changed due to clamping, recalculate the score
        # Use a very small tolerance to catch even tiny threshold changes
        # that can affect scoring
        if abs(threshold - original_threshold) > 0:
            if not inclusive:
                final_pred_mask = p > threshold
            else:
                final_pred_mask = p >= threshold

            if sample_weight is not None:
                final_tp = float(np.sum(weights * (final_pred_mask & (y == 1))))
                final_tn = float(np.sum(weights * (~final_pred_mask & (y == 0))))
                final_fp = float(np.sum(weights * (final_pred_mask & (y == 0))))
                final_fn = float(np.sum(weights * (~final_pred_mask & (y == 1))))
            else:
                final_tp = float(np.sum(final_pred_mask & (y == 1)))
                final_tn = float(np.sum(~final_pred_mask & (y == 0)))
                final_fp = float(np.sum(final_pred_mask & (y == 0)))
                final_fn = float(np.sum(~final_pred_mask & (y == 1)))

            actual_score = float(
                metric_fn(
                    np.array([final_tp]),
                    np.array([final_tn]),
                    np.array([final_fp]),
                    np.array([final_fn]),
                )[0]
            )

    k_real = _realized_k(p_sorted, threshold, inclusive)
    return threshold, actual_score, k_real
