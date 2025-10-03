"""Expected metric optimization under calibration assumption.

This module implements expected threshold optimization using Dinkelbach's algorithm.
Under the calibration assumption, expected metrics can be optimized exactly by
finding optimal thresholds on calibrated probabilities.

Key simplifications:
- Single Dinkelbach implementation for all metrics
- Direct computation instead of coefficient abstraction
- Clear separation: core algorithm vs metrics vs multiclass
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

# ============================================================================
# Core Algorithm - Single Dinkelbach implementation
# ============================================================================


def dinkelbach_optimize(
    probabilities: np.ndarray[Any, Any],
    numerator_fn: callable,
    denominator_fn: callable,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Core Dinkelbach algorithm for ratio optimization.

    Solves: max_t numerator(t) / denominator(t)

    Parameters
    ----------
    probabilities : array of shape (n,)
        Calibrated probabilities
    numerator_fn : callable(threshold) -> float
        Computes numerator at given threshold
    denominator_fn : callable(threshold) -> float
        Computes denominator at given threshold
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    threshold : float
        Optimal threshold
    score : float
        Optimal ratio value
    """
    # Sort probabilities once
    p = np.sort(probabilities)

    # Initial lambda
    lam = 0.5

    for _ in range(max_iter):
        # Find threshold that maximizes: numerator - lambda * denominator
        def objective(t, lambda_val=lam):
            return numerator_fn(t) - lambda_val * denominator_fn(t)

        # Simple grid search over unique probabilities
        # (This is fast enough for most cases)
        candidates = np.unique(p)
        best_t = 0.5
        best_val = -np.inf

        for t in candidates:
            val = objective(t)
            if val > best_val:
                best_val = val
                best_t = t

        # Update lambda
        num = numerator_fn(best_t)
        den = denominator_fn(best_t)

        if den == 0:
            break

        new_lam = num / den

        if abs(new_lam - lam) < tol:
            return float(best_t), float(new_lam)

        lam = new_lam

    return float(best_t), float(lam)


# ============================================================================
# Metric-specific expected optimization
# ============================================================================


def dinkelbach_expected_fbeta_binary(
    y_prob: np.ndarray[Any, Any],
    beta: float = 1.0,
    sample_weight: np.ndarray[Any, Any] | None = None,
    comparison: str = ">",
) -> tuple[float, float]:
    """Expected F-beta optimization under calibration.

    Parameters
    ----------
    y_prob : array of shape (n,)
        Calibrated probabilities for positive class
    beta : float
        F-beta parameter
    sample_weight : array of shape (n,), optional
        Sample weights
    comparison : str
        Comparison operator (kept for backward compatibility)

    Returns
    -------
    threshold : float
        Optimal threshold
    score : float
        Expected F-beta score
    """
    p = np.asarray(y_prob, dtype=np.float64)

    if sample_weight is None:
        weights = np.ones_like(p)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)

    # Validate inputs
    if not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

    # Expected confusion matrix components as functions of threshold
    # Under calibration: E[TP|p>t] = sum(w_i * p_i) for i where p_i > t

    # Pre-compute for efficiency
    wp = weights * p  # weighted probabilities
    w1mp = weights * (1 - p)  # weighted (1-p)

    def expected_tp(t):
        mask = p > t
        return np.sum(wp[mask])

    def expected_fp(t):
        mask = p > t
        return np.sum(w1mp[mask])

    def expected_fn(t):
        mask = p <= t
        return np.sum(wp[mask])

    # F-beta numerator and denominator
    beta2 = beta**2

    def numerator(t):
        tp = expected_tp(t)
        return (1 + beta2) * tp

    def denominator(t):
        tp = expected_tp(t)
        fp = expected_fp(t)
        fn = expected_fn(t)
        return (1 + beta2) * tp + fp + beta2 * fn

    return dinkelbach_optimize(p, numerator, denominator)


def expected_precision(
    probabilities: np.ndarray[Any, Any], weights: np.ndarray[Any, Any] | None = None
) -> tuple[float, float]:
    """Expected precision optimization.

    Precision = TP / (TP + FP)
    """
    p = np.asarray(probabilities, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(p)

    wp = weights * p
    w1mp = weights * (1 - p)

    def numerator(t):
        mask = p > t
        return np.sum(wp[mask])  # Expected TP

    def denominator(t):
        mask = p > t
        return np.sum(wp[mask]) + np.sum(w1mp[mask])  # Expected TP + FP

    return dinkelbach_optimize(p, numerator, denominator)


def expected_jaccard(
    probabilities: np.ndarray[Any, Any], weights: np.ndarray[Any, Any] | None = None
) -> tuple[float, float]:
    """Expected Jaccard/IoU optimization.

    Jaccard = TP / (TP + FP + FN)
    """
    p = np.asarray(probabilities, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(p)

    wp = weights * p

    def numerator(t):
        mask = p > t
        return np.sum(wp[mask])  # Expected TP

    def denominator(t):
        # TP + FP + FN = all predicted positive + all actual positive - TP
        mask = p > t
        tp = np.sum(wp[mask])
        predicted_pos = np.sum(weights[mask])
        actual_pos = np.sum(wp)  # Total expected positives
        return predicted_pos + actual_pos - tp

    return dinkelbach_optimize(p, numerator, denominator)


# ============================================================================
# Multiclass/Multilabel wrapper
# ============================================================================


def dinkelbach_expected_fbeta_multilabel(
    y_prob: np.ndarray[Any, Any],
    beta: float = 1.0,
    sample_weight: np.ndarray[Any, Any] | None = None,
    average: Literal["macro", "micro", "weighted"] = "macro",
    comparison: str = ">",
) -> dict[str, Any]:
    """Expected F-beta optimization for multilabel/multiclass.

    Parameters
    ----------
    y_prob : array of shape (n_samples, n_classes)
        Class probabilities
    beta : float
        F-beta parameter
    sample_weight : array of shape (n_samples,), optional
        Sample weights
    average : str
        Averaging strategy:
        - "macro": Per-class thresholds, unweighted mean
        - "micro": Single global threshold
        - "weighted": Per-class thresholds, weighted mean
    comparison : str
        Comparison operator (kept for backward compatibility)

    Returns
    -------
    dict
        Results with 'thresholds' and 'score' keys
    """
    P = np.asarray(y_prob, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {P.shape}")

    n_samples, n_classes = P.shape

    if average == "micro":
        # Flatten all probabilities into single binary problem
        p_flat = P.ravel()

        if sample_weight is not None:
            # Repeat weights for each class
            w_flat = np.repeat(sample_weight, n_classes)
        else:
            w_flat = None

        threshold, score = dinkelbach_expected_fbeta_binary(p_flat, beta, w_flat)

        return {"threshold": threshold, "score": score}

    else:  # macro or weighted
        # Optimize per-class thresholds
        thresholds = np.zeros(n_classes)
        scores = np.zeros(n_classes)

        for k in range(n_classes):
            thresholds[k], scores[k] = dinkelbach_expected_fbeta_binary(
                P[:, k], beta, sample_weight
            )

        # Compute average score
        if average == "macro":
            avg_score = np.mean(scores)
        else:  # weighted
            # Weight by class frequency
            if sample_weight is not None:
                # This is a simplification - proper weighting would need true labels
                class_weights = np.sum(P * sample_weight[:, None], axis=0)
            else:
                class_weights = np.sum(P, axis=0)

            class_weights /= class_weights.sum()
            avg_score = np.average(scores, weights=class_weights)

        return {
            "thresholds": thresholds,
            "per_class": scores,
            "score": float(avg_score),
        }


def expected_optimize_multiclass(
    probabilities: np.ndarray[Any, Any],
    metric: str = "f1",
    average: Literal["macro", "micro", "weighted"] = "macro",
    weights: np.ndarray[Any, Any] | None = None,
    **metric_params,
) -> dict[str, Any]:
    """Expected optimization for multiclass/multilabel.

    Parameters
    ----------
    probabilities : array of shape (n_samples, n_classes)
        Class probabilities
    metric : str
        Metric to optimize ("f1", "precision", "jaccard")
    average : str
        Averaging strategy
    weights : array of shape (n_samples,), optional
        Sample weights
    **metric_params
        Additional parameters (e.g., beta for F-beta)

    Returns
    -------
    dict
        Results with 'thresholds' and 'score' keys
    """
    P = np.asarray(probabilities, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {P.shape}")

    n_samples, n_classes = P.shape

    # Select metric function
    if metric.lower() in {"f1", "fbeta"}:
        beta = metric_params.get("beta", 1.0)

        def metric_fn(p, w):
            return dinkelbach_expected_fbeta_binary(p, beta, w)
    elif metric.lower() == "precision":
        metric_fn = expected_precision
    elif metric.lower() in {"jaccard", "iou"}:
        metric_fn = expected_jaccard
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Handle different averaging strategies
    if average == "micro":
        # Flatten all probabilities into single binary problem
        p_flat = P.ravel()

        if weights is not None:
            # Repeat weights for each class
            w_flat = np.repeat(weights, n_classes)
        else:
            w_flat = None

        threshold, score = metric_fn(p_flat, w_flat)

        return {"threshold": threshold, "score": score}

    else:  # macro or weighted
        # Optimize per-class thresholds
        thresholds = np.zeros(n_classes)
        scores = np.zeros(n_classes)

        for k in range(n_classes):
            thresholds[k], scores[k] = metric_fn(P[:, k], weights)

        # Compute average score
        if average == "macro":
            avg_score = np.mean(scores)
        else:  # weighted
            # Weight by class frequency
            if weights is not None:
                class_weights = np.sum(P * weights[:, None], axis=0)
            else:
                class_weights = np.sum(P, axis=0)

            class_weights /= class_weights.sum()
            avg_score = np.average(scores, weights=class_weights)

        return {
            "thresholds": thresholds,
            "per_class": scores,
            "score": float(avg_score),
        }


# ============================================================================
# Simple API
# ============================================================================


def optimize_expected_threshold(
    probabilities: np.ndarray[Any, Any], metric: str = "f1", **kwargs
) -> float | np.ndarray[Any, Any]:
    """Simple API for expected threshold optimization.

    Parameters
    ----------
    probabilities : array
        Probabilities (1D for binary, 2D for multiclass)
    metric : str
        Metric to optimize
    **kwargs
        Additional parameters

    Returns
    -------
    float or array
        Optimal threshold(s)
    """
    p = np.asarray(probabilities)

    if p.ndim == 1:
        # Binary case
        if metric.lower() in {"f1", "fbeta"}:
            threshold, _ = dinkelbach_expected_fbeta_binary(
                p, beta=kwargs.get("beta", 1.0)
            )
        elif metric.lower() == "precision":
            threshold, _ = expected_precision(p)
        elif metric.lower() in {"jaccard", "iou"}:
            threshold, _ = expected_jaccard(p)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return threshold

    else:
        # Multiclass case
        result = expected_optimize_multiclass(p, metric, **kwargs)

        if "threshold" in result:
            return result["threshold"]
        else:
            return result["thresholds"]
