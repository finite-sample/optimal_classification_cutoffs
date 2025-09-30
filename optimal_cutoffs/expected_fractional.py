"""Generalized Dinkelbach method for fractional-linear expected metrics.

This module implements a general framework for optimizing any metric that can be
expressed as a ratio of two affine (linear + constant) functions of confusion matrix
counts under the calibration assumption. This includes F-beta, precision, recall,
Jaccard/IoU, Tversky index, accuracy, and specificity.

The key insight is that under calibration, the expected metric optimization decomposes
into finding a single probability threshold, which can be solved efficiently using
Dinkelbach's algorithm in O(n log n) time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class FractionalLinearCoeffs:
    """Coefficients for a fractional-linear metric of the form numerator/denominator.

    The metric is expressed as:
    M = (alpha_0 + alpha_tp*TP + alpha_tn*TN + alpha_fp*FP + alpha_fn*FN) /
        (beta_0 + beta_tp*TP + beta_tn*TN + beta_fp*FP + beta_fn*FN)

    where TP, TN, FP, FN are the confusion matrix counts.
    """

    # Numerator coefficients: alpha_0 + alpha⋅[TP,TN,FP,FN]
    alpha_tp: float = 0.0
    alpha_tn: float = 0.0
    alpha_fp: float = 0.0
    alpha_fn: float = 0.0
    alpha0: float = 0.0

    # Denominator coefficients: beta_0 + beta⋅[TP,TN,FP,FN] (must be > 0)
    beta_tp: float = 0.0
    beta_tn: float = 0.0
    beta_fp: float = 0.0
    beta_fn: float = 0.0
    beta0: float = 0.0


def coeffs_for_metric(
    metric: str,
    *,
    beta: float = 1.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
) -> FractionalLinearCoeffs:
    """Return FractionalLinearCoeffs for common ratio-of-affine metrics.

    All metrics are for the 'positive' class; use One-vs-Rest for multiclass/multilabel.

    Parameters
    ----------
    metric : str
        Name of the metric. Supported: 'precision', 'recall', 'specificity',
        'jaccard', 'iou', 'fbeta', 'f1', 'f2', 'tversky', 'accuracy'.
    beta : float, default=1.0
        Beta parameter for F-beta score. beta=1 gives F1, beta<1 emphasizes precision,
        beta>1 emphasizes recall.
    tversky_alpha : float, default=0.5
        Alpha parameter for Tversky index controlling FP penalty.
    tversky_beta : float, default=0.5
        Beta parameter for Tversky index controlling FN penalty.

    Returns
    -------
    FractionalLinearCoeffs
        Coefficients defining the metric as a ratio of affine functions.

    Raises
    ------
    ValueError
        If the metric is not supported or parameters are invalid.

    Examples
    --------
    >>> coeffs = coeffs_for_metric("precision")
    >>> print(f"Precision = {coeffs.alpha_tp}*TP / "
    ...       f"({coeffs.beta_tp}*TP + {coeffs.beta_fp}*FP)")
    Precision = 1.0*TP / (1.0*TP + 1.0*FP)

    >>> coeffs = coeffs_for_metric("f1")
    >>> # F1 = 2*TP / (2*TP + FP + FN)
    """
    m = metric.lower()

    if m in {"precision", "ppv", "positive_predictive_value"}:
        # Precision = TP / (TP + FP)
        return FractionalLinearCoeffs(alpha_tp=1.0, beta_tp=1.0, beta_fp=1.0)

    if m in {"recall", "sensitivity", "tpr", "true_positive_rate"}:
        # Recall = TP / (TP + FN)
        # Note: denominator is constant across selections, making this trivial
        return FractionalLinearCoeffs(alpha_tp=1.0, beta_tp=1.0, beta_fn=1.0)

    if m in {"specificity", "tnr", "true_negative_rate"}:
        # Specificity = TN / (TN + FP)
        return FractionalLinearCoeffs(alpha_tn=1.0, beta_tn=1.0, beta_fp=1.0)

    if m in {"jaccard", "iou", "intersection_over_union"}:
        # Jaccard = TP / (TP + FP + FN)
        return FractionalLinearCoeffs(
            alpha_tp=1.0, beta_tp=1.0, beta_fp=1.0, beta_fn=1.0
        )

    if m in {"fbeta", "f_beta"}:
        # F_beta = (1+beta^2)*TP / ((1+beta^2)*TP + beta^2*FN + FP)
        if beta < 0:
            raise ValueError(f"Beta parameter must be non-negative, got {beta}")
        b2 = float(beta) ** 2
        return FractionalLinearCoeffs(
            alpha_tp=1.0 + b2, beta_tp=1.0 + b2, beta_fn=b2, beta_fp=1.0
        )

    if m in {"f1"}:
        # F1 = 2*TP / (2*TP + FN + FP)
        return FractionalLinearCoeffs(
            alpha_tp=2.0, beta_tp=2.0, beta_fn=1.0, beta_fp=1.0
        )

    if m in {"f2"}:
        # F2 = 5*TP / (5*TP + 4*FN + FP)
        return FractionalLinearCoeffs(
            alpha_tp=5.0, beta_tp=5.0, beta_fn=4.0, beta_fp=1.0
        )

    if m in {"tversky"}:
        # Tversky(alpha,beta) = TP / (TP + alpha*FP + beta*FN)
        if tversky_alpha < 0 or tversky_beta < 0:
            raise ValueError(
                f"Tversky parameters must be non-negative, got "
                f"alpha={tversky_alpha}, beta={tversky_beta}"
            )
        return FractionalLinearCoeffs(
            alpha_tp=1.0,
            beta_tp=1.0,
            beta_fp=float(tversky_alpha),
            beta_fn=float(tversky_beta),
        )

    if m in {"accuracy", "acc"}:
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        # Note: denominator is constant across selections for binary classification
        return FractionalLinearCoeffs(
            alpha_tp=1.0,
            alpha_tn=1.0,
            beta_tp=1.0,
            beta_tn=1.0,
            beta_fp=1.0,
            beta_fn=1.0,
        )

    raise ValueError(
        f"Metric '{metric}' not supported as fractional-linear. "
        f"Supported: precision, recall, specificity, jaccard, iou, fbeta, f1, f2, "
        f"tversky, accuracy."
    )


def dinkelbach_expected_fractional_binary(
    y_prob: np.ndarray[Any, Any],
    coeffs: FractionalLinearCoeffs,
    sample_weight: np.ndarray[Any, Any] | None = None,
    comparison: Literal[">", ">="] = ">",
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, float, Literal[">", "<"]]:
    """General expected fractional-linear optimization (binary, calibrated).

    Solves the optimization problem:
    max_{S} (alpha_0 + sum_{i in S} alpha_i * w_i * expected_confusion_i) /
            (beta_0 + sum_{i in S} beta_i * w_i * expected_confusion_i)

    where S is the set of samples predicted positive, and expected confusion counts
    are computed under the calibration assumption: E[TP] = sum(w_i * p_i) for i in S.

    Parameters
    ----------
    y_prob : ndarray of shape (n_samples,)
        Calibrated probabilities for the positive class in [0, 1].
    coeffs : FractionalLinearCoeffs
        Coefficients defining the fractional-linear metric.
    sample_weight : ndarray of shape (n_samples,), optional
        Non-negative sample weights. If None, uses uniform weights.
    comparison : {">", ">="}, default=">"
        Tie-breaking convention for threshold comparison.
    max_iter : int, default=100
        Maximum number of Dinkelbach iterations.
    tol : float, default=1e-12
        Convergence tolerance for Dinkelbach algorithm.

    Returns
    -------
    t_star : float
        Optimal probability threshold in [0, 1].
    m_star : float
        Optimal expected metric value.
    direction : {">", "<"}
        Use '>' if predict positive when p > t_star, else '<' if predict positive
        when p < t_star (rare but possible for exotic coefficients).

    Raises
    ------
    ValueError
        If probabilities are outside [0,1] or sample weights are invalid.

    Examples
    --------
    >>> # Expected precision optimization
    >>> p = np.array([0.1, 0.3, 0.7, 0.9])
    >>> coeffs = coeffs_for_metric("precision")
    >>> t_star, prec_star, direction = dinkelbach_expected_fractional_binary(p, coeffs)
    >>> print(f"Optimal threshold: {t_star:.3f}, Expected precision: {prec_star:.3f}")

    Notes
    -----
    This method assumes perfect calibration: p_i = P(y_i = 1 | p_i).
    The algorithm typically converges in fewer than 10 iterations.
    For most common metrics, direction will be ">".
    """
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be in [0, 1].")

    if len(p) == 0:
        return 0.0, 0.0, ">"

    # Handle sample weights
    w = (
        np.ones_like(p, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    if w.shape != p.shape:
        raise ValueError("sample_weight must have shape (n_samples,).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")

    # Precompute totals and sort once (ascending p)
    order = np.argsort(p, kind="mergesort")
    p_s = p[order]
    w_s = w[order]
    wp_s = w_s * p_s  # w * p
    w1m_s = w_s - wp_s  # w * (1 - p)

    # Cumulative sums for efficient range queries
    csum_wp = np.cumsum(wp_s)  # prefix sums of w*p
    csum_w1m = np.cumsum(w1m_s)  # prefix sums of w*(1-p)
    csum_w = np.cumsum(w_s)  # prefix sums of w

    WP_tot = float(csum_wp[-1])  # total w*p
    W1M_tot = float(csum_w1m[-1])  # total w*(1-p)
    W_tot = float(csum_w[-1])  # total w

    # Coefficient differences for optimization
    a_tp, a_tn, a_fp, a_fn, a0 = (
        coeffs.alpha_tp,
        coeffs.alpha_tn,
        coeffs.alpha_fp,
        coeffs.alpha_fn,
        coeffs.alpha0,
    )
    b_tp, b_tn, b_fp, b_fn, b0 = (
        coeffs.beta_tp,
        coeffs.beta_tn,
        coeffs.beta_fp,
        coeffs.beta_fn,
        coeffs.beta0,
    )

    # Terms that multiply expected counts in decision set S
    ap = a_tp - a_fn  # coefficient for w*p when moving to S
    an = a_fp - a_tn  # coefficient for w*(1-p) when moving to S
    bp = b_tp - b_fn  # coefficient for w*p in denominator
    bn = b_fp - b_tn  # coefficient for w*(1-p) in denominator

    # Constant parts (baseline: predict all negative)
    NUM_const = a0 + a_fn * WP_tot + a_tn * W1M_tot
    DEN_const = b0 + b_fn * WP_tot + b_tn * W1M_tot

    def sums_for_threshold(
        t: float, direction: Literal[">", "<"]
    ) -> tuple[float, float, float]:
        """Compute sums over decision set S defined by threshold t and direction."""
        if direction == ">":
            # S = {i : p_i > t} or {i : p_i >= t} based on comparison
            idx = int(
                np.searchsorted(p_s, t, side="right" if comparison == ">" else "left")
            )
            # S is suffix [idx:]
            WP_S = WP_tot - (csum_wp[idx - 1] if idx > 0 else 0.0)
            W1M_S = W1M_tot - (csum_w1m[idx - 1] if idx > 0 else 0.0)
            W_S = W_tot - (csum_w[idx - 1] if idx > 0 else 0.0)
            return WP_S, W1M_S, W_S
        else:  # "<"
            # S = {i : p_i < t} or {i : p_i <= t} based on comparison
            idx = int(
                np.searchsorted(p_s, t, side="left" if comparison == ">" else "right")
            )
            # S is prefix [:idx]
            WP_S = csum_wp[idx - 1] if idx > 0 else 0.0
            W1M_S = csum_w1m[idx - 1] if idx > 0 else 0.0
            W_S = csum_w[idx - 1] if idx > 0 else 0.0
            return WP_S, W1M_S, W_S

    def update_lambda(WP_S: float, W1M_S: float) -> float:
        """Update lambda based on current decision set."""
        num = NUM_const + ap * WP_S + an * W1M_S
        den = DEN_const + bp * WP_S + bn * W1M_S
        if den <= 0.0:
            # Outside admissible region; use tiny positive to continue
            den = 1e-18
        return float(num / den)

    # Dinkelbach iterations
    lam = 0.0  # Initial lambda
    direction: Literal[">", "<"] = ">"
    t_clamped = 0.5  # Default threshold

    for _ in range(max_iter):
        # Compute threshold from current lambda
        c1 = ap - lam * bp  # coefficient multiplying p
        c0 = an - lam * bn  # coefficient multiplying (1 - p)
        slope = c1 - c0

        if abs(slope) < 1e-18:
            # No p-dependence -> either select all or none based on sign
            direction = ">"  # arbitrary; t will be 0 or 1
            if c0 > 0.0:
                WP_S, W1M_S, _ = WP_tot, W1M_tot, W_tot
            else:
                WP_S, W1M_S, _ = 0.0, 0.0, 0.0
            new_lam = update_lambda(WP_S, W1M_S)
            if abs(new_lam - lam) <= tol:
                lam = new_lam
                t_star = 0.0 if c0 > 0.0 else 1.0
                return float(t_star), float(lam), direction
            lam = new_lam
            continue

        # Compute threshold from linear program solution
        t = -c0 / slope
        direction = ">" if slope > 0 else "<"

        # Clamp to [0,1] for searchsorted
        t_clamped = max(0.0, min(1.0, float(t)))

        WP_S, W1M_S, _ = sums_for_threshold(t_clamped, direction)
        new_lam = update_lambda(WP_S, W1M_S)

        if abs(new_lam - lam) <= tol:
            lam = new_lam
            return float(t_clamped), float(lam), direction
        lam = new_lam

    # Max iterations reached
    return float(t_clamped), float(lam), direction


def dinkelbach_expected_fractional_ovr(
    y_prob: np.ndarray[Any, Any],
    metric: str,
    *,
    beta: float = 1.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    average: Literal["macro", "weighted", "micro"] = "macro",
    sample_weight: np.ndarray[Any, Any] | None = None,
    class_weight: np.ndarray[Any, Any] | None = None,
    comparison: Literal[">", ">="] = ">",
    max_iter: int = 100,
    tol: float = 1e-12,
) -> dict[str, np.ndarray[Any, Any] | float | str]:
    """Expected fractional-linear optimization for multilabel or multiclass-OvR.

    Applies the generalized Dinkelbach method to optimize the expected value of
    any fractional-linear metric under calibration, supporting multiple averaging
    strategies.

    Parameters
    ----------
    y_prob : ndarray of shape (n_samples, K)
        Calibrated probabilities per class.
    metric : str
        Metric name. See coeffs_for_metric() for supported metrics.
    beta : float, default=1.0
        Beta parameter for F-beta score.
    tversky_alpha, tversky_beta : float, default=0.5
        Parameters for Tversky index.
    average : {"macro", "weighted", "micro"}, default="macro"
        Averaging strategy:
        - "macro": per-class thresholds, unweighted mean score
        - "weighted": per-class thresholds, weighted by class_weight
        - "micro": single global threshold across all classes/instances
    sample_weight : ndarray of shape (n_samples,), optional
        Non-negative weights per sample.
    class_weight : ndarray of shape (K,), optional
        Non-negative weights per class (used when average="weighted").
    comparison : {">", ">="}, default=">"
        Tie-breaking convention.
    max_iter : int, default=100
        Maximum Dinkelbach iterations per class.
    tol : float, default=1e-12
        Convergence tolerance.

    Returns
    -------
    result : dict
        For "macro"/"weighted":
        {
            "thresholds": ndarray of shape (K,),
            "per_class": ndarray of shape (K,),  # per-class scores
            "score": float,  # averaged score
            "directions": ndarray of shape (K,)  # ">" or "<" per class
        }
        For "micro":
        {
            "threshold": float,
            "score": float,
            "direction": str  # ">" or "<"
        }

    Examples
    --------
    >>> # Multilabel Jaccard with macro averaging
    >>> y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])
    >>> result = dinkelbach_expected_fractional_ovr(y_prob, "jaccard", average="macro")
    >>> print(f"Per-class thresholds: {result['thresholds']}")
    >>> print(f"Macro Jaccard: {result['score']:.3f}")

    Notes
    -----
    Micro averaging flattens all class-instance pairs and treats them as a
    single binary problem. This can be more computationally efficient and
    provides a global decision boundary.
    """
    P = np.asarray(y_prob, dtype=float)
    if P.ndim != 2:
        raise ValueError("y_prob must have shape (n_samples, K).")

    n, K = P.shape

    # Validate sample weights
    sw = (
        np.ones(n, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if sw.shape != (n,):
        raise ValueError("sample_weight must have shape (n_samples,).")
    if np.any(sw < 0):
        raise ValueError("sample_weight must be non-negative.")

    # Get coefficients for the metric
    coeffs = coeffs_for_metric(
        metric, beta=beta, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta
    )

    if average == "micro":
        # Flatten all (sample, class) pairs
        p_flat = P.reshape(-1)
        sw_flat = np.repeat(sw, K)

        t, m, direction = dinkelbach_expected_fractional_binary(
            p_flat,
            coeffs,
            sample_weight=sw_flat,
            comparison=comparison,
            max_iter=max_iter,
            tol=tol,
        )
        return {"threshold": t, "score": m, "direction": direction}

    # For macro and weighted averaging, solve per-class
    thresholds = np.zeros(K, dtype=float)
    scores = np.zeros(K, dtype=float)
    directions = np.empty(K, dtype=object)

    for k in range(K):
        t_k, m_k, dir_k = dinkelbach_expected_fractional_binary(
            P[:, k],
            coeffs,
            sample_weight=sw,
            comparison=comparison,
            max_iter=max_iter,
            tol=tol,
        )
        thresholds[k] = t_k
        scores[k] = m_k
        directions[k] = dir_k

    # Compute averaged score
    if average == "macro":
        avg_score = float(np.mean(scores))
    elif average == "weighted":
        cw = (
            np.ones(K, dtype=float)
            if class_weight is None
            else np.asarray(class_weight, dtype=float)
        )
        if cw.shape != (K,):
            raise ValueError("class_weight must have shape (K,).")
        if np.any(cw < 0):
            raise ValueError("class_weight must be non-negative.")

        # Normalize weights
        w_norm = cw / (cw.sum() if cw.sum() > 0 else 1.0)
        avg_score = float(np.sum(w_norm * scores))
    else:
        raise ValueError('average must be one of {"macro", "weighted", "micro"}.')

    return {
        "thresholds": thresholds,
        "per_class": scores,
        "score": avg_score,
        "directions": directions,
    }
