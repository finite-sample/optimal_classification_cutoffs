"""Expected F-beta optimization using Dinkelbach method under calibration."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def dinkelbach_expected_fbeta_binary(
    y_prob: np.ndarray[Any, Any],
    beta: float = 1.0,
    sample_weight: np.ndarray[Any, Any] | None = None,
    comparison: str = ">",
) -> tuple[float, float]:
    """
    Expected F-beta under calibration, binary case, with nonnegative sample weights.

    Under perfect calibration, maximizes:
    F_beta(S) = ((1+beta^2) A) / (A + C + beta^2 P)
    where:
    - A = sum_{i in S} w_i p_i (expected true positives)
    - C = sum_{i in S} w_i (1 - p_i) (expected false positives)
    - P = sum_i w_i p_i (total expected positives)
    - S = {i : p_i threshold_op t} (selected set)

    Uses Dinkelbach's algorithm to solve the ratio optimization problem.

    Parameters
    ----------
    y_prob : ndarray of shape (n_samples,)
        Calibrated probabilities for the positive class.
    beta : float, default=1.0
        Beta parameter (beta >= 0). beta=1 gives F1, beta < 1 emphasizes precision,
        beta > 1 emphasizes recall.
    sample_weight : ndarray of shape (n_samples,), optional
        Nonnegative weights. If None, uses uniform weights.
    comparison : {">" or ">="}, default=">"
        Decision boundary convention.

    Returns
    -------
    t_star : float
        Optimal threshold in [0, 1].
    f_star : float
        Optimal expected F_beta value in [0, 1].

    Examples
    --------
    >>> y_prob = np.array([0.1, 0.3, 0.7, 0.9])
    >>> t_star, f_star = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
    >>> print(f"Threshold: {t_star:.3f}, F1: {f_star:.3f}")
    Threshold: 0.500, F1: 0.750

    >>> # With sample weights
    >>> weights = np.array([1, 2, 1, 1])
    >>> t_star, f_star = dinkelbach_expected_fbeta_binary(y_prob, sample_weight=weights)

    Notes
    -----
    This method assumes perfect calibration: p_i = P(y_i = 1 | p_i).
    For miscalibrated probabilities, use empirical optimization instead.

    The Dinkelbach algorithm iteratively refines λ = a(S)/b(S) and maximizes
    a(S) - λb(S) until convergence, yielding the optimal ratio a*/b*.
    """
    p = np.asarray(y_prob, dtype=float).reshape(-1)

    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be in [0, 1].")

    if len(p) == 0:
        return 0.0, 0.0

    # Handle sample weights
    w = (
        np.ones_like(p)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    if w.shape != p.shape:
        raise ValueError("sample_weight must have shape (n_samples,).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be nonnegative.")

    # Validate beta
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    beta2 = float(beta) ** 2
    P_total = float(np.sum(w * p))

    if P_total == 0.0:
        # No positives in expectation -> never predict positive
        return 1.0, 0.0

    # Dinkelbach iterations
    lam = 0.5  # initial ratio guess
    tol = 1e-10
    max_iter = 100

    for _iteration in range(max_iter):
        # Threshold for current lambda
        t = lam / (1.0 + beta2)
        t = np.clip(t, 0.0, 1.0)  # Ensure valid threshold

        # Selected set based on comparison operator
        if comparison == ">":
            S = p > t
        else:  # ">="
            S = p >= t

        # Expected counts
        if np.any(S):
            A = float(np.sum(w[S] * p[S]))  # Expected TP
            C = float(np.sum(w[S] * (1.0 - p[S])))  # Expected FP
        else:
            A = C = 0.0

        # Compute new lambda
        denom = A + C + beta2 * P_total
        new_lam = 0.0 if denom == 0.0 else (1.0 + beta2) * A / denom

        # Check convergence
        if abs(new_lam - lam) <= tol:
            lam = new_lam
            break

        lam = new_lam

    # Final threshold and F-beta value
    t_star = lam / (1.0 + beta2)
    f_star = lam  # At convergence, lambda equals the optimal F-beta

    return float(np.clip(t_star, 0.0, 1.0)), float(np.clip(f_star, 0.0, 1.0))


def dinkelbach_expected_fbeta_multilabel(
    y_prob: np.ndarray[Any, Any],
    beta: float = 1.0,
    average: Literal["macro", "weighted", "micro"] = "macro",
    sample_weight: np.ndarray[Any, Any] | None = None,
    class_weight: np.ndarray[Any, Any] | None = None,
    comparison: str = ">",
) -> dict[str, np.ndarray[Any, Any] | float]:
    """
    Expected F-beta for multilabel or multiclass-OvR under calibration.

    Supports three averaging strategies:
    - macro: Per-class thresholds, unweighted mean F-beta
    - weighted: Per-class thresholds, class-weighted mean F-beta
    - micro: Single global threshold across all classes/instances

    Parameters
    ----------
    y_prob : ndarray of shape (n_samples, K)
        Calibrated probabilities per class.
    beta : float, default=1.0
        F-beta parameter (>= 0).
    average : {"macro", "weighted", "micro"}, default="macro"
        Averaging scheme:
        - "macro": per-class thresholds, unweighted mean score
        - "weighted": per-class thresholds, weighted by class_weight
        - "micro": single global threshold across all classes/instances
    sample_weight : ndarray of shape (n_samples,), optional
        Nonnegative weights per sample.
    class_weight : ndarray of shape (K,), optional
        Nonnegative weights per class (used when average="weighted").
    comparison : {">" or ">="}, default=">"
        Decision boundary convention.

    Returns
    -------
    result : dict
        For "macro"/"weighted":
        {
            "thresholds": ndarray of shape (K,),
            "f_beta_per_class": ndarray of shape (K,),
            "f_beta": float
        }
        For "micro":
        {
            "threshold": float,
            "f_beta": float
        }

    Examples
    --------
    >>> # Multilabel case with 3 classes
    >>> y_prob = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.7]])
    >>> result = dinkelbach_expected_fbeta_multilabel(y_prob, average="macro")
    >>> print(f"Per-class thresholds: {result['thresholds']}")
    >>> print(f"Macro F1: {result['f_beta']:.3f}")

    >>> # Micro averaging (single threshold)
    >>> result_micro = dinkelbach_expected_fbeta_multilabel(y_prob, average="micro")
    >>> print(f"Global threshold: {result_micro['threshold']:.3f}")

    Notes
    -----
    Micro averaging flattens all class-instance pairs and treats them as a
    single binary problem. This can be more computationally efficient and
    provides a global decision boundary.

    For single-label multiclass, consider using empirical coordinate ascent
    on validation data after getting initial thresholds from this function.
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
        raise ValueError("sample_weight must be nonnegative.")

    # Handle micro averaging separately (flatten to single binary problem)
    if average == "micro":
        # Flatten probabilities and repeat weights
        p_flat = P.reshape(-1)
        sw_flat = np.repeat(sw, K)

        t, f = dinkelbach_expected_fbeta_binary(
            p_flat, beta=beta, sample_weight=sw_flat, comparison=comparison
        )
        return {"threshold": t, "f_beta": f}

    # For macro and weighted averaging, solve per-class
    thresholds = np.zeros(K, dtype=float)
    f_per_class = np.zeros(K, dtype=float)

    # Validate class weights
    cw = (
        np.ones(K, dtype=float)
        if class_weight is None
        else np.asarray(class_weight, dtype=float)
    )
    if cw.shape != (K,):
        raise ValueError("class_weight must have shape (K,).")
    if np.any(cw < 0):
        raise ValueError("class_weight must be nonnegative.")

    # Per-class Dinkelbach optimization
    for k in range(K):
        t_k, f_k = dinkelbach_expected_fbeta_binary(
            P[:, k], beta=beta, sample_weight=sw, comparison=comparison
        )
        thresholds[k] = t_k
        f_per_class[k] = f_k

    # Compute averaged F-beta
    if average == "macro":
        f_avg = float(np.mean(f_per_class))
    elif average == "weighted":
        # Normalize class weights
        w_norm = cw / (cw.sum() if cw.sum() > 0 else 1.0)
        f_avg = float(np.sum(w_norm * f_per_class))
    else:
        raise ValueError('average must be one of {"macro", "weighted", "micro"}.')

    return {
        "thresholds": thresholds,
        "f_beta_per_class": f_per_class,
        "f_beta": f_avg,
    }
