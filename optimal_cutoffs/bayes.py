"""Bayes-optimal decisions and thresholds under calibrated probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def bayes_decision_from_utility_matrix(
    y_prob: np.ndarray[Any, Any],
    U: np.ndarray[Any, Any],
    labels: Sequence[Any] | None = None,
    return_scores: bool = False,
) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Multiclass Bayes-optimal decisions under calibrated probabilities.

    Under perfect calibration, the Bayes-optimal decision is:
    ŷ(x) = argmax_{d ∈ D} Σ_y U[d,y] p(y|x)

    This generalizes binary thresholds to multiclass scenarios with arbitrary
    utility matrices, including support for abstain decisions.

    Parameters
    ----------
    y_prob : ndarray of shape (n_samples, K)
        Calibrated class probabilities; rows should sum to 1.
    U : ndarray of shape (D, K)
        Utility matrix: U[d, y] = utility of choosing decision d when true class is y.
        Use D=K for standard K-way classification. You may include an extra row for
        an 'abstain' decision (D=K+1) if desired.
    labels : sequence of length D, optional
        Labels for decisions. Defaults to range(K) (and -1 for abstain if D=K+1).
    return_scores : bool, default=False
        If True, also return expected-utility scores for each decision.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Bayes-optimal decisions (using provided labels or integer indices).
    scores : ndarray of shape (n_samples, D), optional
        Expected utilities per decision; returned if `return_scores=True`.

    Examples
    --------
    >>> # Standard 3-class classification
    >>> y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    >>> U = np.eye(3)  # Identity matrix: correct = +1, incorrect = 0
    >>> decisions = bayes_decision_from_utility_matrix(y_prob, U)
    >>> decisions
    array([0, 1])

    >>> # Classification with abstain option
    >>> U_abstain = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
    >>> decisions, scores = bayes_decision_from_utility_matrix(
    ...     y_prob, U_abstain, labels=[0, 1, 2, -1], return_scores=True
    ... )
    """
    y_prob = np.asarray(y_prob, dtype=float)
    U = np.asarray(U, dtype=float)

    if y_prob.ndim != 2:
        raise ValueError("y_prob must have shape (n_samples, K).")

    n, K = y_prob.shape
    D, K_U = U.shape

    if K_U != K:
        raise ValueError(f"U has {K_U} columns but y_prob has {K} classes.")

    # Expected utility for each decision: S[i, d] = sum_y U[d, y] * p_i(y)
    S = y_prob @ U.T  # (n, K) @ (K, D) -> (n, D)

    # Argmax to find Bayes-optimal decision
    idx = np.argmax(S, axis=1)

    # Map to labels
    if labels is None:
        # -1 for abstain option
        labels_list = list(range(K)) if D == K else list(range(K)) + [-1]
    else:
        labels_list = list(labels)
    labels_array = np.asarray(labels_list)

    if len(labels_array) != D:
        raise ValueError(
            f"labels must have length {D} to match utility matrix dimensions."
        )

    y_pred = labels_array[idx]

    return (y_pred, S) if return_scores else y_pred


def bayes_thresholds_from_costs_vector(
    fp_cost: np.ndarray[Any, Any] | list[float],
    fn_cost: np.ndarray[Any, Any] | list[float],
    tp_benefit: np.ndarray[Any, Any] | list[float] | None = None,
    tn_benefit: np.ndarray[Any, Any] | list[float] | None = None,
) -> np.ndarray[Any, Any]:
    """
    Per-class Bayes thresholds for OvR (multi-label/multiclass-OvR) under calibration.

    For each class k, computes the optimal threshold:
    τ_k = (U_tn,k - U_fp,k) / [(U_tn,k - U_fp,k) + (U_tp,k - U_fn,k)]

    This is the closed-form solution for one-vs-rest classification where each
    class is treated as an independent binary problem.

    Parameters
    ----------
    fp_cost, fn_cost : array-like of shape (K,)
        Costs per class for false positives and false negatives.
        Typically negative values (costs).
    tp_benefit, tn_benefit : array-like of shape (K,), optional
        Benefits per class for true positives and true negatives.
        Defaults to 0 if None. Typically positive values (benefits).

    Returns
    -------
    tau : ndarray of shape (K,)
        Per-class Bayes-optimal thresholds in [0, 1].

    Examples
    --------
    >>> # Equal costs across classes
    >>> fp_cost = [-1, -1, -1]
    >>> fn_cost = [-5, -3, -2]  # Different FN costs per class
    >>> thresholds = bayes_thresholds_from_costs_vector(fp_cost, fn_cost)
    >>> thresholds
    array([0.16666667, 0.25      , 0.33333333])

    >>> # With benefits
    >>> tp_benefit = [2, 2, 2]
    >>> tn_benefit = [1, 1, 1]
    >>> thresholds = bayes_thresholds_from_costs_vector(
    ...     fp_cost, fn_cost, tp_benefit, tn_benefit
    ... )

    Notes
    -----
    This function handles degenerate cases where the denominator is zero:
    - If (tp - fn) <= 0: never predict positive -> tau=1
    - If (tp - fn) > 0: always predict positive -> tau=0
    """
    fp = np.asarray(fp_cost, dtype=float)
    fn = np.asarray(fn_cost, dtype=float)
    tp = (
        np.zeros_like(fp) if tp_benefit is None else np.asarray(tp_benefit, dtype=float)
    )
    tn = (
        np.zeros_like(fp) if tn_benefit is None else np.asarray(tn_benefit, dtype=float)
    )

    # Validate shapes
    if not (fp.shape == fn.shape == tp.shape == tn.shape):
        raise ValueError("All cost/benefit arrays must have the same shape.")

    # Compute threshold components
    num = tn - fp
    den = (tn - fp) + (tp - fn)

    # Initialize threshold array
    tau = np.empty_like(num)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau[:] = np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))

    # Handle degenerate cases where denominator is zero
    mask = ~np.isfinite(tau)  # nan or inf
    if np.any(mask):
        # If (tp - fn) <= 0: never predict positive -> tau=1
        # If (tp - fn) > 0: always predict positive -> tau=0
        tau[mask] = np.where((tp - fn)[mask] <= 0.0, 1.0, 0.0)

    clipped_result: np.ndarray[Any, Any] = np.clip(tau, 0.0, 1.0)
    return clipped_result


def bayes_threshold_from_costs_scalar(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    comparison: str = ">",
) -> float:
    """
    Binary Bayes threshold from scalar costs/benefits (backward compatibility).

    This is equivalent to the existing bayes_threshold_from_utility function
    but with a costs/benefits interface for consistency with the vector version.

    Parameters
    ----------
    fp_cost, fn_cost : float
        Scalar costs for false positives and false negatives.
    tp_benefit, tn_benefit : float, default=0.0
        Scalar benefits for true positives and true negatives.
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application.

    Returns
    -------
    float
        Optimal threshold in [0, 1], adjusted for comparison operator.

    Examples
    --------
    >>> # Classic cost-sensitive case: FN costs 5x more than FP
    >>> threshold = bayes_threshold_from_costs_scalar(-1, -5)
    >>> round(threshold, 4)
    0.1667
    """
    # Use the vector version with single values
    thresholds = bayes_thresholds_from_costs_vector(
        [fp_cost], [fn_cost], [tp_benefit], [tn_benefit]
    )
    threshold = float(thresholds[0])

    # Handle comparison operator adjustment (for boundary cases)
    if comparison == ">=" and threshold > 0:
        # For inclusive comparison, slightly reduce threshold to handle edge cases
        threshold = float(np.nextafter(threshold, 0.0, dtype=np.float64))

    return threshold
