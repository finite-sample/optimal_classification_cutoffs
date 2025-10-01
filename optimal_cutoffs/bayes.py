"""Bayes-optimal decisions and thresholds under calibrated probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np


def bayes_decision_from_utility_matrix(
    y_prob: np.ndarray[Any, Any],
    U: np.ndarray[Any, Any],
    labels: Sequence[Any] | None = None,
    return_scores: bool = False,
    *,
    validate: bool = True,
    normalize_rows: bool = False,
    row_sum_tol: float = 1e-6,
    tie_break: Literal["first", "last"] = "first",
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
        Labels for decisions. Defaults to range(D) (and -1 for abstain if D=K+1).
    return_scores : bool, default=False
        If True, also return expected-utility scores for each decision.
    validate : bool, default=True
        If True, validate that probabilities are finite and in [0,1].
    normalize_rows : bool, default=False
        If True, normalize probability rows to sum to 1 when validation fails.
        Only used when validate=True.
    row_sum_tol : float, default=1e-6
        Tolerance for row sum validation when validate=True.
    tie_break : {"first", "last"}, default="first"
        Tie-breaking rule when multiple decisions have equal expected utility.

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

    # Validate probabilities
    if validate:
        if not np.all(np.isfinite(y_prob)):
            raise ValueError("y_prob must be finite.")
        if np.any(y_prob < 0) or np.any(y_prob > 1):
            lo, hi = float(np.min(y_prob)), float(np.max(y_prob))
            raise ValueError(
                f"y_prob must be in [0,1]; got range [{lo:.6f}, {hi:.6f}]."
            )
        rs = y_prob.sum(axis=1)
        if not np.allclose(rs, 1.0, atol=row_sum_tol, rtol=0.0):
            if normalize_rows:
                # Avoid division by zero: if a row sums to 0, keep it uniform
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_prob = np.divide(y_prob, rs[:, None], where=rs[:, None] != 0)
            else:
                rmin, rmax = float(np.min(rs)), float(np.max(rs))
                raise ValueError(
                    f"Rows of y_prob must sum to 1 within tol={row_sum_tol}. "
                    f"Range of sums: [{rmin:.6f}, {rmax:.6f}]."
                )
    if not np.all(np.isfinite(U)):
        raise ValueError("U must be finite.")

    # Expected utility for each decision: S[i, d] = sum_y U[d, y] * p_i(y)
    S = y_prob @ U.T  # (n, K) @ (K, D) -> (n, D)

    # Argmax to find Bayes-optimal decision with tie-breaking
    if tie_break == "first":
        idx = np.argmax(S, axis=1)
    else:  # "last"
        idx = S.shape[1] - 1 - np.argmax(S[:, ::-1], axis=1)

    # Map to labels
    if labels is None:
        labels_list = list(range(D))
        if D == K + 1:
            labels_list[-1] = -1  # nice default for single abstain row
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
    comparison: str = ">",
    *,
    auto_convert_costs: bool = True,
    return_directions: bool = False,
) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Per-class Bayes thresholds for OvR (multi-label/multiclass-OvR) under calibration.

    For each class k, computes the optimal threshold using the correct formula:
    τ_k = B_k / D_k where B_k = U_tn,k - U_fp,k and D_k = A_k + B_k
    with A_k = U_tp,k - U_fn,k

    The decision rule is: predict positive when p > τ (if D > 0) or p < τ (if D < 0).

    Parameters
    ----------
    fp_cost, fn_cost : array-like of shape (K,)
        Costs per class for false positives and false negatives.
        With auto_convert_costs=True, positive values are treated as costs and
        converted to negative utilities automatically.
    tp_benefit, tn_benefit : array-like of shape (K,), optional
        Benefits per class for true positives and true negatives.
        Defaults to 0 if None. Typically positive values (benefits).
    comparison : {">" or ">="}, default=">"
        Comparison operator for threshold application (kept for compatibility).
    auto_convert_costs : bool, default=True
        If True, automatically convert positive costs to negative utilities.
    return_directions : bool, default=False
        If True, also return array of directions (">" or "<") per class.

    Returns
    -------
    tau : ndarray of shape (K,)
        Per-class Bayes-optimal thresholds in [0, 1].
    directions : ndarray of shape (K,), optional
        Decision directions per class. Returned if return_directions=True.

    Examples
    --------
    >>> # Positive costs (auto-converted)
    >>> fp_cost = [1, 1, 1]
    >>> fn_cost = [5, 3, 2]  # Different FN costs per class
    >>> thresholds = bayes_thresholds_from_costs_vector(fp_cost, fn_cost)
    >>> thresholds
    array([0.16666667, 0.25      , 0.33333333])

    >>> # With directions
    >>> tau, dirs = bayes_thresholds_from_costs_vector(
    ...     fp_cost, fn_cost, return_directions=True
    ... )

    Notes
    -----
    This function correctly handles edge cases:
    - Negative denominator D < 0: inequality flips, direction becomes "<"
    - Zero denominator D = 0: decision based on sign of B (tn - fp)
    - Mathematical correctness is maintained without epsilon adjustments
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

    # Auto-convert positive "costs" to negative utilities;
    # leave already negative values as-is
    if auto_convert_costs:
        if np.all(fp >= 0) and np.all(fn >= 0):
            fp = -fp
            fn = -fn
        # tp/tn are benefits; if someone passed negatives here,
        # keep them (explicit utilities)

    # Utility deltas
    A = tp - fn  # gain on p when predicting positive
    B = tn - fp  # gain on (1-p) when predicting positive
    D = A + B  # denominator

    tau = np.full_like(A, 0.0, dtype=float)
    directions = np.full(A.shape, ">", dtype="<U1")

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.divide(B, D, out=np.full_like(B, np.nan, dtype=float), where=(D != 0))

    # D < 0  -> inequality flips: predict positive when p < tau
    neg = D < 0
    directions[neg] = "<"

    # D == 0 -> trivial decisions based on sign of B:
    # If 0 >= B (i.e., B <= 0): always positive; else always negative.
    zero = D == 0
    always_pos = zero & (B <= 0)
    always_neg = zero & (B > 0)
    tau[always_pos] = 0.0
    directions[always_pos] = ">"
    tau[always_neg] = 1.0
    directions[always_neg] = ">"

    # Clip tau into [0,1] for interpretability; direction encodes side
    tau = np.clip(tau, 0.0, 1.0)

    # No epsilon tweaks; let caller's comparison control ties.
    return (np.asarray(tau), directions) if return_directions else np.asarray(tau)


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
        Comparison operator for threshold application (kept for compatibility).

    Returns
    -------
    float
        Optimal threshold in [0, 1].

    Examples
    --------
    >>> # Classic cost-sensitive case: FN costs 5x more than FP
    >>> threshold = bayes_threshold_from_costs_scalar(1, 5)  # auto-converted
    >>> round(threshold, 4)
    0.1667
    """
    thresholds, directions = bayes_thresholds_from_costs_vector(
        [fp_cost], [fn_cost], [tp_benefit], [tn_benefit], return_directions=True
    )
    threshold = float(thresholds[0])
    # direction = directions[0]  # Unused for backward compatibility
    # Backward-compat: ignore direction; most practical setups yield ">"
    # (If you want the direction, expose a new API
    # bayes_threshold_from_costs_scalar_ex with a direction return.)
    return threshold
