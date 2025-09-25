"""Coordinate-ascent multiclass threshold optimization for single-label
consistent predictions.

This module implements a coordinate-ascent algorithm for optimizing per-class thresholds
in multiclass classification. Unlike One-vs-Rest approaches that optimize each class
independently, this algorithm couples classes through joint assignment using argmax
of shifted scores: argmax_j (p_ij - tau_j).

The algorithm iteratively updates each class threshold tau_c while fixing others,
using an efficient O(n log n) sort-and-scan approach to find the optimal tau_c
that maximizes the macro-F1 score of the resulting single-label predictions.

Key benefits:
- Single-label consistency: Predictions respect mutual exclusivity
- Coupled optimization: Classes are optimized jointly, not independently
- Monotone convergence: Each update improves or maintains objective
- Efficient implementation: O(n log n) per coordinate update
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]


def _macro_f1_from_assignments(y_true: Array, y_pred: Array, n_classes: int) -> float:
    """Compute macro-F1 from class assignments.

    Parameters
    ----------
    y_true : Array
        True class labels (integers 0, 1, ..., n_classes-1).
    y_pred : Array
        Predicted class labels (integers 0, 1, ..., n_classes-1).
    n_classes : int
        Number of classes.

    Returns
    -------
    float
        Macro-averaged F1 score.
    """
    tp = np.zeros(n_classes, dtype=int)
    pp = np.zeros(n_classes, dtype=int)  # predicted positives
    pos = np.bincount(y_true, minlength=n_classes)  # true positives per class

    for yi, pi in zip(y_true, y_pred, strict=False):
        pp[pi] += 1
        if yi == pi:
            tp[pi] += 1

    fn = pos - tp
    fp = pp - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0
    )
    return float(np.mean(f1))


def _assign_labels_shifted(P: Array, tau: Array) -> Array:
    """Assign labels using argmax of shifted scores.

    Computes y_pred = argmax_c (P[:,c] - tau[c]) for each sample.
    This ensures single-label predictions that respect class coupling.

    Parameters
    ----------
    P : Array
        Probability matrix of shape (n_samples, n_classes).
    tau : Array
        Per-class threshold vector of shape (n_classes,).

    Returns
    -------
    Array
        Predicted class labels of shape (n_samples,).
    """
    return np.argmax(P - tau[None, :], axis=1).astype(int)  # type: ignore[no-any-return]


def _init_thresholds_ovr_sortscan(
    y_true: Array, P: Array, sortscan_fn: Callable[[Array, Array], float]
) -> Array:
    """Initialize thresholds using One-vs-Rest sort-scan optimization.

    This provides a good starting point for the coordinate ascent algorithm
    by optimizing each class threshold independently using the binary F1
    optimization for the One-vs-Rest problem.

    Parameters
    ----------
    y_true : Array
        True class labels.
    P : Array
        Probability matrix of shape (n_samples, n_classes).
    sortscan_fn : Callable
        Function that takes (y_binary, p_class) and returns optimal threshold.

    Returns
    -------
    Array
        Initial threshold vector of shape (n_classes,).
    """
    C = P.shape[1]
    tau = np.zeros(C, dtype=float)
    for c in range(C):
        y_c = (y_true == c).astype(int)
        tau[c] = sortscan_fn(y_c, P[:, c])
    return tau


def optimal_multiclass_thresholds_coord_ascent(
    y_true: Array,
    P: Array,
    *,
    sortscan_metric_fn: Callable[[Array, Array, Array, Array], Array],
    sortscan_kernel: Callable[
        [Array, Array, Callable[[Array, Array, Array, Array], Array]],
        tuple[float, float, int],
    ],
    max_iter: int = 20,
    init: str = "ovr_sortscan",
    tol_stops: int = 1,
) -> tuple[Array, float, list[float]]:
    """
    Coordinate ascent optimization for multiclass thresholds in single-label setting.

    This algorithm iteratively optimizes per-class thresholds tau by cycling through
    classes. For each class c, it fixes other thresholds {tau_j}_{j≠c} and finds
    the optimal tau_c by analyzing breakpoints where instance assignments change.

    The key insight is that assignments change only at instance-specific breakpoints:
    b_i = p_ic - max_{j≠c}(p_ij - tau_j)

    By sorting these breakpoints and scanning in O(n log n), we can efficiently
    find the optimal tau_c that maximizes macro-F1.

    Parameters
    ----------
    y_true : Array
        True class labels of shape (n_samples,).
    P : Array
        Probability matrix of shape (n_samples, n_classes).
    sortscan_metric_fn : Callable
        Vectorized metric function for sort-scan optimization.
    sortscan_kernel : Callable
        Sort-scan kernel function (optimal_threshold_sortscan).
    max_iter : int, default=20
        Maximum number of coordinate ascent iterations.
    init : str, default="ovr_sortscan"
        Initialization strategy:
        - "ovr_sortscan": Initialize using One-vs-Rest sort-scan
        - "zeros": Initialize all thresholds to zero
    tol_stops : int, default=1
        Number of iterations without improvement before stopping.

    Returns
    -------
    Tuple[Array, float, List[float]]
        - tau: Optimal threshold vector of shape (n_classes,)
        - best_macro_f1: Best macro-F1 score achieved
        - history: List of macro-F1 scores at each iteration

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid parameters.
    """
    y_true = np.asarray(y_true, dtype=int)
    P = np.asarray(P, dtype=float)
    n, C = P.shape

    if y_true.shape[0] != n:
        raise ValueError("y_true and P must have compatible shapes.")
    if C < 2:
        raise ValueError("C >= 2 required for multiclass optimization.")

    # Initialize tau
    if init == "ovr_sortscan":

        def _ss(y_bin: Array, p_c: Array) -> float:
            thr, _, _ = sortscan_kernel(y_bin, p_c, sortscan_metric_fn)
            return thr

        tau = _init_thresholds_ovr_sortscan(y_true, P, _ss)
    elif init == "zeros":
        tau = np.zeros(C, dtype=float)
    else:
        raise ValueError(f"Unknown init '{init}'")

    history: list[float] = []
    no_improve_streak = 0
    best_macro = -np.inf

    for _it in range(max_iter):
        improved = False

        for c in range(C):
            # Current best other-class scores (independent of tau_c)
            others = P - tau[None, :]
            others[:, c] = -np.inf  # Exclude class c
            jstar = np.argmax(others, axis=1)  # Best non-c class for each instance
            best_other = others[np.arange(n), jstar]

            # Breakpoints: tau_c <= b[i] means instance i is assigned to class c
            b = P[:, c] - best_other
            order = np.argsort(-b, kind="mergesort")  # Sort by descending breakpoints

            # Start from tau_c > max(b) -> nobody assigned to class c
            y_pred = jstar.copy()
            macro = _macro_f1_from_assignments(y_true, y_pred, C)
            best_macro_c = macro
            best_k = -1

            # Initialize confusion counts for incremental updates
            tp = np.zeros(C, dtype=int)
            pp = np.zeros(C, dtype=int)  # predicted positives
            pos = np.bincount(y_true, minlength=C)

            # Initialize counts for current y_pred (all assigned to jstar classes)
            for yi, pi in zip(y_true, y_pred, strict=False):
                pp[pi] += 1
                if yi == pi:
                    tp[pi] += 1

            fn = pos - tp
            fp = pp - tp

            def _f1_class(
                k: int,
                tp_vals: Array = tp,
                fp_vals: Array = fp,
                fn_vals: Array = fn,
            ) -> float:
                """Compute F1 score for class k."""
                denom = 2 * tp_vals[k] + fp_vals[k] + fn_vals[k]
                return 0.0 if denom == 0 else (2.0 * tp_vals[k]) / denom

            # Scan: progressively move instances into class c by decreasing tau_c
            for rank, i in enumerate(order):
                r = y_pred[i]  # Previous class assignment (best non-c)
                if r == c:  # Shouldn't happen in this construction
                    continue

                # Update counts: move instance i from class r to class c
                pp[r] -= 1
                if y_true[i] == r:
                    tp[r] -= 1
                fp[r] = pp[r] - tp[r]
                fn[r] = pos[r] - tp[r]

                pp[c] += 1
                if y_true[i] == c:
                    tp[c] += 1
                fp[c] = pp[c] - tp[c]
                fn[c] = pos[c] - tp[c]

                y_pred[i] = c

                # Recompute macro-F1 (could be optimized by tracking changes)
                macro = _macro_f1_from_assignments(y_true, y_pred, C)

                if macro > best_macro_c:
                    best_macro_c = macro
                    best_k = rank

            # Set tau_c to midpoint between optimal breakpoints
            if best_k == -1:
                # Nobody assigned to class c is optimal
                new_tau_c = np.nextafter(np.max(b), np.inf) if b.size > 0 else 1.0
            elif best_k + 1 < b.size:
                # Midpoint between b[best_k] and b[best_k+1]
                b_sorted = b[order]
                new_tau_c = 0.5 * (b_sorted[best_k] + b_sorted[best_k + 1])
            else:
                # Everyone who can be assigned to c should be
                new_tau_c = np.nextafter(np.min(b), -np.inf) if b.size > 0 else 0.0

            # Apply the new tau_c and check global macro-F1
            old_tau_c = tau[c]
            tau[c] = new_tau_c
            y_pred_global = _assign_labels_shifted(P, tau)
            macro_global = _macro_f1_from_assignments(y_true, y_pred_global, C)

            if macro_global > best_macro + 1e-12:
                best_macro = macro_global
                improved = True
            else:
                # Revert if no global improvement (maintains monotonic ascent)
                tau[c] = old_tau_c

        history.append(best_macro)

        if improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            if no_improve_streak >= tol_stops:
                break

    return tau, best_macro, history
