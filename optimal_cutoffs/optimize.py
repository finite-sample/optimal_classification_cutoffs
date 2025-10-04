"""Unified threshold optimization for binary and multiclass classification.

This module consolidates all threshold optimization functionality into a single,
streamlined interface. It includes high-performance Numba kernels, multiple
optimization algorithms, and support for both binary and multiclass problems.

Key features:
- Fast Numba kernels with Python fallbacks
- Binary and multiclass threshold optimization
- Multiple algorithms: sort-scan, scipy, gradient, coordinate ascent
- Sample weight support
- Direct functional API without over-engineered abstractions
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import optimize

from .numba_utils import NUMBA_AVAILABLE, float64, int32, jit, prange

# ============================================================================
# Data Validation
# ============================================================================


def validate_binary_data(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and normalize binary classification data.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    weights : array-like, optional
        Sample weights

    Returns
    -------
    tuple
        (validated_labels, validated_scores, validated_weights)
    """
    # Convert and validate labels/scores
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be 1D")
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: {len(labels)} vs {len(scores)}")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be binary (0 or 1)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite")

    # Handle weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(labels):
            raise ValueError("Invalid weights shape")
        if not np.all(weights >= 0):
            raise ValueError("Weights must be non-negative")
        if np.sum(weights) == 0:
            raise ValueError("Weights sum to zero")

    return labels, scores, weights


# ============================================================================
# Fast Numba Kernels
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, cache=True)
    def compute_confusion_matrix_weighted(
        labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None
    ) -> tuple[float64, float64, float64, float64]:
        """Compute weighted confusion matrix elements."""
        tp = tn = fp = fn = 0.0

        if weights is None:
            for i in prange(len(labels)):
                if labels[i] == 1:
                    if predictions[i]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predictions[i]:
                        fp += 1
                    else:
                        tn += 1
        else:
            for i in prange(len(labels)):
                w = weights[i]
                if labels[i] == 1:
                    if predictions[i]:
                        tp += w
                    else:
                        fn += w
                else:
                    if predictions[i]:
                        fp += w
                    else:
                        tn += w

        return tp, tn, fp, fn

    @jit(nopython=True, fastmath=True, cache=True)
    def fast_f1_score(tp: float64, tn: float64, fp: float64, fn: float64) -> float64:
        """Compute F1 score from confusion matrix."""
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    @jit(nopython=True, parallel=True, cache=True)
    def sort_scan_kernel(
        labels: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray | None,
        inclusive: bool,
    ) -> tuple[float64, float64]:
        """Pure Numba sort-and-scan implementation for binary optimization."""
        n = len(labels)

        # Sort by scores (descending for easier logic)
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        sorted_weights = weights[order] if weights is not None else None

        # Initial state: all negative (threshold > max score)
        tp = 0.0
        fn = 0.0
        fp = 0.0
        tn = 0.0

        # Count initial state
        for i in range(n):
            if sorted_labels[i] == 1:
                fn += sorted_weights[i] if sorted_weights is not None else 1.0
            else:
                tn += sorted_weights[i] if sorted_weights is not None else 1.0

        best_threshold = sorted_scores[0] + 1e-10
        best_score = fast_f1_score(tp, tn, fp, fn)

        # Scan through thresholds (decreasing scores)
        for i in range(n):
            # Update confusion matrix by moving threshold to include this sample
            if sorted_labels[i] == 1:
                tp += sorted_weights[i] if sorted_weights is not None else 1.0
                fn -= sorted_weights[i] if sorted_weights is not None else 1.0
            else:
                fp += sorted_weights[i] if sorted_weights is not None else 1.0
                tn -= sorted_weights[i] if sorted_weights is not None else 1.0

            # Compute score
            score = fast_f1_score(tp, tn, fp, fn)

            # Update best
            if score > best_score:
                best_score = score
                if i < n - 1:
                    # Midpoint between current and next
                    best_threshold = 0.5 * (sorted_scores[i] + sorted_scores[i + 1])
                else:
                    # After last score
                    best_threshold = max(0.0, sorted_scores[i] - 1e-10)

        return best_threshold, best_score

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def coordinate_ascent_kernel(
        y_true: np.ndarray,  # (n,) int32
        probs: np.ndarray,  # (n, k) float64
        max_iter: int,
        tol: float64,
    ) -> tuple[np.ndarray, float64, np.ndarray]:
        """Pure Numba implementation of coordinate ascent for multiclass."""
        n, k = probs.shape
        thresholds = np.zeros(k, dtype=float64)
        history = np.zeros(max_iter, dtype=float64)

        # Precompute class counts for efficiency
        class_counts = np.zeros(k, dtype=int32)
        for i in range(n):
            class_counts[y_true[i]] += 1

        best_score = -1.0
        no_improve = 0

        for iteration in range(max_iter):
            improved = False

            for c in range(k):
                # Find breakpoints for class c
                breakpoints = np.zeros(n, dtype=float64)
                alternatives = np.zeros(n, dtype=int32)

                for i in prange(n):
                    max_other = -np.inf
                    max_other_idx = -1

                    for j in range(k):
                        if j != c:
                            score = probs[i, j] - thresholds[j]
                            if score > max_other:
                                max_other = score
                                max_other_idx = j

                    breakpoints[i] = probs[i, c] - max_other
                    alternatives[i] = max_other_idx

                # Sort breakpoints
                order = np.argsort(-breakpoints)

                # Scan for optimal threshold
                tp = np.zeros(k, dtype=int32)
                fp = np.zeros(k, dtype=int32)

                # Initial state: all assigned to alternatives
                for i in range(n):
                    pred = alternatives[i]
                    if y_true[i] == pred:
                        tp[pred] += 1
                    else:
                        fp[pred] += 1

                current_best = _compute_macro_f1_numba(tp, fp, class_counts)
                best_idx = -1

                # Scan through breakpoints
                for rank in range(n):
                    idx = order[rank]
                    old_pred = alternatives[idx]

                    # Update counts
                    if y_true[idx] == old_pred:
                        tp[old_pred] -= 1
                    else:
                        fp[old_pred] -= 1

                    if y_true[idx] == c:
                        tp[c] += 1
                    else:
                        fp[c] += 1

                    score = _compute_macro_f1_numba(tp, fp, class_counts)
                    if score > current_best:
                        current_best = score
                        best_idx = rank

                # Update threshold
                if best_idx >= 0:
                    sorted_breaks = breakpoints[order]
                    if best_idx + 1 < n:
                        new_threshold = 0.5 * (
                            sorted_breaks[best_idx] + sorted_breaks[best_idx + 1]
                        )
                    else:
                        new_threshold = sorted_breaks[best_idx] - 1e-6

                    if current_best > best_score + tol:
                        thresholds[c] = new_threshold
                        best_score = current_best
                        improved = True

            history[iteration] = best_score

            if not improved:
                no_improve += 1
                if no_improve >= 2:
                    return thresholds, best_score, history[: iteration + 1]
            else:
                no_improve = 0

        return thresholds, best_score, history

    @jit(nopython=True, fastmath=True, inline="always")
    def _compute_macro_f1_numba(
        tp: np.ndarray, fp: np.ndarray, support: np.ndarray
    ) -> float64:
        """Compute macro F1 score in Numba."""
        f1_sum = 0.0
        n_classes = len(tp)

        for c in range(n_classes):
            fn = support[c] - tp[c]
            denom = 2 * tp[c] + fp[c] + fn
            if denom > 0:
                f1_sum += 2.0 * tp[c] / denom

        return f1_sum / n_classes

else:
    # Pure Python fallbacks when Numba is not available
    def compute_confusion_matrix_weighted(
        labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None
    ) -> tuple[float, float, float, float]:
        """Compute weighted confusion matrix elements (Python fallback)."""
        tp = tn = fp = fn = 0.0

        if weights is None:
            for i in range(len(labels)):
                if labels[i] == 1:
                    if predictions[i]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predictions[i]:
                        fp += 1
                    else:
                        tn += 1
        else:
            for i in range(len(labels)):
                w = weights[i]
                if labels[i] == 1:
                    if predictions[i]:
                        tp += w
                    else:
                        fn += w
                else:
                    if predictions[i]:
                        fp += w
                    else:
                        tn += w

        return tp, tn, fp, fn

    def fast_f1_score(tp: float, tn: float, fp: float, fn: float) -> float:
        """Compute F1 score from confusion matrix (Python fallback)."""
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    def sort_scan_kernel(
        labels: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray | None,
        inclusive: bool,
    ) -> tuple[float, float]:
        """Pure Python sort-and-scan implementation."""
        n = len(labels)

        # Sort by scores (descending for easier logic)
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        sorted_weights = weights[order] if weights is not None else None

        # Initial state: all negative (threshold > max score)
        tp = 0.0
        fn = float(
            np.sum(sorted_weights[sorted_labels == 1])
            if sorted_weights is not None
            else np.sum(sorted_labels)
        )
        fp = 0.0
        tn = float(
            np.sum(sorted_weights[sorted_labels == 0])
            if sorted_weights is not None
            else np.sum(1 - sorted_labels)
        )

        best_threshold = float(sorted_scores[0] + 1e-10)
        best_score = fast_f1_score(tp, tn, fp, fn)

        # Scan through thresholds (decreasing scores)
        for i in range(n):
            # Update confusion matrix by moving threshold to include this sample
            w = sorted_weights[i] if sorted_weights is not None else 1.0
            if sorted_labels[i] == 1:
                tp += w
                fn -= w
            else:
                fp += w
                tn -= w

            # Compute score
            score = fast_f1_score(tp, tn, fp, fn)

            # Update best
            if score > best_score:
                best_score = score
                if i < n - 1:
                    # Midpoint between current and next
                    best_threshold = 0.5 * (sorted_scores[i] + sorted_scores[i + 1])
                else:
                    # After last score
                    best_threshold = max(0.0, float(sorted_scores[i] - 1e-10))

        return best_threshold, best_score

    def coordinate_ascent_kernel(
        y_true: np.ndarray, probs: np.ndarray, max_iter: int, tol: float
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Pure Python fallback implementation for coordinate ascent."""
        n, k = probs.shape
        thresholds = np.zeros(k, dtype=np.float64)
        history = []

        # Precompute class counts
        class_counts = np.bincount(y_true, minlength=k)

        best_score = -1.0
        no_improve = 0

        for _iteration in range(max_iter):
            improved = False

            for c in range(k):
                # Find breakpoints for class c
                breakpoints = np.zeros(n)
                alternatives = np.zeros(n, dtype=int)

                for i in range(n):
                    other_scores = probs[i] - thresholds
                    other_scores[c] = -np.inf
                    max_other_idx = np.argmax(other_scores)
                    max_other = other_scores[max_other_idx]

                    breakpoints[i] = probs[i, c] - max_other
                    alternatives[i] = max_other_idx

                # Sort breakpoints
                order = np.argsort(-breakpoints)

                # Scan for optimal threshold
                tp = np.zeros(k, dtype=int)
                fp = np.zeros(k, dtype=int)

                # Initial state: all assigned to alternatives
                for i in range(n):
                    pred = alternatives[i]
                    if y_true[i] == pred:
                        tp[pred] += 1
                    else:
                        fp[pred] += 1

                current_best = _compute_macro_f1_python(tp, fp, class_counts)
                best_idx = -1

                # Scan through breakpoints
                for rank in range(n):
                    idx = order[rank]
                    old_pred = alternatives[idx]

                    # Update counts
                    if y_true[idx] == old_pred:
                        tp[old_pred] -= 1
                    else:
                        fp[old_pred] -= 1

                    if y_true[idx] == c:
                        tp[c] += 1
                    else:
                        fp[c] += 1

                    score = _compute_macro_f1_python(tp, fp, class_counts)
                    if score > current_best:
                        current_best = score
                        best_idx = rank

                # Update threshold
                if best_idx >= 0:
                    sorted_breaks = breakpoints[order]
                    if best_idx + 1 < n:
                        new_threshold = 0.5 * (
                            sorted_breaks[best_idx] + sorted_breaks[best_idx + 1]
                        )
                    else:
                        new_threshold = sorted_breaks[best_idx] - 1e-6

                    if current_best > best_score + tol:
                        thresholds[c] = new_threshold
                        best_score = current_best
                        improved = True

            history.append(best_score)

            if not improved:
                no_improve += 1
                if no_improve >= 2:
                    break
            else:
                no_improve = 0

        return thresholds, best_score, np.array(history)

    def _compute_macro_f1_python(
        tp: np.ndarray, fp: np.ndarray, support: np.ndarray
    ) -> float:
        """Compute macro F1 score in pure Python."""
        f1_sum = 0.0
        n_classes = len(tp)

        for c in range(n_classes):
            fn = support[c] - tp[c]
            denom = 2 * tp[c] + fp[c] + fn
            if denom > 0:
                f1_sum += 2.0 * tp[c] / denom

        return f1_sum / n_classes


# ============================================================================
# Binary Optimization Algorithms
# ============================================================================


def optimize_sort_scan(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    operator: str = ">=",
) -> tuple[float, float]:
    """Sort-and-scan optimization for piecewise constant metrics.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    metric : str
        Metric to optimize (currently optimized for F1)
    weights : array-like, optional
        Sample weights
    operator : str
        Comparison operator (">=" or ">")

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_score)
    """
    labels, scores, weights = validate_binary_data(labels, scores, weights)

    # Use fast Numba kernel for F1
    if metric.lower() in ("f1", "f1_score"):
        threshold, score = sort_scan_kernel(
            labels, scores, weights, inclusive=(operator == ">=")
        )
        return float(threshold), float(score)

    # Generic implementation for other metrics
    return _generic_sort_scan(labels, scores, metric, weights, operator)


def _generic_sort_scan(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None,
    operator: str,
) -> tuple[float, float]:
    """Generic sort-and-scan implementation for any metric."""
    # Import metric function
    from .metrics import METRIC_REGISTRY

    metric_fn = METRIC_REGISTRY[metric]

    # Sort by scores
    order = np.argsort(scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    best_threshold = max(0.0, float(sorted_scores[0] - 1e-10))
    best_score = -np.inf

    # Scan through unique thresholds
    unique_scores = np.unique(sorted_scores)

    for i, score_val in enumerate(unique_scores):
        # Find threshold position
        if operator == ">=":
            predictions = sorted_scores >= score_val
        else:
            predictions = sorted_scores > score_val

        # Compute metric
        tp, tn, fp, fn = compute_confusion_matrix_weighted(
            sorted_labels, predictions, weights
        )
        score = metric_fn(int(tp), int(tn), int(fp), int(fn))

        if score > best_score:
            best_score = score
            if i > 0:
                best_threshold = 0.5 * (unique_scores[i - 1] + score_val)
            else:
                best_threshold = max(0.0, float(score_val - 1e-10))

    return best_threshold, best_score


def optimize_scipy(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    operator: str = ">=",
    method: str = "bounded",
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Scipy-based optimization for smooth metrics.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    metric : str
        Metric to optimize
    weights : array-like, optional
        Sample weights
    operator : str
        Comparison operator (">=" or ">")
    method : str
        Scipy optimization method
    tol : float
        Tolerance for convergence

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_score)
    """
    labels, scores, weights = validate_binary_data(labels, scores, weights)

    # Import metric function
    from .metrics import METRIC_REGISTRY

    metric_fn = METRIC_REGISTRY[metric]

    def objective(threshold: float) -> float:
        """Objective to minimize (negative metric)."""
        predictions = scores >= threshold if operator == ">=" else scores > threshold
        tp, tn, fp, fn = compute_confusion_matrix_weighted(labels, predictions, weights)
        score = metric_fn(int(tp), int(tn), int(fp), int(fn))
        return -score  # Minimize negative for maximization

    # Determine score bounds
    score_min, score_max = float(np.min(scores)), float(np.max(scores))
    bounds = (max(0.0, score_min - 1e-10), min(1.0, score_max + 1e-10))

    try:
        result = optimize.minimize_scalar(
            objective, bounds=bounds, method="bounded", options={"xatol": tol}
        )
        optimal_threshold = float(result.x)
        optimal_score = -float(result.fun)
    except Exception:
        # Fallback to sort_scan
        warnings.warn(
            "Scipy optimization failed, falling back to sort_scan", stacklevel=2
        )
        return optimize_sort_scan(labels, scores, metric, weights, operator)

    return optimal_threshold, optimal_score


def optimize_gradient(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    operator: str = ">=",
    learning_rate: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Simple gradient ascent optimization.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    metric : str
        Metric to optimize
    weights : array-like, optional
        Sample weights
    operator : str
        Comparison operator (">=" or ">")
    learning_rate : float
        Learning rate for gradient ascent
    max_iter : int
        Maximum iterations
    tol : float
        Tolerance for convergence

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_score)
    """
    labels, scores, weights = validate_binary_data(labels, scores, weights)

    # Import metric function
    from .metrics import METRIC_REGISTRY

    metric_fn = METRIC_REGISTRY[metric]

    # Check if metric is piecewise constant
    from .metrics import is_piecewise_metric

    if is_piecewise_metric(metric):
        warnings.warn(
            "Gradient optimization is ineffective for piecewise-constant metrics. "
            "Use sort_scan instead.",
            stacklevel=2,
        )

    # Start with median score
    threshold = float(np.median(scores))

    def evaluate_metric(t: float) -> float:
        predictions = scores >= t if operator == ">=" else scores > t
        tp, tn, fp, fn = compute_confusion_matrix_weighted(labels, predictions, weights)
        return metric_fn(int(tp), int(tn), int(fp), int(fn))

    for _ in range(max_iter):
        # Simple finite difference gradient
        h = 1e-8
        grad = (evaluate_metric(threshold + h) - evaluate_metric(threshold - h)) / (
            2 * h
        )

        if abs(grad) < tol:
            break

        threshold += learning_rate * grad

        # Keep threshold in reasonable bounds
        min_bound = max(0.0, np.min(scores) - 1e-10)
        max_bound = min(1.0, np.max(scores) + 1e-10)
        threshold = np.clip(threshold, min_bound, max_bound)

    final_score = evaluate_metric(threshold)
    return threshold, final_score


# ============================================================================
# Multiclass Optimization
# ============================================================================


def _assign_labels_shifted(P: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Assign labels using argmax of shifted scores.

    Parameters
    ----------
    P : array-like of shape (n_samples, n_classes)
        Probability matrix
    tau : array-like of shape (n_classes,)
        Per-class thresholds

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Predicted class labels
    """
    return np.argmax(P - tau[None, :], axis=1)


def find_optimal_threshold_multiclass(
    true_labs: np.ndarray,
    pred_prob: np.ndarray,
    metric: str = "f1",
    method: str = "auto",
    average: str = "macro",
    sample_weight: np.ndarray | None = None,
    comparison: str = ">",
) -> np.ndarray:
    """Find optimal per-class thresholds for multiclass classification.

    Uses One-vs-Rest (OvR) strategy where each class is treated as a separate
    binary classification problem, or coordinate ascent for coupled optimization.

    Parameters
    ----------
    true_labs : array-like of shape (n_samples,)
        True class labels (0, 1, ..., n_classes-1)
    pred_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities
    metric : str, default="f1"
        Metric to optimize
    method : str, default="auto"
        Optimization method ("auto", "sort_scan", "scipy", "gradient", "coord_ascent")
    average : str, default="macro"
        Averaging strategy ("macro", "micro", "weighted", "none")
    sample_weight : array-like, optional
        Sample weights
    comparison : str, default=">"
        Comparison operator (">" or ">=")

    Returns
    -------
    np.ndarray of shape (n_classes,)
        Optimal threshold for each class
    """
    from .validation import validate_multiclass_classification

    true_labs, pred_prob, _ = validate_multiclass_classification(
        true_labs, pred_prob
    )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    n_samples, n_classes = pred_prob.shape

    # Handle coordinate ascent method
    if method == "coord_ascent":
        # Additional validation for coordinate ascent - requires consecutive labels
        from .validation import validate_multiclass_labels

        # This will raise ValueError if validation fails
        validate_multiclass_labels(true_labs, n_classes=n_classes)

        # Coordinate ascent limitations
        if sample_weight is not None:
            raise NotImplementedError(
                "Coordinate ascent does not support sample weights. "
                "This limitation could be lifted in future versions."
            )
        if comparison != ">":
            raise NotImplementedError(
                "Coordinate ascent only supports '>' comparison. "
                "Support for '>=' could be added in future versions."
            )
        if metric != "f1":
            raise NotImplementedError(
                "Coordinate ascent only supports 'f1' metric. "
                "Support for other piecewise metrics could be added in future versions."
            )

        # Convert inputs to proper types for Numba
        true_labs_int32 = np.asarray(true_labs, dtype=np.int32)
        pred_prob_float64 = np.asarray(pred_prob, dtype=np.float64, order="C")

        # Call the optimized kernel directly
        thresholds, _, _ = coordinate_ascent_kernel(
            true_labs_int32, pred_prob_float64, max_iter=20, tol=1e-12
        )

        return thresholds

    # Map method to binary optimization function
    if method == "auto":
        # Choose best method based on metric
        from .metrics import is_piecewise_metric

        if is_piecewise_metric(metric):
            optimize_fn = optimize_sort_scan
        else:
            optimize_fn = optimize_scipy
    elif method == "sort_scan":
        optimize_fn = optimize_sort_scan
    elif method == "scipy":
        optimize_fn = optimize_scipy
    elif method == "gradient":
        optimize_fn = optimize_gradient
    else:
        optimize_fn = optimize_sort_scan  # Default fallback

    operator = ">=" if comparison == ">=" else ">"

    if average == "micro":
        # Micro averaging: use single global threshold
        # Flatten all class probabilities and create binary labels
        true_binary_flat = np.zeros((n_samples * n_classes,), dtype=int)
        pred_prob_flat = pred_prob.ravel()

        # Create binary labels for micro averaging
        for i in range(n_samples):
            for j in range(n_classes):
                idx = i * n_classes + j
                true_binary_flat[idx] = 1 if true_labs[i] == j else 0

        # Sample weights for micro averaging
        if sample_weight is not None:
            sample_weight_flat = np.repeat(sample_weight, n_classes)
        else:
            sample_weight_flat = None

        # Find single optimal threshold
        optimal_threshold, _ = optimize_fn(
            true_binary_flat, pred_prob_flat, metric, sample_weight_flat, operator
        )

        # Return same threshold for all classes
        return np.full(n_classes, optimal_threshold, dtype=float)

    else:
        # Macro/weighted/none averaging: optimize per-class thresholds independently
        optimal_thresholds = np.zeros(n_classes, dtype=float)

        # Create binary labels for each class (One-vs-Rest)
        true_binary_all = np.zeros((n_samples, n_classes), dtype=int)
        for class_idx in range(n_classes):
            true_binary_all[:, class_idx] = (true_labs == class_idx).astype(int)

        # Optimize each class independently
        for class_idx in range(n_classes):
            threshold, _ = optimize_fn(
                true_binary_all[:, class_idx],
                pred_prob[:, class_idx],
                metric,
                sample_weight,
                operator,
            )
            optimal_thresholds[class_idx] = threshold

        return optimal_thresholds


# ============================================================================
# Main API Functions
# ============================================================================


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
    weights: np.ndarray | None = None,
    strategy: str = "auto",
    operator: str = ">=",
    require_probability: bool = True,
) -> tuple[float, float]:
    """Simple functional interface for binary threshold optimization.

    Parameters
    ----------
    labels : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    metric : str, default="f1"
        Metric to optimize
    weights : array-like, optional
        Sample weights
    strategy : str, default="auto"
        Optimization strategy ("auto", "sort_scan", "scipy", "gradient")
    operator : str, default=">="
        Comparison operator (">=" or ">")
    require_probability : bool, default=True
        Whether to require scores in [0, 1]

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_score)
    """
    # Validate probability requirement
    if require_probability:
        scores = np.asarray(scores)
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Scores must be in [0, 1] when require_probability=True")

    # Select optimization function
    if strategy == "auto":
        # Choose best method based on metric
        from .metrics import is_piecewise_metric

        if is_piecewise_metric(metric):
            optimize_fn = optimize_sort_scan
        else:
            optimize_fn = optimize_scipy
    elif strategy == "sort_scan":
        optimize_fn = optimize_sort_scan
    elif strategy == "scipy":
        optimize_fn = optimize_scipy
    elif strategy == "gradient":
        optimize_fn = optimize_gradient
    else:
        optimize_fn = optimize_sort_scan  # Default fallback

    return optimize_fn(labels, scores, metric, weights, operator)


# ============================================================================
# Performance Information
# ============================================================================


def get_performance_info() -> dict[str, Any]:
    """Get information about performance optimizations available."""
    return {
        "numba_available": NUMBA_AVAILABLE,
        "numba_version": (
            None
            if not NUMBA_AVAILABLE
            else getattr(__import__("numba"), "__version__", "unknown")
        ),
        "expected_speedup": "10-100x" if NUMBA_AVAILABLE else "1x (Python fallback)",
        "parallel_processing": NUMBA_AVAILABLE,
        "fastmath_enabled": NUMBA_AVAILABLE,
    }
