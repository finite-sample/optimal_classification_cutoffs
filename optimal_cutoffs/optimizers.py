"""Threshold search strategies for optimizing classification metrics."""

from typing import Any, Literal

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from .metrics import (
    METRIC_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    get_vectorized_metric,
    has_vectorized_implementation,
    is_piecewise_metric,
    multiclass_metric,
    multiclass_metric_exclusive,
)
from .multiclass_coord import optimal_multiclass_thresholds_coord_ascent
from .piecewise import optimal_threshold_sortscan
from .types import ArrayLike, AveragingMethod, ComparisonOperator, OptimizationMethod
from .validation import (
    _validate_comparison_operator,
    _validate_inputs,
    _validate_metric_name,
    _validate_optimization_method,
)


def _accuracy(
    prob: np.ndarray[Any, Any],
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    verbose: bool = False,
) -> float:
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print(f"Probability: {prob[0]:0.4f} Accuracy: {accuracy:0.4f}")
    return 1 - accuracy


def _f1(
    prob: np.ndarray[Any, Any],
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    verbose: bool = False,
) -> float:
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    if verbose:
        print(f"Probability: {prob[0]:0.4f} F1 score: {f1:0.4f}")
    return 1 - f1


def get_probability(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    objective: Literal["accuracy", "f1"] = "accuracy",
    verbose: bool = False,
) -> float:
    """Brute-force search for a simple metric's best threshold.

    .. deprecated:: 1.0.0
        :func:`get_probability` is deprecated and will be removed in a future version.
        Use :func:`get_optimal_threshold` instead, which provides a unified API for
        both binary and multiclass classification with more optimization methods
        and additional features like sample weights.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Predicted probabilities from a classifier.
    objective:
        Metric to optimize. Supported values are ``"accuracy"`` and ``"f1"``.
    verbose:
        If ``True``, print intermediate metric values during the search.

    Returns
    -------
    float
        Threshold that maximizes the specified metric.
    """
    import warnings

    warnings.warn(
        "get_probability is deprecated and will be removed in a future version. "
        "Use get_optimal_threshold instead, which provides a unified API for "
        "both binary and multiclass classification with more optimization methods "
        "and additional features like sample weights.",
        DeprecationWarning,
        stacklevel=2,
    )
    if objective == "accuracy":
        prob = optimize.brute(
            _accuracy,
            (slice(0.1, 0.9, 0.1),),
            args=(true_labs, pred_prob, verbose),
            disp=verbose,
        )
    elif objective == "f1":
        prob = optimize.brute(
            _f1,
            (slice(0.1, 0.9, 0.1),),
            args=(true_labs, pred_prob, verbose),
            disp=verbose,
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return float(prob[0] if isinstance(prob, np.ndarray) else prob)


def _metric_score(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    threshold: float,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
) -> float:
    """Compute a metric score for a given threshold using registry metrics.

    Parameters
    ----------
    true_labs:
        Array of true labels.
    pred_prob:
        Array of predicted probabilities.
    threshold:
        Decision threshold.
    metric:
        Name of metric from registry.
    sample_weight:
        Optional array of sample weights.

    Returns
    -------
    float
        Computed metric score.
    """
    tp, tn, fp, fn = get_confusion_matrix(
        true_labs, pred_prob, threshold, sample_weight, comparison
    )
    try:
        metric_func = METRIC_REGISTRY[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {metric}") from exc
    return float(metric_func(tp, tn, fp, fn))


def _multiclass_metric_score(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    thresholds: ArrayLike,
    metric: str = "f1",
    average: AveragingMethod = "macro",
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute a multiclass metric score for given per-class thresholds.

    Parameters
    ----------
    true_labs:
        Array of true class labels.
    pred_prob:
        Array of predicted probabilities.
    thresholds:
        Array of per-class thresholds.
    metric:
        Name of metric from registry.
    average:
        Averaging strategy for multiclass.
    sample_weight:
        Optional array of sample weights.

    Returns
    -------
    float
        Computed multiclass metric score.
    """
    confusion_matrices = get_multiclass_confusion_matrix(
        true_labs, pred_prob, thresholds, sample_weight
    )
    result = multiclass_metric(confusion_matrices, metric, average)
    return float(result) if isinstance(result, np.ndarray) else result


def _optimal_threshold_piecewise(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
) -> float:
    """Find optimal threshold using O(n log n) algorithm for piecewise metrics.

    This function provides a backward-compatible interface to the optimized
    sort-and-scan implementation for piecewise-constant metrics.

    Parameters
    ----------
    true_labs:
        Array of true binary labels.
    pred_prob:
        Array of predicted probabilities.
    metric:
        Name of metric to optimize from METRIC_REGISTRY.
    sample_weight:
        Optional array of sample weights.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    float
        Optimal threshold that maximizes the metric.
    """
    # Check if we have a vectorized implementation
    if has_vectorized_implementation(metric):
        try:
            vectorized_metric = get_vectorized_metric(metric)
            threshold, _, _ = optimal_threshold_sortscan(
                np.asarray(true_labs),
                np.asarray(pred_prob),
                vectorized_metric,
                sample_weight=(
                    np.asarray(sample_weight) if sample_weight is not None else None
                ),
                inclusive=comparison,
            )
            return threshold
        except Exception:
            # Fall back to original implementation if vectorized fails
            pass

    # Fall back to original implementation
    return _optimal_threshold_piecewise_fallback(
        true_labs, pred_prob, metric, sample_weight, comparison
    )


def _optimal_threshold_piecewise_fallback(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
) -> float:
    """Fallback implementation for metrics not yet vectorized.

    This is the original O(k log n) implementation that evaluates at unique
    probabilities.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)

    if len(true_labs) == 0:
        raise ValueError("true_labs cannot be empty")

    if len(true_labs) != len(pred_prob):
        raise ValueError(
            f"Length mismatch: true_labs ({len(true_labs)}) vs "
            f"pred_prob ({len(pred_prob)})"
        )

    # Get metric function
    try:
        metric_func = METRIC_REGISTRY[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {metric}") from exc

    # Handle edge case: single prediction
    if len(pred_prob) == 1:
        return float(pred_prob[0])

    # Sort by predicted probability in descending order for efficiency
    sort_idx = np.argsort(-pred_prob)
    sorted_probs = pred_prob[sort_idx]
    sorted_labels = true_labs[sort_idx]

    # Handle sample weights
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(true_labs):
            raise ValueError(
                f"Length mismatch: sample_weight ({len(sample_weight)}) vs "
                f"true_labs ({len(true_labs)})"
            )
        # Sort weights along with labels and probabilities
        weights_sorted = sample_weight[sort_idx]
    else:
        weights_sorted = np.ones(len(true_labs))

    # Compute total positives and negatives (weighted)
    P = float(np.sum(weights_sorted * sorted_labels))
    N = float(np.sum(weights_sorted * (1 - sorted_labels)))

    # Handle edge case: all same class
    if P == 0 or N == 0:
        return 0.5

    # Find unique probabilities to use as threshold candidates
    unique_probs = np.unique(pred_prob)

    best_score = -np.inf
    best_threshold = 0.5

    # Cumulative sums for TP and FP (weighted)
    cum_tp = np.cumsum(weights_sorted * sorted_labels)
    cum_fp = np.cumsum(weights_sorted * (1 - sorted_labels))

    # Evaluate at each unique threshold
    for threshold in unique_probs:
        # Find position where we switch from positive to negative predictions
        # Apply comparison operator for thresholding
        if comparison == ">":
            pos_mask = sorted_probs > threshold
        else:  # ">="
            pos_mask = sorted_probs >= threshold

        if np.any(pos_mask):
            # Find last position where condition is satisfied
            last_pos = np.where(pos_mask)[0][-1]
            tp = float(cum_tp[last_pos])
            fp = float(cum_fp[last_pos])
        else:
            # No predictions above threshold -> all negative
            tp = fp = 0.0

        fn = P - tp
        tn = N - fp

        # Compute metric score
        score = float(metric_func(int(tp), int(tn), int(fp), int(fn)))

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def _dinkelbach_expected_fbeta(
    y_true: ArrayLike, pred_prob: ArrayLike, beta: float = 1.0
) -> float:
    """Dinkelbach method for exact expected F-beta optimization under calibration.

    **Important**: This method optimizes the *expected* F-beta score under the
    assumption that predicted probabilities are perfectly calibrated. It may not
    give the optimal threshold for the actual F-beta score on the given dataset
    if the probabilities are miscalibrated.

    Solves max_k ((1+β²)S_k) / (β²P + k) where:
    - S_k = sum_{j<=k} p_(j) after sorting probabilities descending
    - P = sum_i y_i (total positive labels)
    - k ranges from 1 to n (number of samples)

    **Mathematical Assumptions**:
    1. Predicted probabilities are well-calibrated (p_i = P(y_i = 1 | p_i))
    2. The expected number of true positives at threshold τ is sum_{p_i > τ} p_i
    3. This differs from actual F-beta which uses actual TP/FP/FN counts

    **When to Use**:
    - When you believe your classifier is well-calibrated
    - When optimizing for expected performance rather than performance on this dataset
    - As a baseline comparison against other methods

    **When NOT to Use**:
    - When probabilities are poorly calibrated (many real-world classifiers)
    - When you need optimal performance on the specific dataset provided
    - When sample weights are required (not supported)

    Parameters
    ----------
    y_true : ArrayLike
        Array of true binary labels (0 or 1).
    pred_prob : ArrayLike
        Predicted probabilities from a classifier. Should be well-calibrated.
    beta : float, default=1.0
        Beta parameter for F-beta score. beta=1.0 gives F1 score.

    Returns
    -------
    float
        Threshold that maximizes expected F-beta score under calibration.

    References
    ----------
    Based on Dinkelbach's algorithm for fractional programming.
    Exact for expected F-beta under perfect calibration assumptions.

    See: Ye, N., Chai, K. M. A., Lee, W. S., & Chieu, H. L. (2012).
    Optimizing F-measures: a tale of two approaches. ICML.
    """
    y = np.asarray(y_true)
    p = np.asarray(pred_prob)

    # Sort probabilities in descending order
    idx = np.argsort(-p, kind="mergesort")
    p_sorted = p[idx]

    # Cumulative sum of sorted probabilities
    S = np.cumsum(p_sorted)
    P = y.sum()  # Total positive labels

    # Compute F-beta objective: (1+β²)S_k / (β²P + k)
    beta2 = beta * beta
    numer = (1.0 + beta2) * S
    denom = beta2 * P + (np.arange(p_sorted.size) + 1)
    f = numer / denom

    # Find k that maximizes the objective
    k = int(np.argmax(f))

    # Return midpoint threshold between k-th and (k+1)-th sorted probabilities
    left = p_sorted[k]
    right = p_sorted[k + 1] if k + 1 < p_sorted.size else left

    if right != left:
        thr = float(0.5 * (left + right))
    else:
        # Handle case where left == right (tied probabilities)
        thr = float(np.nextafter(left, 1.0))

    return thr


def get_optimal_threshold(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
) -> float | np.ndarray[Any, Any]:
    """Find the threshold that optimizes a metric.

    Parameters
    ----------
    true_labs:
        Array of true binary labels or multiclass labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Predicted probabilities from a classifier. For binary: 1D array (n_samples,).
        For multiclass: 2D array (n_samples, n_classes).
    metric:
        Name of a metric registered in :data:`~optimal_cutoffs.metrics.METRIC_REGISTRY`.
    method:
        Strategy used for optimization:
        - ``"auto"``: Automatically selects best method (default)
        - ``"sort_scan"``: O(n log n) algorithm for piecewise metrics with
          vectorized implementation
        - ``"smart_brute"``: Evaluates all unique probabilities
        - ``"minimize"``: Uses ``scipy.optimize.minimize_scalar``
        - ``"gradient"``: Simple gradient ascent
        - ``"dinkelbach"``: Exact expected F-beta optimization (F1 only)
    sample_weight:
        Optional array of sample weights for handling imbalanced datasets.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    float | np.ndarray
        For binary: The threshold that maximizes the chosen metric.
        For multiclass: Array of per-class thresholds.
    """
    # Validate inputs
    true_labs, pred_prob, sample_weight = _validate_inputs(
        true_labs, pred_prob, sample_weight=sample_weight
    )
    _validate_metric_name(metric)
    _validate_optimization_method(method)
    _validate_comparison_operator(comparison)

    # Check if this is multiclass
    if pred_prob.ndim == 2:
        return get_optimal_multiclass_thresholds(
            true_labs,
            pred_prob,
            metric,
            method,
            average="macro",
            sample_weight=sample_weight,
            comparison=comparison,
        )

    # Binary case - implement method routing with auto detection
    if method == "auto":
        # Auto routing: prefer sort_scan for piecewise metrics with vectorized
        # implementation
        if is_piecewise_metric(metric) and has_vectorized_implementation(metric):
            method = "sort_scan"
        else:
            method = "smart_brute"

    if method == "sort_scan":
        # Use O(n log n) sort-and-scan optimization for vectorized piecewise metrics
        if not has_vectorized_implementation(metric):
            raise ValueError(
                f"sort_scan method requires vectorized implementation for "
                f"metric '{metric}'"
            )

        vectorized_metric = get_vectorized_metric(metric)
        threshold, _, _ = optimal_threshold_sortscan(
            true_labs,
            pred_prob,
            vectorized_metric,
            sample_weight=sample_weight,
            inclusive=comparison,
        )
        return threshold

    if method == "smart_brute":
        # Use fast piecewise optimization for piecewise-constant metrics
        if is_piecewise_metric(metric):
            return _optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        else:
            # Fall back to original brute force for non-piecewise metrics
            thresholds = np.unique(pred_prob)
            scores = [
                _metric_score(
                    true_labs, pred_prob, t, metric, sample_weight, comparison
                )
                for t in thresholds
            ]
            return float(thresholds[int(np.argmax(scores))])

    if method == "minimize":
        res = optimize.minimize_scalar(
            lambda t: -_metric_score(
                true_labs, pred_prob, t, metric, sample_weight, comparison
            ),
            bounds=(0, 1),
            method="bounded",
        )
        # ``minimize_scalar`` may return a threshold that is suboptimal for
        # piecewise-constant metrics like F1. To provide a more robust
        # solution, use the same enhanced candidate generation as smart_brute.
        if is_piecewise_metric(metric) and has_vectorized_implementation(metric):
            # For piecewise metrics, use the same optimal threshold as smart_brute
            piecewise_threshold = _optimal_threshold_piecewise(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
            # Compare scipy result with piecewise result
            scipy_score = _metric_score(
                true_labs, pred_prob, res.x, metric, sample_weight, comparison
            )
            piecewise_score = _metric_score(
                true_labs,
                pred_prob,
                piecewise_threshold,
                metric,
                sample_weight,
                comparison,
            )

            if piecewise_score >= scipy_score:
                return float(piecewise_threshold)
            else:
                return float(res.x)
        else:
            # Fall back to original candidate evaluation for non-piecewise metrics
            candidates = np.unique(np.append(pred_prob, res.x))
            scores = [
                _metric_score(
                    true_labs, pred_prob, t, metric, sample_weight, comparison
                )
                for t in candidates
            ]
            return float(candidates[int(np.argmax(scores))])

    if method == "gradient":
        threshold = 0.5
        lr = 0.1
        eps = 1e-5
        for _ in range(100):
            # Ensure evaluation points are within bounds
            thresh_plus = np.clip(threshold + eps, 0.0, 1.0)
            thresh_minus = np.clip(threshold - eps, 0.0, 1.0)

            grad = (
                _metric_score(
                    true_labs, pred_prob, thresh_plus, metric, sample_weight, comparison
                )
                - _metric_score(
                    true_labs,
                    pred_prob,
                    thresh_minus,
                    metric,
                    sample_weight,
                    comparison,
                )
            ) / (2 * eps)
            threshold = np.clip(threshold + lr * grad, 0.0, 1.0)
        # Final safety clip to ensure numerical precision doesn't cause issues
        return float(np.clip(threshold, 0.0, 1.0))

    if method == "dinkelbach":
        # Use Dinkelbach method for exact expected F-beta optimization
        if metric == "f1":
            beta = 1.0
        else:
            raise ValueError(
                f"dinkelbach method currently only supports F1 metric, got '{metric}'"
            )

        if sample_weight is not None:
            raise ValueError("dinkelbach method does not support sample weights")

        return _dinkelbach_expected_fbeta(true_labs, pred_prob, beta)

    raise ValueError(f"Unknown method: {method}")


def get_optimal_multiclass_thresholds(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "auto",
    average: AveragingMethod = "macro",
    sample_weight: ArrayLike | None = None,
    vectorized: bool = False,
    comparison: ComparisonOperator = ">",
) -> np.ndarray[Any, Any] | float:
    """Find optimal per-class thresholds for multiclass classification using
    One-vs-Rest.

    Parameters
    ----------
    true_labs:
        Array of true class labels (0, 1, 2, ..., n_classes-1).
    pred_prob:
        Array of predicted probabilities with shape (n_samples, n_classes).
    metric:
        Name of a metric registered in :data:`~optimal_cutoffs.metrics.METRIC_REGISTRY`.
    method:
        Strategy used for optimization:
        - ``"auto"``: Automatically selects best method (default)
        - ``"sort_scan"``: O(n log n) algorithm for piecewise metrics with
          vectorized implementation
        - ``"smart_brute"``: Evaluates all unique probabilities
        - ``"minimize"``: Uses ``scipy.optimize.minimize_scalar``
        - ``"gradient"``: Simple gradient ascent
        - ``"coord_ascent"``: Coordinate ascent for coupled multiclass
          optimization (single-label consistent)
    average:
        Averaging strategy that affects optimization:
        - "macro"/"none": Optimize each class independently (default behavior)
        - "micro": Optimize to maximize micro-averaged metric across all classes
        - "weighted": Optimize each class independently, same as macro
    sample_weight:
        Optional array of sample weights for handling imbalanced datasets.
    vectorized:
        If True, use vectorized implementation for better performance when possible.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    np.ndarray | float
        For "macro"/"weighted"/"none": Array of optimal thresholds, one per class.
        For "micro" with single threshold strategy: Single optimal threshold.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)

    # Input validation
    if len(true_labs) == 0:
        raise ValueError("true_labs cannot be empty")

    if pred_prob.ndim != 2:
        raise ValueError(f"pred_prob must be 2D for multiclass, got {pred_prob.ndim}D")

    if len(true_labs) != pred_prob.shape[0]:
        raise ValueError(
            f"Length mismatch: true_labs ({len(true_labs)}) vs "
            f"pred_prob ({pred_prob.shape[0]})"
        )

    if np.any(np.isnan(pred_prob)) or np.any(np.isinf(pred_prob)):
        raise ValueError("pred_prob contains NaN or infinite values")

    # Check class labels are valid for One-vs-Rest
    unique_labels = np.unique(true_labs)
    expected_labels = np.arange(len(unique_labels))
    if not np.array_equal(np.sort(unique_labels), expected_labels):
        raise ValueError(
            f"Class labels must be consecutive integers starting from 0. "
            f"Got {unique_labels}, expected {expected_labels}"
        )

    n_classes = pred_prob.shape[1]

    if average == "micro":
        # For micro-averaging, we can either:
        # 1. Pool all OvR problems and optimize a single threshold
        # 2. Optimize per-class thresholds to maximize micro-averaged metric
        # We implement approach 2 for more flexibility
        return _optimize_micro_averaged_thresholds(
            true_labs, pred_prob, metric, method, sample_weight, vectorized, comparison
        )
    elif method == "coord_ascent":
        # Coordinate ascent for coupled multiclass optimization
        if sample_weight is not None:
            raise NotImplementedError(
                "coord_ascent method does not yet support sample weights"
            )
        if comparison != ">":
            raise NotImplementedError(
                "coord_ascent method currently only supports '>' comparison"
            )
        if metric != "f1":
            raise NotImplementedError(
                "coord_ascent method currently only supports F1 metric"
            )

        # Use vectorized F1 metric for sort-scan initialization
        if has_vectorized_implementation(metric):
            vectorized_metric = get_vectorized_metric(metric)
        else:
            raise ValueError(
                f"coord_ascent requires vectorized implementation for metric '{metric}'"
            )

        tau, _, _ = optimal_multiclass_thresholds_coord_ascent(
            true_labs,
            pred_prob,
            sortscan_metric_fn=vectorized_metric,
            sortscan_kernel=optimal_threshold_sortscan,
            max_iter=20,
            init="ovr_sortscan",
            tol_stops=1,
        )
        return tau
    else:
        # For macro, weighted, none: optimize each class independently
        if vectorized and method == "smart_brute" and is_piecewise_metric(metric):
            return _optimize_thresholds_vectorized(
                true_labs, pred_prob, metric, sample_weight, comparison
            )
        else:
            # Standard per-class optimization
            optimal_thresholds = np.zeros(n_classes)
            for class_idx in range(n_classes):
                # One-vs-Rest: current class vs all others
                true_binary = (true_labs == class_idx).astype(int)
                pred_binary_prob = pred_prob[:, class_idx]

                # Optimize threshold for this class
                optimal_thresholds[class_idx] = get_optimal_threshold(
                    true_binary,
                    pred_binary_prob,
                    metric,
                    method,
                    sample_weight,
                    comparison,
                )
            return optimal_thresholds


def _optimize_micro_averaged_thresholds(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str,
    method: OptimizationMethod,
    sample_weight: ArrayLike | None,
    vectorized: bool,
    comparison: ComparisonOperator = ">",
) -> np.ndarray[Any, Any]:
    """Optimize thresholds to maximize micro-averaged metric.

    For micro-averaging, we optimize per-class thresholds jointly to maximize
    the micro-averaged metric score across all classes.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    n_classes = pred_prob.shape[1]

    def objective(thresholds: np.ndarray[Any, Any]) -> float:
        """Objective function: negative micro-averaged metric."""
        if metric == "accuracy":
            # For accuracy, use exclusive single-label metric instead of OvR micro
            score = multiclass_metric_exclusive(
                true_labs, pred_prob, thresholds, metric, comparison, sample_weight
            )
        else:
            # For other metrics, use OvR micro-averaging
            cms = get_multiclass_confusion_matrix(
                true_labs, pred_prob, thresholds, sample_weight, comparison
            )
            score_result = multiclass_metric(cms, metric, "micro")
            score = (
                float(score_result)
                if isinstance(score_result, np.ndarray)
                else score_result
            )
        return -float(score)

    if method == "smart_brute":
        # For micro-averaging with smart_brute, we need to search over combinations
        # of thresholds. Start with independent optimization as initial guess.
        initial_thresholds = np.zeros(n_classes)
        for class_idx in range(n_classes):
            true_binary = (true_labs == class_idx).astype(int)
            pred_binary_prob = pred_prob[:, class_idx]
            initial_thresholds[class_idx] = get_optimal_threshold(
                true_binary,
                pred_binary_prob,
                metric,
                "smart_brute",
                sample_weight,
                comparison,
            )

        # For now, return the independent optimization result
        # TODO: Implement joint optimization for micro-averaging
        return initial_thresholds

    elif method in ["minimize", "gradient"]:
        # Use scipy optimization for joint threshold optimization
        from scipy.optimize import minimize  # type: ignore[import-untyped]

        # Initial guess: independent optimization per class
        initial_guess = np.zeros(n_classes)
        for class_idx in range(n_classes):
            true_binary = (true_labs == class_idx).astype(int)
            pred_binary_prob = pred_prob[:, class_idx]
            initial_guess[class_idx] = get_optimal_threshold(
                true_binary,
                pred_binary_prob,
                metric,
                "minimize",
                sample_weight,
                comparison,
            )

        # Joint optimization
        result = minimize(
            objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=[(0, 1) for _ in range(n_classes)],
        )

        return np.asarray(result.x)

    else:
        raise ValueError(f"Unknown method: {method}")


def _optimize_thresholds_vectorized(
    true_labs: ArrayLike,
    pred_prob: ArrayLike,
    metric: str,
    sample_weight: ArrayLike | None,
    comparison: ComparisonOperator = ">",
) -> np.ndarray[Any, Any]:
    """Vectorized optimization for piecewise metrics.

    This function vectorizes the piecewise threshold optimization
    across all classes for better performance.
    """
    true_labs = np.asarray(true_labs)
    pred_prob = np.asarray(pred_prob)
    n_samples, n_classes = pred_prob.shape

    # Create binary labels for all classes at once: (n_samples, n_classes)
    true_binary_all = (true_labs[:, None] == np.arange(n_classes)).astype(int)

    optimal_thresholds = np.zeros(n_classes)

    # For now, fall back to per-class optimization
    # TODO: Implement fully vectorized version
    for class_idx in range(n_classes):
        optimal_thresholds[class_idx] = _optimal_threshold_piecewise(
            true_binary_all[:, class_idx],
            pred_prob[:, class_idx],
            metric,
            sample_weight,
            comparison,
        )

    return optimal_thresholds


__all__ = [
    "get_probability",
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
]
