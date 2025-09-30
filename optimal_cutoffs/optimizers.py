"""Threshold search strategies for optimizing classification metrics."""

from typing import Any, cast

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from .bayes import (
    bayes_decision_from_utility_matrix,
    bayes_threshold_from_costs_scalar,
    bayes_thresholds_from_costs_vector,
)
from .expected import (
    dinkelbach_expected_fbeta_binary,
    dinkelbach_expected_fbeta_multilabel,
)
from .expected_fractional import (
    coeffs_for_metric,
    dinkelbach_expected_fractional_binary,
    dinkelbach_expected_fractional_ovr,
)
from .metrics import (
    METRIC_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    get_vectorized_metric,
    has_vectorized_implementation,
    is_piecewise_metric,
    make_linear_counts_metric,
    multiclass_metric,
    multiclass_metric_exclusive,
)
from .multiclass_coord import optimal_multiclass_thresholds_coord_ascent
from .piecewise import optimal_threshold_sortscan
from .types import (
    ArrayLike,
    AveragingMethod,
    ComparisonOperator,
    EstimationMode,
    ExpectedResult,
    OptimizationMethod,
    UtilityDict,
    UtilityMatrix,
)
from .validation import (
    _validate_comparison_operator,
    _validate_inputs,
    _validate_metric_name,
    _validate_optimization_method,
)


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
                inclusive=(comparison == ">="),
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

    # Sort by predicted probability in descending order for efficiency (stable sort)
    sort_idx = np.argsort(-pred_prob, kind="mergesort")
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
    if P == 0.0:  # All negatives - optimal threshold should predict all negative
        max_prob = float(np.max(sorted_probs))
        return max_prob if comparison == ">" else float(np.nextafter(max_prob, np.inf))
    if N == 0.0:  # All positives - optimal threshold should predict all positive
        min_prob = float(np.min(sorted_probs))
        if comparison == ">":
            # For exclusive comparison, need threshold < min_prob, but ensure >= 0
            threshold = max(0.0, float(np.nextafter(min_prob, -np.inf)))
        else:
            # For inclusive comparison, threshold = min_prob works
            threshold = min_prob
        return threshold

    # Find unique probabilities to use as threshold candidates
    unique_probs = np.unique(pred_prob)

    best_score = -np.inf
    best_threshold = 0.5

    # Cumulative sums for TP and FP (weighted)
    cum_tp = np.cumsum(weights_sorted * sorted_labels)
    cum_fp = np.cumsum(weights_sorted * (1 - sorted_labels))

    # Evaluate at each unique threshold
    for threshold in unique_probs:
        # Use binary search to find cutoff position (more efficient than O(n) mask)
        # Since sorted_probs is descending, use negative values for searchsorted
        if comparison == ">":
            # Count of probabilities > threshold
            k = int(np.searchsorted(-sorted_probs, -threshold, side="left"))
        else:  # ">="
            # Count of probabilities >= threshold
            k = int(np.searchsorted(-sorted_probs, -threshold, side="right"))

        if k > 0:
            # k samples predicted as positive
            tp = float(cum_tp[k - 1])
            fp = float(cum_fp[k - 1])
        else:
            # No predictions above threshold -> all negative
            tp = fp = 0.0

        fn = P - tp
        tn = N - fp

        # Compute metric score (keep floating-point precision for weighted metrics)
        score = float(metric_func(tp, tn, fp, fn))

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def _dinkelbach_expected_fbeta(
    y_true: ArrayLike,
    pred_prob: ArrayLike,
    beta: float = 1.0,
    comparison: ComparisonOperator = ">",
) -> float:
    """Dinkelbach method for exact expected F-beta optimization under calibration.

    **Important**: This method optimizes the *expected* F-beta score under the
    assumption that predicted probabilities are perfectly calibrated. It may not
    give the optimal threshold for the actual F-beta score on the given dataset
    if the probabilities are miscalibrated.

    Solves max_k ((1+β²)S_k) / (β²P + k) where:
    - S_k = sum_{j<=k} p_(j) after sorting probabilities descending
    - P = sum_i p_i (expected total positive labels)
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
    comparison : ComparisonOperator, default=">"
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).

    Returns
    -------
    float
        Threshold that maximizes expected F-beta score under calibration.
    Notes
    -----
    This routine optimizes the expected Fβ under perfect calibration, and thus
    depends only on the predicted probabilities, not on the realized labels.

    References
    ----------
    Based on Dinkelbach's algorithm for fractional programming.
    Exact for expected F-beta under perfect calibration assumptions.

    See: Ye, N., Chai, K. M. A., Lee, W. S., & Chieu, H. L. (2012).
    Optimizing F-measures: a tale of two approaches. ICML.
    """
    p = np.asarray(pred_prob)

    # Sort probabilities in descending order
    idx = np.argsort(-p, kind="mergesort")
    p_sorted = p[idx]

    # Cumulative sum of sorted probabilities
    S = np.cumsum(p_sorted)
    P = p.sum()  # Expected total positive labels (sum of probabilities)

    # Compute F-beta objective: (1+β²)S_k / (β²P + k)
    beta2 = beta * beta
    numer = (1.0 + beta2) * S
    denom = beta2 * P + (np.arange(p_sorted.size) + 1)
    f = numer / denom

    # Find k that maximizes the objective
    k = int(np.argmax(f))

    # Return threshold between k-th and (k+1)-th sorted probabilities
    left = p_sorted[k]
    right = p_sorted[k + 1] if k + 1 < p_sorted.size else left

    if abs(right - left) > 1e-12:  # Not tied
        # Use midpoint when probabilities are different
        thr = float(0.5 * (left + right))
    else:
        # Handle tied probabilities based on comparison operator
        # For ties, set threshold to tied value and let comparison determine inclusion
        if comparison == ">":
            # With ">", threshold = tied_value excludes all tied elements
            # (since tied_value > tied_value is false)
            thr = float(left)
        else:  # ">="
            # With ">=", threshold = tied_value includes all tied elements
            # (since tied_value >= tied_value is true)
            thr = float(left)

    return thr


def get_optimal_threshold(
    true_labs: ArrayLike | None,
    pred_prob: ArrayLike,
    metric: str = "f1",
    method: OptimizationMethod = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: ComparisonOperator = ">",
    *,
    mode: EstimationMode = "empirical",
    utility: UtilityDict | None = None,
    utility_matrix: UtilityMatrix | None = None,
    minimize_cost: bool | None = None,
    beta: float = 1.0,
    average: AveragingMethod = "macro",
    class_weight: ArrayLike | None = None,
) -> float | np.ndarray[Any, Any] | ExpectedResult | tuple[float, float]:
    """Find the threshold that optimizes a metric or utility function.

    Parameters
    ----------
    true_labs:
        Array of true binary labels or multiclass labels (0, 1, 2, ..., n_classes-1).
        Not required when mode="bayes".
    pred_prob:
        Predicted probabilities from a classifier. For binary: 1D array (n_samples,).
        For multiclass: 2D array (n_samples, n_classes).
    metric:
        Name of a metric registered in :data:`~optimal_cutoffs.metrics.METRIC_REGISTRY`.
        Ignored if utility or minimize_cost is provided.
    method:
        Strategy used for optimization:
        - ``"auto"``: Automatically selects best method (default)
        - ``"sort_scan"``: O(n log n) algorithm for piecewise metrics with
          vectorized implementation
        - ``"unique_scan"``: Evaluates all unique probabilities
        - ``"minimize"``: Uses ``scipy.optimize.minimize_scalar``
        - ``"gradient"``: Simple gradient ascent
        - ``"coord_ascent"``: Coordinate ascent for coupled multiclass optimization
    sample_weight:
        Optional array of sample weights for handling imbalanced datasets.
    comparison:
        Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).
    mode:
        Estimation regime to use:
        - ``"empirical"``: Use method parameter for empirical optimization (default)
        - ``"bayes"``: Return Bayes-optimal threshold/decisions under calibrated
          probabilities
          (requires utility or utility_matrix, ignores method and true_labs)
        - ``"expected"``: Use Dinkelbach method for expected F-beta optimization
          (supports sample weights and multiclass, binary/multilabel)
    utility:
        Optional utility specification for cost/benefit-aware optimization.
        Dict with keys "tp", "tn", "fp", "fn" specifying utilities/costs per outcome.
        For multiclass mode="bayes", can contain per-class vectors.
        Example: ``{"tp": 0, "tn": 0, "fp": -1, "fn": -5}`` for cost-sensitive.
    utility_matrix:
        Alternative to utility dict for multiclass Bayes decisions.
        Shape (D, K) array where D=decisions, K=classes.
        If provided, returns class decisions rather than thresholds.
    minimize_cost:
        If True, interpret utility values as costs and minimize total cost. This
        automatically negates fp/fn values if they're positive.
    beta:
        F-beta parameter for expected mode (beta >= 0). beta=1 gives F1,
        beta < 1 emphasizes precision, beta > 1 emphasizes recall.
        Only used when mode="expected".
    average:
        Averaging strategy for multiclass expected mode:
        - "macro": per-class thresholds, unweighted mean F-beta
        - "weighted": per-class thresholds, class-weighted mean F-beta
        - "micro": single global threshold across all classes/instances
    class_weight:
        Optional per-class weights for weighted averaging in expected mode.
        Shape (K,) array. Only used when mode="expected" and average="weighted".

    Returns
    -------
    float | np.ndarray | dict
        - mode="empirical": float (binary) or ndarray (multiclass thresholds)
        - mode="bayes":
          * float (binary threshold) or ndarray (OvR thresholds) if using utility dict
          * ndarray (class decisions) if using utility_matrix
        - mode="expected":
          * tuple (threshold, f_beta) for binary
          * dict with "thresholds", "f_beta_per_class", "f_beta" for multiclass
            macro/weighted
          * dict with "threshold", "f_beta" for multiclass micro

    Examples
    --------
    >>> # Standard metric optimization
    >>> threshold = get_optimal_threshold(y, p, metric="f1")

    >>> # Cost-sensitive: FN costs 5x more than FP
    >>> threshold = get_optimal_threshold(y, p, utility={"fp": -1, "fn": -5})

    >>> # Bayes-optimal for cost scenario (calibrated)
    >>> threshold = get_optimal_threshold(None, p,
    ...     utility={"fp": -1, "fn": -5}, mode="bayes")

    >>> # Expected F1 optimization under calibration
    >>> threshold, f_beta = get_optimal_threshold(y, p, mode="expected", beta=1.0)

    >>> # Multiclass expected F-beta with macro averaging
    >>> result = get_optimal_threshold(y, p_multiclass, mode="expected",
    ...                              beta=2.0, average="macro")
    >>> print(result["thresholds"])  # Per-class thresholds
    >>> print(result["f_beta"])      # Macro-averaged F-beta

    >>> # Multiclass Bayes decisions with utility matrix
    >>> U = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])  # with abstain
    >>> decisions = get_optimal_threshold(None, p_multiclass,
    ...                                  utility_matrix=U, mode="bayes")
    """

    # Handle mode-based routing using new router pattern
    from .routers import get_router
    
    router = get_router(mode)
    
    return router.route(
        true_labs=true_labs,
        pred_prob=pred_prob,
        metric=metric,
        method=method,
        sample_weight=sample_weight,
        comparison=comparison,
        utility=utility,
        utility_matrix=utility_matrix,
        minimize_cost=minimize_cost,
        beta=beta,
        average=average,
        class_weight=class_weight,
    )


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
        - ``"unique_scan"``: Evaluates all unique probabilities
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

    n_classes = pred_prob.shape[1]

    # Check class labels are valid for the prediction probability matrix
    unique_labels = np.unique(true_labs)
    if np.any((unique_labels < 0) | (unique_labels >= n_classes)):
        raise ValueError(
            f"Labels {unique_labels} must be within [0, {n_classes - 1}] to match "
            f"pred_prob shape {pred_prob.shape}"
        )

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
                "coord_ascent method does not yet support sample weights. "
                "This limitation could be lifted by extending the per-class "
                "sort-scan kernel to handle weighted confusion matrices."
            )
        if comparison != ">":
            raise NotImplementedError(
                "coord_ascent method currently only supports '>' comparison. "
                "Support for '>=' could be added by passing the comparison "
                "parameter through to the per-class optimization kernel."
            )
        if metric != "f1":
            raise NotImplementedError(
                f"coord_ascent method only supports 'f1' metric, got '{metric}'. "
                "Support for other piecewise metrics (precision, recall, accuracy) "
                "could be added by extending the coordinate ascent implementation "
                "to use different vectorized metric functions."
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
        if vectorized and method in ["unique_scan"] and is_piecewise_metric(metric):
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
                result = get_optimal_threshold(
                    true_binary,
                    pred_binary_prob,
                    metric,
                    method,
                    sample_weight,
                    comparison,
                    mode="empirical",
                )
                # mode="empirical" guarantees float return for binary classification
                optimal_thresholds[class_idx] = cast(float, result)
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

    if method in ["unique_scan"]:
        # For micro-averaging with unique_scan, we need to search over combinations
        # of thresholds. Start with independent optimization as initial guess.
        initial_thresholds = np.zeros(n_classes)
        for class_idx in range(n_classes):
            true_binary = (true_labs == class_idx).astype(int)
            pred_binary_prob = pred_prob[:, class_idx]
            result = get_optimal_threshold(
                true_binary,
                pred_binary_prob,
                metric,
                "unique_scan",
                sample_weight,
                comparison,
                mode="empirical",
            )
            # mode="empirical" guarantees float return for binary classification
            initial_thresholds[class_idx] = cast(float, result)

        # LIMITATION: For unique_scan with micro averaging, we currently return
        # independent per-class optimization results (OvR initialization).
        # True joint optimization would require searching over threshold combinations,
        # which is computationally expensive. For joint optimization, use
        # method="minimize" which implements multi-dimensional optimization.
        import warnings

        warnings.warn(
            "unique_scan with micro averaging uses independent per-class optimization "
            "(OvR initialization), not true joint optimization. For joint optimization "
            "of micro-averaged metrics, use method='minimize'.",
            UserWarning,
            stacklevel=3,
        )
        return initial_thresholds

    elif method in ["minimize", "gradient"]:
        # Use scipy optimization for joint threshold optimization
        from scipy.optimize import minimize  # type: ignore[import-untyped]

        # Initial guess: independent optimization per class
        initial_guess = np.zeros(n_classes)
        for class_idx in range(n_classes):
            true_binary = (true_labs == class_idx).astype(int)
            pred_binary_prob = pred_prob[:, class_idx]
            result = get_optimal_threshold(
                true_binary,
                pred_binary_prob,
                metric,
                "minimize",
                sample_weight,
                comparison,
                mode="empirical",
            )
            # mode="empirical" guarantees float return for binary classification
            initial_guess[class_idx] = cast(float, result)

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
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
]
