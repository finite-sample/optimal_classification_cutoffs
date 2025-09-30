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

    # Handle mode-based routing
    if mode == "bayes":
        # Convert probabilities to numpy array
        pred_prob = np.asarray(pred_prob, dtype=float)
        is_binary = (pred_prob.ndim == 1) or (
            pred_prob.ndim == 2 and pred_prob.shape[1] == 1
        )

        # Handle utility matrix case (multiclass decisions)
        if utility_matrix is not None:
            result = bayes_decision_from_utility_matrix(pred_prob, utility_matrix)
            return result  # type: ignore[return-value]

        # Handle utility dict case
        if utility is None:
            raise ValueError(
                "mode='bayes' requires utility parameter or utility_matrix to be "
                "specified"
            )

        if is_binary:
            # Binary case - use scalar closed-form
            u = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
            u.update({k: float(v) for k, v in utility.items()})
            if minimize_cost:
                u["fp"] = -abs(u["fp"])
                u["fn"] = -abs(u["fn"])

            # Handle single-column probability case
            if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
                pred_prob = pred_prob[:, 0]  # Flatten single column

            return bayes_threshold_from_costs_scalar(
                u["fp"], u["fn"], u["tp"], u["tn"], comparison=comparison
            )
        else:
            # Multiclass OvR (including 2-class as multiclass) - expect per-class
            # vectors in utility
            fp_costs = utility.get("fp", None)
            fn_costs = utility.get("fn", None)
            tp_benefits = utility.get("tp", None)
            tn_benefits = utility.get("tn", None)

            if fp_costs is None or fn_costs is None:
                raise ValueError(
                    "Multiclass Bayes requires 'fp' and 'fn' as arrays in utility dict"
                )

            # Convert to arrays if needed
            fp_array = np.asarray(fp_costs) if fp_costs is not None else fp_costs
            fn_array = np.asarray(fn_costs) if fn_costs is not None else fn_costs
            tp_array = (
                np.asarray(tp_benefits) if tp_benefits is not None else tp_benefits
            )
            tn_array = (
                np.asarray(tn_benefits) if tn_benefits is not None else tn_benefits
            )

            return bayes_thresholds_from_costs_vector(
                fp_array, fn_array, tp_array, tn_array
            )

    if mode == "expected":
        # Convert probabilities to numpy array
        pred_prob = np.asarray(pred_prob, dtype=float)
        is_binary = (pred_prob.ndim == 1) or (
            pred_prob.ndim == 2 and pred_prob.shape[1] == 1
        )

        if is_binary:
            # Binary case (1D array or 2D with 1 column)
            if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
                pred_prob = pred_prob[:, 0]  # Flatten single column

            # Convert sample_weight to array if needed
            sw = np.asarray(sample_weight) if sample_weight is not None else None

            # Get coefficients for the requested metric
            try:
                coeffs = coeffs_for_metric(
                    metric,
                    beta=beta,
                    tversky_alpha=0.5,  # Default values
                    tversky_beta=0.5,
                )

                # Use generalized Dinkelbach framework
                threshold, expected_score, direction = (
                    dinkelbach_expected_fractional_binary(
                        pred_prob,
                        coeffs,
                        sample_weight=sw,
                        comparison=comparison,
                    )
                )

                # Verify direction matches comparison (should be ">")
                if direction != ">":
                    # This is rare but possible for exotic metrics
                    pass  # Still return the result

                return (float(threshold), float(expected_score))

            except ValueError:
                # Fallback to F-beta specific implementation for unsupported metrics
                binary_result: tuple[float, float] = dinkelbach_expected_fbeta_binary(
                    pred_prob,
                    beta=beta,
                    sample_weight=sw,
                    comparison=comparison,
                )
                return binary_result
        else:
            # Multiclass/multilabel case (including 2-class as multiclass)
            # Convert "none" to "macro" for expected mode
            avg = "macro" if average == "none" else average

            # Convert sample_weight and class_weight to arrays if needed
            sw = np.asarray(sample_weight) if sample_weight is not None else None
            cw = np.asarray(class_weight) if class_weight is not None else class_weight

            # Try generalized framework first
            try:
                result_dict = dinkelbach_expected_fractional_ovr(
                    pred_prob,
                    metric,
                    beta=beta,
                    tversky_alpha=0.5,
                    tversky_beta=0.5,
                    average=avg,
                    sample_weight=sw,
                    class_weight=cw,
                    comparison=comparison,
                )

                if avg == "micro":
                    # Return dictionary for micro averaging (backward compatibility)
                    return {
                        "threshold": float(result_dict["threshold"]),
                        "f_beta": float(result_dict["score"]),
                    }
                else:
                    # Return dict for macro/weighted averaging (backward compatibility)
                    return {
                        "thresholds": cast(
                            np.ndarray[Any, Any], result_dict["thresholds"]
                        ),
                        "f_beta_per_class": cast(
                            np.ndarray[Any, Any], result_dict["per_class"]
                        ),
                        "f_beta": float(result_dict["score"]),
                    }

            except ValueError:
                # Fallback to F-beta specific implementation for unsupported metrics
                return dinkelbach_expected_fbeta_multilabel(
                    pred_prob,
                    beta=beta,
                    average=avg,
                    sample_weight=sw,
                    class_weight=cw,
                    comparison=comparison,
                )

    # mode="empirical" - handle utility/cost-based optimization first
    if utility is not None or minimize_cost:
        # Convert probabilities to numpy array
        pred_prob = np.asarray(pred_prob, dtype=float)

        # Parse utility dict: accept costs-only for convenience
        u = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        if utility is not None:
            u.update({k: float(v) for k, v in utility.items()})
        if minimize_cost:
            # Treat provided numbers as costs unless explicitly signed; keep tp/tn
            u["fp"] = -abs(u["fp"])
            u["fn"] = -abs(u["fn"])
            # leave tp/tn as-is (benefits), often 0

        # Handle multiclass case
        if pred_prob.ndim == 2:
            raise NotImplementedError(
                "Utility/cost-based optimization not yet implemented for multiclass. "
                "Binary classification only for now."
            )

        # Empirical optimum via sort-scan on linear counts objective
        if true_labs is None:
            raise ValueError("true_labs is required for empirical utility optimization")

        true_labs = np.asarray(true_labs)
        sample_weight_array = (
            np.asarray(sample_weight) if sample_weight is not None else None
        )

        metric_fn = make_linear_counts_metric(
            u["tp"], u["tn"], u["fp"], u["fn"], name="user_utility"
        )
        # Use exact kernel (vectorized, weighted, inclusive/exclusive honored)
        thr, best, _ = optimal_threshold_sortscan(
            true_labs,
            pred_prob,
            metric_fn,
            inclusive=(comparison == ">="),
            sample_weight=sample_weight_array,
        )
        return float(thr)

    # Validate inputs for standard metric-based optimization
    if true_labs is None:
        raise ValueError("true_labs is required for empirical optimization")

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
            method = "unique_scan"

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
            inclusive=(comparison == ">="),
        )
        return threshold

    if method == "unique_scan":
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
        # solution, use the same enhanced candidate generation as unique_scan.
        if is_piecewise_metric(metric) and has_vectorized_implementation(metric):
            # For piecewise metrics, use the same optimal threshold as unique_scan
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
