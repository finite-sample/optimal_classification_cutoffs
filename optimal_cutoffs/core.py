"""Core threshold optimization dispatcher based on mathematical problem taxonomy.

This module provides the main get_optimal_threshold() function that automatically
routes to the appropriate optimization algorithm based on problem characteristics
and theoretical properties.

Design principle: Problem type determines algorithm, not user guessing.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from .types_minimal import OptimizationResult

logger = logging.getLogger(__name__)


def infer_problem_type(
    true_labels: ArrayLike | None,
    pred_proba: ArrayLike,
) -> tuple[str, dict]:
    """Automatically detect problem type from data characteristics.

    Returns
    -------
    problem_type : str
        One of: "binary", "multilabel", "multiclass"
    info : dict
        Additional information about the problem
    """
    pred_proba = np.asarray(pred_proba)

    if pred_proba.ndim == 1:
        return "binary", {"n_classes": 2, "n_samples": len(pred_proba)}

    if pred_proba.ndim == 2:
        n_samples, n_outputs = pred_proba.shape

        if n_outputs == 1:
            return "binary", {"n_classes": 2, "n_samples": n_samples}

        if n_outputs == 2:
            # Could be binary (with [P(0), P(1)]) or 2-class multiclass
            # Check if probabilities sum to 1 (multiclass) or are independent (multilabel)
            if true_labels is not None:
                true_labels = np.asarray(true_labels)
                unique_labels = np.unique(true_labels)

                # If labels are in {0, 1}, could be either
                if set(unique_labels) <= {0, 1}:
                    # Check if probabilities sum to ~1 (multiclass) or not (multilabel)
                    prob_sums = np.sum(pred_proba, axis=1)
                    if np.allclose(prob_sums, 1.0, rtol=0.05):
                        return "multiclass", {"n_classes": 2, "n_samples": n_samples}
                    else:
                        return "multilabel", {"n_classes": 2, "n_samples": n_samples}
                else:
                    return "multiclass", {"n_classes": 2, "n_samples": n_samples}
            else:
                # Without labels, check probability sums
                prob_sums = np.sum(pred_proba, axis=1)
                if np.allclose(prob_sums, 1.0, rtol=0.05):
                    return "multiclass", {"n_classes": 2, "n_samples": n_samples}
                else:
                    return "multilabel", {"n_classes": 2, "n_samples": n_samples}

        # More than 2 outputs
        if true_labels is not None:
            true_labels = np.asarray(true_labels)

            # Check label structure
            if true_labels.ndim == 1:
                # Single label per sample -> multiclass
                unique_labels = np.unique(true_labels)
                expected_classes = len(unique_labels)

                if expected_classes == n_outputs:
                    return "multiclass", {
                        "n_classes": n_outputs,
                        "n_samples": n_samples,
                    }
                else:
                    # Labels might be subset of classes
                    return "multiclass", {
                        "n_classes": n_outputs,
                        "n_samples": n_samples,
                    }

            elif true_labels.ndim == 2:
                # Multiple labels per sample -> multilabel
                return "multilabel", {"n_classes": n_outputs, "n_samples": n_samples}

        # Without labels, use probability sums as heuristic
        prob_sums = np.sum(pred_proba, axis=1)
        if np.allclose(prob_sums, 1.0, rtol=0.05):
            return "multiclass", {"n_classes": n_outputs, "n_samples": n_samples}
        else:
            return "multilabel", {"n_classes": n_outputs, "n_samples": n_samples}

    raise ValueError(f"Cannot infer problem type from shape {pred_proba.shape}")


def _map_legacy_method(method: str) -> str:
    """Map legacy method names to new taxonomy."""
    # Deprecated methods that should be rejected
    deprecated_methods = {"dinkelbach", "smart_brute"}
    if method in deprecated_methods:
        return method  # Return as-is to trigger validation error

    # Valid legacy mappings
    legacy_mapping = {
        "sort_scan": "auto",  # sort_scan was our O(n log n) method, now auto
        "unique_scan": "auto",  # unique_scan was for evaluating unique probabilities
        "grid_search": "auto",  # grid_search -> auto selection
        "brute_force": "auto",  # brute_force -> auto selection
        "minimize": "minimize",  # keep minimize method as-is
        "gradient": "gradient",  # keep gradient method as-is
    }
    return legacy_mapping.get(method, method)


def get_optimal_threshold(
    true_labels: ArrayLike | None,
    pred_proba: ArrayLike,
    *,
    # What to optimize
    metric: str = "f1",
    utility: dict[str, float] | None = None,
    utility_matrix: np.ndarray | None = None,
    cost_matrix: np.ndarray | None = None,
    # How to optimize
    method: str = "auto",
    mode: str = "empirical",
    # Multi-class/label options
    average: str = "macro",
    # Algorithm parameters
    beta: float = 1.0,
    max_iter: int = 30,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Find optimal classification thresholds using mathematical problem taxonomy.

    This function automatically detects the problem type (binary, multilabel, or
    multiclass) and routes to the theoretically appropriate optimization algorithm.

    Problem Classification:
    ----------------------
    - **Binary**: 1D probabilities or 2D with single column
    - **Multilabel**: K independent binary labels (probabilities don't sum to 1)
    - **Multiclass**: K mutually exclusive classes (probabilities sum to 1)

    Algorithm Selection:
    -------------------
    - **Binary F-measures**: Sort-and-scan O(n log n) - exact optimum
    - **Binary utilities**: Closed-form O(1) - Bayes optimal
    - **Multilabel macro**: K independent optimizations - exact
    - **Multilabel micro**: Coordinate ascent - local optimum
    - **Multiclass macro**: Coordinate ascent with margin rule - local optimum
    - **Multiclass micro**: Single threshold optimization - exact
    - **General cost matrix**: Direct Bayes rule O(K²) - exact Bayes optimal

    Parameters
    ----------
    true_labels : array-like or None
        True labels. Shape depends on problem:
        - Binary: (n_samples,) with values in {0, 1}
        - Multilabel: (n_samples, n_labels) binary matrix
        - Multiclass: (n_samples,) with values in {0, ..., K-1}
        Can be None for mode="bayes" with utility_matrix/cost_matrix
    pred_proba : array-like
        Predicted probabilities (must be calibrated!):
        - Binary: (n_samples,) or (n_samples, 1) or (n_samples, 2)
        - Multilabel: (n_samples, n_labels)
        - Multiclass: (n_samples, n_classes) with rows summing to 1
    metric : str, default="f1"
        Metric to optimize: "f1", "precision", "recall", "accuracy", etc.
    utility : dict, optional
        For binary problems: {"tp": 10, "tn": 1, "fp": -1, "fn": -5}
    utility_matrix : array of shape (n_decisions, n_classes), optional
        General utility matrix U[d,y] = utility(decision=d, true_class=y)
    cost_matrix : array of shape (n_decisions, n_classes), optional
        General cost matrix C[d,y] = cost(decision=d, true_class=y)
    method : str, default="auto"
        Optimization method:
        - "auto": Select best method based on problem characteristics
        - "sort_scan": O(n log n) for piecewise metrics
        - "coord_ascent": Coordinate ascent for coupled problems
        - "independent": Independent per-class optimization
        - "minimize": Scipy optimization
    mode : str, default="empirical"
        Optimization mode:
        - "empirical": Standard threshold optimization on observed data
        - "expected": Expected metric optimization using Dinkelbach method
        - "bayes": Bayes-optimal decisions (requires utility/cost specification)
    average : str, default="macro"
        For multiclass/multilabel: "macro", "micro", "weighted"
    beta : float, default=1.0
        F-beta parameter (beta=1 gives F1 score)
    max_iter : int, default=30
        Maximum iterations for coordinate ascent
    sample_weight : array-like, optional
        Sample weights for handling class imbalance
    comparison : str, default=">"
        Comparison operator for thresholding: ">" or ">="
    tolerance : float, default=1e-10
        Numerical tolerance for optimization algorithms

    Returns
    -------
    OptimizationResult
        Unified result with:
        - thresholds: Optimal threshold(s)
        - scores: Metric value(s) at optimal thresholds
        - predict: Function for making predictions
        - metric: Name of optimized metric
        - n_classes: Number of classes/labels

    Examples
    --------
    >>> # Binary F1 optimization
    >>> result = get_optimal_threshold(y_true, y_prob, metric="f1")
    >>> threshold = result.threshold  # Scalar for binary
    >>> predictions = result.predict(y_prob)

    >>> # Binary cost-sensitive (closed form)
    >>> utility = {"tp": 10, "tn": 1, "fp": -1, "fn": -5}
    >>> result = get_optimal_threshold(y_true, y_prob, utility=utility)

    >>> # Multiclass with margin rule (coordinate ascent)
    >>> result = get_optimal_threshold(y_true, y_prob, method="coord_ascent")
    >>> thresholds = result.thresholds  # Per-class
    >>> predictions = result.predict(y_prob)  # argmax(p - tau)

    >>> # General cost matrix (no thresholds!)
    >>> cost_matrix = np.array([[0, 10, 50], [10, 0, 40], [100, 90, 0]])
    >>> result = get_optimal_threshold(None, y_prob, cost_matrix=cost_matrix)
    >>> predictions = result.predict(y_prob)  # Direct Bayes rule

    >>> # Expected F1 under calibration assumption
    >>> result = get_optimal_threshold(None, y_prob, mode="expected", metric="f1")

    Notes
    -----
    **Calibration assumption**: This library assumes predicted probabilities
    are calibrated (E[y|p] = p). Use calibration methods before optimization.

    **Complexity guide**:
    - Binary metrics: O(n log n) via sort-and-scan
    - Binary utilities: O(1) closed form
    - Multilabel macro: O(K·n log n) independent optimization
    - Multilabel micro: O(iter·K·n log n) coordinate ascent
    - Multiclass margin: O(iter·K·n log n) coordinate ascent
    - General costs: O(K²) per prediction (no thresholds needed)

    **When NOT to use thresholds**: For general cost matrices where costs
    depend on both true and predicted class, use cost_matrix parameter
    which implements the exact Bayes rule without thresholds.
    """
    # Handle special case: general cost/utility matrices
    if utility_matrix is not None or cost_matrix is not None:
        if mode != "empirical":
            mode = "bayes"  # Force Bayes mode for matrix-based optimization

        from .bayes import bayes_optimal_decisions

        return bayes_optimal_decisions(
            pred_proba, utility_matrix=utility_matrix, cost_matrix=cost_matrix
        )

    # Validate comparison operator
    if comparison not in (">", ">="):
        raise ValueError(
            f"Invalid comparison operator: '{comparison}'. Must be '>' or '>='"
        )

    # Map legacy method names for backward compatibility
    method = _map_legacy_method(method)

    # Validate method names (after legacy mapping)
    valid_methods = {"auto", "coord_ascent", "independent", "minimize", "gradient"}
    if method not in valid_methods:
        raise ValueError(
            f"Invalid optimization method: '{method}'. Must be one of {valid_methods}"
        )

    # Validate metric names (check registry for registered metrics)
    if mode == "empirical" and utility is None:
        from .metrics import METRICS

        if metric.lower() not in METRICS:
            available_metrics = set(METRICS.keys())
            raise ValueError(
                f"Unknown metric: '{metric}'. Available metrics: {available_metrics}"
            )

    # Detect problem type
    problem_type, problem_info = infer_problem_type(true_labels, pred_proba)
    n_samples = problem_info["n_samples"]
    n_classes = problem_info["n_classes"]
    logger.debug(
        f"Detected {problem_type=} with {n_samples=} samples, {n_classes=} classes"
    )

    # Route based on mode
    logger.debug(f"Using {mode=} with {method=} for {metric=}")
    match mode:
        case "empirical":
            return _optimize_empirical(
                problem_type,
                true_labels,
                pred_proba,
                metric,
                method,
                average,
                sample_weight,
                comparison,
                tolerance,
                max_iter,
                beta,
            )
        case "expected":
            return _optimize_expected(
                problem_type,
                true_labels,
                pred_proba,
                metric,
                average,
                sample_weight,
                comparison,
                tolerance,
                beta,
            )
        case "bayes":
            return _optimize_bayes(
                problem_type, true_labels, pred_proba, utility, sample_weight
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def _optimize_empirical(
    problem_type: str,
    true_labels: ArrayLike | None,
    pred_proba: ArrayLike,
    metric: str,
    method: str,
    average: str,
    sample_weight: ArrayLike | None,
    comparison: str,
    tolerance: float,
    max_iter: int,
    beta: float,
) -> OptimizationResult:
    """Route empirical optimization to appropriate algorithm."""
    if true_labels is None:
        raise ValueError("true_labels required for empirical optimization")

    match problem_type:
        case "binary":
            logger.debug("Routing to binary optimization")
            from .binary import optimize_f1_binary, optimize_metric_binary

            if metric in ("f1", "fbeta"):
                return optimize_f1_binary(
                    true_labels,
                    pred_proba,
                    beta=beta,
                    sample_weight=sample_weight,
                    comparison=comparison,
                )
            else:
                return optimize_metric_binary(
                    true_labels,
                    pred_proba,
                    metric=metric,
                    method=method,
                    sample_weight=sample_weight,
                    comparison=comparison,
                    tolerance=tolerance,
                )

        case "multilabel":
            logger.debug(f"Routing to multilabel optimization with {average=}")
            from .multilabel import optimize_multilabel

            return optimize_multilabel(
                true_labels,
                pred_proba,
                metric=metric,
                average=average,
                method=method,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
            )

        case "multiclass":
            logger.debug(
                f"Routing to multiclass optimization with {average=} and {method=}"
            )
            from .multiclass import optimize_multiclass

            return optimize_multiclass(
                true_labels,
                pred_proba,
                metric=metric,
                average=average,
                method=method,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
            )

        case _:
            raise ValueError(f"Unknown problem type: {problem_type}")


def _optimize_expected(
    problem_type: str,
    true_labels: ArrayLike | None,
    pred_proba: ArrayLike,
    metric: str,
    average: str,
    sample_weight: ArrayLike | None,
    comparison: str,
    tolerance: float,
    beta: float,
) -> OptimizationResult:
    """Route expected optimization to Dinkelbach algorithm."""
    if metric.lower() not in ("f1", "fbeta"):
        raise ValueError("mode='expected' currently supports F-beta only")

    from .expected import (
        dinkelbach_expected_fbeta_binary,
        dinkelbach_expected_fbeta_multilabel,
    )

    pred_proba = np.asarray(pred_proba, dtype=np.float64)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    if problem_type == "binary":
        # Binary expected F-beta
        if pred_proba.ndim == 2 and pred_proba.shape[1] == 2:
            pred_proba = pred_proba[:, 1]  # Extract positive class
        elif pred_proba.ndim == 2 and pred_proba.shape[1] == 1:
            pred_proba = pred_proba.ravel()

        result = dinkelbach_expected_fbeta_binary(
            pred_proba, beta=beta, sample_weight=sw, comparison=comparison
        )

        threshold = result.thresholds[0]
        score = result.scores[0]

        def predict_binary(probs: ArrayLike) -> np.ndarray:
            p = np.asarray(probs)
            if p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]
            elif p.ndim == 2 and p.shape[1] == 1:
                p = p.ravel()

            if comparison == ">=":
                return (p >= threshold).astype(np.int32)
            else:
                return (p > threshold).astype(np.int32)

        return OptimizationResult(
            thresholds=np.array([threshold], dtype=np.float64),
            scores=np.array([score], dtype=np.float64),
            predict=predict_binary,
            metric=f"expected_f{beta}_score" if beta != 1.0 else "expected_f1_score",
            n_classes=2,
        )

    else:
        # Multilabel/multiclass expected F-beta
        avg = average if average in ("macro", "micro", "weighted") else "macro"
        true_labels_arr = (
            None if true_labels is None else np.asarray(true_labels, dtype=int)
        )

        result = dinkelbach_expected_fbeta_multilabel(
            pred_proba,
            beta=beta,
            sample_weight=sw,
            average=avg,
            true_labels=true_labels_arr,
            comparison=comparison,
        )

        n_classes = pred_proba.shape[1]

        if result.thresholds.size == 1:  # Micro averaging
            threshold = float(result.thresholds[0])

            def predict_micro(probs: ArrayLike) -> np.ndarray:
                p = np.asarray(probs, dtype=np.float64)
                if p.ndim != 2 or p.shape[1] != n_classes:
                    raise ValueError(f"Expected shape (n_samples, {n_classes})")

                if comparison == ">=":
                    valid = p >= threshold
                else:
                    valid = p > threshold

                masked = np.where(valid, p, -np.inf)
                pred = np.argmax(masked, axis=1)
                no_valid = ~np.any(valid, axis=1)
                if np.any(no_valid):
                    pred[no_valid] = np.argmax(p[no_valid], axis=1)

                return pred.astype(np.int32)

            return OptimizationResult(
                thresholds=np.array([threshold], dtype=np.float64),
                scores=result.scores,
                predict=predict_micro,
                metric=f"expected_f{beta}_score_micro"
                if beta != 1.0
                else "expected_f1_score_micro",
                n_classes=n_classes,
            )

        else:  # Macro/weighted averaging
            thresholds = np.asarray(result.thresholds, dtype=np.float64)

            def predict_macro(probs: ArrayLike) -> np.ndarray:
                p = np.asarray(probs, dtype=np.float64)
                if p.ndim != 2 or p.shape[1] != n_classes:
                    raise ValueError(f"Expected shape (n_samples, {n_classes})")

                if comparison == ">=":
                    valid = p >= thresholds[None, :]
                else:
                    valid = p > thresholds[None, :]

                masked = np.where(valid, p, -np.inf)
                pred = np.argmax(masked, axis=1)
                no_valid = ~np.any(valid, axis=1)
                if np.any(no_valid):
                    pred[no_valid] = np.argmax(p[no_valid], axis=1)

                return pred.astype(np.int32)

            return OptimizationResult(
                thresholds=thresholds,
                scores=result.scores,
                predict=predict_macro,
                metric=f"expected_f{beta}_score_{avg}"
                if beta != 1.0
                else f"expected_f1_score_{avg}",
                n_classes=n_classes,
            )


def _optimize_bayes(
    problem_type: str,
    true_labels: ArrayLike | None,
    pred_proba: ArrayLike,
    utility: dict[str, float] | None,
    sample_weight: ArrayLike | None,
) -> OptimizationResult:
    """Route Bayes optimization to appropriate algorithm."""
    if problem_type == "binary":
        if utility is None:
            raise ValueError("mode='bayes' requires utility parameter")

        from .binary import optimize_utility_binary

        return optimize_utility_binary(
            true_labels, pred_proba, utility=utility, sample_weight=sample_weight
        )

    else:
        # For multiclass/multilabel, utility should specify per-class costs
        raise NotImplementedError(
            "Per-class utilities from matrix not yet implemented. "
            "Use UtilitySpec for OvR thresholds."
        )


__all__ = [
    "get_optimal_threshold",
    "infer_problem_type",
]
