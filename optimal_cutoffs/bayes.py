"""Bayes-optimal decisions and thresholds for classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray

from .types_minimal import OptimizationResult
from .validation import validate_classification

# ============================================================================
# Utility Specification
# ============================================================================


@dataclass(frozen=True, slots=True)
class UtilitySpec:
    """Complete utility specification for decision theory approaches."""

    tp_utility: float = 1.0
    tn_utility: float = 1.0
    fp_utility: float = -1.0
    fn_utility: float = -1.0

    # Note: compute_utility method temporarily removed to simplify migration

    @classmethod
    def from_costs(cls, fp_cost: float, fn_cost: float) -> Self:
        """Create from misclassification costs (converted to negative utilities)."""
        if not np.isfinite(fp_cost) or not np.isfinite(fn_cost):
            raise ValueError("Costs must be finite")
        return cls(
            tp_utility=0.0,
            tn_utility=0.0,
            fp_utility=-abs(fp_cost),
            fn_utility=-abs(fn_cost),
        )

    @classmethod
    def from_dict(cls, utility_dict: dict[str, float]) -> Self:
        """Create from dictionary with keys 'tp', 'tn', 'fp', 'fn'."""
        required_keys = {"tp", "tn", "fp", "fn"}
        if not all(key in utility_dict for key in required_keys):
            raise ValueError(f"Utility dict must contain keys: {required_keys}")

        # Validate all values are finite
        for key, value in utility_dict.items():
            if key in required_keys and not np.isfinite(value):
                raise ValueError(
                    f"Utility value for '{key}' must be finite, got {value}"
                )

        return cls(
            tp_utility=utility_dict["tp"],
            tn_utility=utility_dict["tn"],
            fp_utility=utility_dict["fp"],
            fn_utility=utility_dict["fn"],
        )


# ============================================================================
# Core Abstractions
# ============================================================================


class DecisionRule(Enum):
    """How to make decisions from utilities."""

    THRESHOLD = auto()  # Binary threshold on probability
    ARGMAX = auto()  # Argmax of expected utilities
    MARGIN = auto()  # Argmax of margin (p - threshold)


@dataclass(frozen=True)
class BayesOptimal:
    """Unified Bayes-optimal decision maker."""

    utility: UtilitySpec | NDArray[np.float64]

    def __post_init__(self):
        """Validate utility specification."""
        if isinstance(self.utility, np.ndarray):
            if self.utility.ndim != 2:
                raise ValueError(f"Utility matrix must be 2D, got {self.utility.ndim}D")
            if not np.all(np.isfinite(self.utility)):
                raise ValueError("Utility matrix must contain finite values")

    @cached_property
    def is_binary(self) -> bool:
        """Check if this is a binary problem."""
        if isinstance(self.utility, UtilitySpec):
            return True
        if self.utility is None:
            raise ValueError("mode='bayes' requires utility parameter")
        return self.utility.shape == (2, 2)

    @cached_property
    def decision_rule(self) -> DecisionRule:
        """Determine optimal decision rule."""
        if self.is_binary:
            return DecisionRule.THRESHOLD
        elif isinstance(self.utility, np.ndarray):
            # Square matrix -> standard classification
            if self.utility.shape[0] == self.utility.shape[1]:
                return DecisionRule.ARGMAX
            # More decisions than classes -> includes abstain
            else:
                return DecisionRule.ARGMAX
        else:
            return DecisionRule.MARGIN

    def compute_threshold(self) -> float:
        """Compute optimal threshold for binary case (only valid when D > 0).

        Returns
        -------
        float
            Optimal probability threshold (not clipped to [0,1])

        Raises
        ------
        ValueError
            If D <= 0, callers must use margin-based decision instead
        """
        if not self.is_binary:
            raise ValueError("Thresholds only defined for binary problems")

        _, B, D = self._binary_params()

        # Only valid for D > 0; for D <= 0 callers must use the margin-based decision.
        if D <= 1e-12:
            raise ValueError(
                "compute_threshold is only valid when (tp-fn)+(tn-fp) > 0. "
                "Use margin-based decision for D <= 0."
            )
        return B / D  # Do NOT clip; preserve semantics

    def _binary_params(self) -> tuple[float, float, float]:
        """Extract binary utility parameters A, B, D."""
        if isinstance(self.utility, UtilitySpec):
            tp, tn, fp, fn = (
                self.utility.tp_utility,
                self.utility.tn_utility,
                self.utility.fp_utility,
                self.utility.fn_utility,
            )
        else:
            if self.utility.shape != (2, 2):
                raise ValueError("Binary decisions require a 2x2 utility matrix")
            tn, fn = float(self.utility[0, 0]), float(self.utility[0, 1])
            fp, tp = float(self.utility[1, 0]), float(self.utility[1, 1])

        A = tp - fn  # Benefit of TP over FN
        B = tn - fp  # Benefit of TN over FP
        D = A + B  # Total utility difference
        return A, B, D

    def _extract_binary_p(self, probs: np.ndarray) -> np.ndarray:
        """Extract P(y=1) from binary probabilities."""
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            return probs  # already P(y=1)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]  # assume column 1 is P(y=1)
        raise ValueError("Binary probabilities must be shape (n,) or (n,2)")

    def _decide_binary(self, probs: np.ndarray) -> np.ndarray:
        """Make binary decisions using margin approach (handles all D cases)."""
        p = self._extract_binary_p(probs)
        _, B, D = self._binary_params()

        if abs(D) < 1e-12:  # D ≈ 0: decision is probability-independent
            # Predict positive if B <= 0, negative if B > 0
            return (
                np.ones_like(p, dtype=np.int32)
                if B <= 0
                else np.zeros_like(p, dtype=np.int32)
            )

        # The margin formula D*p - B >= 0 is universally correct
        # It naturally handles D > 0, D < 0, and gives consistent >= tie-breaking
        margin = D * p - B
        return (margin >= 0.0).astype(np.int32)

    def compute_thresholds(self, n_classes: int) -> NDArray[np.float64]:
        """Compute per-class thresholds for OvR multiclass.

        Parameters
        ----------
        n_classes : int
            Number of classes

        Returns
        -------
        NDArray[np.float64]
            Per-class thresholds
        """
        if isinstance(self.utility, UtilitySpec):
            # Use same utility for all classes
            threshold = self.compute_threshold()
            return np.full(n_classes, threshold)
        else:
            # Need per-class utilities
            raise NotImplementedError(
                "Per-class utilities from matrix not yet implemented. "
                "Use UtilitySpec for OvR thresholds."
            )

    def decide(self, probabilities: NDArray[np.float64]) -> NDArray[np.int32]:
        """Make Bayes-optimal decisions.

        Parameters
        ----------
        probabilities : NDArray[np.float64]
            Probability array. For binary: shape (n,) or (n,2).
            For multiclass: shape (n, n_classes).

        Returns
        -------
        NDArray[np.int32]
            Optimal decisions
        """
        probs = np.asarray(probabilities, dtype=np.float64)

        if self.decision_rule == DecisionRule.THRESHOLD:
            # Use margin-based binary decision
            return self._decide_binary(probs)

        elif self.decision_rule == DecisionRule.ARGMAX:
            if not isinstance(self.utility, np.ndarray):
                raise ValueError("ARGMAX rule requires utility matrix")

            # Expected utilities: E[U|x] = Σ_y U(d,y) P(y|x)
            if probs.ndim != 2:
                raise ValueError("ARGMAX rule requires 2D probability matrix")
            expected = probs @ self.utility.T
            return np.argmax(expected, axis=1).astype(np.int32)

        else:  # MARGIN - implement properly or remove
            if self.is_binary:
                return self._decide_binary(probs)
            else:
                # For multiclass margin: would need per-class thresholds
                if probs.ndim != 2:
                    raise ValueError("Multiclass margin requires 2D probability matrix")
                n_classes = probs.shape[1]
                thresholds = self.compute_thresholds(n_classes)
                # Argmax of margin: p - threshold
                margins = probs - thresholds[None, :]
                return np.argmax(margins, axis=1).astype(np.int32)

    def expected_utility(self, probabilities: NDArray[np.float64]) -> float:
        """Compute expected utility under optimal decisions.

        Parameters
        ----------
        probabilities : NDArray[np.float64]
            Probability array

        Returns
        -------
        float
            Expected utility per sample
        """
        probs = np.asarray(probabilities, dtype=np.float64)

        if self.decision_rule == DecisionRule.ARGMAX:
            expected = probs @ self.utility.T  # (n, k) @ (k, d) -> (n, d)
            chosen = np.max(expected, axis=1)
            return float(np.mean(chosen))

        # Binary (UtilitySpec or 2x2 matrix)
        p = self._extract_binary_p(probs)
        if isinstance(self.utility, UtilitySpec):
            tp, tn, fp, fn = (
                self.utility.tp_utility,
                self.utility.tn_utility,
                self.utility.fp_utility,
                self.utility.fn_utility,
            )
        else:
            tn, fn = float(self.utility[0, 0]), float(self.utility[0, 1])
            fp, tp = float(self.utility[1, 0]), float(self.utility[1, 1])

        eu_pos = p * tp + (1 - p) * fp
        eu_neg = p * fn + (1 - p) * tn
        return float(np.mean(np.maximum(eu_pos, eu_neg)))


# ============================================================================
# Factory Functions
# ============================================================================


def bayes_optimal_threshold(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
) -> OptimizationResult:
    """Compute optimal threshold from costs and benefits.

    Parameters
    ----------
    fp_cost : float
        Cost of false positive (positive value)
    fn_cost : float
        Cost of false negative (positive value)
    tp_benefit : float, default=0.0
        Benefit of true positive
    tn_benefit : float, default=0.0
        Benefit of true negative

    Returns
    -------
    OptimizationResult
        Optimization result with threshold and predict function
    """
    # Convert costs to utilities (negate costs)
    utility = UtilitySpec(
        tp_utility=tp_benefit,
        tn_utility=tn_benefit,
        fp_utility=-abs(fp_cost),
        fn_utility=-abs(fn_cost),
    )

    optimizer = BayesOptimal(utility)
    threshold = optimizer.compute_threshold()

    # Compute expected utility as "score"
    # For now, use a placeholder since we need probabilities to compute expected utility
    expected_utility = 0.0

    # Create prediction function (closure captures threshold)
    def predict_binary(probs):
        p = np.asarray(probs)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]  # Use positive class probabilities
        elif p.ndim == 2 and p.shape[1] == 1:
            p = p.ravel()
        return (p >= threshold).astype(np.int32)

    return OptimizationResult(
        thresholds=np.array([threshold]),
        scores=np.array([expected_utility]),
        predict=predict_binary,
        metric="expected_utility",
        n_classes=2,
    )


def bayes_optimal_decisions(
    probabilities: NDArray[np.float64], utility_matrix: NDArray[np.float64]
) -> OptimizationResult:
    """Compute optimal decisions from utility matrix.

    Parameters
    ----------
    probabilities : array of shape (n_samples, n_classes)
        Class probabilities
    utility_matrix : array of shape (n_decisions, n_classes)
        Utility matrix U[d,y] = utility(decision=d, true=y)

    Returns
    -------
    OptimizationResult
        Optimization result with decision strategy and predict function
    """
    # Convert to arrays
    probs = np.asarray(probabilities, dtype=np.float64)
    utility = np.asarray(utility_matrix, dtype=np.float64)

    # Validate shapes
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2D array")
    if utility.ndim != 2:
        raise ValueError("utility_matrix must be 2D array")
    if probs.shape[1] != utility.shape[1]:
        raise ValueError(
            f"probabilities has {probs.shape[1]} classes but "
            f"utility_matrix has {utility.shape[1]}"
        )

    n_decisions = utility.shape[0]
    n_classes = utility.shape[1]

    # Compute expected utilities: E[U|x] = Σ_y U(d,y) P(y|x)
    expected = (
        probs @ utility.T
    )  # (n_samples, n_classes) @ (n_classes, n_decisions) -> (n_samples, n_decisions)

    # Compute maximum expected utility for each sample as "scores"
    max_expected_utilities = np.max(expected, axis=1)
    mean_expected_utility = np.mean(max_expected_utilities)

    # Create prediction function (closure captures utility matrix)
    def predict_bayes_decisions(probabilities_new):
        p = np.asarray(probabilities_new, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError("probabilities must be 2D array")
        if p.shape[1] != n_classes:
            raise ValueError(f"Expected {n_classes} classes, got {p.shape[1]}")

        # Compute expected utilities and return argmax decisions
        expected_new = p @ utility.T
        return np.argmax(expected_new, axis=1).astype(np.int32)

    # For utility-based decisions, we don't have traditional "thresholds",
    # but we can use zeros as placeholders since the predict function handles everything
    thresholds = np.zeros(n_decisions, dtype=np.float64)
    scores = np.full(n_decisions, mean_expected_utility, dtype=np.float64)

    return OptimizationResult(
        thresholds=thresholds,
        scores=scores,
        predict=predict_bayes_decisions,
        metric="expected_utility",
        n_classes=n_decisions,
    )


def bayes_thresholds_from_costs(
    fp_costs: NDArray[np.float64] | list[float],
    fn_costs: NDArray[np.float64] | list[float],
) -> OptimizationResult:
    """Compute per-class Bayes thresholds from costs (vectorized).

    Parameters
    ----------
    fp_costs : array-like
        False positive costs per class (can be positive or negative)
    fn_costs : array-like
        False negative costs per class (can be positive or negative)

    Returns
    -------
    OptimizationResult
        Optimization result with per-class thresholds and predict function

    Notes
    -----
    Uses the formula: threshold = |fp_cost| / (|fp_cost| + |fn_cost|)
    Costs can be negative (representing utilities) or positive.
    """
    fp = np.asarray(fp_costs, dtype=np.float64)
    fn = np.asarray(fn_costs, dtype=np.float64)
    if fp.shape != fn.shape:
        raise ValueError("fp_costs and fn_costs must have same shape")

    # Validate inputs are finite
    if not np.all(np.isfinite(fp)) or not np.all(np.isfinite(fn)):
        raise ValueError("All costs must be finite")

    # Take absolute values to handle negative costs (utilities)
    fp_abs = np.abs(fp)
    fn_abs = np.abs(fn)

    den = fp_abs + fn_abs
    if np.any(den <= 0):
        raise ValueError("All |fp_cost| + |fn_cost| must be > 0")

    thresholds = fp_abs / den
    n_classes = len(thresholds)

    # Compute expected cost as "score" (negative values for costs)
    expected_costs = -(fp_abs + fn_abs)  # Negative for minimization

    # Create prediction function for per-class thresholds
    def predict_multiclass_bayes(probs):
        p = np.asarray(probs)
        if p.ndim != 2:
            raise ValueError("Multiclass requires 2D probabilities")
        if p.shape[1] != n_classes:
            raise ValueError(f"Expected {n_classes} classes, got {p.shape[1]}")

        # Apply per-class thresholds and predict highest valid probability
        valid = p >= thresholds[None, :]
        masked = np.where(valid, p, -np.inf)
        predictions = np.argmax(masked, axis=1).astype(np.int32)

        # For samples where no class is above threshold, predict highest probability
        no_valid = np.all(~valid, axis=1)
        if np.any(no_valid):
            predictions[no_valid] = np.argmax(p[no_valid], axis=1)

        return predictions

    return OptimizationResult(
        thresholds=thresholds,
        scores=expected_costs,
        predict=predict_multiclass_bayes,
        metric="expected_cost",
        n_classes=n_classes,
    )


# ============================================================================
# Integration with Optimization Pipeline
# ============================================================================


def optimize_bayes_thresholds(
    labels, predictions, utility: UtilitySpec | dict[str, float], weights=None
) -> OptimizationResult:
    """Optimize thresholds using Bayes decision theory.

    Parameters
    ----------
    labels : array-like
        True labels
    predictions : array-like
        Predicted probabilities
    utility : UtilitySpec or dict
        Utility specification
    weights : array-like, optional
        Sample weights

    Returns
    -------
    OptimizationResult
        Unified optimization result with thresholds, scores, and predict function
    """
    # Validate inputs
    labels, predictions, weights, problem_type = validate_classification(
        labels, predictions, weights
    )

    # Convert utility if needed
    if isinstance(utility, dict):
        utility = UtilitySpec.from_dict(utility)

    # Create optimizer
    optimizer = BayesOptimal(utility)

    # Compute thresholds based on problem type
    if problem_type == "binary":
        thresholds = np.array([optimizer.compute_threshold()])
        n_classes = 2
    else:
        n_classes = predictions.shape[1]
        thresholds = optimizer.compute_thresholds(n_classes)

    # Compute expected utility if we have probabilities
    # probs = Probabilities.from_array(predictions)
    # expected_util = optimizer.expected_utility(probs)
    expected_util = 0.0  # Temporary placeholder

    # Create prediction function based on problem type
    if problem_type == "binary":
        threshold = float(thresholds[0])
        
        def predict_binary(probs):
            p = np.asarray(probs)
            if p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]  # Use positive class probabilities
            elif p.ndim == 2 and p.shape[1] == 1:
                p = p.ravel()
            return (p >= threshold).astype(np.int32)
        
        predict_fn = predict_binary
    else:
        def predict_multiclass(probs):
            p = np.asarray(probs)
            if p.ndim != 2 or p.shape[1] != len(thresholds):
                raise ValueError("Multiclass probabilities must be (n_samples, n_classes)")

            mask = p >= thresholds[None, :]
            masked = np.where(mask, p, -np.inf)
            pred = np.argmax(masked, axis=1)
            none_pass = ~np.any(mask, axis=1)
            if np.any(none_pass):
                # Fallback: pure argmax of probabilities
                pred[none_pass] = np.argmax(p[none_pass], axis=1)
            return pred.astype(np.int32)
        
        predict_fn = predict_multiclass

    return OptimizationResult(
        thresholds=thresholds,
        scores=np.full(n_classes, expected_util),
        predict=predict_fn,
        metric="expected_utility",
        n_classes=n_classes,
    )


# ============================================================================
# Simple API
# ============================================================================


def compute_bayes_threshold(
    costs: dict[str, float], benefits: dict[str, float] | None = None
) -> float:
    """Simple API for computing Bayes-optimal threshold.

    Parameters
    ----------
    costs : dict
        Dictionary with 'fp' and 'fn' keys for costs
    benefits : dict, optional
        Dictionary with 'tp' and 'tn' keys for benefits

    Returns
    -------
    float
        Optimal threshold

    Examples
    --------
    >>> # FN costs 5x more than FP
    >>> threshold = compute_bayes_threshold({'fp': 1, 'fn': 5})
    >>> print(f"{threshold:.3f}")
    0.167
    """
    fp_cost = costs.get("fp", 1.0)
    fn_cost = costs.get("fn", 1.0)
    tp_benefit = benefits.get("tp", 0.0) if benefits else 0.0
    tn_benefit = benefits.get("tn", 0.0) if benefits else 0.0

    result = bayes_optimal_threshold(fp_cost, fn_cost, tp_benefit, tn_benefit)
    return result.threshold
