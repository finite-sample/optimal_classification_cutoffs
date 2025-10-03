"""Bayes-optimal decisions and thresholds for classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray

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
        """Compute optimal threshold for binary case.

        Returns
        -------
        float
            Optimal probability threshold in [0, 1]
        """
        if not self.is_binary:
            raise ValueError("Thresholds only defined for binary problems")

        if isinstance(self.utility, UtilitySpec):
            u = self.utility
        else:
            # Extract from 2x2 matrix
            u = UtilitySpec(
                tn_utility=self.utility[0, 0],
                fn_utility=self.utility[0, 1],
                fp_utility=self.utility[1, 0],
                tp_utility=self.utility[1, 1],
            )

        # Compute threshold using correct formula
        A = u.tp_utility - u.fn_utility  # Benefit of TP over FN
        B = u.tn_utility - u.fp_utility  # Benefit of TN over FP
        D = A + B

        # Handle edge cases
        if abs(D) < 1e-10:
            # Decision independent of probability
            return 0.0 if B <= 0 else 1.0

        # Standard case: threshold = B/D
        threshold = B / D

        # Note: We don't handle D < 0 (flipped inequality) here
        # That's a rare pathological case better handled by rejecting
        # such utility specifications
        if D < 0:
            raise ValueError(
                "Utility specification leads to inverted decision rule (D < 0). "
                "This typically indicates misspecified utilities."
            )

        return np.clip(threshold, 0.0, 1.0)

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
        probabilities : Probabilities
            Validated probability array

        Returns
        -------
        NDArray[np.int32]
            Optimal decisions
        """
        if self.decision_rule == DecisionRule.THRESHOLD:
            threshold = self.compute_threshold()
            if probabilities.is_binary:
                return (probabilities.data > threshold).astype(np.int32)
            else:
                # Apply to positive class probabilities
                pos_probs = probabilities.get_class_probabilities(1)
                return (pos_probs > threshold).astype(np.int32)

        elif self.decision_rule == DecisionRule.ARGMAX:
            if not isinstance(self.utility, np.ndarray):
                raise ValueError("ARGMAX rule requires utility matrix")

            # Expected utilities: E[U|x] = Σ_y U(d,y) P(y|x)
            expected = probabilities.data @ self.utility.T
            return np.argmax(expected, axis=1).astype(np.int32)

        else:  # MARGIN
            if probabilities.is_binary:
                threshold = self.compute_threshold()
                return (probabilities.data > threshold).astype(np.int32)
            else:
                thresholds = self.compute_thresholds(probabilities.n_classes)
                # Argmax of margin: p - threshold
                margins = probabilities.data - thresholds[None, :]
                return np.argmax(margins, axis=1).astype(np.int32)

    def expected_utility(self, probabilities: NDArray[np.float64]) -> float:
        """Compute expected utility under optimal decisions.

        Parameters
        ----------
        probabilities : Probabilities
            Validated probability array

        Returns
        -------
        float
            Expected utility per sample
        """
        decisions = self.decide(probabilities)

        if isinstance(self.utility, UtilitySpec):
            # Binary case - compute from decisions
            # This would need the true labels to compute actual utility
            # For now, return expected utility based on threshold
            if probabilities.is_binary:
                # Expected utility calculation would go here
                return 0.0  # Placeholder
        else:
            # Matrix case - compute expected utilities
            expected = probabilities.data @ self.utility.T
            # Take utility of chosen decision for each sample
            n_samples = len(decisions)
            chosen_utils = expected[np.arange(n_samples), decisions]
            return float(np.mean(chosen_utils))

        return 0.0


# ============================================================================
# Factory Functions
# ============================================================================


def bayes_optimal_threshold(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
) -> float:
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
    float
        Optimal threshold in [0, 1]
    """
    # Convert costs to utilities (negate costs)
    utility = UtilitySpec(
        tp_utility=tp_benefit,
        tn_utility=tn_benefit,
        fp_utility=-abs(fp_cost),
        fn_utility=-abs(fn_cost),
    )

    optimizer = BayesOptimal(utility)
    return optimizer.compute_threshold()


def bayes_optimal_decisions(
    probabilities: NDArray[np.float64], utility_matrix: NDArray[np.float64]
) -> NDArray[np.int32]:
    """Compute optimal decisions from utility matrix.

    Parameters
    ----------
    probabilities : array of shape (n_samples, n_classes)
        Class probabilities
    utility_matrix : array of shape (n_decisions, n_classes)
        Utility matrix U[d,y] = utility(decision=d, true=y)

    Returns
    -------
    array of shape (n_samples,)
        Optimal decisions
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

    # Compute expected utilities: E[U|x] = Σ_y U(d,y) P(y|x)
    expected = (
        probs @ utility.T
    )  # (n_samples, n_classes) @ (n_classes, n_decisions) -> (n_samples, n_decisions)

    # Return argmax decisions
    return np.argmax(expected, axis=1).astype(np.int32)


def bayes_thresholds_from_costs(
    fp_costs: NDArray[np.float64] | list[float],
    fn_costs: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]:
    """Compute per-class Bayes thresholds from costs.

    Parameters
    ----------
    fp_costs : array-like
        False positive costs per class
    fn_costs : array-like
        False negative costs per class

    Returns
    -------
    NDArray[np.float64]
        Per-class optimal thresholds
    """
    fp_arr = np.asarray(fp_costs, dtype=np.float64)
    fn_arr = np.asarray(fn_costs, dtype=np.float64)

    if fp_arr.shape != fn_arr.shape:
        raise ValueError("fp_costs and fn_costs must have same shape")

    # Compute threshold for each class
    thresholds = []
    for fp_cost, fn_cost in zip(fp_arr.flat, fn_arr.flat, strict=False):
        threshold = bayes_optimal_threshold(fp_cost, fn_cost)
        thresholds.append(threshold)

    return np.array(thresholds, dtype=np.float64).reshape(fp_arr.shape)


# ============================================================================
# Integration with Optimization Pipeline
# ============================================================================


@dataclass(frozen=True)
class BayesThresholdResult:
    """Result from Bayes threshold optimization."""

    thresholds: NDArray[np.float64]
    utility_spec: UtilitySpec
    expected_utility: float | None = None

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary result."""
        return len(self.thresholds) == 1

    def apply(self, probabilities: NDArray[np.float64]) -> NDArray[np.int32]:
        """Apply thresholds to get predictions."""
        probs = np.asarray(probabilities, dtype=np.float64)

        if self.is_binary:
            threshold = self.thresholds[0]
            if probs.ndim == 1:
                return (probs > threshold).astype(np.int32)
            else:
                # Apply to positive class
                return (probs[:, 1] > threshold).astype(np.int32)
        else:
            # Multiclass OvR
            if probs.ndim == 2:
                # Apply per-class thresholds
                predictions = np.zeros((probs.shape[0], probs.shape[1]), dtype=bool)
                for i in range(probs.shape[1]):
                    class_probs = probs[:, i]
                    predictions[:, i] = class_probs > self.thresholds[i]
                # Return class with highest probability among those above threshold
                return np.argmax(np.where(predictions, probs, -np.inf), axis=1).astype(
                    np.int32
                )
            else:
                raise ValueError(
                    "Cannot apply multiclass thresholds to binary probabilities"
                )


def optimize_bayes_thresholds(
    labels, predictions, utility: UtilitySpec | dict[str, float], weights=None
) -> BayesThresholdResult:
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
    BayesThresholdResult
        Optimal thresholds
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

    return BayesThresholdResult(
        thresholds=thresholds, utility_spec=utility, expected_utility=expected_util
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

    return bayes_optimal_threshold(fp_cost, fn_cost, tp_benefit, tn_benefit)
