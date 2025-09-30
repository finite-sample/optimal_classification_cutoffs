"""High-level wrapper for threshold optimization."""

from typing import Any, Self, cast

import numpy as np

from .multiclass_coord import _assign_labels_shifted
from .optimizers import get_optimal_threshold
from .types import (
    ArrayLike,
    ComparisonOperator,
    EstimationMode,
    OptimizationMethod,
    SampleWeightLike,
    UtilityDict,
    UtilityMatrix,
)


class ThresholdOptimizer:
    """Optimizer for classification thresholds supporting both binary and multiclass.

    The class wraps threshold optimization functions and exposes a scikit-learn
    style ``fit``/``predict`` API. For multiclass, uses One-vs-Rest strategy.
    """

    def __init__(
        self,
        metric: str | None = None,
        verbose: bool = False,
        method: OptimizationMethod = "auto",
        comparison: ComparisonOperator = ">",
        *,
        mode: EstimationMode = "empirical",
        utility: UtilityDict | None = None,
        utility_matrix: UtilityMatrix | None = None,
        minimize_cost: bool | None = None,
        beta: float = 1.0,
        class_weight: ArrayLike | None = None,
    ) -> None:
        """Create a new optimizer.

        Parameters
        ----------
        metric:
            Metric to optimize, e.g. ``"accuracy"``, ``"f1"``, ``"precision"``,
            ``"recall"``.
        verbose:
            If ``True``, print progress during threshold search.
        method:
            Optimization method:
            - ``"auto"``: Automatically selects best method (default)
            - ``"sort_scan"``: O(n log n) algorithm for piecewise metrics with
              vectorized implementation
            - ``"unique_scan"``: Evaluates all unique probabilities
            - ``"minimize"``: Uses ``scipy.optimize.minimize_scalar``
            - ``"gradient"``: Simple gradient ascent
            - ``"coord_ascent"``: Coordinate ascent for coupled multiclass
              optimization (single-label consistent)
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
            Dict with keys "tp", "tn", "fp", "fn" specifying utilities/costs per
            outcome.
            For multiclass mode="bayes", can contain per-class vectors.
            Example: ``{"tp": 0, "tn": 0, "fp": -1, "fn": -5}`` for cost-sensitive.
        utility_matrix:
            Alternative to utility dict for multiclass Bayes decisions.
            Shape (D, K) array where D=decisions, K=classes.
            If provided with mode="bayes", returns class decisions rather than
            thresholds.
        minimize_cost:
            If True, interpret utility values as costs and minimize total cost. This
            automatically negates fp/fn values if they're positive.
        beta:
            F-beta parameter for expected mode (beta >= 0). beta=1 gives F1,
            beta < 1 emphasizes precision, beta > 1 emphasizes recall.
            Only used when mode="expected".
        class_weight:
            Optional per-class weights for weighted averaging in expected mode.
            Shape (K,) array. Only used when mode="expected" and average="weighted".
        """
        if metric is None:
            metric = "accuracy"

        self.metric = metric
        self.verbose = verbose
        self.method = method
        self.comparison = comparison
        self.mode = mode
        self.utility = utility
        self.utility_matrix = utility_matrix
        self.minimize_cost = minimize_cost
        self.beta = beta
        self.class_weight = class_weight
        self.threshold_: (
            float | np.ndarray[Any, Any] | dict[str, Any] | tuple[float, float] | None
        ) = None
        self.is_multiclass_: bool = False

    def fit(
        self,
        true_labs: ArrayLike,
        pred_prob: ArrayLike,
        sample_weight: SampleWeightLike = None,
    ) -> Self:
        """Estimate the optimal threshold(s) from labeled data.

        Parameters
        ----------
        true_labs:
            Array of true labels. For binary: (0, 1). For multiclass:
            (0, 1, 2, ..., n_classes-1).
        pred_prob:
            Predicted probabilities from a classifier. For binary: 1D array
            (n_samples,).
            For multiclass: 2D array (n_samples, n_classes).
        sample_weight:
            Optional array of sample weights for handling imbalanced datasets.

        Returns
        -------
        Self
            Fitted instance with ``threshold_`` attribute set.
        """
        pred_prob = np.asarray(pred_prob)

        # Check if multiclass
        self.is_multiclass_ = pred_prob.ndim == 2

        if (
            self.is_multiclass_
            or self.metric not in ["accuracy", "f1"]
            or sample_weight is not None
            or self.mode != "empirical"
            or self.utility is not None
            or self.utility_matrix is not None
            or self.minimize_cost is not None
        ):
            # Use the more general optimizer
            result = get_optimal_threshold(
                true_labs,
                pred_prob,
                self.metric,
                self.method,
                sample_weight,
                self.comparison,
                mode=self.mode,
                utility=self.utility,
                utility_matrix=self.utility_matrix,
                minimize_cost=self.minimize_cost,
                beta=self.beta,
                class_weight=self.class_weight,
            )

            # Handle tuple return from mode='expected'
            if isinstance(result, tuple):
                self.threshold_, self.expected_score_ = result
            else:
                self.threshold_ = result
        else:
            # Use standard optimizer for simple binary cases
            self.threshold_ = get_optimal_threshold(
                true_labs,
                pred_prob,
                self.metric,
                self.method,
                comparison=self.comparison,
            )

        return self

    def predict(self, pred_prob: ArrayLike) -> np.ndarray[Any, Any]:
        """Convert probabilities to class predictions using the learned threshold(s).

        Parameters
        ----------
        pred_prob:
            Array of predicted probabilities to be thresholded.

        Returns
        -------
        np.ndarray[Any, Any]
            For binary: Boolean array of predicted class labels.
            For multiclass: Integer array of predicted class labels.
        """
        if self.threshold_ is None:
            raise RuntimeError("ThresholdOptimizer has not been fitted.")

        pred_prob = np.asarray(pred_prob)

        if self.is_multiclass_:
            # Multiclass prediction strategy depends on optimization method
            if self.method == "coord_ascent":
                # Coordinate ascent uses argmax(P - tau) for single-label consistency
                return _assign_labels_shifted(
                    pred_prob, cast(np.ndarray[Any, Any], self.threshold_)
                )
            else:
                # One-vs-Rest prediction using per-class thresholds
                n_samples, n_classes = pred_prob.shape
                if self.comparison == ">":
                    binary_predictions = pred_prob > self.threshold_
                else:  # ">="
                    binary_predictions = pred_prob >= self.threshold_

                # For each sample, predict the class with highest probability among
                # those above threshold
                # If no classes above threshold, predict the class with highest
                # probability
                predictions = np.zeros(n_samples, dtype=int)

                for i in range(n_samples):
                    above_threshold = np.where(binary_predictions[i])[0]
                    if len(above_threshold) > 0:
                        # Among classes above threshold, pick the one with highest
                        # probability
                        predictions[i] = above_threshold[
                            np.argmax(pred_prob[i, above_threshold])
                        ]
                    else:
                        # No class above threshold, pick highest probability class
                        predictions[i] = np.argmax(pred_prob[i])

                return predictions
        else:
            # Binary prediction
            if self.comparison == ">":
                return pred_prob > self.threshold_
            else:  # ">="
                return pred_prob >= self.threshold_
