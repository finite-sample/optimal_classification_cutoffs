"""High-level wrapper for threshold optimization."""

from typing import Any, Literal, Self, cast

import numpy as np

from .multiclass_coord import _assign_labels_shifted
from .optimizers import get_optimal_threshold, get_probability
from .types import ArrayLike, ComparisonOperator, OptimizationMethod, SampleWeightLike


class ThresholdOptimizer:
    """Optimizer for classification thresholds supporting both binary and multiclass.

    The class wraps threshold optimization functions and exposes a scikit-learn
    style ``fit``/``predict`` API. For multiclass, uses One-vs-Rest strategy.
    """

    def __init__(
        self,
        objective: str = "accuracy",
        verbose: bool = False,
        method: OptimizationMethod = "auto",
        comparison: ComparisonOperator = ">",
    ) -> None:
        """Create a new optimizer.

        Parameters
        ----------
        objective:
            Metric to optimize, e.g. ``"accuracy"``, ``"f1"``, ``"precision"``,
            ``"recall"``.
        verbose:
            If ``True``, print progress during threshold search.
        method:
            Optimization method:
            - ``"auto"``: Automatically selects best method (default)
            - ``"sort_scan"``: O(n log n) algorithm for piecewise metrics with
              vectorized implementation
            - ``"smart_brute"``: Evaluates all unique probabilities
            - ``"minimize"``: Uses ``scipy.optimize.minimize_scalar``
            - ``"gradient"``: Simple gradient ascent
            - ``"coord_ascent"``: Coordinate ascent for coupled multiclass
              optimization (single-label consistent)
        comparison:
            Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).
        """
        self.objective = objective
        self.verbose = verbose
        self.method = method
        self.comparison = comparison
        self.threshold_: float | np.ndarray[Any, Any] | None = None
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
            or self.objective not in ["accuracy", "f1"]
            or sample_weight is not None
        ):
            # Use the more general optimizer
            self.threshold_ = get_optimal_threshold(
                true_labs,
                pred_prob,
                self.objective,
                self.method,
                sample_weight,
                self.comparison,
            )
        else:
            # Use legacy optimizer for backward compatibility (only when no sample
            # weights)
            self.threshold_ = get_probability(
                true_labs,
                pred_prob,
                cast(Literal["accuracy", "f1"], self.objective),
                self.verbose,
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
