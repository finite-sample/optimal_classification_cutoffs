"""High-level wrapper for threshold optimization."""

from __future__ import annotations

import numpy as np
from .optimizers import get_probability, get_optimal_threshold


class ThresholdOptimizer:
    """Optimizer for classification thresholds supporting both binary and multiclass.

    The class wraps threshold optimization functions and exposes a scikit-learn
    style ``fit``/``predict`` API. For multiclass, uses One-vs-Rest strategy.
    """

    def __init__(
        self,
        objective: str = "accuracy",
        verbose: bool = False,
        method: str = "smart_brute",
    ):
        """Create a new optimizer.

        Parameters
        ----------
        objective:
            Metric to optimize, e.g. ``"accuracy"``, ``"f1"``, ``"precision"``, ``"recall"``.
        verbose:
            If ``True``, print progress during threshold search.
        method:
            Optimization method: ``"smart_brute"``, ``"minimize"``, or ``"gradient"``.
        """
        self.objective = objective
        self.verbose = verbose
        self.method = method
        self.threshold_ = None
        self.is_multiclass_ = False

    def fit(self, true_labs, pred_prob):
        """Estimate the optimal threshold(s) from labeled data.

        Parameters
        ----------
        true_labs:
            Array of true labels. For binary: (0, 1). For multiclass: (0, 1, 2, ..., n_classes-1).
        pred_prob:
            Predicted probabilities from a classifier. For binary: 1D array (n_samples,).
            For multiclass: 2D array (n_samples, n_classes).

        Returns
        -------
        ThresholdOptimizer
            Fitted instance with ``threshold_`` attribute set.
        """
        pred_prob = np.asarray(pred_prob)

        # Check if multiclass
        self.is_multiclass_ = pred_prob.ndim == 2

        if self.is_multiclass_ or self.objective not in ["accuracy", "f1"]:
            # Use the more general optimizer
            self.threshold_ = get_optimal_threshold(
                true_labs, pred_prob, self.objective, self.method
            )
        else:
            # Use legacy optimizer for backward compatibility
            self.threshold_ = get_probability(
                true_labs, pred_prob, self.objective, self.verbose
            )

        return self

    def predict(self, pred_prob):
        """Convert probabilities to class predictions using the learned threshold(s).

        Parameters
        ----------
        pred_prob:
            Array of predicted probabilities to be thresholded.

        Returns
        -------
        numpy.ndarray
            For binary: Boolean array of predicted class labels.
            For multiclass: Integer array of predicted class labels.
        """
        if self.threshold_ is None:
            raise RuntimeError("ThresholdOptimizer has not been fitted.")

        pred_prob = np.asarray(pred_prob)

        if self.is_multiclass_:
            # Multiclass prediction using One-vs-Rest thresholds
            n_samples, n_classes = pred_prob.shape
            binary_predictions = pred_prob > self.threshold_

            # For each sample, predict the class with highest probability among those above threshold
            # If no classes above threshold, predict the class with highest probability
            predictions = np.zeros(n_samples, dtype=int)

            for i in range(n_samples):
                above_threshold = np.where(binary_predictions[i])[0]
                if len(above_threshold) > 0:
                    # Among classes above threshold, pick the one with highest probability
                    predictions[i] = above_threshold[
                        np.argmax(pred_prob[i, above_threshold])
                    ]
                else:
                    # No class above threshold, pick highest probability class
                    predictions[i] = np.argmax(pred_prob[i])

            return predictions
        else:
            # Binary prediction
            return pred_prob > self.threshold_
