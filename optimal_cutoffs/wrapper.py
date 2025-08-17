"""High-level wrapper for threshold optimization."""

from __future__ import annotations

from .optimizers import get_probability


class ThresholdOptimizer:
    """Brute-force optimizer for classification thresholds.

    The class wraps :func:`optimal_cutoffs.optimizers.get_probability` and
    exposes a scikit-learn style ``fit``/``predict`` API.
    """

    def __init__(self, objective: str = "accuracy", verbose: bool = False):
        """Create a new optimizer.

        Parameters
        ----------
        objective:
            Metric to optimize, e.g. ``"accuracy"`` or ``"f1"``.
        verbose:
            If ``True``, print progress during threshold search.
        """
        self.objective = objective
        self.verbose = verbose
        self.threshold_ = None

    def fit(self, true_labs, pred_prob):
        """Estimate the optimal threshold from labeled data.

        Parameters
        ----------
        true_labs:
            Array of true binary labels.
        pred_prob:
            Predicted probabilities from a classifier.

        Returns
        -------
        ThresholdOptimizer
            Fitted instance with ``threshold_`` attribute set.
        """
        self.threshold_ = get_probability(true_labs, pred_prob, self.objective, self.verbose)
        return self

    def predict(self, pred_prob):
        """Convert probabilities to class predictions using the learned threshold.

        Parameters
        ----------
        pred_prob:
            Array of predicted probabilities to be thresholded.

        Returns
        -------
        numpy.ndarray
            Boolean array of predicted class labels.
        """
        if self.threshold_ is None:
            raise RuntimeError("ThresholdOptimizer has not been fitted.")
        return pred_prob > self.threshold_
