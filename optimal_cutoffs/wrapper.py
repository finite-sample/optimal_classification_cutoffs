"""High-level wrapper for threshold optimization."""

from __future__ import annotations

from .optimizers import get_probability


class ThresholdOptimizer:
    """Brute-force optimizer for classification thresholds."""

    def __init__(self, objective: str = "accuracy", verbose: bool = False):
        self.objective = objective
        self.verbose = verbose
        self.threshold_ = None

    def fit(self, true_labs, pred_prob):
        """Estimate the optimal threshold from labeled data."""
        self.threshold_ = get_probability(true_labs, pred_prob, self.objective, self.verbose)
        return self

    def predict(self, pred_prob):
        """Convert probabilities to class predictions using learned threshold."""
        if self.threshold_ is None:
            raise RuntimeError("ThresholdOptimizer has not been fitted.")
        return pred_prob > self.threshold_
