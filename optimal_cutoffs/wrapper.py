"""High level wrapper for threshold optimization."""

from dataclasses import dataclass
from typing import Optional

from .optimizers import (
    brute_threshold,
    smart_brute_threshold,
    minimize_threshold,
    gradient_threshold,
)

_METHODS = {
    "brute": brute_threshold,
    "smart_brute": smart_brute_threshold,
    "minimize": minimize_threshold,
    "gradient": gradient_threshold,
}


@dataclass
class ThresholdOptimizer:
    """Optimize a probability threshold for classification.

    Parameters
    ----------
    metric : str, default="accuracy"
        Metric to optimize. Currently supports ``"accuracy"`` and ``"f1"``.
    method : str, default="brute"
        Optimization method. Options are ``"brute"``, ``"smart_brute"``,
        ``"minimize"``, or ``"gradient"``.
    kwargs : dict
        Additional keyword arguments passed to the optimizer.
    """

    metric: str = "accuracy"
    method: str = "brute"
    kwargs: Optional[dict] = None
    threshold_: Optional[float] = None

    def fit(self, y_true, y_prob):
        """Find the optimal threshold given true labels and predicted probabilities."""
        opt_func = _METHODS[self.method]
        params = self.kwargs or {}
        self.threshold_ = opt_func(y_true, y_prob, metric=self.metric, **params)
        return self

    def predict(self, y_prob):
        """Convert probabilities to labels using the learned threshold."""
        if self.threshold_ is None:
            raise ValueError("ThresholdOptimizer.fit must be called before predict.")
        return (y_prob > self.threshold_).astype(int)
