"""Top-level package for optimal classification cutoff utilities."""

from .metrics import (
    get_confusion_matrix,
    METRIC_REGISTRY,
    register_metric,
    register_metrics,
)
from .optimizers import get_probability, get_optimal_threshold
from .cv import cv_threshold_optimization, nested_cv_threshold_optimization
from .wrapper import ThresholdOptimizer

__all__ = [
    "get_confusion_matrix",
    "METRIC_REGISTRY",
    "register_metric",
    "register_metrics",
    "get_probability",
    "get_optimal_threshold",
    "cv_threshold_optimization",
    "nested_cv_threshold_optimization",
    "ThresholdOptimizer",
]
