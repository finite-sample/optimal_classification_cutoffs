"""Top-level package for optimal classification cutoff utilities."""

from .metrics import (
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    multiclass_metric,
    METRIC_REGISTRY,
    register_metric,
    register_metrics,
)
from .optimizers import get_probability, get_optimal_threshold, get_optimal_multiclass_thresholds
from .cv import cv_threshold_optimization, nested_cv_threshold_optimization
from .wrapper import ThresholdOptimizer

__all__ = [
    "get_confusion_matrix",
    "get_multiclass_confusion_matrix", 
    "multiclass_metric",
    "METRIC_REGISTRY",
    "register_metric",
    "register_metrics",
    "get_probability",
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
    "cv_threshold_optimization",
    "nested_cv_threshold_optimization",
    "ThresholdOptimizer",
]
