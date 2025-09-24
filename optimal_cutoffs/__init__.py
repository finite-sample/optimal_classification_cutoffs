"""Top-level package for optimal classification cutoff utilities."""

__version__ = "0.1.0"

from .cv import cv_threshold_optimization, nested_cv_threshold_optimization
from .metrics import (
    METRIC_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    is_piecewise_metric,
    multiclass_metric,
    register_metric,
    register_metrics,
)
from .optimizers import (
    get_optimal_multiclass_thresholds,
    get_optimal_threshold,
    get_probability,
)
from .wrapper import ThresholdOptimizer

__all__ = [
    "__version__",
    "get_confusion_matrix",
    "get_multiclass_confusion_matrix",
    "multiclass_metric",
    "METRIC_REGISTRY",
    "register_metric",
    "register_metrics",
    "is_piecewise_metric",
    "get_probability",
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
    "cv_threshold_optimization",
    "nested_cv_threshold_optimization",
    "ThresholdOptimizer",
]
