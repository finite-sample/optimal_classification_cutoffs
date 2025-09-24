"""Top-level package for optimal classification cutoff utilities."""

__version__ = "0.1.0"

from .cv import cv_threshold_optimization, nested_cv_threshold_optimization
from .metrics import (
    METRIC_REGISTRY,
    VECTORIZED_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    get_vectorized_metric,
    has_vectorized_implementation,
    is_piecewise_metric,
    multiclass_metric,
    needs_probability_scores,
    register_metric,
    register_metrics,
    should_maximize_metric,
)
from .optimizers import (
    get_optimal_multiclass_thresholds,
    get_optimal_threshold,
    get_probability,
)
from .types import MulticlassMetricReturn
from .wrapper import ThresholdOptimizer

__all__ = [
    "__version__",
    "get_confusion_matrix",
    "get_multiclass_confusion_matrix",
    "multiclass_metric",
    "METRIC_REGISTRY",
    "VECTORIZED_REGISTRY",
    "register_metric",
    "register_metrics",
    "is_piecewise_metric",
    "should_maximize_metric",
    "needs_probability_scores",
    "get_vectorized_metric",
    "has_vectorized_implementation",
    "get_probability",
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
    "cv_threshold_optimization",
    "nested_cv_threshold_optimization",
    "ThresholdOptimizer",
    "MulticlassMetricReturn",
]
