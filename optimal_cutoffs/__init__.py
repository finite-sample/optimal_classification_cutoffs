"""Top-level package for optimal classification cutoff utilities."""

# Single source of truth for version in pyproject.toml
try:
    from importlib.metadata import version

    __version__ = version("optimal-classification-cutoffs")
except Exception:
    # Fallback for development: read from pyproject.toml
    import pathlib
    import tomllib  # Python 3.11+ stdlib

    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    else:
        __version__ = "unknown"

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
