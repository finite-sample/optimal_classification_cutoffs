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

from .bayes import (
    bayes_decision_from_utility_matrix,
    bayes_threshold_from_costs_scalar,
    bayes_thresholds_from_costs_vector,
)
from .cv import cv_threshold_optimization, nested_cv_threshold_optimization
from .expected import (
    dinkelbach_expected_fbeta_binary,
    dinkelbach_expected_fbeta_multilabel,
)
from .metrics import (
    METRIC_REGISTRY,
    VECTORIZED_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    get_vectorized_metric,
    has_vectorized_implementation,
    is_piecewise_metric,
    make_cost_metric,
    make_linear_counts_metric,
    multiclass_metric,
    multiclass_metric_exclusive,
    needs_probability_scores,
    register_metric,
    register_metrics,
    should_maximize_metric,
)
from .optimizers import (
    get_optimal_multiclass_thresholds,
    get_optimal_threshold,
)
from .types import MulticlassMetricReturn
from .wrapper import ThresholdOptimizer

__all__ = [
    "__version__",
    # Enhanced Bayes functions
    "bayes_decision_from_utility_matrix",
    "bayes_thresholds_from_costs_vector",
    "bayes_threshold_from_costs_scalar",
    # Enhanced expected functions
    "dinkelbach_expected_fbeta_binary",
    "dinkelbach_expected_fbeta_multilabel",
    # Metrics and confusion matrix
    "get_confusion_matrix",
    "get_multiclass_confusion_matrix",
    "make_cost_metric",
    "make_linear_counts_metric",
    "multiclass_metric",
    "multiclass_metric_exclusive",
    "METRIC_REGISTRY",
    "VECTORIZED_REGISTRY",
    "register_metric",
    "register_metrics",
    "is_piecewise_metric",
    "should_maximize_metric",
    "needs_probability_scores",
    "get_vectorized_metric",
    "has_vectorized_implementation",
    # Core optimization functions
    "get_optimal_threshold",
    "get_optimal_multiclass_thresholds",
    # Cross-validation
    "cv_threshold_optimization",
    "nested_cv_threshold_optimization",
    # High-level wrapper
    "ThresholdOptimizer",
    # Types
    "MulticlassMetricReturn",
]
