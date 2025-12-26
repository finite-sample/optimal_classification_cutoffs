"""Optimal threshold selection for classification problems.

Clean, minimal API focused on user jobs-to-be-done rather than
internal mathematical taxonomy.
"""

# Single source of truth for version
try:
    from importlib.metadata import version

    __version__ = version("optimal-classification-cutoffs")
except Exception:
    import pathlib
    import tomllib

    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    else:
        __version__ = "unknown"

# NEW CLEAN API - 8 exports total
# Import namespaces directly - fix circular imports within them
from . import algorithms, bayes, cv
from .api import optimize_decisions, optimize_thresholds
from .core import Average, OptimizationResult, Task


# Import metrics separately to handle the circular import
def _import_metrics():
    """Import metrics namespace avoiding circular dependency."""
    import importlib.util
    import os

    # Load metrics namespace module
    spec = importlib.util.spec_from_file_location(
        "optimal_cutoffs.metrics",
        os.path.join(os.path.dirname(__file__), "metrics", "__init__.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from . import metrics
except ImportError:
    # Handle circular import by loading manually
    metrics = _import_metrics()

__all__ = [
    "__version__",
    # === Core API ===
    "optimize_thresholds",  # THE canonical entry point
    "optimize_decisions",  # For cost matrices (no thresholds)
    "OptimizationResult",  # Unified result type
    "Task",
    "Average",  # Enums for explicit choices
    # === Namespaced Power Tools ===
    "metrics",  # metrics.get(), metrics.register(), etc.
    "bayes",  # bayes.threshold(), bayes.policy(), etc.
    "cv",  # cv.cross_validate(), cv.nested_cross_validate()
    "algorithms",  # algorithms.multiclass.ovr_margin(), etc.
]
