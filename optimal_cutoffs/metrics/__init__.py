"""Clean metrics API.

Provides access to metric registry and built-in metrics without polluting
the root namespace.
"""

# Import from the original metrics module to avoid circular imports
import importlib

# Dynamic import to avoid circular dependency
_metrics_module = importlib.import_module("optimal_cutoffs.metrics_core")
METRICS = _metrics_module.METRICS
get_metric_function = _metrics_module.get_metric_function
register_metric = _metrics_module.register_metric
is_piecewise_metric = _metrics_module.is_piecewise_metric
should_maximize_metric = _metrics_module.should_maximize_metric
compute_metric_at_threshold = _metrics_module.compute_metric_at_threshold


# Clean functional interface
def get(name: str):
    """Get metric function by name."""
    return get_metric_function(name)


def register(
    name: str, func, *, maximize: bool = True, is_piecewise: bool = False, **kwargs
):
    """Register a new metric."""
    return register_metric(
        name, func, maximize=maximize, is_piecewise=is_piecewise, **kwargs
    )


def list_available():
    """List all available metric names."""
    return list(METRICS.keys())


def info(name: str):
    """Get information about a metric."""
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}")

    return {
        "name": name,
        "maximized": should_maximize_metric(name),
        "piecewise": is_piecewise_metric(name),
        "function": get_metric_function(name),
    }


# Common built-in metrics (for convenience)
f1_score = _metrics_module.f1_score
precision_score = _metrics_module.precision_score
recall_score = _metrics_module.recall_score
accuracy_score = _metrics_module.accuracy_score
iou_score = _metrics_module.iou_score
specificity_score = _metrics_module.specificity_score

# Keep the underlying registry available but not prominent
_registry = METRICS
