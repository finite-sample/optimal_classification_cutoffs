"""Optimal Classification Thresholds - Mathematical Taxonomy Implementation.

This library provides mathematically principled threshold optimization for
classification problems, organized by problem type:

**Binary Classification:**
- optimize_f1_binary(): O(n log n) exact F1 optimization
- optimize_utility_binary(): O(1) closed-form Bayes optimal
- optimize_metric_binary(): General metric optimization with auto method selection

**Multi-Label Classification:**
- optimize_macro_multilabel(): K independent optimizations (exact)
- optimize_micro_multilabel(): Coordinate ascent for coupled thresholds
- optimize_multilabel(): Auto-routing based on averaging strategy

**Multi-Class Classification:**
- optimize_ovr_independent(): Independent per-class optimization (can predict multiple)
- optimize_ovr_margin(): Margin rule with coordinate ascent (single-label)
- optimize_micro_multiclass(): Single threshold for all classes
- optimize_multiclass(): Auto-routing based on method and averaging

**Bayes-Optimal Decisions:**
- bayes_optimal_threshold(): Binary closed-form from costs/utilities
- bayes_thresholds_from_costs(): Per-class OvR thresholds (closed-form)
- bayes_optimal_decisions(): General cost matrix optimization (no thresholds!)

**Expected Metrics (Dinkelbach):**
- dinkelbach_expected_fbeta_binary(): Expected F-beta optimization
- dinkelbach_expected_fbeta_multilabel(): Multilabel expected F-beta

**Convenience Function:**
- get_optimal_threshold(): Auto-detect problem type and route (for quick use)

Design Principle: Problem type determines algorithm choice based on mathematical properties.
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

# Binary classification functions
# Bayes-optimal decisions
from .bayes import (
    BayesOptimal,
    UtilitySpec,
    bayes_optimal_decisions,
    bayes_optimal_threshold,
    bayes_thresholds_from_costs,
    compute_bayes_threshold,
)
from .binary import (
    optimize_f1_binary,
    optimize_metric_binary,
    optimize_utility_binary,
)

# Convenience dispatcher (auto problem type detection)
from .core import (
    get_optimal_threshold,
    infer_problem_type,
)

# Cross-validation utilities
from .cv import (
    cv_threshold_optimization,
    nested_cv_threshold_optimization,
)

# Expected metrics (Dinkelbach)
from .expected import (
    dinkelbach_expected_fbeta_binary,
    dinkelbach_expected_fbeta_multilabel,
)

# Metrics and utilities
from .metrics import (
    METRICS,
    accuracy_score,
    compute_metric_at_threshold,
    compute_multiclass_metrics_from_labels,
    confusion_matrix_at_threshold,
    confusion_matrix_from_predictions,
    f1_score,
    get_metric_function,
    has_vectorized_implementation,
    iou_score,
    is_piecewise_metric,
    make_cost_metric,
    make_linear_counts_metric,
    multiclass_confusion_matrices_at_thresholds,
    multiclass_metric_ovr,
    multiclass_metric_single_label,
    needs_probability_scores,
    precision_score,
    recall_score,
    register_metric,
    register_metrics,
    should_maximize_metric,
    specificity_score,
)

# Multi-class classification functions
from .multiclass import (
    optimize_micro_multiclass,
    optimize_multiclass,
    optimize_ovr_independent,
    optimize_ovr_margin,
)

# Multi-label classification functions
from .multilabel import (
    optimize_macro_multilabel,
    optimize_micro_multilabel,
    optimize_multilabel,
)

# Core result type
from .types_minimal import OptimizationResult

# ============================================================================
# Main API Exports - Organized by Problem Type
# ============================================================================

__all__ = [
    "__version__",
    # === Problem Type Detection ===
    "infer_problem_type",
    # === Binary Classification ===
    "optimize_f1_binary",  # O(n log n) exact F1
    "optimize_utility_binary",  # O(1) closed-form Bayes
    "optimize_metric_binary",  # General metric with auto method
    # === Multi-Label Classification ===
    "optimize_macro_multilabel",  # K independent (exact)
    "optimize_micro_multilabel",  # Coordinate ascent (coupled)
    "optimize_multilabel",  # Auto-route by averaging
    # === Multi-Class Classification ===
    "optimize_ovr_independent",  # Independent per-class (can predict multiple)
    "optimize_ovr_margin",  # Margin rule + coord ascent (single-label)
    "optimize_micro_multiclass",  # Single threshold for all classes
    "optimize_multiclass",  # Auto-route by method/averaging
    # === Bayes-Optimal Decisions ===
    "bayes_optimal_threshold",  # Binary closed-form from costs
    "bayes_thresholds_from_costs",  # Per-class OvR (closed-form)
    "bayes_optimal_decisions",  # General cost matrix (O(K²), no thresholds!)
    "BayesOptimal",  # Class-based interface
    "UtilitySpec",  # Utility specification helper
    "compute_bayes_threshold",  # Simple API for binary costs
    # === Expected Metrics (Dinkelbach) ===
    "dinkelbach_expected_fbeta_binary",  # Binary expected F-beta
    "dinkelbach_expected_fbeta_multilabel",  # Multilabel expected F-beta
    # === Cross-Validation ===
    "cv_threshold_optimization",  # Standard CV for threshold validation
    "nested_cv_threshold_optimization",  # Nested CV for unbiased estimates
    # === Convenience Auto-Dispatcher ===
    "get_optimal_threshold",  # Auto-detect problem type (convenience only)
    # === Metrics and Utilities ===
    "METRICS",  # Global metric registry
    "accuracy_score",
    "f1_score",
    "precision_score",
    "recall_score",
    "iou_score",
    "specificity_score",
    "register_metric",
    "register_metrics",
    "get_metric_function",
    "is_piecewise_metric",
    "should_maximize_metric",
    "needs_probability_scores",
    "has_vectorized_implementation",
    "make_cost_metric",
    "make_linear_counts_metric",
    "compute_metric_at_threshold",
    "confusion_matrix_at_threshold",
    "confusion_matrix_from_predictions",
    "multiclass_confusion_matrices_at_thresholds",
    "multiclass_metric_ovr",
    "multiclass_metric_single_label",
    "compute_multiclass_metrics_from_labels",
    # === Types ===
    "OptimizationResult",
]


# ============================================================================
# Quick Start Guide for Common Use Cases
# ============================================================================


def _print_quick_start():
    """Print quick start guide organized by problem type."""
    print("""
Optimal Classification Thresholds - Quick Start by Problem Type

=== Binary Classification ===
from optimal_cutoffs import optimize_f1_binary, optimize_utility_binary

# F1 optimization (exact, O(n log n))
result = optimize_f1_binary(y_true, y_prob)

# Cost-sensitive (closed-form, O(1))
utility = {"tp": 10, "tn": 1, "fp": -1, "fn": -5}
result = optimize_utility_binary(y_true, y_prob, utility=utility)

=== Multi-Label Classification ===
from optimal_cutoffs import optimize_multilabel

# Macro: independent per-label (exact)
result = optimize_multilabel(y_true, y_prob, average="macro")

# Micro: coupled via global TP/FP/FN (coordinate ascent)
result = optimize_multilabel(y_true, y_prob, average="micro")

=== Multi-Class Classification ===
from optimal_cutoffs import optimize_multiclass

# Margin rule (single-label, coordinate ascent)
result = optimize_multiclass(y_true, y_prob, method="coord_ascent")

# Independent per-class (can predict multiple)
result = optimize_multiclass(y_true, y_prob, method="independent")

=== General Cost Matrix (No Thresholds!) ===
from optimal_cutoffs import bayes_optimal_decisions

cost_matrix = np.array([[0, 10, 50], [10, 0, 40], [100, 90, 0]])
result = bayes_optimal_decisions(y_prob, cost_matrix=cost_matrix)
# Uses direct Bayes rule: argmin_j Σ_i p_i * C(i,j)

=== Auto-Detection (Convenience) ===
from optimal_cutoffs import get_optimal_threshold

# Detects problem type automatically
result = get_optimal_threshold(y_true, y_prob, metric="f1")

See documentation for full taxonomy and algorithm details.
    """)


# ============================================================================
# Backward Compatibility Aliases (Legacy API)
# ============================================================================


# Legacy function aliases for backward compatibility
def get_optimal_binary_threshold(*args, **kwargs):
    """Legacy alias for get_optimal_threshold() with binary data."""
    return get_optimal_threshold(*args, **kwargs)


def get_optimal_multiclass_thresholds(*args, **kwargs):
    """Legacy alias for get_optimal_threshold() with multiclass data."""
    return get_optimal_threshold(*args, **kwargs)


def get_optimal_multilabel_thresholds(*args, **kwargs):
    """Legacy alias for get_optimal_threshold() with multilabel data."""
    return get_optimal_threshold(*args, **kwargs)


# Add legacy aliases to exports
__all__.extend(
    [
        "get_optimal_binary_threshold",
        "get_optimal_multiclass_thresholds",
        "get_optimal_multilabel_thresholds",
    ]
)

# Uncomment to show quick start on import
# _print_quick_start()
