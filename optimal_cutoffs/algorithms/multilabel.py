"""Multilabel classification algorithms.

Specific algorithms moved from the top-level API.
"""

from numpy.typing import ArrayLike

from ..core import OptimizationResult


def macro_independent(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Independent per-label optimization (macro averaging).

    Each label optimized separately as binary problem.
    Moved from optimize_macro_multilabel().
    """
    from ..multilabel import optimize_macro_multilabel

    return optimize_macro_multilabel(
        y_true,
        y_score,
        metric=metric,
        method=method,
        sample_weight=sample_weight,
        **kwargs,
    )


def micro_coordinate_ascent(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    max_iter: int = 30,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Coordinate ascent for coupled multilabel optimization (micro averaging).

    Thresholds coupled through global TP/FP/FN counts.
    Moved from optimize_micro_multilabel().
    """
    from ..multilabel import optimize_micro_multilabel

    return optimize_micro_multilabel(
        y_true,
        y_score,
        metric=metric,
        max_iter=max_iter,
        sample_weight=sample_weight,
        **kwargs,
    )


def general_multilabel(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    average: str = "macro",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """General multilabel optimization with auto-routing.

    Moved from optimize_multilabel().
    """
    from ..multilabel import optimize_multilabel

    return optimize_multilabel(
        y_true,
        y_score,
        metric=metric,
        average=average,
        method=method,
        sample_weight=sample_weight,
        **kwargs,
    )
