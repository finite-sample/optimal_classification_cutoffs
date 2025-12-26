"""Expected metric optimization (Dinkelbach method).

Advanced algorithms for optimizing expected values of metrics
under probability distributions.
"""

from numpy.typing import ArrayLike

from ..core import OptimizationResult


def dinkelbach_fbeta_binary(
    y_score: ArrayLike,
    *,
    beta: float = 1.0,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Expected F-beta optimization for binary classification.

    Moved from dinkelbach_expected_fbeta_binary().
    Requires calibrated probabilities.
    """
    from ..expected import dinkelbach_expected_fbeta_binary

    return dinkelbach_expected_fbeta_binary(
        y_score, beta=beta, sample_weight=sample_weight, **kwargs
    )


def dinkelbach_fbeta_multilabel(
    y_score: ArrayLike,
    *,
    beta: float = 1.0,
    average: str = "macro",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Expected F-beta optimization for multilabel classification.

    Moved from dinkelbach_expected_fbeta_multilabel().
    Requires calibrated probabilities.
    """
    from ..expected import dinkelbach_expected_fbeta_multilabel

    return dinkelbach_expected_fbeta_multilabel(
        y_score, beta=beta, average=average, sample_weight=sample_weight, **kwargs
    )
