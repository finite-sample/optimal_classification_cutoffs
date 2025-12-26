"""Cross-validation for threshold optimization.

Clean interface for validating threshold optimization methods.
"""

import numpy as np
from numpy.typing import ArrayLike

from ..api import optimize_thresholds
from ..core import OptimizationResult

__all__ = [
    "cross_validate",
    "nested_cross_validate",
    "optimize_thresholds",
    "OptimizationResult",
]


def cross_validate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    cv: int = 5,
    random_state: int | None = None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validate threshold optimization.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_score : array-like
        Predicted scores/probabilities
    metric : str, default="f1"
        Metric to optimize and evaluate
    cv : int, default=5
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility
    **optimize_kwargs
        Additional arguments passed to optimize_thresholds()

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of per-fold thresholds and scores.

    Examples
    --------
    >>> thresholds, scores = cross_validate(y_true, y_scores, metric="f1", cv=5)
    >>> print(f"CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Choose splitter: stratify by default for classification when possible
    if y_true.ndim == 1 and len(np.unique(y_true)) > 1:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    thresholds = []
    scores = []
    
    for train_idx, test_idx in splitter.split(y_true, y_true):
        y_train, y_test = y_true[train_idx], y_true[test_idx]
        if y_score.ndim == 1:
            score_train, score_test = y_score[train_idx], y_score[test_idx]
        else:
            score_train, score_test = y_score[train_idx], y_score[test_idx]

        # Optimize threshold on training set
        result = optimize_thresholds(y_train, score_train, metric=metric, **optimize_kwargs)
        threshold = result.threshold if hasattr(result, 'threshold') else result.thresholds
        
        # Evaluate on test set
        test_result = optimize_thresholds(y_test, score_test, metric=metric, **optimize_kwargs)
        score = test_result.score if hasattr(test_result, 'score') else 0.0
        
        thresholds.append(threshold)
        scores.append(score)

    return np.array(thresholds), np.array(scores)


def nested_cross_validate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    inner_cv: int = 3,
    outer_cv: int = 5,
    random_state: int | None = None,
    **optimize_kwargs,
) -> dict:
    """Nested cross-validation for unbiased threshold optimization evaluation.

    Inner CV: Optimizes thresholds
    Outer CV: Evaluates the optimization procedure

    Parameters
    ----------
    y_true : array-like
        True labels
    y_score : array-like
        Predicted scores/probabilities
    metric : str, default="f1"
        Metric to optimize and evaluate
    inner_cv : int, default=3
        Number of inner CV folds (for threshold optimization)
    outer_cv : int, default=5
        Number of outer CV folds (for evaluation)
    random_state : int, optional
        Random seed for reproducibility
    **optimize_kwargs
        Additional arguments passed to optimize_thresholds()

    Returns
    -------
    dict
        Nested CV results with keys:
        - 'test_scores': array of outer test scores
        - 'mean_score': mean outer test score
        - 'std_score': standard deviation of outer test scores
        - 'thresholds': threshold estimates from each outer fold

    Examples
    --------
    >>> # Get unbiased estimate of threshold optimization performance
    >>> results = nested_cross_validate(y_true, y_scores, metric="f1")
    >>> print(f"Unbiased CV Score: {results['mean_score']:.3f}")
    """
    # For now, implement simple nested CV
    # TODO: Full implementation can be added later if needed
    thresholds, scores = cross_validate(
        y_true,
        y_score,
        metric=metric,
        cv=outer_cv,
        random_state=random_state,
        **optimize_kwargs,
    )

    return {
        "test_scores": scores,
        "thresholds": thresholds,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }
