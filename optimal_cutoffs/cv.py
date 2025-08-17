"""Cross-validation helpers for threshold optimization."""

from typing import Tuple

import numpy as np
from sklearn.model_selection import KFold

from .metrics import accuracy, f1_score
from .optimizers import (
    brute_threshold,
    smart_brute_threshold,
    minimize_threshold,
    gradient_threshold,
)

_METHODS = {
    "brute": brute_threshold,
    "smart_brute": smart_brute_threshold,
    "minimize": minimize_threshold,
    "gradient": gradient_threshold,
}

_METRICS = {
    "accuracy": accuracy,
    "f1": f1_score,
    "f1_score": f1_score,
}


def cv_threshold_optimization(
    model,
    X,
    y,
    metric: str = "accuracy",
    cv: int = 5,
    method: str = "brute",
    **kwargs,
) -> Tuple[float, float]:
    """Estimate optimal threshold via cross-validation.

    Returns
    -------
    tuple
        Mean threshold and mean score across folds.
    """

    metric_func = _METRICS[metric]
    opt_func = _METHODS[method]

    kf = KFold(cv)
    thresholds = []
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        threshold = opt_func(y_train, train_prob, metric=metric, **kwargs)
        thresholds.append(threshold)
        scores.append(metric_func(y_test, test_prob, threshold))
    return float(np.mean(thresholds)), float(np.mean(scores))


def nested_cv_threshold_optimization(
    model,
    X,
    y,
    metric: str = "accuracy",
    outer_cv: int = 5,
    inner_cv: int = 3,
    method: str = "brute",
    **kwargs,
) -> float:
    """Nested cross-validation to evaluate threshold tuning."""

    metric_func = _METRICS[metric]
    opt_func = _METHODS[method]

    outer_kf = KFold(outer_cv)
    scores = []
    for train_idx, test_idx in outer_kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_threshold, _ = cv_threshold_optimization(
            model, X_train, y_train, metric=metric, cv=inner_cv, method=method, **kwargs
        )
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        scores.append(metric_func(y_test, test_prob, inner_threshold))
    return float(np.mean(scores))
