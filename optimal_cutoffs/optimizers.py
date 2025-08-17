"""Threshold search algorithms."""

import numpy as np
from scipy import optimize

from .metrics import accuracy, f1_score


_METRICS = {
    "accuracy": accuracy,
    "f1": f1_score,
    "f1_score": f1_score,
}


def _get_metric(metric):
    try:
        return _METRICS[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric '{metric}'.") from exc


def brute_threshold(y_true, y_prob, metric="accuracy", step=0.1, verbose=False):
    """Search threshold using :func:`scipy.optimize.brute`."""

    metric_func = _get_metric(metric)

    def objective(thresh):
        return -metric_func(y_true, y_prob, thresh[0])

    result = optimize.brute(objective, (slice(0, 1, step),), disp=verbose)
    return float(result[0])


def smart_brute_threshold(
    y_true,
    y_prob,
    metric="accuracy",
    coarse_step=0.1,
    fine_step=0.01,
    verbose=False,
):
    """Two-stage brute-force search with a coarse then fine grid."""

    metric_func = _get_metric(metric)
    thresholds = np.arange(0, 1 + coarse_step, coarse_step)
    scores = [metric_func(y_true, y_prob, t) for t in thresholds]
    best = thresholds[int(np.argmax(scores))]

    start = max(best - coarse_step, 0)
    end = min(best + coarse_step, 1)
    fine_thresholds = np.arange(start, end + fine_step, fine_step)
    fine_scores = [metric_func(y_true, y_prob, t) for t in fine_thresholds]
    return float(fine_thresholds[int(np.argmax(fine_scores))])


def minimize_threshold(y_true, y_prob, metric="accuracy", verbose=False):
    """Use :func:`scipy.optimize.minimize_scalar` to find threshold."""

    metric_func = _get_metric(metric)

    def objective(thresh):
        return -metric_func(y_true, y_prob, thresh)

    result = optimize.minimize_scalar(objective, bounds=(0, 1), method="bounded")
    if verbose:
        print(result)
    return float(result.x)


def gradient_threshold(
    y_true,
    y_prob,
    metric="accuracy",
    lr=0.1,
    n_iter=100,
):
    """Simple gradient-descent search for the best threshold."""

    metric_func = _get_metric(metric)
    thresh = 0.5
    for _ in range(n_iter):
        grad = (
            metric_func(y_true, y_prob, min(thresh + 1e-5, 1))
            - metric_func(y_true, y_prob, max(thresh - 1e-5, 0))
        ) / 2e-5
        thresh = np.clip(thresh + lr * grad, 0, 1)
    return float(thresh)
