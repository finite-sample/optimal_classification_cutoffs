from scipy import optimize
from metrics import METRICS


def get_optimal_threshold(true_labs, pred_prob, objective: str = "accuracy", verbose: bool = False):
    """Find the probability threshold that optimizes a chosen metric."""
    try:
        metric = METRICS[objective]
    except KeyError as exc:
        raise ValueError(f"`objective` must be one of {list(METRICS)}") from exc
    prob = optimize.brute(
        metric,
        (slice(0.1, 0.9, 0.1),),
        args=(true_labs, pred_prob, verbose),
        disp=verbose,
    )
    return prob[0]


# Backwards compatibility
get_probability = get_optimal_threshold
