"""Utility functions for optimizing classification thresholds."""

import numpy as np
from scipy import optimize


def _accuracy(prob, true_labs, pred_prob, verbose=False):
    """Helper for brute-force search that returns `1 - accuracy`.

    Parameters
    ----------
    prob : array_like
        Candidate probability threshold provided by :func:`scipy.optimize.brute`.
    true_labs : array_like
        Array of true binary labels.
    pred_prob : array_like
        Predicted probabilities for the positive class.
    verbose : bool, optional
        If ``True``, prints intermediate accuracy values.

    Returns
    -------
    float
        ``1 - accuracy`` for the supplied ``prob``.
    """

    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print("Probability: {0:0.4f} Accuracy: {1:0.4f}".format(prob[0], accuracy))
    return 1 - accuracy


def _f1(prob, true_labs, pred_prob, verbose=False):
    """Helper for brute-force search that returns ``1 - F1``.

    Parameters
    ----------
    prob : array_like
        Candidate probability threshold provided by :func:`scipy.optimize.brute`.
    true_labs : array_like
        Array of true binary labels.
    pred_prob : array_like
        Predicted probabilities for the positive class.
    verbose : bool, optional
        If ``True``, prints intermediate F1 values.

    Returns
    -------
    float
        ``1 - F1`` for the supplied ``prob``.
    """

    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("Probability: {0:0.4f} F1 score: {1:0.4f}".format(prob[0], f1))
    return 1 - f1


def get_confusion_matrix(true_labs, pred_prob, prob):
    """Compute confusion-matrix counts for a given probability threshold.

    Parameters
    ----------
    true_labs : array_like
        Array of true binary labels.
    pred_prob : array_like
        Predicted probabilities for the positive class.
    prob : float
        Probability threshold used to convert ``pred_prob`` into predicted labels.

    Returns
    -------
    tuple
        Counts of true positives, true negatives, false positives, and false negatives.
    """

    pred_labs = pred_prob > prob
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp = np.sum(np.logical_and(pred_labs == 1, true_labs == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.sum(np.logical_and(pred_labs == 0, true_labs == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(pred_labs == 1, true_labs == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.sum(np.logical_and(pred_labs == 0, true_labs == 1))

    return tp, tn, fp, fn


def get_probability(true_labs, pred_prob, objective='accuracy', verbose=False):
    """Find the probability threshold that optimizes a binary metric.

    Parameters
    ----------
    true_labs : array_like
        Array of true binary labels.
    pred_prob : array_like
        Predicted probabilities for the positive class.
    objective : {'accuracy', 'f1'}, optional
        Metric to optimize. Defaults to ``'accuracy'``.
    verbose : bool, optional
        If ``True``, prints intermediate metric values during the search.

    Returns
    -------
    float
        Probability threshold that minimizes ``1 - metric`` (i.e. maximizes the metric).
    """

    if objective == 'accuracy':
        prob = optimize.brute(
            _accuracy, (slice(0.1, 0.9, 0.1),), args=(true_labs, pred_prob, verbose), disp=verbose
        )
    elif objective == 'f1':
        prob = optimize.brute(
            _f1, (slice(0.1, 0.9, 0.1),), args=(true_labs, pred_prob, verbose), disp=verbose
        )
    else:
        raise ValueError('`objective` must be `accuracy` or `f1`')
    return prob[0]


class ThresholdOptimizer:
    """Brute-force optimizer for classification thresholds.

    Parameters
    ----------
    objective : {'accuracy', 'f1'}, optional
        Metric to optimize. Defaults to ``'accuracy'``.
    verbose : bool, optional
        If ``True``, prints intermediate metric values during fitting.
    """

    def __init__(self, objective: str = 'accuracy', verbose: bool = False):
        self.objective = objective
        self.verbose = verbose
        self.threshold_ = None

    def fit(self, true_labs, pred_prob):
        """Estimate the optimal threshold from labeled data.

        Parameters
        ----------
        true_labs : array_like
            Array of true binary labels.
        pred_prob : array_like
            Predicted probabilities for the positive class.

        Returns
        -------
        ThresholdOptimizer
            The fitted instance with ``threshold_`` attribute set.
        """

        self.threshold_ = get_probability(true_labs, pred_prob, self.objective, self.verbose)
        return self

    def predict(self, pred_prob):
        """Convert probabilities to class predictions using learned threshold.

        Parameters
        ----------
        pred_prob : array_like
            Predicted probabilities for the positive class.

        Returns
        -------
        numpy.ndarray
            Boolean array of class predictions where ``True`` indicates the positive class.

        Raises
        ------
        RuntimeError
            If the optimizer has not been fitted yet.
        """

        if self.threshold_ is None:
            raise RuntimeError('ThresholdOptimizer has not been fitted.')
        return pred_prob > self.threshold_
