"""Metrics and confusion matrix utilities."""

import numpy as np


def confusion_matrix(y_true, y_prob, threshold):
    """Return confusion matrix counts for a probability threshold.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float
        Probability threshold used to convert probabilities to labels.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob > threshold

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def accuracy(y_true, y_prob, threshold):
    """Classification accuracy for a given threshold."""
    tp, tn, fp, fn = confusion_matrix(y_true, y_prob, threshold)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total else 0.0


def f1_score(y_true, y_prob, threshold):
    """F1 score for a given threshold."""
    tp, tn, fp, fn = confusion_matrix(y_true, y_prob, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0
