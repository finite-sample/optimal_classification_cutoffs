import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from optimal_cut_offs import get_confusion_matrix, precision_recall_f1


def test_confusion_matrix_and_metrics():
    true_labs = [1, 1, 0, 0]
    pred_prob = [0.9, 0.2, 0.6, 0.4]
    threshold = 0.5
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, threshold)
    assert (tp, tn, fp, fn) == (1, 1, 1, 1)
    precision, recall, f1 = precision_recall_f1(true_labs, pred_prob, threshold)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
