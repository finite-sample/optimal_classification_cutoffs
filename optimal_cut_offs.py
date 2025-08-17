"""Utilities for finding optimal probability thresholds without external deps."""

def get_confusion_matrix(true_labs, pred_prob, prob):
    """Compute confusion matrix counts for a given threshold."""
    pred_labs = [1 if p > prob else 0 for p in pred_prob]
    tp = sum(1 for pl, tl in zip(pred_labs, true_labs) if pl == 1 and tl == 1)
    tn = sum(1 for pl, tl in zip(pred_labs, true_labs) if pl == 0 and tl == 0)
    fp = sum(1 for pl, tl in zip(pred_labs, true_labs) if pl == 1 and tl == 0)
    fn = sum(1 for pl, tl in zip(pred_labs, true_labs) if pl == 0 and tl == 1)
    return tp, tn, fp, fn

def precision_recall_f1(true_labs, pred_prob, prob):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1

def accuracy(true_labs, pred_prob, prob):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0

def get_optimal_threshold(true_labs, pred_prob, method='smart_brute', objective='f1'):
    """Find the threshold maximizing the chosen metric."""
    def metric(thr):
        if objective == 'f1':
            return precision_recall_f1(true_labs, pred_prob, thr)[2]
        elif objective == 'accuracy':
            return accuracy(true_labs, pred_prob, thr)
        else:
            raise ValueError('Unknown objective')
    if method == 'smart_brute':
        thresholds = [i / 100 for i in range(1, 100)]
        return max(thresholds, key=metric)
    elif method == 'minimize':
        candidates = sorted(set(pred_prob))
        candidates = [0.0] + candidates + [1.0]
        midpoints = [(candidates[i] + candidates[i + 1]) / 2 for i in range(len(candidates) - 1)]
        return max(midpoints, key=metric)
    elif method == 'gradient':
        thr = max([i / 100 for i in range(1, 100)], key=metric)
        step = 0.1
        while step > 1e-3:
            improved = False
            for cand in (thr - step, thr + step):
                if 0 < cand < 1 and metric(cand) > metric(thr):
                    thr = cand
                    improved = True
                    break
            if not improved:
                step /= 2
        return thr
    else:
        raise ValueError('Unknown method')

def cross_validate_thresholds(true_labs, pred_prob, method='smart_brute', objective='f1', k=5):
    """Simple k-fold cross validation for threshold optimization."""
    n = len(true_labs)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    thresholds = []
    scores = []
    indices = list(range(n))
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = indices[start:stop]
        train_idx = indices[:start] + indices[stop:]
        train_true = [true_labs[i] for i in train_idx]
        train_prob = [pred_prob[i] for i in train_idx]
        test_true = [true_labs[i] for i in test_idx]
        test_prob = [pred_prob[i] for i in test_idx]
        thr = get_optimal_threshold(train_true, train_prob, method, objective)
        thresholds.append(thr)
        if objective == 'f1':
            _, _, score = precision_recall_f1(test_true, test_prob, thr)
        else:
            score = accuracy(test_true, test_prob, thr)
        scores.append(score)
        start = stop
    return thresholds, scores

def get_probability(true_labs, pred_prob, objective='accuracy', verbose=False):
    """Backward compatible wrapper returning optimal probability."""
    obj = 'accuracy' if objective == 'accuracy' else 'f1'
    return get_optimal_threshold(true_labs, pred_prob, 'smart_brute', obj)
