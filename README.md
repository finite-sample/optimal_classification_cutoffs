# Optimal Classification Cut-Offs

This package helps you pick the best probability threshold for binary classifiers.
Probabilistic models emit scores between 0 and 1, but the default threshold of 0.5 rarely
produces the most useful predictions. `optimal_cut_offs` performs a simple brute-force search
over candidate thresholds and returns the value that maximizes metrics such as accuracy or the F₁ score.

## Quick start

```python
from optimal_cut_offs import ThresholdOptimizer

# true binary labels and predicted probabilities
y_true = ...
y_prob = ...

optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)
```

## API

### `get_confusion_matrix(true_labs, pred_prob, prob)`
Return counts of true/false positives and negatives for a probability threshold.

### `get_probability(true_labs, pred_prob, objective='accuracy', verbose=False)`
Brute-force search for the threshold that maximizes accuracy or F₁.

### `ThresholdOptimizer(objective='accuracy', verbose=False)`
Convenience wrapper around `get_probability` with `fit`/`predict` methods.

## Examples

- [Cross-validation and gradient methods](comscore.ipynb)

## Authors

Suriyan Laohaprapanon and Gaurav Sood
