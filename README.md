# Optimal Classification Cut-Offs

Probabilistic classifiers output per-class probabilities, and fixed cutoffs such as ``0.5`` rarely maximize metrics like accuracy or the F\ :sub:`1` score.
This package provides utilities to **select optimal probability cutoffs for each class**, supporting both multi-class and binary classifiers.
Optimization methods include brute-force search, numerical techniques, and gradient-based approaches.
Binary thresholding at a single cutoff remains fully supported as a special case.

## Quick start

```python
from optimal_cutoffs import ThresholdOptimizer

# true binary labels and predicted probabilities
y_true = ...
y_prob = ...

optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)
```

## API

### `get_confusion_matrix(true_labs, pred_prob, threshold)`
- **Purpose:** Compute confusion-matrix counts for a threshold.
- **Args:** arrays of true labels and probabilities, plus the decision threshold.
- **Returns:** `(tp, tn, fp, fn)` counts.

### `register_metric(name=None, func=None)`
- **Purpose:** Add a metric function to the global registry.
- **Args:** optional metric name and callable; can also be used as a decorator.
- **Returns:** the registered function or decorator.

### `register_metrics(metrics)`
- **Purpose:** Register multiple metric functions at once.
- **Args:** dictionary mapping names to callables.
- **Returns:** `None`.

### `get_probability(true_labs, pred_prob, objective='accuracy', verbose=False)`
- **Purpose:** Brute-force search for the threshold that maximizes accuracy or F\ :sub:`1`.
- **Args:** true labels, predicted probabilities, metric name, and verbosity flag.
- **Returns:** optimal threshold.

### `get_optimal_threshold(true_labs, pred_prob, metric='f1', method='smart_brute')`
- **Purpose:** Optimize any registered metric using different strategies
  (brute force, ``minimize``, or ``gradient``).
- **Args:** true labels, probabilities, metric name, and optimization method.
- **Returns:** optimal threshold.

### `cv_threshold_optimization(true_labs, pred_prob, metric='f1', method='smart_brute', cv=5, random_state=None)`
- **Purpose:** Estimate thresholds via cross-validation and report per-fold scores.
- **Returns:** arrays of thresholds and scores.

### `nested_cv_threshold_optimization(true_labs, pred_prob, metric='f1', method='smart_brute', inner_cv=5, outer_cv=5, random_state=None)`
- **Purpose:** Perform nested cross-validation for threshold estimation and
  unbiased performance evaluation.
- **Returns:** arrays of outer-fold thresholds and scores.

### `ThresholdOptimizer(objective='accuracy', verbose=False)`
- **Purpose:** High-level wrapper with ``fit``/``predict`` methods.
- **Args:** metric name and verbosity flag.
- **Returns:** fitted instance with ``threshold_`` attribute after calling ``fit``.

## Examples

- [Cross-validation and gradient methods](examples/comscore.ipynb)

## Authors

Suriyan Laohaprapanon and Gaurav Sood
