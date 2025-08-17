## Optimal Classification Cut-Offs

[![PyPI version](https://img.shields.io/pypi/v/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)
[![PyPI Downloads](https://static.pepy.tech/badge/optimal-classification-cutoffs)](https://pepy.tech/projects/optimal-classification-cutoffs)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Probabilistic classifiers output per-class probabilities, and fixed cutoffs such as ``0.5`` rarely maximize metrics like accuracy or the F\ :sub:`1` score.
This package provides utilities to **select optimal probability cutoffs for both binary and multiclass classification**.
For multiclass problems, the package uses a One-vs-Rest strategy to optimize per-class thresholds independently.
Optimization methods include brute-force search, numerical techniques, and gradient-based approaches.

## Quick start

### Binary Classification
```python
from optimal_cutoffs import ThresholdOptimizer

# true binary labels and predicted probabilities
y_true = [0, 1, 1, 0, 1]
y_prob = [0.2, 0.8, 0.7, 0.3, 0.9]

optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)
```

### Multiclass Classification
```python
import numpy as np
from optimal_cutoffs import ThresholdOptimizer

# true multiclass labels and predicted probability matrix
y_true = [0, 1, 2, 0, 1]
y_prob = np.array([
    [0.7, 0.2, 0.1],  # probabilities for classes 0, 1, 2
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
])

optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)  # returns class indices
```

## API

### `get_confusion_matrix(true_labs, pred_prob, threshold)`
- **Purpose:** Compute confusion-matrix counts for a threshold.
- **Args:** arrays of true binary labels and probabilities, plus the decision threshold.
- **Returns:** `(tp, tn, fp, fn)` counts.

### `get_multiclass_confusion_matrix(true_labs, pred_prob, thresholds)`
- **Purpose:** Compute per-class confusion-matrix counts for multiclass classification using One-vs-Rest.
- **Args:** true class labels (0, 1, 2, ...), probability matrix (n_samples, n_classes), and per-class thresholds.
- **Returns:** List of per-class `(tp, tn, fp, fn)` tuples.

### `register_metric(name=None, func=None)`
- **Purpose:** Add a metric function to the global registry.
- **Args:** optional metric name and callable; can also be used as a decorator.
- **Returns:** the registered function or decorator.

### `register_metrics(metrics)`
- **Purpose:** Register multiple metric functions at once.
- **Args:** dictionary mapping names to callables.
- **Returns:** `None`.

### `multiclass_metric(confusion_matrices, metric_name, average="macro")`
- **Purpose:** Compute multiclass metrics from per-class confusion matrices.
- **Args:** list of confusion matrices, metric name, averaging strategy ("macro", "micro", "weighted").
- **Returns:** aggregated metric score.

### `get_probability(true_labs, pred_prob, objective='accuracy', verbose=False)`
- **Purpose:** Brute-force search for the threshold that maximizes accuracy or F\ :sub:`1` using scipy.optimize.brute.
- **Args:** true labels, predicted probabilities, objective ("accuracy" or "f1"), and verbosity flag.
- **Returns:** optimal threshold.

### `get_optimal_threshold(true_labs, pred_prob, metric='f1', method='smart_brute')`
- **Purpose:** Optimize any registered metric using different strategies: "smart_brute" (evaluates all unique probabilities), "minimize" (scipy.optimize.minimize_scalar), or "gradient" (simple gradient ascent). **Automatically detects binary vs multiclass inputs.**
- **Args:** true labels (binary or multiclass), probabilities (1D for binary, 2D for multiclass), metric name, and optimization method.
- **Returns:** optimal threshold (float for binary, array for multiclass).

### `get_optimal_multiclass_thresholds(true_labs, pred_prob, metric='f1', method='smart_brute', average='macro')`
- **Purpose:** Find optimal per-class thresholds for multiclass classification using One-vs-Rest strategy.
- **Args:** true class labels, probability matrix (n_samples, n_classes), metric name, optimization method, and averaging strategy.
- **Returns:** array of optimal thresholds, one per class.

### `cv_threshold_optimization(true_labs, pred_prob, metric='f1', method='smart_brute', cv=5, random_state=None)`
- **Purpose:** Estimate thresholds via cross-validation and report per-fold scores.
- **Returns:** arrays of thresholds and scores.

### `nested_cv_threshold_optimization(true_labs, pred_prob, metric='f1', method='smart_brute', inner_cv=5, outer_cv=5, random_state=None)`
- **Purpose:** Perform nested cross-validation for threshold estimation and
  unbiased performance evaluation.
- **Returns:** arrays of outer-fold thresholds and scores.

### `ThresholdOptimizer(objective='accuracy', verbose=False, method='smart_brute')`
- **Purpose:** High-level wrapper with ``fit``/``predict`` methods using scikit-learn style API. **Supports both binary and multiclass classification.**
- **Args:** objective metric name (e.g., "accuracy", "f1", "precision", "recall"), verbosity flag, and optimization method.
- **Returns:** fitted instance with ``threshold_`` attribute (float for binary, array for multiclass). The ``predict`` method returns boolean predictions for binary, class indices for multiclass.

## Examples

- [Basic binary usage](examples/basic_usage.py)
- [Advanced binary usage with sklearn](examples/advanced_usage.ipynb)  
- [Multiclass classification](examples/multiclass_usage.py)
- [Cross-validation and gradient methods](examples/comscore.ipynb)

## Authors

Suriyan Laohaprapanon and Gaurav Sood
