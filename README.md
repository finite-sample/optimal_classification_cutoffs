## Optimal Classification Cut-Offs


[![Python application](https://github.com/finite-sample/optimal_classification_cutoffs/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/optimal_classification_cutoffs/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/optimal_classification_cutoffs/)
[![PyPI version](https://img.shields.io/pypi/v/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)
[![PyPI Downloads](https://static.pepy.tech/badge/optimal-classification-cutoffs)](https://pepy.tech/projects/optimal-classification-cutoffs)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/optimal_classification_cutoffs/master/pyproject.toml&query=$.project.requires-python&label=Python)](https://github.com/finite-sample/optimal_classification_cutoffs)
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

# Auto-selection (uses sort_scan for piecewise metrics like F1)
optimizer = ThresholdOptimizer(objective="f1", method="auto")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)

# Explicit sort_scan for maximum performance on piecewise metrics
optimizer_fast = ThresholdOptimizer(objective="f1", method="sort_scan")
optimizer_fast.fit(y_true, y_prob)
y_pred_fast = optimizer_fast.predict(y_prob)
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

# Standard One-vs-Rest optimization
optimizer = ThresholdOptimizer(objective="f1", method="auto")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)

# For imbalanced datasets, try coordinate ascent
optimizer_coord = ThresholdOptimizer(objective="f1", method="coord_ascent")
optimizer_coord.fit(y_true, y_prob)
y_pred_coord = optimizer_coord.predict(y_prob)  # often better macro-F1
```

## Understanding Piecewise-Constant Metrics

Classification metrics like **F1 score, accuracy, precision, and recall are piecewise-constant functions** with respect to the decision threshold. This means they only change values when the threshold crosses one of the unique predicted probabilities—they remain constant between these "breakpoints."

### Why Continuous Optimizers Can Miss the Maximum

Standard optimization methods like `scipy.optimize.minimize_scalar` assume smooth functions and use gradient-based techniques. However, piecewise-constant functions have:

- **Zero gradients** everywhere except at breakpoints (where they're undefined)
- **Flat regions** that provide no directional information to guide optimization
- **Step discontinuities** that can trap optimizers in suboptimal regions

The figure below illustrates this phenomenon:

![F1 Score Piecewise Behavior](https://finite-sample.github.io/optimal_classification_cutoffs/piecewise_f1_demo.png)

*F1 score only changes at unique probability values (red dots). Continuous optimizers may converge anywhere within the flat regions, potentially missing the true optimum.*

### Our Solution: Optimized Algorithms for Piecewise Metrics

This library addresses the piecewise-constant challenge through multiple specialized algorithms:

1. **Sort-and-Scan (`method="sort_scan"`)**: Our fastest exact algorithm for piecewise-constant metrics in binary classification. Uses O(n log n) sorting with vectorized confusion matrix computation, providing 50-100x performance improvements while guaranteeing the global optimum.

2. **Smart Brute Force (`method="smart_brute"`)**: Fallback algorithm that evaluates all unique probability values as threshold candidates. Still significantly faster than naive approaches through optimized implementations.

3. **Coordinate Ascent (`method="coord_ascent"`)**: Specialized multiclass optimizer that maintains single-label consistency through coupled threshold optimization. Particularly effective for imbalanced datasets where it often improves macro-F1 compared to independent One-vs-Rest optimization.

4. **Auto Selection (`method="auto"`)**: Intelligently selects the best algorithm based on your data and metric characteristics.

5. **Fallback Mechanisms**: Including `scipy.optimize.minimize_scalar` with unique probability evaluation for non-piecewise metrics.

For detailed mathematical explanation and interactive visualizations, see our [theoretical documentation](https://finite-sample.github.io/optimal_classification_cutoffs/theory.html).

## When to Use Calibration

**Threshold optimization and probability calibration serve different purposes and are complementary:**

### Use Calibration When:
- You need reliable probability estimates (e.g., "this prediction has 70% confidence")
- Comparing models based on probability quality
- Converting arbitrary scores to meaningful probabilities

### Use Threshold Optimization When:
- Maximizing specific classification metrics (F1, precision, recall)
- Making binary decisions for deployment
- Handling imbalanced datasets (where 0.5 threshold is suboptimal)

### Best Practice: Use Both Together

```python
from sklearn.calibration import CalibratedClassifierCV
from optimal_cutoffs import ThresholdOptimizer

# 1. Train and calibrate your classifier
calibrated_model = CalibratedClassifierCV(base_model, cv=3)
calibrated_model.fit(X_train, y_train)
y_prob_cal = calibrated_model.predict_proba(X_val)[:, 1]

# 2. Optimize threshold on calibrated probabilities
optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_val, y_prob_cal)

# 3. Use both for final predictions
y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]  # Calibrated probabilities
y_pred_test = optimizer.predict(y_prob_test)                # Optimized decisions
```

**Key insight:** Calibration improves probability quality; threshold optimization maximizes classification metrics. Using both gives you reliable probabilities *and* optimal decisions.

For more details on calibration methods (Platt scaling, isotonic regression) and when to use them, see our [full documentation](https://finite-sample.github.io/optimal_classification_cutoffs/theory.html#calibration-and-threshold-optimization).

## Advanced Methods and Future Enhancements

**Expected F-beta optimization via Dinkelbach method:** For scenarios requiring optimization of expected F-beta under calibrated probabilities, the Dinkelbach fractional programming method provides an ultra-fast exact solution. This leverages the F1 threshold identity that states the optimal threshold equals the ratio of false negatives to false positives at optimality. This mathematically elegant approach can be significantly faster than iterative optimization for calibrated probability distributions. Implementation of this method is planned for a future release, building on the theoretical foundation established in the threshold optimization literature.

## API

`get_confusion_matrix(true_labs, pred_prob, threshold)`
- **Purpose:** Compute confusion-matrix counts for a threshold.
- **Args:** arrays of true binary labels and probabilities, plus the decision threshold.
- **Returns:** `(tp, tn, fp, fn)` counts.

`get_multiclass_confusion_matrix(true_labs, pred_prob, thresholds)`
- **Purpose:** Compute per-class confusion-matrix counts for multiclass classification using One-vs-Rest.
- **Args:** true class labels (0, 1, 2, ...), probability matrix (n_samples, n_classes), and per-class thresholds.
- **Returns:** List of per-class `(tp, tn, fp, fn)` tuples.

`register_metric(name=None, func=None)`
- **Purpose:** Add a metric function to the global registry.
- **Args:** optional metric name and callable; can also be used as a decorator.
- **Returns:** the registered function or decorator.

`register_metrics(metrics)`
- **Purpose:** Register multiple metric functions at once.
- **Args:** dictionary mapping names to callables.
- **Returns:** `None`.

`multiclass_metric(confusion_matrices, metric_name, average="macro")`
- **Purpose:** Compute multiclass metrics from per-class confusion matrices.
- **Args:** list of confusion matrices, metric name, averaging strategy ("macro", "micro", "weighted").
- **Returns:** aggregated metric score.

`get_probability(true_labs, pred_prob, objective='accuracy', verbose=False)`
- **Purpose:** Brute-force search for the threshold that maximizes accuracy or F\ :sub:`1` using scipy.optimize.brute.
- **Args:** true labels, predicted probabilities, objective ("accuracy" or "f1"), and verbosity flag.
- **Returns:** optimal threshold.

`get_optimal_threshold(true_labs, pred_prob, metric='f1', method='auto', sample_weight=None, comparison='>')`
- **Purpose:** Optimize any registered metric using different strategies. **Automatically detects binary vs multiclass inputs.**
- **Args:** true labels (binary or multiclass), probabilities (1D for binary, 2D for multiclass), metric name, optimization method, optional sample weights, and comparison operator.
- **Methods:**
  - `"auto"`: Automatically selects the best method based on metric characteristics (default)
  - `"sort_scan"`: Exact O(n log n) algorithm for piecewise-constant metrics in binary classification. Provides 50-100x speedup while guaranteeing global optimum
  - `"smart_brute"`: Evaluates all unique probabilities (fallback when sort_scan unavailable)
  - `"minimize"`: Uses scipy.optimize.minimize_scalar with fallback evaluation
  - `"gradient"`: Simple gradient ascent
- **Comparison:** `">"` (exclusive, default) or `">="` (inclusive) for handling tied probabilities. The sort-and-scan algorithm uses midpoint thresholds to make this robust either way.
- **Returns:** optimal threshold (float for binary, array for multiclass).

`get_optimal_multiclass_thresholds(true_labs, pred_prob, metric='f1', method='auto', average='macro', sample_weight=None, comparison='>')`
- **Purpose:** Find optimal per-class thresholds for multiclass classification.
- **Args:** true class labels, probability matrix (n_samples, n_classes), metric name, optimization method, averaging strategy, optional sample weights, and comparison operator.
- **Methods:** Same as `get_optimal_threshold` plus:
  - `"coord_ascent"`: Coordinate ascent for coupled multiclass optimization (single-label consistent). Iteratively optimizes per-class thresholds while maintaining single-label predictions via argmax(P - tau). Often improves macro-F1 on imbalanced datasets compared to independent One-vs-Rest optimization
- **Strategies:** "macro"/"weighted" (One-vs-Rest independent), "micro" (joint optimization), "coord_ascent" (coupled optimization)
- **Returns:** array of optimal thresholds, one per class.

`cv_threshold_optimization(true_labs, pred_prob, metric='f1', method='auto', cv=5, random_state=None)`
- **Purpose:** Estimate thresholds via cross-validation and report per-fold scores.
- **Args:** Same parameters as `get_optimal_threshold`, plus cross-validation folds and random state.
- **Returns:** arrays of thresholds and scores.

`nested_cv_threshold_optimization(true_labs, pred_prob, metric='f1', method='auto', inner_cv=5, outer_cv=5, random_state=None)`
- **Purpose:** Perform nested cross-validation for threshold estimation and
  unbiased performance evaluation.
- **Returns:** arrays of outer-fold thresholds and scores.

`ThresholdOptimizer(objective='accuracy', verbose=False, method='auto', comparison='>')`
- **Purpose:** High-level wrapper with ``fit``/``predict`` methods using scikit-learn style API. **Supports both binary and multiclass classification.**
- **Args:** objective metric name (e.g., "accuracy", "f1", "precision", "recall"), verbosity flag, optimization method, and comparison operator.
- **Methods:** All methods from `get_optimal_threshold` and `get_optimal_multiclass_thresholds`, including `"sort_scan"` for binary and `"coord_ascent"` for multiclass.
- **Comparison:** `">"` (exclusive) or `">="` (inclusive) for threshold comparison behavior.
- **Returns:** fitted instance with ``threshold_`` attribute (float for binary, array for multiclass). The ``predict`` method returns boolean predictions for binary, class indices for multiclass.

## Examples

- [Basic binary usage](examples/basic_usage.py)
- [Advanced binary usage with sklearn](examples/advanced_usage.ipynb)  
- [Multiclass classification](examples/multiclass_usage.py)
- [Cross-validation and gradient methods](examples/comscore.ipynb)

## Authors

Suriyan Laohaprapanon and Gaurav Sood
