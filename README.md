# Optimal Classification Cut-Offs

[![Python application](https://github.com/finite-sample/optimal_classification_cutoffs/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/optimal_classification_cutoffs/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/optimal_classification_cutoffs/)
[![PyPI version](https://img.shields.io/pypi/v/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)
[![PyPI Downloads](https://static.pepy.tech/badge/optimal-classification-cutoffs)](https://pepy.tech/projects/optimal-classification-cutoffs)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/optimal_classification_cutoffs/master/pyproject.toml&query=$.project.requires-python&label=Python)](https://github.com/finite-sample/optimal_classification_cutoffs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Select optimal probability thresholds for binary and multiclass classification.**  
Maximize F1, precision, recall, accuracy, or custom cost-sensitive metrics using efficient algorithms designed for piecewise-constant classification metrics.

---

## 🚀 Quick Start

### Installation
```bash
pip install optimal-classification-cutoffs
```

### Binary Classification
```python
from optimal_cutoffs import ThresholdOptimizer

# Your true labels and predicted probabilities
y_true = [0, 1, 1, 0, 1]
y_prob = [0.2, 0.8, 0.7, 0.3, 0.9]

# Find optimal threshold for F1 score
optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_true, y_prob)
print(f"Optimal threshold: {optimizer.threshold_:.3f}")

# Make predictions
y_pred = optimizer.predict(y_prob)
print(f"Predictions: {y_pred}")  # [0, 1, 1, 0, 1]
```

### Multiclass Classification
```python
import numpy as np

# Multiclass data: 3 classes, 5 samples
y_true = [0, 1, 2, 0, 1] 
y_prob = np.array([
    [0.7, 0.2, 0.1],  # Sample 0: most likely class 0
    [0.1, 0.8, 0.1],  # Sample 1: most likely class 1
    [0.1, 0.1, 0.8],  # Sample 2: most likely class 2
    [0.6, 0.3, 0.1],  # Sample 3: most likely class 0
    [0.2, 0.7, 0.1],  # Sample 4: most likely class 1
])

# Find optimal per-class thresholds
optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_true, y_prob)
print(f"Per-class thresholds: {optimizer.threshold_}")

# Make predictions
y_pred = optimizer.predict(y_prob)
print(f"Predicted classes: {y_pred}")
```

### Cost-Sensitive Optimization ✨ *New in v0.3.0*
```python
# Medical diagnosis: false negatives cost 10x more than false positives
threshold = get_optimal_threshold(
    y_true, y_prob, 
    utility={"tp": 50, "tn": 0, "fp": -1, "fn": -10}
)

# Or for calibrated probabilities (no training data needed)
from optimal_cutoffs import bayes_threshold_from_costs
optimal_threshold = bayes_threshold_from_costs(
    cost_fp=1.0,    # Cost of false positive
    cost_fn=10.0,   # Cost of false negative (10x worse)
    benefit_tp=50.0 # Benefit of catching true positive
)
```

---

## 🔧 Key Features

### ⚡ Optimized Algorithms for Piecewise Metrics
Classification metrics like F1, accuracy, precision, and recall are **piecewise-constant functions** that only change when thresholds cross unique probability values. Standard optimizers fail because these metrics have zero gradients everywhere.

**Our solution:** Specialized algorithms that guarantee global optima:

- **`sort_scan`**: O(n log n) exact algorithm, 50-100x faster than naive approaches
- **`coord_ascent`**: Advanced multiclass optimizer for coupled single-label predictions  
- **`auto`**: Intelligent method selection based on your data

![F1 Score Piecewise Behavior](docs/piecewise_f1_demo.png)

*F1 score only changes at unique probability values. Our algorithms find the true optimum.*

### 💰 Cost-Sensitive Optimization ✨ *New in v0.3.0*

Handle scenarios where different errors have different costs:

```python
# Financial fraud: missing fraud (FN) costs much more than false alarms (FP)
from optimal_cutoffs import make_cost_metric

cost_metric = make_cost_metric(
    cost_fp=1.0,     # False positive cost (false alarm)
    cost_fn=100.0,   # False negative cost (missed fraud)
    benefit_tp=500.0  # True positive benefit (caught fraud)
)

threshold = get_optimal_threshold(y_true, y_prob, metric=cost_metric)
```

**Bayes-Optimal Thresholds:** For calibrated probabilities, calculate optimal thresholds directly without training data:

```python
from optimal_cutoffs import bayes_threshold_from_utility

# Direct calculation for calibrated probabilities
threshold = bayes_threshold_from_utility(
    U_tp=50,  U_tn=0,   # Utilities for correct predictions
    U_fp=-1,  U_fn=-10  # Utilities for errors  
)
```

### 🎯 Multiclass Strategies

**One-vs-Rest (Default)**: Independent per-class thresholds
```python
thresholds = get_optimal_multiclass_thresholds(y_true, y_prob, method="auto")
```

**Coordinate Ascent**: Coupled optimization for single-label consistency
```python
# Better for imbalanced datasets
thresholds = get_optimal_multiclass_thresholds(y_true, y_prob, method="coord_ascent")
```

---

## 🤔 When to Use What?

### Threshold Optimization vs Calibration

| **Use Threshold Optimization When:** | **Use Calibration When:** |
|--------------------------------------|---------------------------|
| Maximizing classification metrics (F1, precision) | Need reliable probability estimates |
| Making binary decisions for deployment | Comparing model confidence |
| Handling class imbalance | Converting scores to probabilities |

**Best Practice: Use Both Together**
```python
from sklearn.calibration import CalibratedClassifierCV

# 1. Calibrate probabilities first
calibrated_model = CalibratedClassifierCV(base_model)
y_prob = calibrated_model.predict_proba(X_val)[:, 1]

# 2. Optimize threshold on calibrated probabilities  
optimizer = ThresholdOptimizer(objective="f1")
optimizer.fit(y_val, y_prob)

# Result: Reliable probabilities AND optimal decisions
```

### Cost-Sensitive vs Metric Optimization

| **Use Cost-Sensitive When:** | **Use Metric Optimization When:** |
|-------------------------------|-----------------------------------|
| Different errors have different costs | All errors are equally bad |
| Business impact varies by error type | Optimizing standard metrics (F1, accuracy) |
| Medical, financial, safety applications | General ML model evaluation |

### Method Selection Guide

| **Method** | **Best For** | **Speed** | **Guarantees** |
|------------|--------------|-----------|----------------|
| `"auto"` | Most cases | Fast | Selects best method automatically |
| `"sort_scan"` | Binary piecewise metrics | Very Fast | Exact global optimum |
| `"coord_ascent"` | Multiclass, imbalanced data | Medium | Local optimum, single-label consistent |
| `"minimize"` | Custom smooth metrics | Medium | Local optimum |

---

## 📖 API Reference

### Core Functions

#### `ThresholdOptimizer(objective="f1", method="auto")`
**Scikit-learn style threshold optimization**
```python
optimizer = ThresholdOptimizer(objective="f1", method="auto")
optimizer.fit(y_true, y_prob)
y_pred = optimizer.predict(y_prob)
```
- **Auto-detects** binary vs multiclass inputs
- **Methods**: `"auto"`, `"sort_scan"`, `"coord_ascent"`, `"minimize"`, `"gradient"`
- **Returns**: Fitted object with `threshold_` attribute

#### `get_optimal_threshold(y_true, y_prob, metric="f1", method="auto", **kwargs)`
**Functional interface for threshold optimization**
```python
threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
```
- **New in v0.3.0**: `utility`, `minimize_cost`, and `bayes` parameters
- **Returns**: Optimal threshold (float for binary, array for multiclass)

### Cost-Sensitive Functions ✨ *New in v0.3.0*

#### `bayes_threshold_from_utility(U_tp, U_tn, U_fp, U_fn)`
**Calculate Bayes-optimal threshold for calibrated probabilities**
```python
threshold = bayes_threshold_from_utility(U_tp=1, U_tn=0, U_fp=-1, U_fn=-5)
```

#### `bayes_threshold_from_costs(cost_fp, cost_fn, benefit_tp=0, benefit_tn=0)`
**Convenience wrapper for cost-based optimization**
```python
threshold = bayes_threshold_from_costs(cost_fp=1, cost_fn=10, benefit_tp=50)
```

#### `make_cost_metric(cost_fp, cost_fn, benefit_tp=0, benefit_tn=0)`
**Create custom cost-sensitive metrics**
```python
custom_metric = make_cost_metric(cost_fp=1, cost_fn=5, benefit_tp=10)
threshold = get_optimal_threshold(y_true, y_prob, metric=custom_metric)
```

#### `make_linear_counts_metric(w_tp=0, w_tn=0, w_fp=0, w_fn=0)`
**Create metrics from confusion matrix weights**
```python
profit_metric = make_linear_counts_metric(w_tp=100, w_fp=-10, w_fn=-50)
```

### Multiclass Functions

#### `get_optimal_multiclass_thresholds(y_true, y_prob, metric="f1", method="auto")`
**Multiclass threshold optimization**
```python
thresholds = get_optimal_multiclass_thresholds(y_true, y_prob, method="coord_ascent")
```

### Utility Functions

<details>
<summary>Click to expand utility functions</summary>

#### `get_confusion_matrix(y_true, y_prob, threshold)`
```python
tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, 0.5)
```

#### `get_multiclass_confusion_matrix(y_true, y_prob, thresholds)`  
```python
cms = get_multiclass_confusion_matrix(y_true, y_prob, [0.3, 0.5, 0.7])
```

#### `register_metric(name, func)` and `register_metrics(metrics_dict)`
```python
@register_metric("custom_f2")
def f2_score(tp, tn, fp, fn):
    return (5 * tp) / (5 * tp + 4 * fn + fp)
```

</details>

---

## 📊 Examples

### Basic Examples
- [Binary classification](examples/basic_usage.py) - Getting started
- [Multiclass classification](examples/multiclass_usage.py) - Multi-class optimization

### Advanced Examples  
- [Cost-sensitive medical diagnosis](examples/cost_sensitive_medical.py) ✨ *New*
- [Financial fraud detection](examples/cost_sensitive_finance.py) ✨ *New*
- [Cross-validation workflows](examples/comscore.ipynb) - Robust evaluation
- [Integration with sklearn](examples/advanced_usage.ipynb) - Production pipelines

---

## 🧮 Theory & Background

**Why do standard optimizers fail?** Classification metrics are piecewise-constant functions with zero gradients everywhere except at breakpoints. Traditional optimizers get trapped in flat regions and miss the global optimum.

**Our innovation:** Exact algorithms that leverage the mathematical structure of classification metrics. The sort-and-scan method achieves O(n log n) complexity while guaranteeing global optimality for piecewise metrics.

**For detailed mathematical explanations** and interactive visualizations, see our [comprehensive documentation](https://finite-sample.github.io/optimal_classification_cutoffs/).

---

## 🔬 Advanced Methods

**Coordinate Ascent for Multiclass:** Unlike One-vs-Rest approaches, our coordinate ascent method maintains single-label consistency by coupling classes through `argmax(P - τ)` decision rules. This often improves macro-F1 on imbalanced datasets.

**Dinkelbach Fractional Programming:** For expected F-beta optimization under calibrated probabilities, the Dinkelbach method provides ultra-fast exact solutions using the F1 threshold identity. Future release planned.

---

## 👨‍💻 Authors

Suriyan Laohaprapanon and Gaurav Sood