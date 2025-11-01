# Optimal Classification Thresholds

[![Python application](https://github.com/finite-sample/optimal-classification-cutoffs/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/optimal-classification-cutoffs/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/optimal-classification-cutoffs/)
[![PyPI version](https://img.shields.io/pypi/v/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)

**Exact, fast threshold optimization under calibrated probabilities.**

This library finds **optimal decision thresholds** by exploiting the **piecewise structure** of common metrics and assuming **calibrated probabilities**. It supports binary, multi-label, and multi-class problems; cost/utility objectives; and general cost matrices.

---

## Contents

* [Why thresholds matter](#why-thresholds-matter)
* [Installation](#installation)
* [90‑second tour](#90-second-tour)
* [Quick start](#quick-start)
* [Decision rules — cheat sheet](#decision-rules--cheat-sheet)
* [Problem types & functions](#problem-types--functions)
* [Expected metrics (under calibration)](#expected-metrics-under-calibration)
* [Validation & recommended workflow](#validation--recommended-workflow)
* [Determinism (ties) & assumptions](#determinism-ties--assumptions)
* [Performance](#performance)
* [When *not* to use thresholds](#when-not-to-use-thresholds)

---

## Why thresholds matter

Most classifiers output probabilities `p = P(y=1|x)`, but decisions need thresholds τ. The default `τ = 0.5` is rarely optimal for real objectives (F1, precision/recall, costs). Under calibration and piecewise structure, the **exact** optimum can be found efficiently.

---

## Installation

```bash
pip install optimal-classification-cutoffs

# With performance extras (Numba)
pip install optimal-classification-cutoffs[performance]
```

---

## 90‑second tour

* **Exact piecewise search:** For metrics like F1/precision/recall/accuracy, the objective is **piecewise-constant** in τ. Sorting scores once gives **O(n log n)** exact optimization.
* **Bayes utilities:** With outcome utilities `(u_tp,u_tn,u_fp,u_fn)`, the optimal binary threshold has a **closed form**.
* **Multiclass:**

  * **OvR metrics:** optimize per-class thresholds (macro/micro) or use a **weighted margin** rule under per-class costs.
  * **General cost matrices:** skip thresholds and apply Bayes rule directly on the probability vector.
* **Expected metrics:** Under perfect calibration, optimize expected Fβ without labels.
* **Cross-validation:** Thresholds are hyperparameters—validate them.
* **Acceleration:** Numba-backed kernels with pure Python fallback.

---

## Quick start

### Binary: optimize F1

```python
from optimal_cutoffs import optimize_f1_binary

y_true = [0, 1, 1, 0, 1]
y_prob = [0.2, 0.8, 0.7, 0.3, 0.9]

result = optimize_f1_binary(y_true, y_prob)
print(f"Optimal threshold: {result.threshold:.3f}")
print(f"F1: {result.score:.3f}")

y_pred = result.predict(y_prob)
```

### Binary: cost-sensitive (closed form)

```python
from optimal_cutoffs import optimize_utility_binary

# False negatives 10× costlier than false positives
utility = {"tp": 0.0, "tn": 0.0, "fp": -1.0, "fn": -10.0}
result = optimize_utility_binary(y_true, y_prob, utility=utility)
print(f"Optimal τ ≈ {result.threshold:.3f}")  # ≈ 0.091 (= 1/(1+10))
```

### Multiclass: macro-F1 with independent OvR thresholds

```python
import numpy as np
from optimal_cutoffs import optimize_ovr_independent

y_true = [0, 1, 2, 0, 1]
y_prob = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
])

result = optimize_ovr_independent(y_true, y_prob, metric="f1")
print(result.thresholds)
y_pred = result.predict(y_prob)
```

### Multiclass: general cost matrix (no thresholds)

```python
import numpy as np
from optimal_cutoffs import bayes_optimal_decisions

cost_matrix = np.array([
    [0,  10,  50],
    [10,  0,  40],
    [100, 90,  0],
])

result = bayes_optimal_decisions(y_prob, cost_matrix=cost_matrix)
y_pred = result.predict(y_prob)
```

---

## Decision rules — cheat sheet

### Binary Bayes with utilities

```
τ* = (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)]
```

* **Costs only:** set `u_fp = -c_fp`, `u_fn = -c_fn`, `u_tp = u_tn = 0` ➞ `τ* = c_fp / (c_fp + c_fn)`.
* Independent of class priors.

### Multiclass OvR costs (must pick exactly one class)

```
τ_j = c_j / (c_j + r_j)
ŷ = argmax_j [ (c_j + r_j) * (p_j - τ_j) ]
```

### General multiclass costs

```
ŷ = argmin_j ∑_i p_i * C[i, j]
```

---

## Problem types & functions

| Problem        | Objective                       | Thresholds Coupled? | Function                        |        Complexity | Optimality    |
| -------------- | ------------------------------- | ------------------: | ------------------------------- | ----------------: | ------------- |
| Binary         | Utility (cost/benefit)          |                   — | `optimize_utility_binary()`     |              O(1) | Bayes-optimal |
| Binary         | F-measures / precision / recall |                   — | `optimize_f1_binary()`          |        O(n log n) | Exact         |
| Multi‑label    | Macro‑F1                        |                  No | `optimize_macro_multilabel()`   |      O(K·n log n) | Exact         |
| Multi‑label    | Micro‑F1                        |                 Yes | `optimize_micro_multilabel()`   | O(iter·K·n log n) | Local optimum |
| Multiclass OvR | Macro‑F1 (indep.)               |                  No | `optimize_ovr_independent()`    |      O(K·n log n) | Exact         |
| Multiclass OvR | OvR costs (single‑label)        |                  No | `bayes_thresholds_from_costs()` |              O(1) | Bayes-optimal |
| Multiclass     | General cost matrix             |                   — | `bayes_optimal_decisions()`     |  O(K²) per sample | Bayes-optimal |

---

## Expected metrics (under calibration)

Under perfect calibration, **expected Fβ** can be optimized **without labels**.

```python
from optimal_cutoffs import dinkelbach_expected_fbeta_binary

result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
print(result.threshold, result.score)
```

---

## Validation & recommended workflow

Thresholds are **hyperparameters**. Validate them on held‑out data.

```python
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from optimal_cutoffs import optimize_f1_binary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

clf = SomeClassifier().fit(X_train, y_train)
clf_cal = CalibratedClassifierCV(clf, cv='prefit').fit(X_val, y_val)

p_val = clf_cal.predict_proba(X_val)[:, 1]
tau = optimize_f1_binary(y_val, p_val).threshold
p_test = clf_cal.predict_proba(X_test)[:, 1]
y_hat = (p_test >= tau).astype(int)
```

---

## Determinism (ties) & assumptions

* **Ties:** fix `comparison` (">" vs ">=") across train/val/test.
* **Calibration:** ensure `E[y|p]=p` (binary) or `E[1{y=j}|p]=p_j` (multiclass).
* **Prior shift:** cost-based thresholds are prior‑invariant; F‑metric thresholds are not.

---

## Performance

* **Numba acceleration:** 10–100× speedups.
* **Vectorized scans:** O(n log n) dominated by sort.
* **Pure Python fallback:** supported.

---

## When *not* to use thresholds

* Uncalibrated probabilities ➞ calibrate first.
* General cost matrices ➞ use `bayes_optimal_decisions()`.
* Need probabilistic outputs ➞ keep probabilities.

---

## References

* Lipton et al. (2014) *Optimal Thresholding of Classifiers to Maximize F1*
* Dinkelbach (1967) *Nonlinear fractional programming*
* Elkan (2001) *The Foundations of Cost-Sensitive Learning*
* Platt (1999), Zadrozny & Elkan (2002) *Calibration papers*
