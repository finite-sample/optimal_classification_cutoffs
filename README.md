# Optimal Classification Cut-Offs

[![Python application](https://github.com/finite-sample/optimal_classification_cutoffs/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/optimal_classification_cutoffs/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/optimal_classification_cutoffs/)
[![PyPI version](https://img.shields.io/pypi/v/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)
[![PyPI Downloads](https://static.pepy.tech/badge/optimal-classification-cutoffs)](https://pepy.tech/projects/optimal-classification-cutoffs)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/optimal_classification_cutoffs/master/pyproject.toml&query=$.project.requires-python&label=Python)](https://github.com/finite-sample/optimal_classification_cutoffs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Select optimal probability thresholds for binary and multiclass classification.**  
Maximize F1, precision, recall, accuracy, or custom cost-sensitive utilities with algorithms designed for **piecewise‑constant** classification metrics.

---

## Why thresholds—and what are we optimizing?

Most probabilistic classifiers output **scores or probabilities** `p = P(y=1|x)` (binary) or a **probability vector** over classes (multiclass). Turning those into decisions requires **thresholds**:

- **Binary:** predict 1 if `p > τ`, else 0.  
- **Multiclass:** predict class `argmax_k p_k` or use **per‑class thresholds** `τ_k`.

The default `τ = 0.5` is rarely optimal for your objective (e.g., F1 under imbalance, cost asymmetry, etc.). Because metrics like F1/precision/recall/accuracy **only change when thresholds cross unique probability values**, they are *piecewise‑constant*. That structure lets us compute **globally optimal thresholds** quickly and exactly.

---

## Methods at a glance (from basic → advanced)

**Intuition:** we want the cut(s) over sorted probabilities that maximize your objective.

- **Smart brute (unique cuts)** — *baseline / safe*: evaluate the metric at **all unique predicted probabilities** and pick the best. Competitive when `n_unique` is moderate.  
  Method: `"smart_brute"`.

- **Sort & scan (exact, fast)** — *recommended for piecewise metrics*: sort probabilities once and compute all candidate scores with **vectorized cumulative counts**. **O(n log n)**, exact optimum for F1/precision/recall/accuracy.  
  Method: `"sort_scan"`.

- **Dinkelbach (expected Fβ; calibrated)** — *analytical, fastest when valid*: solves a **fractional program** for **expected** Fβ under **perfect calibration**. Currently supports F1. Use when you trust calibration and want the expected‑metric optimum.  
  Method: `"dinkelbach"`.

- **Continuous optimizers** — *for non‑piecewise targets or micro‑averaged multiclass joint objectives*: fallback to `scipy.optimize` or simple gradient heuristics. Not guaranteed optimal for stepwise metrics.  
  Methods: `"minimize"`, `"gradient"`.

**Multiclass strategies:**

- **One‑vs‑Rest (OvR)** — optimize each class’s threshold independently (macro/weighted/none averaging). Simple and effective; by default we predict the **highest‑probability class above its threshold**, falling back to `argmax` if none pass.  
  Method: `"auto"`, `"smart_brute"`, `"sort_scan"`, `"minimize"`, `"gradient"`.

- **Coordinate Ascent (coupled, single‑label consistent)** — optimizes F1 for the **single‑label** rule `argmax_k (p_k − τ_k)`. Typically better for **imbalanced** problems; currently F1 only, comparison `">"` only, and no sample weights.  
  Method: `"coord_ascent"`.

---

## Practical validation: holdout & cross‑validation

Thresholds are **hyperparameters**. To estimate a threshold you can trust:

1. **Split**: Train your model; reserve **validation** data (or use **cross‑validation**) to choose `τ`.  
2. **(Optional) Calibrate** probabilities (`CalibratedClassifierCV`) for better transportability.  
3. **Select** thresholds on validation/CV using this library.  
4. **Freeze** the threshold and **evaluate** on a held‑out test set.

This repository includes **cross‑validation** utilities to estimate thresholds and quantify uncertainty.

---

## 🚀 Quick start

### Install
```bash
pip install optimal-classification-cutoffs
```

### Binary

```python
from optimal_cutoffs import ThresholdOptimizer

y_true = [0, 1, 1, 0, 1]
y_prob = [0.2, 0.8, 0.7, 0.3, 0.9]

# Optimize F1 threshold
opt = ThresholdOptimizer(objective="f1", method="auto")
opt.fit(y_true, y_prob)
print(opt.threshold_)            # e.g. 0.7...
y_pred = opt.predict(y_prob)     # boolean labels
```

### Multiclass (OvR thresholds)

```python
import numpy as np
from optimal_cutoffs import ThresholdOptimizer

y_true = [0, 1, 2, 0, 1]
y_prob = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
])

opt = ThresholdOptimizer(objective="f1")   # auto-detects multiclass
opt.fit(y_true, y_prob)
print(opt.threshold_)                      # per-class τ_k
y_pred = opt.predict(y_prob)               # integer class labels
```

### Cost-Sensitive Binary

```python
from optimal_cutoffs import get_optimal_threshold, bayes_threshold_from_costs

# Empirical (finite-sample) optimum from labeled data
tau = get_optimal_threshold(
    y_true, y_prob,
    utility={"tp": 50.0, "tn": 0.0, "fp": -1.0, "fn": -10.0},  # benefits/costs
)

# Closed-form Bayes threshold (calibrated probabilities)
tau_bayes = bayes_threshold_from_costs(
    fp_cost=1.0, fn_cost=10.0, tp_benefit=50.0, tn_benefit=0.0
)
```

## API Decision Stack

1. Problem: binary or multiclass (auto‑detected).

2. Objective: metric ("f1", "precision", "recall", "accuracy") or utility/cost (binary‑only).

3. Estimation regime (choose one):
    • Empirical (finite sample) — optimize on labeled data.
    • Expected under calibration —
      – Bayes (utility, closed‑form; binary‑only), or
      – Dinkelbach (expected F1; no weights).

4. Method (empirical only): "auto", "sort_scan", "smart_brute", "minimize", "gradient"; multiclass adds "coord_ascent". For expected F1, use "dinkelbach".

5. Validation: holdout or cross‑validation (cv_threshold_optimization, nested_cv_threshold_optimization).

## Examples

* Empirical metric (binary):

```
get_optimal_threshold(y, p, metric="f1", method="auto")
```

* Empirical utility (binary):
```
get_optimal_threshold(y, p, utility={"fp":-1, "fn":-5}, method="sort_scan")
```

* Bayes utility (calibrated, binary):
```
bayes_threshold_from_costs(fp_cost=1, fn_cost=5) # or
get_optimal_threshold(None, p, utility={"fp":-1,"fn":-5}, bayes=True)
```

* Expected F1 via Dinkelbach (calibrated, binary):

```
get_optimal_threshold(y, p, metric="f1", method="dinkelbach")
```
