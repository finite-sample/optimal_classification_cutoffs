## Optimal Cut-Offs

Probabilities from classification models can have two problems:

1. Miscalibration: A p of .9 often doesn't mean a 90% chance of 1 (assuming a dichotomous y). (You can calibrate it using isotonic regression.)
2. Optimal cut-offs: For multi-class classifiers, we do not know what probability value will maximize the accuracy or F1 score. Or any metric for which you need to trade-off between FP and FN.

Here we share a solution for #2. It involves running the outputs through search routines that find the probability threshold that optimizes a chosen metric. A simple wrapper makes it easy to use.

### Package

The ``optimal_cutoffs`` package contains:

* ``metrics`` – confusion-matrix utilities and metric functions.
* ``optimizers`` – several threshold-search algorithms (brute force, smart brute, gradient, etc.).
* ``cv`` – helpers for cross-validation and nested cross-validation.
* ``wrapper`` – a ``ThresholdOptimizer`` class for high-level use.

### Usage

```python
from optimal_cutoffs import ThresholdOptimizer

# y_true are true labels, y_prob are predicted probabilities
opt = ThresholdOptimizer(metric="accuracy", method="brute")
opt.fit(y_true, y_prob)
labels = opt.predict(y_prob)
```

### Illustration

Check out the [Jupyter notebook](examples/comscore.ipynb) to see the package in action.

### Authors

Suriyan Laohaprapanon and Gaurav Sood
