Cross-Validation Utilities
==========================

The cv module provides functions for robust threshold estimation using cross-validation techniques.

Cross-Validation Functions
--------------------------

.. autofunction:: optimal_cutoffs.cv.cv_threshold_optimization

.. autofunction:: optimal_cutoffs.cv.nested_cv_threshold_optimization

Usage Examples
--------------

Basic Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import cv_threshold_optimization
   import numpy as np
   
   # Your data
   y_true = np.random.randint(0, 2, 1000)
   y_prob = np.random.uniform(0, 1, 1000)
   
   # 5-fold cross-validation
   thresholds, scores = cv_threshold_optimization(
       y_true, y_prob,
       metric='f1',
       cv=5,
       method='auto'
   )
   
   print(f"CV thresholds: {thresholds}")
   print(f"CV scores: {scores}")
   print(f"Mean threshold: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")

Stratified Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold
   
   # Use stratified splits for imbalanced data
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
   thresholds, scores = cv_threshold_optimization(
       y_true, y_prob,
       metric='f1',
       cv=cv,  # Pass custom CV splitter
       method='auto'
   )

Nested Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import nested_cv_threshold_optimization
   
   # Nested CV for unbiased performance estimation
   outer_scores, inner_results = nested_cv_threshold_optimization(
       y_true, y_prob,
       metric='f1',
       outer_cv=5,
       inner_cv=3,
       method='auto'
   )
   
   print(f"Outer CV scores: {outer_scores}")
   print(f"Mean performance: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")

Custom Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   
   # Time series cross-validation
   tscv = TimeSeriesSplit(n_splits=5)
   
   thresholds, scores = cv_threshold_optimization(
       y_true, y_prob,
       metric='precision',
       cv=tscv,
       method='smart_brute'
   )

With Sample Weights
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sample weights for imbalanced data
   sample_weights = np.where(y_true == 1, 0.5, 2.0)  # Upweight minority class
   
   thresholds, scores = cv_threshold_optimization(
       y_true, y_prob,
       metric='f1',
       cv=5,
       sample_weight=sample_weights
   )

Multiclass Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multiclass data
   y_true_mc = np.random.randint(0, 3, 1000)
   y_prob_mc = np.random.dirichlet([1, 1, 1], 1000)  # 3 classes
   
   # Returns list of threshold arrays (one per fold)
   thresholds_list, scores = cv_threshold_optimization(
       y_true_mc, y_prob_mc,
       metric='f1',
       cv=5,
       average='macro'  # Macro-averaged F1
   )
   
   # Average thresholds across folds
   mean_thresholds = np.mean(thresholds_list, axis=0)
   print(f"Mean per-class thresholds: {mean_thresholds}")

Best Practices
--------------

Choosing CV Strategy
~~~~~~~~~~~~~~~~~~~

* **Balanced data**: Use standard ``KFold`` or ``cv=5``
* **Imbalanced data**: Use ``StratifiedKFold`` to preserve class ratios
* **Time series**: Use ``TimeSeriesSplit`` to respect temporal order
* **Small datasets**: Use ``LeaveOneOut`` or higher k in k-fold

Threshold Aggregation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multiple strategies for combining CV thresholds
   thresholds, scores = cv_threshold_optimization(y_true, y_prob, metric='f1', cv=10)
   
   # Different aggregation methods
   mean_threshold = np.mean(thresholds)
   median_threshold = np.median(thresholds)
   
   # Weighted by CV scores
   weights = scores / np.sum(scores)
   weighted_threshold = np.average(thresholds, weights=weights)
   
   # Choose best single fold
   best_idx = np.argmax(scores)
   best_threshold = thresholds[best_idx]

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bootstrap confidence intervals
   from scipy import stats
   
   thresholds, scores = cv_threshold_optimization(y_true, y_prob, metric='f1', cv=10)
   
   # 95% confidence interval for threshold
   threshold_mean = np.mean(thresholds)
   threshold_se = stats.sem(thresholds)
   ci_lower, ci_upper = stats.t.interval(0.95, len(thresholds)-1, 
                                        loc=threshold_mean, scale=threshold_se)
   
   print(f"Threshold: {threshold_mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")