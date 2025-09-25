Advanced Topics
===============

This section covers advanced features and techniques for sophisticated use cases.

Cost-Aware Threshold Optimization
---------------------------------

Beyond Standard Metrics
~~~~~~~~~~~~~~~~~~~~~~~

While metrics like F1 score are useful, real-world applications often have specific costs associated with different types of errors:

.. code-block:: python

   from optimal_cutoffs import get_optimal_threshold
   import numpy as np
   
   # Medical screening example
   y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
   y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.6, 0.4])
   
   # Define utility matrix (benefits positive, costs negative)
   medical_costs = {
       "tp": 1000,    # Benefit: Early detection saves $1000 in treatment
       "tn": 0,       # No additional cost for correct negative
       "fp": -200,    # Cost: Unnecessary procedure costs $200
       "fn": -5000    # Cost: Missed diagnosis costs $5000 in delayed treatment
   }
   
   # Optimize for expected utility
   threshold_cost = get_optimal_threshold(
       y_true, y_prob, 
       utility=medical_costs
   )
   
   print(f"Cost-optimized threshold: {threshold_cost:.3f}")

Bayes-Optimal Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~

For well-calibrated probabilities, you can calculate the theoretical optimum without training data:

.. code-block:: python

   from optimal_cutoffs import bayes_threshold_from_utility, bayes_threshold_from_costs
   
   # Method 1: Direct utility specification
   threshold_bayes = bayes_threshold_from_utility(
       U_tp=1000, U_tn=0, U_fp=-200, U_fn=-5000
   )
   
   # Method 2: Cost/benefit specification (cleaner interface)
   threshold_bayes2 = bayes_threshold_from_costs(
       fp_cost=200, fn_cost=5000, tp_benefit=1000
   )
   
   print(f"Bayes optimal threshold: {threshold_bayes:.3f}")
   print(f"Equivalent calculation: {threshold_bayes2:.3f}")
   
   # Use Bayes threshold directly (no training labels needed)
   threshold_empirical = get_optimal_threshold(
       None, y_prob,  # None for true labels
       utility=medical_costs,
       bayes=True
   )

The Bayes-optimal threshold is: ``t* = (U_tn - U_fp) / [(U_tn - U_fp) + (U_tp - U_fn)]``

Probability Calibration
----------------------

Threshold optimization assumes well-calibrated probabilities. If your model's probabilities are poorly calibrated, consider calibration first:

.. code-block:: python

   from sklearn.calibration import CalibratedClassifierCV
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_classification
   
   # Generate data
   X, y = make_classification(n_samples=2000, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
   # Train base classifier
   base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
   
   # Apply calibration
   calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
   calibrated_clf.fit(X_train, y_train)
   
   # Compare calibrated vs uncalibrated probabilities
   base_clf.fit(X_train, y_train)
   
   y_prob_uncalibrated = base_clf.predict_proba(X_train)[:, 1]
   y_prob_calibrated = calibrated_clf.predict_proba(X_train)[:, 1]
   
   # Optimize thresholds for both
   threshold_uncal = get_optimal_threshold(y_train, y_prob_uncalibrated, metric='f1')
   threshold_cal = get_optimal_threshold(y_train, y_prob_calibrated, metric='f1')
   
   print(f"Uncalibrated threshold: {threshold_uncal:.3f}")
   print(f"Calibrated threshold: {threshold_cal:.3f}")

Advanced Multiclass Strategies
------------------------------

Coordinate Ascent Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multiclass problems requiring single-label consistency (exactly one prediction per sample):

.. code-block:: python

   # Standard One-vs-Rest (default)
   thresholds_ovr = get_optimal_threshold(y_true, y_prob, metric='f1', method='auto')
   
   # Coordinate ascent for single-label consistency
   thresholds_coord = get_optimal_threshold(y_true, y_prob, metric='f1', method='coord_ascent')
   
   print(f"OvR thresholds: {thresholds_ovr}")
   print(f"Coordinate ascent thresholds: {thresholds_coord}")

The coordinate ascent method couples classes through joint assignment using ``argmax(p_j - tau_j)``, ensuring exactly one prediction per sample.

Custom Multiclass Costs
~~~~~~~~~~~~~~~~~~~~~~~

For different costs across classes (planned feature):

.. code-block:: python

   # Current workaround: Optimize each class separately
   def optimize_multiclass_with_costs(y_true, y_prob, class_costs):
       """Optimize multiclass thresholds with different costs per class."""
       n_classes = y_prob.shape[1]
       thresholds = []
       
       for class_idx in range(n_classes):
           # Convert to binary problem
           y_binary = (y_true == class_idx).astype(int)
           y_prob_binary = y_prob[:, class_idx]
           
           # Get costs for this class
           costs = class_costs.get(class_idx, {"fp": -1.0, "fn": -1.0})
           
           threshold = get_optimal_threshold(
               y_binary, y_prob_binary,
               utility=costs
           )
           thresholds.append(threshold)
       
       return np.array(thresholds)
   
   # Example usage
   class_specific_costs = {
       0: {"fp": -1.0, "fn": -2.0},    # Class 0: FN twice as costly
       1: {"fp": -5.0, "fn": -1.0},    # Class 1: FP five times as costly
       2: {"fp": -1.0, "fn": -10.0},   # Class 2: FN ten times as costly
   }
   
   # custom_thresholds = optimize_multiclass_with_costs(y_true, y_prob, class_specific_costs)

Performance Optimization
------------------------

Algorithm Selection
~~~~~~~~~~~~~~~~~~~

Understanding when to use each optimization method:

.. code-block:: python

   import time
   
   def benchmark_methods(y_true, y_prob, methods=['auto', 'sort_scan', 'smart_brute', 'minimize']):
       """Benchmark different optimization methods."""
       results = {}
       
       for method in methods:
           start_time = time.time()
           try:
               threshold = get_optimal_threshold(y_true, y_prob, metric='f1', method=method)
               elapsed = time.time() - start_time
               results[method] = {'threshold': threshold, 'time': elapsed}
           except Exception as e:
               results[method] = {'threshold': None, 'time': None, 'error': str(e)}
       
       return results
   
   # Test with different data sizes
   for size in [1000, 5000, 10000]:
       y = np.random.randint(0, 2, size)
       y_prob = np.random.uniform(0, 1, size)
       
       results = benchmark_methods(y, y_prob)
       print(f"\nDataset size: {size}")
       for method, result in results.items():
           if result.get('time'):
               print(f"{method:>12}: {result['time']:.4f}s, threshold={result['threshold']:.3f}")
           else:
               print(f"{method:>12}: FAILED ({result.get('error', 'Unknown error')})")

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large datasets:

.. code-block:: python

   def chunked_optimization(y_true, y_prob, chunk_size=10000, method='auto'):
       """Process large datasets in chunks."""
       n_samples = len(y_true)
       chunk_thresholds = []
       
       for start_idx in range(0, n_samples, chunk_size):
           end_idx = min(start_idx + chunk_size, n_samples)
           
           chunk_true = y_true[start_idx:end_idx]
           chunk_prob = y_prob[start_idx:end_idx]
           
           threshold = get_optimal_threshold(
               chunk_true, chunk_prob,
               metric='f1',
               method=method
           )
           chunk_thresholds.append(threshold)
       
       # Combine results (various strategies possible)
       final_threshold = np.median(chunk_thresholds)  # Robust to outliers
       return final_threshold, chunk_thresholds
   
   # Example with large synthetic dataset
   large_y = np.random.randint(0, 2, 100000)
   large_prob = np.random.uniform(0, 1, 100000)
   
   threshold, chunk_results = chunked_optimization(large_y, large_prob)
   print(f"Final threshold: {threshold:.3f}")
   print(f"Chunk variation: {np.std(chunk_results):.3f}")

Custom Metrics and Vectorization
--------------------------------

Advanced Metric Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs.metrics import register_metric
   import numpy as np
   
   # Define a complex custom metric
   def balanced_accuracy(tp, tn, fp, fn):
       """Balanced accuracy: average of sensitivity and specificity."""
       sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
       specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
       return (sensitivity + specificity) / 2
   
   # Vectorized version for O(n log n) optimization
   def balanced_accuracy_vectorized(tp, tn, fp, fn):
       """Vectorized balanced accuracy computation."""
       sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), 
                             where=(tp + fn) > 0)
       specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float),
                             where=(tn + fp) > 0)
       return (sensitivity + specificity) / 2
   
   # Register with vectorized version
   register_metric(
       'balanced_accuracy',
       balanced_accuracy,
       vectorized_func=balanced_accuracy_vectorized,
       is_piecewise=True,
       maximize=True,
       needs_proba=False
   )
   
   # Use the custom metric
   threshold = get_optimal_threshold(y_true, y_prob, metric='balanced_accuracy')
   print(f"Balanced accuracy optimized threshold: {threshold:.3f}")

Non-Piecewise Metrics
~~~~~~~~~~~~~~~~~~~~

For metrics that are not piecewise-constant:

.. code-block:: python

   def log_loss_metric(tp, tn, fp, fn):
       """Not piecewise - depends on actual probability values."""
       # This is just an example - real log loss needs probabilities
       return -(tp * np.log(0.9) + tn * np.log(0.9) + fp * np.log(0.1) + fn * np.log(0.1))
   
   register_metric(
       'log_loss_approx',
       log_loss_metric,
       is_piecewise=False,  # Not piecewise-constant
       maximize=False,      # Minimize log loss
       needs_proba=True     # Requires probability values
   )
   
   # This will use continuous optimization methods
   threshold = get_optimal_threshold(y_true, y_prob, metric='log_loss_approx', method='minimize')

Nested Cross-Validation
-----------------------

For unbiased performance estimation:

.. code-block:: python

   from optimal_cutoffs import nested_cv_threshold_optimization
   
   # Nested cross-validation for unbiased performance estimation
   outer_scores, inner_results = nested_cv_threshold_optimization(
       y_true, y_prob,
       metric='f1',
       outer_cv=5,
       inner_cv=3,
       method='auto'
   )
   
   print(f"Outer CV scores: {outer_scores}")
   print(f"Unbiased performance estimate: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
   
   # Analyze inner fold variation
   for fold_idx, inner_result in enumerate(inner_results):
       thresholds = inner_result['thresholds']
       scores = inner_result['scores']
       print(f"Outer fold {fold_idx}: threshold={np.mean(thresholds):.3f}, score={np.mean(scores):.3f}")

Production Deployment Considerations
-----------------------------------

Model Serialization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import joblib
   from optimal_cutoffs import ThresholdOptimizer
   
   # Train and optimize
   optimizer = ThresholdOptimizer(metric='f1', method='auto')
   optimizer.fit(y_train, y_prob_train)
   
   # Save complete model
   model_package = {
       'base_classifier': trained_classifier,  # Your trained classifier
       'threshold_optimizer': optimizer,
       'threshold': optimizer.threshold_,
       'training_metrics': {
           'f1_score': optimizer.score_,
           'threshold_std': 0.05  # From CV if available
       },
       'calibration_method': 'isotonic',  # If calibration was used
       'version': '1.0'
   }
   
   joblib.dump(model_package, 'production_model.pkl')
   
   # Load and use in production
   loaded_model = joblib.load('production_model.pkl')
   base_clf = loaded_model['base_classifier']
   optimizer = loaded_model['threshold_optimizer']
   
   # Make predictions
   def predict_production(X_new):
       y_prob = base_clf.predict_proba(X_new)[:, 1]
       y_pred = optimizer.predict(y_prob)
       return y_pred, y_prob

Monitoring and Drift Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def monitor_threshold_performance(y_true, y_prob, reference_threshold, 
                                   reference_score, tolerance=0.05):
       """Monitor if optimal threshold has drifted significantly."""
       from optimal_cutoffs.metrics import get_confusion_matrix, f1_score
       
       # Calculate current optimal threshold
       current_threshold = get_optimal_threshold(y_true, y_prob, metric='f1')
       
       # Calculate performance with reference threshold
       tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, reference_threshold)
       current_score_ref = f1_score(tp, tn, fp, fn)
       
       # Calculate performance with current optimal threshold
       tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, current_threshold)
       current_score_opt = f1_score(tp, tn, fp, fn)
       
       # Check for significant drift
       threshold_drift = abs(current_threshold - reference_threshold)
       performance_drop = reference_score - current_score_ref
       potential_improvement = current_score_opt - current_score_ref
       
       drift_detected = (threshold_drift > tolerance or 
                        performance_drop > tolerance or
                        potential_improvement > tolerance)
       
       return {
           'drift_detected': drift_detected,
           'current_optimal_threshold': current_threshold,
           'threshold_drift': threshold_drift,
           'performance_with_reference': current_score_ref,
           'performance_drop': performance_drop,
           'potential_improvement': potential_improvement,
           'recommendation': 'retrain' if drift_detected else 'continue'
       }
   
   # Example monitoring
   monitoring_result = monitor_threshold_performance(
       y_test, y_prob_test, 
       reference_threshold=0.3, 
       reference_score=0.85
   )
   
   print(f"Drift detected: {monitoring_result['drift_detected']}")
   print(f"Recommendation: {monitoring_result['recommendation']}")

A/B Testing Optimal Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def ab_test_thresholds(y_true, y_prob, threshold_a, threshold_b, 
                         metric_func, confidence_level=0.95):
       """Statistical comparison of two thresholds."""
       from scipy import stats
       from optimal_cutoffs.metrics import get_confusion_matrix
       
       # Calculate performance for both thresholds
       tp_a, tn_a, fp_a, fn_a = get_confusion_matrix(y_true, y_prob, threshold_a)
       tp_b, tn_b, fp_b, fn_b = get_confusion_matrix(y_true, y_prob, threshold_b)
       
       score_a = metric_func(tp_a, tn_a, fp_a, fn_a)
       score_b = metric_func(tp_b, tn_b, fp_b, fn_b)
       
       # Bootstrap confidence intervals
       n_bootstrap = 1000
       scores_a_boot = []
       scores_b_boot = []
       
       for _ in range(n_bootstrap):
           # Resample with replacement
           indices = np.random.choice(len(y_true), len(y_true), replace=True)
           y_boot = y_true[indices]
           prob_boot = y_prob[indices]
           
           tp_a_boot, tn_a_boot, fp_a_boot, fn_a_boot = get_confusion_matrix(
               y_boot, prob_boot, threshold_a)
           tp_b_boot, tn_b_boot, fp_b_boot, fn_b_boot = get_confusion_matrix(
               y_boot, prob_boot, threshold_b)
           
           scores_a_boot.append(metric_func(tp_a_boot, tn_a_boot, fp_a_boot, fn_a_boot))
           scores_b_boot.append(metric_func(tp_b_boot, tn_b_boot, fp_b_boot, fn_b_boot))
       
       # Statistical test
       statistic, p_value = stats.ttest_rel(scores_b_boot, scores_a_boot)
       
       alpha = 1 - confidence_level
       significant = p_value < alpha
       
       return {
           'threshold_a': threshold_a,
           'threshold_b': threshold_b,
           'score_a': score_a,
           'score_b': score_b,
           'score_difference': score_b - score_a,
           'p_value': p_value,
           'significant': significant,
           'better_threshold': 'B' if score_b > score_a else 'A',
           'confidence_level': confidence_level
       }
   
   # Example A/B test
   from optimal_cutoffs.metrics import f1_score
   
   result = ab_test_thresholds(
       y_test, y_prob_test,
       threshold_a=0.5,      # Default threshold
       threshold_b=0.3,      # Optimized threshold
       metric_func=f1_score
   )
   
   print(f"Threshold A: {result['threshold_a']:.3f}, Score: {result['score_a']:.3f}")
   print(f"Threshold B: {result['threshold_b']:.3f}, Score: {result['score_b']:.3f}")
   print(f"Difference: {result['score_difference']:.3f}")
   print(f"Statistically significant: {result['significant']} (p={result['p_value']:.3f})")
   print(f"Recommendation: Use threshold {result['better_threshold']}")