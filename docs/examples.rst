Examples
========

This page provides comprehensive examples showing how to use optimal-classification-cutoffs in various scenarios.

Complete Binary Classification Example
--------------------------------------

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, confusion_matrix
   from optimal_cutoffs import get_optimal_threshold, ThresholdOptimizer

   # Generate synthetic dataset
   X, y = make_classification(
       n_samples=1000,
       n_features=10,
       n_classes=2,
       weights=[0.9, 0.1],  # Imbalanced classes
       random_state=42
   )

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42, stratify=y
   )

   # Train classifier
   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)

   # Get probabilities
   y_prob_train = clf.predict_proba(X_train)[:, 1]
   y_prob_test = clf.predict_proba(X_test)[:, 1]

   # Method 1: Direct threshold optimization
   print("=== Direct Threshold Optimization ===")
   threshold_f1 = get_optimal_threshold(y_train, y_prob_train, metric='f1')
   threshold_precision = get_optimal_threshold(y_train, y_prob_train, metric='precision')
   threshold_recall = get_optimal_threshold(y_train, y_prob_train, metric='recall')

   print(f"Optimal F1 threshold: {threshold_f1:.3f}")
   print(f"Optimal Precision threshold: {threshold_precision:.3f}")
   print(f"Optimal Recall threshold: {threshold_recall:.3f}")

   # Make predictions with F1-optimized threshold
   y_pred_optimized = (y_prob_test >= threshold_f1).astype(int)
   y_pred_default = (y_prob_test >= 0.5).astype(int)

   print(f"\n=== Performance Comparison ===")
   print("With default 0.5 threshold:")
   print(classification_report(y_test, y_pred_default))

   print("With optimized threshold:")
   print(classification_report(y_test, y_pred_optimized))

   # Method 2: Using ThresholdOptimizer
   print("=== Using ThresholdOptimizer ===")
   optimizer = ThresholdOptimizer(metric='f1', method='auto', verbose=True)
   optimizer.fit(y_train, y_prob_train)

   y_pred_optimizer = optimizer.predict(y_prob_test)
   print(f"Optimized threshold: {optimizer.threshold_:.3f}")
   print(f"Training F1 score: {optimizer.score_:.3f}")
   print(classification_report(y_test, y_pred_optimizer))

Multiclass Classification Example
---------------------------------

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.multiclass import OneVsRestClassifier
   from sklearn.linear_model import LogisticRegression
   import matplotlib.pyplot as plt

   # Generate multiclass dataset
   X, y = make_classification(
       n_samples=2000,
       n_features=10,
       n_classes=4,
       n_informative=8,
       n_redundant=0,
       n_clusters_per_class=1,
       weights=[0.4, 0.3, 0.2, 0.1],  # Imbalanced
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42, stratify=y
   )

   # Train multiclass classifier
   clf = OneVsRestClassifier(LogisticRegression(random_state=42))
   clf.fit(X_train, y_train)

   # Get class probabilities
   y_prob_train = clf.predict_proba(X_train)
   y_prob_test = clf.predict_proba(X_test)

   print("=== Multiclass Threshold Optimization ===")

   # Optimize thresholds for each class
   thresholds_macro = get_optimal_threshold(
       y_train, y_prob_train,
       metric='f1',
       average='macro'
   )

   thresholds_weighted = get_optimal_threshold(
       y_train, y_prob_train,
       metric='f1',
       average='weighted'
   )

   print(f"Macro-averaged thresholds: {thresholds_macro}")
   print(f"Weighted thresholds: {thresholds_weighted}")

   # Using ThresholdOptimizer for multiclass
   optimizer = ThresholdOptimizer(metric='f1', average='macro')
   optimizer.fit(y_train, y_prob_train)

   y_pred_default = clf.predict(X_test)  # Uses argmax
   y_pred_optimized = optimizer.predict(y_prob_test)

   print(f"\nDefault accuracy: {np.mean(y_pred_default == y_test):.3f}")
   print(f"Optimized accuracy: {np.mean(y_pred_optimized == y_test):.3f}")

   # Per-class analysis
   from sklearn.metrics import f1_score
   f1_default = f1_score(y_test, y_pred_default, average=None)
   f1_optimized = f1_score(y_test, y_pred_optimized, average=None)

   print(f"\nPer-class F1 scores:")
   print(f"Default:   {f1_default}")
   print(f"Optimized: {f1_optimized}")

Cost-Sensitive Classification Example
-------------------------------------

.. code-block:: python

   # Medical diagnosis scenario: False negatives are much more costly
   print("=== Cost-Sensitive Medical Diagnosis Example ===")

   # Simulate medical diagnosis data
   np.random.seed(42)
   n_patients = 1000

   # Features: age, test_result_1, test_result_2, symptoms_score
   X_medical = np.random.randn(n_patients, 4)
   X_medical[:, 0] = np.random.uniform(20, 80, n_patients)  # Age

   # True disease status (10% positive)
   disease_prob = 1 / (1 + np.exp(-(X_medical[:, 1] + X_medical[:, 2] - 1)))
   y_disease = np.random.binomial(1, disease_prob * 0.1)  # Low prevalence

   # Train diagnostic model
   X_train, X_test, y_train, y_test = train_test_split(
       X_medical, y_disease, test_size=0.3, random_state=42, stratify=y_disease
   )

   clf = LogisticRegression(random_state=42)
   clf.fit(X_train, y_train)

   y_prob_train = clf.predict_proba(X_train)[:, 1]
   y_prob_test = clf.predict_proba(X_test)[:, 1]

   # Define costs
   # - Missing a disease (FN) costs $50,000 in treatment delays
   # - False alarm (FP) costs $1,000 in unnecessary procedures
   # - Correct diagnosis has no additional cost

   cost_matrix = {
       "tp": 0,      # Correct positive diagnosis
       "tn": 0,      # Correct negative diagnosis
       "fp": -1000,  # False positive cost
       "fn": -50000  # False negative cost (very high!)
   }

   # Standard F1 optimization
   threshold_f1 = get_optimal_threshold(y_train, y_prob_train, metric='f1')

   # Cost-sensitive optimization
   threshold_cost = get_optimal_threshold(
       y_train, y_prob_train,
       utility=cost_matrix
   )

   # Bayes optimal (for calibrated probabilities)
   threshold_bayes = get_optimal_threshold(
       None, y_prob_train,  # No labels needed for Bayes
       utility=cost_matrix,
       bayes=True
   )

   print(f"F1-optimized threshold: {threshold_f1:.3f}")
   print(f"Cost-optimized threshold: {threshold_cost:.3f}")
   print(f"Bayes-optimal threshold: {threshold_bayes:.3f}")

   # Evaluate different strategies
   strategies = {
       'Default (0.5)': 0.5,
       'F1-Optimized': threshold_f1,
       'Cost-Optimized': threshold_cost,
       'Bayes-Optimal': threshold_bayes
   }

   print(f"\n{'Strategy':<15} {'Threshold':<10} {'FP':<5} {'FN':<5} {'Cost':<10}")
   print("-" * 50)

   for name, thresh in strategies.items():
       y_pred = (y_prob_test >= thresh).astype(int)

       # Calculate confusion matrix components
       tp = np.sum((y_pred == 1) & (y_test == 1))
       tn = np.sum((y_pred == 0) & (y_test == 0))
       fp = np.sum((y_pred == 1) & (y_test == 0))
       fn = np.sum((y_pred == 0) & (y_test == 1))

       # Calculate total cost
       total_cost = fp * 1000 + fn * 50000

       print(f"{name:<15} {thresh:<10.3f} {fp:<5} {fn:<5} ${total_cost:<10,}")

Cross-Validation Example
------------------------

.. code-block:: python

   from optimal_cutoffs import cv_threshold_optimization
   from sklearn.model_selection import StratifiedKFold

   print("=== Cross-Validation Example ===")

   # Generate imbalanced dataset
   X, y = make_classification(
       n_samples=5000,
       n_features=20,
       n_classes=2,
       weights=[0.95, 0.05],  # Very imbalanced
       random_state=42
   )

   # Train classifier
   clf = RandomForestClassifier(n_estimators=50, random_state=42)
   clf.fit(X, y)
   y_prob = clf.predict_proba(X)[:, 1]

   # Standard 5-fold CV
   thresholds, scores = cv_threshold_optimization(
       y, y_prob,
       metric='f1',
       cv=5,
       method='auto'
   )

   print(f"5-fold CV results:")
   print(f"Thresholds: {thresholds}")
   print(f"F1 scores: {scores}")
   print(f"Mean threshold: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")
   print(f"Mean F1 score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

   # Stratified CV for imbalanced data
   stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
   thresholds_strat, scores_strat = cv_threshold_optimization(
       y, y_prob,
       metric='f1',
       cv=stratified_cv,
       method='smart_brute'
   )

   print(f"\n10-fold Stratified CV results:")
   print(f"Mean threshold: {np.mean(thresholds_strat):.3f} ± {np.std(thresholds_strat):.3f}")
   print(f"Mean F1 score: {np.mean(scores_strat):.3f} ± {np.std(scores_strat):.3f}")

   # Compare with different metrics
   metrics = ['f1', 'precision', 'recall', 'accuracy']
   cv_results = {}

   for metric in metrics:
       thresholds, scores = cv_threshold_optimization(
           y, y_prob, metric=metric, cv=5
       )
       cv_results[metric] = {
           'mean_threshold': np.mean(thresholds),
           'std_threshold': np.std(thresholds),
           'mean_score': np.mean(scores),
           'std_score': np.std(scores)
       }

   print(f"\n{'Metric':<10} {'Threshold':<15} {'Score':<15}")
   print("-" * 40)
   for metric, results in cv_results.items():
       thresh_str = f"{results['mean_threshold']:.3f} ± {results['std_threshold']:.3f}"
       score_str = f"{results['mean_score']:.3f} ± {results['std_score']:.3f}"
       print(f"{metric:<10} {thresh_str:<15} {score_str:<15}")

Custom Metric Example
---------------------

.. code-block:: python

   from optimal_cutoffs.metrics import register_metric

   print("=== Custom Metric Example ===")

   # Define a custom metric: Geometric mean of precision and recall
   def geometric_mean_score(tp, tn, fp, fn):
       """Geometric mean of precision and recall."""
       precision = tp / (tp + fp) if tp + fp > 0 else 0.0
       recall = tp / (tp + fn) if tp + fn > 0 else 0.0
       return np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0.0

   # Vectorized version for O(n log n) optimization
   def geometric_mean_vectorized(tp, tn, fp, fn):
       """Vectorized geometric mean computation."""
       precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                           where=(tp + fp) > 0)
       recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float),
                        where=(tp + fn) > 0)

       # Only compute sqrt where both precision and recall > 0
       valid = (precision > 0) & (recall > 0)
       result = np.zeros_like(tp, dtype=float)
       result[valid] = np.sqrt(precision[valid] * recall[valid])
       return result

   # Register the custom metric
   register_metric(
       'geometric_mean',
       geometric_mean_score,
       vectorized_func=geometric_mean_vectorized,
       is_piecewise=True,
       maximize=True
   )

   # Use the custom metric
   X, y = make_classification(n_samples=1000, weights=[0.7, 0.3], random_state=42)
   clf = LogisticRegression(random_state=42)
   clf.fit(X, y)
   y_prob = clf.predict_proba(X)[:, 1]

   # Optimize using custom metric
   threshold_custom = get_optimal_threshold(y, y_prob, metric='geometric_mean')
   threshold_f1 = get_optimal_threshold(y, y_prob, metric='f1')

   print(f"Geometric mean optimized threshold: {threshold_custom:.3f}")
   print(f"F1 optimized threshold: {threshold_f1:.3f}")

   # Compare performance
   y_pred_custom = (y_prob >= threshold_custom).astype(int)
   y_pred_f1 = (y_prob >= threshold_f1).astype(int)

   # Calculate both metrics for comparison
   from optimal_cutoffs.metrics import get_confusion_matrix

   tp_c, tn_c, fp_c, fn_c = get_confusion_matrix(y, y_prob, threshold_custom)
   tp_f, tn_f, fp_f, fn_f = get_confusion_matrix(y, y_prob, threshold_f1)

   gm_custom = geometric_mean_score(tp_c, tn_c, fp_c, fn_c)
   gm_f1 = geometric_mean_score(tp_f, tn_f, fp_f, fn_f)

   from optimal_cutoffs.metrics import f1_score
   f1_custom = f1_score(tp_c, tn_c, fp_c, fn_c)
   f1_f1 = f1_score(tp_f, tn_f, fp_f, fn_f)

   print(f"\nCustom threshold - Geometric mean: {gm_custom:.3f}, F1: {f1_custom:.3f}")
   print(f"F1 threshold - Geometric mean: {gm_f1:.3f}, F1: {f1_f1:.3f}")

Performance Comparison Example
-----------------------------

.. code-block:: python

   import time
   from optimal_cutoffs.piecewise import optimal_threshold_sortscan
   from optimal_cutoffs.metrics import get_vectorized_metric

   print("=== Performance Comparison Example ===")

   # Generate large dataset
   np.random.seed(42)
   sizes = [1000, 5000, 10000, 50000]
   methods = ['smart_brute', 'sort_scan', 'minimize']

   print(f"{'Size':<8} {'Method':<12} {'Time (s)':<10} {'Threshold':<12}")
   print("-" * 50)

   for size in sizes:
       y = np.random.randint(0, 2, size)
       y_prob = np.random.uniform(0, 1, size)

       results = {}

       for method in methods:
           start_time = time.time()
           try:
               threshold = get_optimal_threshold(
                   y, y_prob, metric='f1', method=method
               )
               elapsed = time.time() - start_time
               results[method] = (threshold, elapsed)

               print(f"{size:<8} {method:<12} {elapsed:<10.4f} {threshold:<12.4f}")

           except Exception as e:
               print(f"{size:<8} {method:<12} {'FAILED':<10} {'N/A':<12}")

       print()  # Empty line between sizes

Real-World Integration Example
------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import GridSearchCV
   from sklearn.ensemble import RandomForestClassifier

   print("=== Real-World Integration Example ===")

   # Load a real dataset (using make_classification as proxy)
   X, y = make_classification(
       n_samples=2000,
       n_features=15,
       n_classes=2,
       weights=[0.8, 0.2],
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42, stratify=y
   )

   # Create a complete ML pipeline
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('classifier', RandomForestClassifier(random_state=42))
   ])

   # Hyperparameter tuning
   param_grid = {
       'classifier__n_estimators': [50, 100],
       'classifier__max_depth': [10, 20, None]
   }

   # Grid search with cross-validation
   grid_search = GridSearchCV(
       pipeline, param_grid,
       cv=5, scoring='roc_auc',
       n_jobs=-1, verbose=1
   )

   print("Training pipeline with grid search...")
   grid_search.fit(X_train, y_train)

   # Get best model and predictions
   best_pipeline = grid_search.best_estimator_
   y_prob_train = best_pipeline.predict_proba(X_train)[:, 1]
   y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]

   print(f"Best parameters: {grid_search.best_params_}")

   # Apply threshold optimization
   optimizer = ThresholdOptimizer(metric='f1', method='auto')
   optimizer.fit(y_train, y_prob_train)

   # Final evaluation
   y_pred_default = best_pipeline.predict(X_test)
   y_pred_optimized = optimizer.predict(y_prob_test)

   from sklearn.metrics import classification_report, roc_auc_score

   print(f"\n=== Final Results ===")
   print(f"Optimized threshold: {optimizer.threshold_:.3f}")
   print(f"ROC AUC: {roc_auc_score(y_test, y_prob_test):.3f}")

   print(f"\nDefault predictions (0.5 threshold):")
   print(classification_report(y_test, y_pred_default))

   print(f"Optimized predictions:")
   print(classification_report(y_test, y_pred_optimized))

   # Save the complete solution
   import joblib

   # In practice, you would save both the trained pipeline and optimizer
   solution = {
       'pipeline': best_pipeline,
       'threshold_optimizer': optimizer,
       'threshold': optimizer.threshold_,
       'training_score': optimizer.score_
   }

   # joblib.dump(solution, 'complete_model.pkl')
   print(f"\nModel ready for deployment with threshold: {optimizer.threshold_:.3f}")
