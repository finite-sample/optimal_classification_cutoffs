Quick Start
===========

This guide will get you up and running with optimal-classification-cutoffs in just a few minutes.

Basic Binary Classification
----------------------------

The simplest use case is finding the optimal threshold for binary classification:

.. code-block:: python

   from optimal_cutoffs import get_optimal_threshold
   import numpy as np

   # Your binary classification data
   y_true = np.array([0, 0, 1, 1, 0, 1])
   y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
   
   # Find optimal threshold for F1 score
   threshold = get_optimal_threshold(y_true, y_prob, metric='f1')
   print(f"Optimal threshold: {threshold:.3f}")
   
   # Make predictions
   predictions = (y_prob >= threshold).astype(int)
   print(f"Predictions: {predictions}")

Other Metrics
~~~~~~~~~~~~~

You can optimize for different metrics:

.. code-block:: python

   # Optimize for accuracy
   threshold_acc = get_optimal_threshold(y_true, y_prob, metric='accuracy')
   
   # Optimize for precision
   threshold_prec = get_optimal_threshold(y_true, y_prob, metric='precision')
   
   # Optimize for recall
   threshold_rec = get_optimal_threshold(y_true, y_prob, metric='recall')
   
   print(f"Accuracy threshold: {threshold_acc:.3f}")
   print(f"Precision threshold: {threshold_prec:.3f}")
   print(f"Recall threshold: {threshold_rec:.3f}")

Multiclass Classification
-------------------------

For multiclass problems, the library automatically detects the problem type and returns per-class thresholds:

.. code-block:: python

   # Multiclass example with 3 classes
   y_true = np.array([0, 1, 2, 0, 1, 2])
   y_prob = np.array([
       [0.7, 0.2, 0.1],  # Sample 1: likely class 0
       [0.1, 0.8, 0.1],  # Sample 2: likely class 1
       [0.1, 0.1, 0.8],  # Sample 3: likely class 2
       [0.6, 0.3, 0.1],  # Sample 4: likely class 0
       [0.2, 0.7, 0.1],  # Sample 5: likely class 1
       [0.1, 0.2, 0.7]   # Sample 6: likely class 2
   ])
   
   # Get per-class optimal thresholds
   thresholds = get_optimal_threshold(y_true, y_prob, metric='f1')
   print(f"Optimal thresholds per class: {thresholds}")

Using the Scikit-learn Interface
--------------------------------

For integration with scikit-learn pipelines, use the ``ThresholdOptimizer`` class:

.. code-block:: python

   from optimal_cutoffs import ThresholdOptimizer
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   # Generate sample data
   X = np.random.randn(1000, 5)
   y = (X[:, 0] + X[:, 1] > 0).astype(int)
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
   # Train a classifier
   clf = RandomForestClassifier(random_state=42)
   clf.fit(X_train, y_train)
   y_prob_train = clf.predict_proba(X_train)[:, 1]
   y_prob_test = clf.predict_proba(X_test)[:, 1]
   
   # Optimize threshold
   optimizer = ThresholdOptimizer(metric='f1', method='smart_brute')
   optimizer.fit(y_train, y_prob_train)
   
   # Make optimized predictions
   y_pred = optimizer.predict(y_prob_test)
   
   print(f"Optimal threshold: {optimizer.threshold_:.3f}")
   print(f"Test accuracy: {np.mean(y_pred == y_test):.3f}")

Optimization Methods
--------------------

The library provides several optimization methods:

.. code-block:: python

   # Auto method selection (recommended)
   threshold = get_optimal_threshold(y_true, y_prob, metric='f1', method='auto')
   
   # Fast O(n log n) algorithm for piecewise metrics
   threshold = get_optimal_threshold(y_true, y_prob, metric='f1', method='sort_scan')
   
   # Brute force evaluation of all unique probabilities
   threshold = get_optimal_threshold(y_true, y_prob, metric='f1', method='smart_brute')
   
   # Scipy-based continuous optimization
   threshold = get_optimal_threshold(y_true, y_prob, metric='f1', method='minimize')

Cost-Sensitive Optimization
---------------------------

For applications where different types of errors have different costs:

.. code-block:: python

   # False negatives cost 5x more than false positives
   threshold = get_optimal_threshold(
       y_true, y_prob, 
       utility={"fp": -1.0, "fn": -5.0}
   )
   
   # With benefits for correct predictions
   threshold = get_optimal_threshold(
       y_true, y_prob,
       utility={"tp": 2.0, "tn": 1.0, "fp": -1.0, "fn": -5.0}
   )

Next Steps
----------

* Read the :doc:`user_guide` for detailed explanations and advanced features
* Check out :doc:`examples` for more comprehensive examples
* Explore :doc:`advanced` topics like cross-validation and custom metrics
* Understand the :doc:`theory` behind why this approach works better than standard methods