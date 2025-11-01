Scikit-learn Interface
=====================

The wrapper module provides a scikit-learn compatible interface for threshold optimization.

ThresholdOptimizer Class
------------------------

.. autoclass:: optimal_cutoffs.wrapper.ThresholdOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import ThresholdOptimizer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   # Train your classifier
   clf = RandomForestClassifier()
   clf.fit(X_train, y_train)

   # Get probabilities
   y_prob_train = clf.predict_proba(X_train)[:, 1]  # Binary case
   y_prob_test = clf.predict_proba(X_test)[:, 1]

   # Optimize threshold
   optimizer = ThresholdOptimizer(metric='f1', method='auto')
   optimizer.fit(y_train, y_prob_train)

   # Make predictions
   y_pred = optimizer.predict(y_prob_test)

Multiclass Usage
~~~~~~~~~~~~~~~~

.. code-block:: python

   # For multiclass problems, pass full probability matrix
   y_prob_train = clf.predict_proba(X_train)  # Shape: (n_samples, n_classes)
   y_prob_test = clf.predict_proba(X_test)

   # Optimizer automatically detects multiclass
   optimizer = ThresholdOptimizer(metric='f1')
   optimizer.fit(y_train, y_prob_train)  # y_train has integer class labels

   # Returns class predictions
   y_pred = optimizer.predict(y_prob_test)

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   # Create pipeline with threshold optimization
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('classifier', RandomForestClassifier()),
       ('threshold', ThresholdOptimizer(metric='f1'))
   ])

   # Note: This requires custom pipeline steps for probability extraction
   # See advanced examples for full implementation

Attributes
----------

After fitting, the ``ThresholdOptimizer`` instance has several useful attributes:

* ``threshold_``: The optimized threshold(s)
* ``score_``: The metric value achieved at the optimal threshold
* ``n_classes_``: Number of classes detected (1 for binary, >1 for multiclass)
* ``classes_``: Array of unique class labels
