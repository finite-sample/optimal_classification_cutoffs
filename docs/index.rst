Optimal Classification Cutoffs
===============================

A Python library for computing optimal classification thresholds for binary and multiclass classification problems.

Features
--------

* Automatic detection of binary vs multiclass problems
* Multiple optimization methods (brute force, scipy minimize, gradient ascent)
* Support for custom metrics
* Cross-validation utilities
* Scikit-learn compatible API
* One-vs-Rest strategy for multiclass problems

Installation
------------

.. code-block:: bash

   pip install optimal-classification-cutoffs

Quick Start
-----------

Binary Classification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import get_optimal_threshold
   import numpy as np

   # Binary classification example
   y_true = np.array([0, 0, 1, 1])
   y_prob = np.array([0.1, 0.4, 0.35, 0.8])
   
   threshold = get_optimal_threshold(y_true, y_prob, metric='f1')
   print(f"Optimal threshold: {threshold}")

Multiclass Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import get_optimal_threshold
   import numpy as np

   # Multiclass classification example
   y_true = np.array([0, 1, 2, 0, 1, 2])
   y_prob = np.array([
       [0.7, 0.2, 0.1],
       [0.1, 0.8, 0.1], 
       [0.1, 0.1, 0.8],
       [0.6, 0.3, 0.1],
       [0.2, 0.7, 0.1],
       [0.1, 0.2, 0.7]
   ])
   
   thresholds = get_optimal_threshold(y_true, y_prob, metric='f1')
   print(f"Optimal thresholds per class: {thresholds}")

Using the Scikit-learn Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import ThresholdOptimizer
   from sklearn.model_selection import train_test_split

   # Initialize optimizer
   optimizer = ThresholdOptimizer(metric='f1', method='smart_brute')
   
   # Fit on training data
   optimizer.fit(y_train, y_prob_train)
   
   # Predict on test data
   y_pred = optimizer.predict(y_prob_test)

Theory and Background
=====================

Understanding why standard optimization methods can fail for classification metrics:

.. toctree::
   :maxdepth: 2
   
   theory

API Reference
=============

Core Functions
--------------

.. automodule:: optimal_cutoffs.optimizers
   :members:

Threshold Optimizer Class
-------------------------

.. automodule:: optimal_cutoffs.wrapper
   :members:

Metrics
-------

.. automodule:: optimal_cutoffs.metrics
   :members:

Cross-Validation
----------------

.. automodule:: optimal_cutoffs.cv
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`