Optimal Classification Cutoffs
===============================

A Python library for computing optimal classification thresholds for binary and multiclass classification problems.

**Key Features:**

* **🚀 Fast O(n log n) algorithms** for exact threshold optimization
* **📊 Multiple metrics** - F1, accuracy, precision, recall, and custom metrics
* **💰 Cost-sensitive optimization** with utility-based thresholds
* **🎯 Multiclass support** with One-vs-Rest and coordinate ascent strategies
* **🔬 Cross-validation** utilities for robust threshold estimation
* **🛠️ Scikit-learn compatible** API for seamless integration
* **⚡ Auto method selection** - intelligent algorithm choice for best performance

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide
   examples
   advanced

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Background
   
   theory

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   
   faq

Why Optimize Classification Thresholds?
=======================================

Most classifiers use a default threshold of 0.5, but this is often suboptimal for:

🏥 **Medical Diagnosis**
   False negatives (missed diseases) cost far more than false positives

🏦 **Fraud Detection**  
   Missing fraud has higher cost than investigating legitimate transactions

📧 **Spam Detection**
   Blocking legitimate emails is worse than letting some spam through

📊 **Imbalanced Datasets**
   Default thresholds perform poorly when classes have very different frequencies

The Problem with Standard Optimization
======================================

Classification metrics like F1 score are **piecewise-constant functions** that create challenges for traditional optimization methods:

.. image:: piecewise_f1_demo.png
   :alt: F1 Score Piecewise Behavior
   :width: 600px
   :align: center

Standard optimizers fail because these functions have:

* **Zero gradients** everywhere except at breakpoints
* **Flat regions** providing no directional information  
* **Step discontinuities** that trap optimizers

Our solution uses specialized algorithms designed for piecewise-constant optimization.

Quick Example
=============

.. code-block:: python

   from optimal_cutoffs import get_optimal_threshold
   import numpy as np
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_classification

   # Generate imbalanced dataset
   X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

   # Train classifier
   clf = RandomForestClassifier().fit(X_train, y_train)
   y_prob = clf.predict_proba(X_test)[:, 1]

   # Find optimal threshold (automatic algorithm selection)
   threshold = get_optimal_threshold(y_test, y_prob, metric='f1')
   print(f"Optimal F1 threshold: {threshold:.3f}")

   # Compare with default 0.5 threshold
   default_pred = (y_prob >= 0.5).astype(int)
   optimal_pred = (y_prob >= threshold).astype(int)

   from sklearn.metrics import f1_score
   print(f"Default F1: {f1_score(y_test, default_pred):.3f}")
   print(f"Optimal F1: {f1_score(y_test, optimal_pred):.3f}")

Performance Comparison
=====================

The library's specialized algorithms significantly outperform standard optimization:

+------------------+------------------+------------------+------------------+
| Dataset Size     | sort_scan        | smart_brute      | scipy minimize   |
+==================+==================+==================+==================+
| 1,000 samples    | 0.001s ⚡       | 0.003s ⚡       | 0.050s          |
+------------------+------------------+------------------+------------------+
| 10,000 samples   | 0.008s ⚡       | 0.025s ⚡       | 0.200s          |
+------------------+------------------+------------------+------------------+
| 100,000 samples  | 0.080s ⚡       | 2.100s          | 5.000s          |
+------------------+------------------+------------------+------------------+

✅ **sort_scan**: O(n log n) exact algorithm for piecewise metrics  
✅ **smart_brute**: Evaluates only unique probability values  
⚠️ **minimize**: Standard scipy optimization (often suboptimal)

Citation
========

If you use this library in academic research, please cite:

.. code-block:: bibtex

   @software{optimal_classification_cutoffs,
     author = {Laohaprapanon, Suriyan and Sood, Gaurav},
     title = {Optimal Classification Cutoffs: Fast algorithms for threshold optimization},
     url = {https://github.com/finite-sample/optimal_classification_cutoffs},
     year = {2024}
   }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`