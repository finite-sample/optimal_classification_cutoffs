Metrics System
==============

The metrics module provides built-in classification metrics and a flexible system for registering custom metrics.

Metric Registration
-------------------

.. autofunction:: optimal_cutoffs.metrics.register_metric

.. autofunction:: optimal_cutoffs.metrics.register_metrics

Metric Properties
-----------------

.. autofunction:: optimal_cutoffs.metrics.is_piecewise_metric

.. autofunction:: optimal_cutoffs.metrics.should_maximize_metric

.. autofunction:: optimal_cutoffs.metrics.needs_probability_scores

.. autofunction:: optimal_cutoffs.metrics.has_vectorized_implementation

.. autofunction:: optimal_cutoffs.metrics.get_vectorized_metric

Built-in Metrics
-----------------

.. autofunction:: optimal_cutoffs.metrics.f1_score

.. autofunction:: optimal_cutoffs.metrics.accuracy_score

.. autofunction:: optimal_cutoffs.metrics.precision_score

.. autofunction:: optimal_cutoffs.metrics.recall_score

Cost-Sensitive Metrics
-----------------------

.. autofunction:: optimal_cutoffs.metrics.make_linear_counts_metric

.. autofunction:: optimal_cutoffs.metrics.make_cost_metric

Confusion Matrix Utilities
---------------------------

.. autofunction:: optimal_cutoffs.metrics.get_confusion_matrix

.. autofunction:: optimal_cutoffs.metrics.get_multiclass_confusion_matrix

Multiclass Metrics
------------------

.. autofunction:: optimal_cutoffs.metrics.multiclass_metric

.. autofunction:: optimal_cutoffs.metrics.multiclass_metric_exclusive

Global Registries
------------------

.. autodata:: optimal_cutoffs.metrics.METRIC_REGISTRY
   :annotation: 

   Dictionary mapping metric names to metric functions. All registered metrics are stored here.

.. autodata:: optimal_cutoffs.metrics.VECTORIZED_REGISTRY
   :annotation: 

   Dictionary mapping metric names to vectorized implementations for O(n log n) optimization.

.. autodata:: optimal_cutoffs.metrics.METRIC_PROPERTIES
   :annotation: 

   Dictionary storing properties (piecewise, maximize, needs_proba) for each registered metric.