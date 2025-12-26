Metrics System
==============

The metrics module provides built-in classification metrics and a flexible system for registering custom metrics.

Metric Functions
----------------

.. autofunction:: optimal_cutoffs.metrics.get

.. autofunction:: optimal_cutoffs.metrics.register

.. autofunction:: optimal_cutoffs.metrics.list_available

.. autofunction:: optimal_cutoffs.metrics.info

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from optimal_cutoffs import metrics

   # List all available metrics
   print(metrics.list_available())

   # Get information about a metric
   print(metrics.info('f1'))

   # Register a custom metric
   def custom_metric(tp, tn, fp, fn):
       return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

   metrics.register('custom', custom_metric)

   # Use the custom metric
   metric_fn = metrics.get('custom')
   score = metric_fn(10, 5, 2, 3)
