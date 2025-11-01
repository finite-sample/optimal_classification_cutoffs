Core Functions
==============

This module contains the main optimization functions that form the core of the library.

Main Optimization Function
---------------------------

.. autofunction:: optimal_cutoffs.optimizers.get_optimal_threshold

Multiclass Optimization
-----------------------

.. autofunction:: optimal_cutoffs.optimizers.get_optimal_multiclass_thresholds

Cost-Sensitive Optimization
---------------------------

.. autofunction:: optimal_cutoffs.optimizers.bayes_threshold_from_utility

.. autofunction:: optimal_cutoffs.optimizers.bayes_threshold_from_costs

Legacy Functions
----------------

.. autofunction:: optimal_cutoffs.optimizers.get_probability

Internal Functions
------------------

These functions are used internally but may be useful for advanced users:

.. automodule:: optimal_cutoffs.piecewise
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: optimal_cutoffs.multiclass_coord
   :members:
   :undoc-members:
   :show-inheritance:
