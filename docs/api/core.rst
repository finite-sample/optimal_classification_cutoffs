Core Functions
==============

This module contains the main optimization functions that form the core of the library.

Main Optimization Functions
----------------------------

.. autofunction:: optimal_cutoffs.api.optimize_thresholds

.. autofunction:: optimal_cutoffs.api.optimize_decisions

Binary Classification
---------------------

.. autofunction:: optimal_cutoffs.binary.optimize_f1_binary

.. autofunction:: optimal_cutoffs.binary.optimize_metric_binary

.. autofunction:: optimal_cutoffs.binary.optimize_utility_binary

Multiclass Classification
-------------------------

.. autofunction:: optimal_cutoffs.multiclass.optimize_multiclass

.. autofunction:: optimal_cutoffs.multiclass.optimize_ovr_independent

.. autofunction:: optimal_cutoffs.multiclass.optimize_ovr_margin

.. autofunction:: optimal_cutoffs.multiclass.optimize_micro_multiclass

Multilabel Classification
-------------------------

.. autofunction:: optimal_cutoffs.multilabel.optimize_multilabel

.. autofunction:: optimal_cutoffs.multilabel.optimize_macro_multilabel

.. autofunction:: optimal_cutoffs.multilabel.optimize_micro_multilabel

Bayes-Optimal Decisions
------------------------

.. autofunction:: optimal_cutoffs.bayes.threshold

.. autofunction:: optimal_cutoffs.bayes.thresholds_from_costs

.. autofunction:: optimal_cutoffs.bayes.policy

Internal Functions
------------------

These functions are used internally but may be useful for advanced users:

.. automodule:: optimal_cutoffs.piecewise
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: optimal_cutoffs.optimize
   :members:
   :undoc-members:
   :show-inheritance:
