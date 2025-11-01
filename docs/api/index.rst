API Reference
=============

The optimal-classification-cutoffs library provides a comprehensive API for threshold optimization across binary and multiclass classification problems.

Overview
--------

The main entry points are:

* :func:`~optimal_cutoffs.optimizers.get_optimal_threshold` - Core optimization function for binary and multiclass problems
* :class:`~optimal_cutoffs.wrapper.ThresholdOptimizer` - Scikit-learn compatible wrapper class
* :func:`~optimal_cutoffs.cv.cv_threshold_optimization` - Cross-validation utilities

Core Functions
--------------

The primary interface for threshold optimization:

.. toctree::
   :maxdepth: 1

   core

Metrics System
--------------

Built-in metrics and utilities for custom metric registration:

.. toctree::
   :maxdepth: 1

   metrics

Scikit-learn Interface
----------------------

High-level wrapper for integration with scikit-learn workflows:

.. toctree::
   :maxdepth: 1

   wrapper

Cross-Validation
----------------

Utilities for robust threshold estimation:

.. toctree::
   :maxdepth: 1

   cv

Multiclass Support
------------------

Specialized functionality for multiclass classification:

.. toctree::
   :maxdepth: 1

   multiclass

Type Definitions
----------------

Type hints and protocols used throughout the library:

.. automodule:: optimal_cutoffs.types
   :members:
   :undoc-members:
   :show-inheritance:
