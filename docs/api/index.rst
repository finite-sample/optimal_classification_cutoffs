API Reference
=============

The optimal-classification-cutoffs library provides a comprehensive API for threshold optimization across binary and multiclass classification problems.

Overview
--------

The main entry points are:

* :func:`~optimal_cutoffs.api.optimize_thresholds` - Core optimization function for binary and multiclass problems
* :func:`~optimal_cutoffs.cv.cross_validate` - Cross-validation utilities

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
