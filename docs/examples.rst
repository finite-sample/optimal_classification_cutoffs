Examples
========

This section provides comprehensive examples showing how to use optimal-classification-cutoffs in various scenarios. All examples are available as interactive Jupyter notebooks that demonstrate the power of API 2.0.0.

.. toctree::
   :maxdepth: 1
   :caption: Interactive Examples

   examples/01_quickstart.ipynb
   examples/02_business_value.ipynb  
   examples/03_multiclass.ipynb
   examples/04_interactive_demo.ipynb

Quick Links to Notebooks
-------------------------

* :doc:`examples/01_quickstart` - **Quickstart**: See a 40%+ performance improvement in just 3 lines of code
* :doc:`examples/02_business_value` - **Business Value**: Learn how to optimize for dollars rather than statistical metrics with cost-sensitive optimization  
* :doc:`examples/03_multiclass` - **Multiclass**: Handle complex multi-class scenarios with advanced threshold strategies
* :doc:`examples/04_interactive_demo` - **Interactive Demo**: Deep dive into the mathematical foundations with interactive exploration

Learning Path
-------------

For the best learning experience, follow this order:

1. **Quickstart** → See immediate 40%+ improvements with minimal code
2. **Business Value** → Understand cost-sensitive optimization for real ROI
3. **Multiclass** → Master advanced multiclass threshold strategies  
4. **Interactive Demo** → Explore mathematical foundations interactively

Each example builds on the previous ones while being self-contained enough to run independently.

Running the Examples
--------------------

To run these examples locally:

1. **Install with examples dependencies**:

   .. code-block:: bash

      pip install optimal-classification-cutoffs[examples]

2. **Download the notebooks** from the `GitHub repository <https://github.com/finite-sample/optimal-classification-cutoffs/tree/master/docs/examples>`_

3. **Launch Jupyter**:

   .. code-block:: bash

      jupyter notebook

4. **Open and run** any of the example notebooks

What You'll Learn
------------------

**API 2.0.0 Features Demonstrated:**

- Progressive disclosure design (simple → advanced)
- Explainable auto-selection with reasoning
- Enum-based explicit control (Task, Average)
- Namespace organization (metrics/, cv/, bayes/, algorithms/)
- Modern match/case performance optimizations
- Zero backward compatibility - clean slate approach

**Real-World Applications:**

- Fraud detection with cost-sensitive optimization
- Medical diagnosis with asymmetric error costs  
- Document classification with multiclass strategies
- A/B testing with threshold validation
- Business ROI calculation from model improvements

Each notebook contains working code you can run immediately to see 40%+ performance improvements over default 0.5 thresholds.