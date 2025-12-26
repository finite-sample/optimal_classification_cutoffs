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

Learning Path
-------------

For the best learning experience, follow this order:

1. **Quickstart** - See a 40%+ performance improvement in just 3 lines of code
2. **Business Value** - Learn how to optimize for dollars rather than statistical metrics with cost-sensitive optimization
3. **Multiclass** - Handle complex multi-class scenarios with advanced threshold strategies
4. **Interactive Demo** - Deep dive into the mathematical foundations with interactive exploration

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