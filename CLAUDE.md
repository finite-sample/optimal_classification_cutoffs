# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Installation
```bash
# Install in development mode
python -m pip install -e .

# Install with example dependencies
python -m pip install -e ".[examples]"
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_multiclass.py -v

# Run specific test
python -m pytest tests/test_multiclass.py::test_multiclass_confusion_matrix -v
```

### Examples
```bash
# Run basic binary example
python examples/basic_usage.py

# Run multiclass example
python examples/multiclass_usage.py
```

## Architecture Overview

This library provides optimal threshold selection for binary and multiclass classification problems. The core architecture is built around several key components that work together:

### Binary vs Multiclass Detection
The library automatically detects whether input is binary (1D probabilities) or multiclass (2D probability matrix) and routes to appropriate implementations. Binary classification uses legacy optimizers for backward compatibility, while multiclass uses the newer general framework.

### Multiclass Strategy: One-vs-Rest
For multiclass problems, the library implements a One-vs-Rest (OvR) strategy where each class gets its own optimized threshold. This approach:
- Treats each class as a separate binary problem (class vs all others)
- Optimizes thresholds independently per class
- Handles class imbalance better than simple argmax approaches
- Allows different metrics to be optimized for different classes

### Core Components

**metrics.py**: Contains the metric registry system and confusion matrix utilities. The `METRIC_REGISTRY` is a global dictionary that allows registration of custom metrics that accept `(tp, tn, fp, fn)` parameters. For multiclass, special handling ensures proper micro averaging (avoiding TN double-counting in OvR).

**optimizers.py**: Implements threshold optimization algorithms. `get_optimal_threshold()` serves as the main entry point that auto-detects binary vs multiclass and routes accordingly. Optimization methods include "smart_brute" (evaluates all unique probabilities), "minimize" (scipy optimization), and "gradient" (simple gradient ascent).

**wrapper.py**: Provides the high-level `ThresholdOptimizer` class with scikit-learn style fit/predict API. The predict method for multiclass implements a sophisticated decision rule: if multiple classes exceed their thresholds, predict the one with highest probability; if none exceed thresholds, fall back to argmax.

**cv.py**: Cross-validation utilities that work with the threshold optimization functions to provide robust threshold estimates.

### Input Validation
The library includes comprehensive input validation that ensures:
- Class labels are consecutive integers starting from 0 (required for OvR)
- No NaN or infinite values in probability arrays
- Matching dimensions between labels and probabilities
- Proper array shapes for binary (1D) vs multiclass (2D) inputs

### Prediction Logic for Multiclass
The multiclass prediction strategy balances threshold-based decisions with practical fallbacks:
1. Apply per-class thresholds to get binary predictions for each class
2. For each sample, if multiple classes are above threshold, predict the one with highest probability
3. If no classes are above threshold, predict the class with highest probability (standard argmax)

This approach maintains the benefits of optimized thresholds while ensuring every sample gets a prediction.