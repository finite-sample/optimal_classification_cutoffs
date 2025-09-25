# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Version Information
- **Current Version**: 0.2.1 (in development)
- **Python Support**: 3.11, 3.12, 3.13

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
# Run all tests (205 tests as of v0.2.0)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_multiclass.py -v

# Run specific test
python -m pytest tests/test_multiclass.py::test_multiclass_confusion_matrix -v

# Run tests for new features
python -m pytest tests/test_piecewise.py -v              # O(n log n) optimization
python -m pytest tests/test_multiclass_coord.py -v      # Coordinate ascent
python -m pytest tests/test_registry_integration.py -v  # Registry system
python -m pytest tests/test_property_based.py -v        # Property-based tests
```

### Code Quality
```bash
# Run linting (passes with 0 errors)
ruff check optimal_cutoffs/

# Apply code formatting
ruff format optimal_cutoffs/

# Type checking
mypy optimal_cutoffs/
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

### Multiclass Strategies
The library supports two main multiclass approaches:

**One-vs-Rest (OvR)** - Default strategy where each class gets its own optimized threshold:
- Treats each class as a separate binary problem (class vs all others)
- Optimizes thresholds independently per class
- Handles class imbalance better than simple argmax approaches
- Allows different metrics to be optimized for different classes

**Coordinate Ascent** - Advanced strategy for single-label consistency:
- Couples classes through joint assignment using argmax(P - tau)
- Ensures exactly one prediction per sample (single-label)
- Iteratively optimizes per-class thresholds while fixing others
- Better suited for applications requiring strict single-label predictions

### Core Components

**metrics.py**: Contains the enhanced metric registry system and confusion matrix utilities. The `METRIC_REGISTRY` is a global dictionary that allows registration of custom metrics that accept `(tp, tn, fp, fn)` parameters. The `VECTORIZED_REGISTRY` stores optimized vectorized implementations for O(n log n) algorithms. Built-in metrics (F1, accuracy, precision, recall) have both scalar and vectorized versions. For multiclass, special handling ensures proper micro averaging (avoiding TN double-counting in OvR).

**optimizers.py**: Implements threshold optimization algorithms. `get_optimal_threshold()` serves as the main entry point that auto-detects binary vs multiclass and routes accordingly. Optimization methods include:
- "auto" (automatically selects best method, default)
- "sort_scan" (O(n log n) algorithm for piecewise metrics with vectorized implementations)
- "smart_brute" (evaluates all unique probabilities)
- "minimize" (scipy optimization with enhanced fallbacks)
- "gradient" (simple gradient ascent)
- "coord_ascent" (coordinate ascent for coupled multiclass optimization, single-label consistent)

**piecewise.py**: Implements the O(n log n) sort-and-scan optimization kernel for piecewise-constant metrics. This module provides exact optimization using cumulative sums and vectorized operations, achieving significant performance improvements for large datasets.

**multiclass_coord.py**: Implements coordinate-ascent optimization for multiclass thresholds that ensures single-label consistency. Unlike One-vs-Rest approaches, this couples classes through joint assignment using argmax of shifted scores: argmax_j (p_ij - tau_j).

**wrapper.py**: Provides the high-level `ThresholdOptimizer` class with scikit-learn style fit/predict API. The predict method for multiclass implements sophisticated decision rules that vary by optimization method (OvR vs coordinate ascent).

**validation.py**: Comprehensive input validation utilities ensuring robust API behavior with clear error messages.

**types.py**: Type definitions and protocols for better code maintainability and IDE support.

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

## Version 0.2.0 Key Improvements

### Performance Enhancements
- **O(n log n) Sort-and-Scan Algorithm**: New `sort_scan` method provides exact optimization for piecewise-constant metrics (F1, accuracy, precision, recall) with significant performance improvements
- **Vectorized Metric Registry**: Built-in metrics have optimized vectorized implementations for batch processing
- **Auto Method Selection**: The "auto" method intelligently selects the best optimization algorithm based on metric properties

### New Optimization Methods
- **sort_scan**: O(n log n) exact optimization for piecewise metrics
- **coord_ascent**: Coordinate ascent for single-label consistent multiclass predictions
- **Enhanced minimize**: Improved scipy-based optimization with smart fallbacks

### Comparison Operators
- Support for both ">" (exclusive) and ">=" (inclusive) threshold comparisons
- Proper handling of tied probabilities with different comparison operators
- Consistent behavior across all optimization methods

### Robustness & Testing
- **205 comprehensive tests** including property-based testing with Hypothesis
- **Edge case handling** for tied probabilities, extreme class imbalances, and numerical precision
- **Input validation** with clear, informative error messages
- **Sample weight support** across all optimization methods

### Code Quality
- **100% ruff compliance** with comprehensive linting rules (E, W, F, I, B, C4, UP)
- **Type annotations** with mypy support for better IDE integration
- **Comprehensive documentation** with detailed docstrings and examples

### Backward Compatibility
- All existing APIs remain unchanged
- Legacy `get_probability()` function maintained with deprecation warnings
- Automatic fallbacks ensure robust behavior

## Development Notes

### Key Design Principles
- **Performance**: Prioritize O(n log n) algorithms where possible
- **Robustness**: Comprehensive input validation and error handling
- **Flexibility**: Support multiple optimization strategies and metrics
- **Compatibility**: Maintain backward compatibility while adding new features

### Testing Strategy
- **Unit tests**: Core functionality with edge cases
- **Property-based tests**: Mathematical properties using Hypothesis
- **Integration tests**: End-to-end workflows and API compatibility
- **Performance tests**: Algorithmic complexity verification
- **Regression tests**: Prevent breaking changes

### Code Organization Best Practices
- **Modular design**: Separate concerns (optimization, metrics, validation)
- **Registry pattern**: Extensible metric system
- **Type safety**: Comprehensive type annotations
- **Documentation**: Docstrings follow NumPy style
- **Error handling**: Informative error messages with context

### Performance Considerations
- Use `method="auto"` for optimal performance (automatically selects best algorithm)
- For large datasets (n > 10,000), the `sort_scan` method provides significant speedups
- Sample weights are supported but may reduce performance for some methods
- The `coord_ascent` method is currently limited to F1 metric and ">" comparison

### Common Patterns
```python
# Recommended usage pattern
from optimal_cutoffs import get_optimal_threshold

# Binary classification - auto method selection
threshold = get_optimal_threshold(y_true, y_prob, metric="f1", method="auto")

# Multiclass classification - coordinate ascent for single-label consistency  
thresholds = get_optimal_threshold(y_true, y_prob, metric="f1", method="coord_ascent")

# Performance-critical applications
threshold = get_optimal_threshold(y_true, y_prob, metric="f1", method="sort_scan")
```