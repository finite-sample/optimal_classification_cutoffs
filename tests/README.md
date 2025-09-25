# Test Suite Documentation

This directory contains comprehensive tests for the optimal threshold classification library. The test suite is designed with multiple testing strategies to ensure robustness, performance, and mathematical correctness.

## Test Organization

### Core Functionality Tests
- **`test_metrics.py`** - Basic metric functions and registry system
- **`test_optimizers.py`** - Core optimization algorithms and method routing
- **`test_piecewise.py`** - O(n log n) sort-and-scan algorithm (584 lines, comprehensive)
- **`test_multiclass.py`** - Multiclass classification support

### Advanced Features Tests
- **`test_multiclass_coord.py`** - Coordinate ascent optimization for coupled multiclass
- **`test_multiclass_averaging.py`** - Different averaging strategies (macro, micro, weighted)
- **`test_sample_weights.py`** - Sample weight handling across all methods
- **`test_comparison_operators.py`** - ">" vs ">=" threshold comparison handling
- **`test_dinkelbach.py`** - Dinkelbach method for expected F-beta optimization

### Robustness & Edge Case Tests
- **`test_edge_cases.py`** - Comprehensive boundary conditions and extreme scenarios (600+ lines)
- **`test_tied_probabilities.py`** - Tied probability value handling
- **`test_validation.py`** - Input validation and error handling (370+ lines)

### Advanced Testing Approaches
- **`test_property_based.py`** - Property-based testing with Hypothesis (1000+ lines!)
- **`test_cross_method_validation.py`** - Systematic comparison across optimization methods
- **`test_performance_benchmarks.py`** - Performance characteristics and O(n log n) verification

### Integration & System Tests
- **`test_registry_integration.py`** - Metric registry system integration
- **`test_averaging_identities.py`** - Mathematical identity validation
- **`test_regression_minimize_fallback.py`** - Fallback mechanism testing

## Test Statistics

- **Total Files**: 17 test files
- **Total Lines**: ~5,700 lines of test code
- **Total Tests**: 200+ individual test cases
- **Coverage**: Outstanding coverage of all major functionality

## Testing Strategies

### 1. Unit Testing
Standard unit tests for individual functions and methods, ensuring correct behavior under normal conditions.

### 2. Property-Based Testing
Using Hypothesis framework to generate thousands of test cases with random but controlled inputs, verifying mathematical properties hold across all cases.

### 3. Edge Case Testing
Systematic testing of boundary conditions, extreme inputs, and degenerate cases that could break algorithms.

### 4. Performance Testing
Verification of algorithmic complexity guarantees, particularly the O(n log n) sort-scan algorithm performance.

### 5. Integration Testing
Testing interactions between components, method consistency, and end-to-end workflows.

### 6. Regression Testing
Ensuring that changes don't break existing functionality and that performance characteristics are maintained.

## Key Test Categories

### Mathematical Correctness
- **Metric calculations** - Verify F1, accuracy, precision, recall calculations
- **Threshold computation** - Ensure optimal thresholds are found
- **Multiclass averaging** - Test macro, micro, weighted averaging strategies
- **Property invariants** - Mathematical properties that must always hold

### Algorithm Robustness
- **Tied probabilities** - Multiple samples with identical predicted probabilities
- **Extreme imbalance** - 99:1 class ratios and beyond
- **Numerical precision** - Values at machine epsilon, floating-point limits
- **Degenerate cases** - All same class, single samples, empty arrays

### Performance Characteristics
- **Scaling behavior** - O(n log n) complexity verification
- **Memory usage** - Reasonable memory consumption with large datasets
- **Worst-case scenarios** - Performance under adversarial conditions

### Error Handling
- **Input validation** - Clear error messages for invalid inputs
- **Graceful degradation** - Reasonable behavior when optimization is difficult
- **Exception safety** - No crashes or undefined behavior

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Specific Test Categories
```bash
# Core functionality
python -m pytest tests/test_optimizers.py tests/test_metrics.py -v

# Edge cases and robustness
python -m pytest tests/test_edge_cases.py tests/test_tied_probabilities.py -v

# Performance and scaling
python -m pytest tests/test_performance_benchmarks.py -v

# Property-based testing (slow but comprehensive)
python -m pytest tests/test_property_based.py -v
```

### Quick Smoke Test
```bash
python -m pytest tests/test_optimizers.py::TestBasicOptimization -v
```

## Test Design Principles

### 1. Comprehensive Coverage
Every major code path and edge case is tested, with particular attention to:
- Algorithm correctness under all conditions
- Numerical stability and precision
- Error handling and input validation

### 2. Performance Validation
Tests verify not just correctness but also performance characteristics:
- Algorithmic complexity guarantees
- Memory usage patterns
- Scaling behavior

### 3. Mathematical Rigor
Property-based testing ensures mathematical invariants hold:
- Metric calculations are always correct
- Optimal thresholds truly optimize the target metric
- Multiclass strategies maintain mathematical consistency

### 4. Real-World Scenarios
Tests include realistic edge cases encountered in practice:
- Highly imbalanced datasets
- Noisy probability estimates
- Large-scale data processing

### 5. Clear Documentation
Each test file includes comprehensive documentation explaining:
- What is being tested and why
- Expected behaviors and edge cases
- Mathematical properties being verified

## Contributing to Tests

When adding new functionality:

1. **Add unit tests** for the core functionality
2. **Add edge case tests** for boundary conditions
3. **Add property-based tests** for mathematical properties
4. **Add performance tests** if introducing new algorithms
5. **Update documentation** to reflect new test coverage

The test suite is designed to catch regressions early and ensure the library maintains its high standards for correctness, performance, and robustness.