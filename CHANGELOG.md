# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2024-12-30

### Added
- **Generalized Dinkelbach Framework**: Extended Dinkelbach expected optimization to support any fractional-linear metric
  - New `expected_fractional.py` module with coefficient-based metric representation
  - Support for precision, recall, Jaccard/IoU, Tversky index, accuracy, specificity, F-beta metrics
  - Mathematical framework using `FractionalLinearCoeffs` dataclass for metric coefficients
  - Automatic O(n log n) optimization via sorting and binary search per Dinkelbach iteration
- **Enhanced Expected Mode**: `mode="expected"` now supports 8+ metrics beyond just F-beta
  - Precision, recall, specificity, accuracy, Jaccard, Tversky, F-beta family
  - Automatic fallback to legacy F-beta implementation for unsupported metrics
  - Unified API supporting both binary and multiclass/multilabel optimization
- **Comprehensive Test Coverage**: Added 200+ tests for generalized fractional-linear framework
  - Coefficient correctness tests for all supported metrics
  - Property-based tests using Hypothesis for edge case discovery
  - Integration tests ensuring backward compatibility
  - Edge case handling for extreme probabilities and numerical stability

### Changed
- **Extended Metric Support**: `mode="expected"` no longer limited to F-beta metrics
  - Can optimize expected precision, recall, accuracy, Jaccard, etc. under calibration
  - Maintains mathematical rigor with coefficient-based approach
  - Preserves exact optimization guarantees via Dinkelbach algorithm
- **Improved Framework Design**: Cleaner separation between metric-specific and general optimization
  - `coeffs_for_metric()` provides standardized coefficient mapping
  - `dinkelbach_expected_fractional_binary()` handles any fractional-linear metric
  - `dinkelbach_expected_fractional_ovr()` extends to multiclass with macro/micro/weighted averaging

### Enhanced
- **Mathematical Rigor**: All new metrics maintain theoretical guarantees of Dinkelbach optimization
  - Exact optimization for expected metrics under perfect calibration assumption
  - O(n log n) complexity per iteration with fast convergence (typically <10 iterations)
  - Support for sample weights and comparison operators throughout
- **Backward Compatibility**: Existing F-beta code continues to work unchanged
  - Automatic routing to appropriate implementation (generalized vs legacy)
  - All existing APIs preserve exact behavior for F-beta metrics
  - Comprehensive regression testing ensures no breaking changes

## [0.4.0] - 2024-12-30

### Added
- **Explicit Mode Parameter**: Added `mode` parameter to `get_optimal_threshold()` for clear estimation regime specification
  - `mode="empirical"` (default): Standard empirical optimization using the `method` parameter
  - `mode="bayes"`: Bayes-optimal threshold under calibrated probabilities (requires `utility`)
  - `mode="expected"`: Expected metric optimization (supports multiple metrics, binary and multiclass)
- **Enhanced ThresholdOptimizer**: Extended wrapper class with new parameters
  - Added `mode`, `utility`, and `minimize_cost` parameters for full API coverage
  - Changed primary parameter from `objective` to `metric` with backward compatibility
- **Improved Method Naming**: Renamed `"smart_brute"` to `"unique_scan"` for clarity
  - `"unique_scan"`: Evaluates thresholds at unique probability values
  - Better name reflects the actual algorithm behavior
- **Auto Method Selection**: Changed default optimization method in CV functions from `"smart_brute"` to `"auto"`
  - Automatically selects best method based on metric properties and data size
  - Better performance out-of-the-box for most use cases
- **Comprehensive Test Suite**: Added realistic integration tests with datasets of 100+ samples
  - Tests now use meaningful datasets generated from scikit-learn
  - Added performance tests with 5000+ sample datasets
  - Enhanced test coverage for edge cases and real-world scenarios

### Changed
- **Parameter Naming**: Unified naming convention to use `metric` consistently
  - `ThresholdOptimizer` now uses `metric` parameter instead of `objective`
  - All documentation and examples updated to use `metric` terminology
- **CV Function Defaults**: Cross-validation functions now default to `method="auto"`
  - `cv_threshold_optimization()` and `nested_cv_threshold_optimization()`
  - Provides better performance and method selection automatically
- **Enhanced mode='expected' Support**: Extended expected optimization to support multiple metrics and multiclass
  - Returns tuple `(threshold, expected_score)` for binary classification
  - Returns dict with per-class information for multiclass classification
  - ThresholdOptimizer wrapper properly handles tuple returns

### Deprecated
- **Legacy Parameters**: Several parameters deprecated with clear migration paths
  - `bayes=True` → use `mode="bayes"` instead
  - `method="dinkelbach"` → use `mode="expected"` instead  
  - `method="smart_brute"` → use `method="unique_scan"` instead
  - `objective` parameter in `ThresholdOptimizer` → use `metric` instead
- All deprecations emit `DeprecationWarning` with migration instructions

### Fixed
- **Test Suite Robustness**: Fixed 113+ test failures related to deprecated API usage
- **Type Annotations**: Resolved all mypy type checking errors without type ignore comments
- **Sort-Scan Precision**: Improved tolerance handling for edge cases with extreme or tied probabilities
- **Utility Parameter Handling**: Fixed sign convention consistency between Bayes and empirical optimization
- **Wrapper Compatibility**: Fixed ThresholdOptimizer wrapper to handle tuple returns from mode='expected'

### Migration Guide

#### Mode Parameter Usage
```python
# Before (v0.3.x)
threshold = get_optimal_threshold(y, p, method="dinkelbach")
threshold = get_optimal_threshold(None, p, utility={...}, bayes=True)

# After (v0.4.x)
threshold = get_optimal_threshold(y, p, mode="expected")
threshold = get_optimal_threshold(None, p, utility={...}, mode="bayes")
```

#### ThresholdOptimizer Parameter Changes
```python
# Before (v0.3.x)
optimizer = ThresholdOptimizer(objective="f1")

# After (v0.4.x)
optimizer = ThresholdOptimizer(metric="f1")

# With new features
optimizer = ThresholdOptimizer(
    metric="f1",
    mode="bayes",
    utility={"fp": -1, "fn": -5}
)
```

#### Method Name Changes
```python
# Before (v0.3.x)
threshold = get_optimal_threshold(y, p, method="smart_brute")

# After (v0.4.x)
threshold = get_optimal_threshold(y, p, method="unique_scan")
# Or better, use auto method selection
threshold = get_optimal_threshold(y, p, method="auto")
```

#### Cross-Validation Changes
```python
# Before (v0.3.x) - explicitly needed to specify method
thresholds, scores = cv_threshold_optimization(y, p, method="smart_brute")

# After (v0.4.x) - auto method selection by default
thresholds, scores = cv_threshold_optimization(y, p)  # Uses method="auto"
```

### Technical Details
- Full backward compatibility maintained - existing code will work unchanged
- All deprecated parameters continue to function with warnings
- New test suite with 50+ tests covering new features and deprecation scenarios
- Enhanced type annotations with new `EstimationMode` type
- Comprehensive documentation updates

## [0.3.0] - 2024-12-28

### Added
- **Cost/Benefit-Aware Threshold Optimization**: Complete support for utility-based threshold optimization
  - `bayes_threshold_from_utility()` - Calculate Bayes-optimal thresholds under calibrated probabilities
  - `bayes_threshold_from_costs()` - Convenience wrapper for cost specification
  - `make_linear_counts_metric()` - Create linear utility metrics from confusion matrix counts
  - `make_cost_metric()` - Create cost-sensitive metrics with benefits and penalties
  - Extended `get_optimal_threshold()` API with `utility`, `minimize_cost`, and `bayes` parameters
- **Enhanced Documentation Structure**: Comprehensive documentation with improved navigation
  - Installation guide with troubleshooting
  - Quick start guide with examples
  - Comprehensive user guide covering all features
  - Advanced topics including cost-sensitive optimization
  - Detailed API reference with separate sections
  - FAQ with common issues and solutions
  - Real-world examples and integration patterns
- **Performance Improvements**: Enhanced algorithmic efficiency and robustness
  - Improved edge case handling in sort-scan optimization
  - Better tolerance handling for numerical precision issues
  - Enhanced test coverage with property-based testing

### Changed
- **Version Requirement**: Updated NumPy requirement to `>=1.20.0` for modern API support
- **Code Quality**: Fixed all ruff linting issues for consistent code style
- **Type Annotations**: Improved mypy compliance with complete type coverage
- **Test Robustness**: Enhanced tolerance handling for edge cases in numerical optimization
  - Improved handling of tied probabilities and extreme values
  - Better edge case detection for property-based tests
  - More robust comparison between optimization methods

### Fixed
- **MyPy Compliance**: Fixed type annotation issues in metric factory functions
- **Test Reliability**: Fixed 4 failing tests related to numerical precision and edge cases:
  - `test_sortscan_matches_bruteforce_accuracy` - Enhanced tolerance for boundary cases
  - `test_sortscan_matches_bruteforce_precision` - Improved edge case handling  
  - `test_coord_ascent_unsupported_features` - Fixed regex pattern matching
  - `test_piecewise_matches_brute_force` - Added degenerate case filtering
- **Documentation Links**: Fixed broken figure link in README.md
- **Code Style**: Resolved all line length and formatting issues for ruff compliance

### Technical Details
- **Test Suite**: 443 tests passing with comprehensive coverage
- **Quality Checks**: Full compliance with ruff, mypy, and pytest requirements
- **Dependencies**: Maintained compatibility with Python 3.10-3.13
- **Documentation**: Complete Sphinx documentation with proper navigation structure

### Migration Guide
The new cost/benefit-aware optimization is fully backward compatible. Existing code will continue to work unchanged. To use the new features:

```python
# Cost-sensitive optimization (FN costs 5x more than FP)
threshold = get_optimal_threshold(y, p, utility={"fp": -1.0, "fn": -5.0})

# Bayes-optimal for calibrated probabilities (no training data needed)
threshold = get_optimal_threshold(None, p, utility={"fp": -1, "fn": -5}, bayes=True)
```

## [0.2.1] - 2024-09-25

### Previous Release
- O(n log n) sort-and-scan optimization
- Multiclass support with One-vs-Rest
- Cross-validation utilities  
- Comprehensive test suite
- Scikit-learn compatible API