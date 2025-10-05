"""Test fixtures and utilities for optimal cutoff testing.

This module provides standardized data generation, realistic datasets,
assertion helpers, and hypothesis testing strategies for consistent
testing across all test modules.
"""

# Export all data generators (simple numpy-based)
from .data_generators import (
    generate_binary_data,
    generate_multiclass_data,
    generate_calibrated_probabilities,
    generate_tied_probabilities,
)

# Export all realistic datasets (sklearn-based)
from .realistic_datasets import (
    BinaryDataset,
    MulticlassDataset,
    make_realistic_binary_dataset,
    make_imbalanced_binary_dataset,
    make_well_separated_binary_dataset,
    make_overlapping_binary_dataset,
    make_realistic_multiclass_dataset,
    make_calibrated_binary_dataset,
    make_large_binary_dataset,
    # Standard dataset constants
    STANDARD_BINARY,
    IMBALANCED_BINARY,
    WELL_SEPARATED_BINARY,
    OVERLAPPING_BINARY,
    CALIBRATED_BINARY,
    LARGE_BINARY,
    STANDARD_MULTICLASS,
    IMBALANCED_MULTICLASS,
)

# Export assertion helpers
from .assertions import (
    assert_valid_threshold,
    assert_valid_confusion_matrix,
    assert_valid_metric_score,
    assert_monotonic_increase,
    assert_arrays_close,
    assert_probability_matrix_valid,
    assert_labels_valid,
    assert_optimization_successful,
    assert_method_consistency,
)

# Export hypothesis strategies
from .hypothesis_strategies import (
    beta_bernoulli_calibrated,
    labels_binary_like,
    multiclass_labels_and_probs,
    tied_probabilities,
    extreme_probabilities,
    rational_weights,
)

# Make all available at package level
__all__ = [
    # Data generators (simple)
    "generate_binary_data",
    "generate_multiclass_data", 
    "generate_calibrated_probabilities",
    "generate_tied_probabilities",
    # Realistic datasets (sklearn-based)
    "BinaryDataset",
    "MulticlassDataset",
    "make_realistic_binary_dataset",
    "make_imbalanced_binary_dataset", 
    "make_well_separated_binary_dataset",
    "make_overlapping_binary_dataset",
    "make_realistic_multiclass_dataset",
    "make_calibrated_binary_dataset",
    "make_large_binary_dataset",
    # Standard dataset constants
    "STANDARD_BINARY",
    "IMBALANCED_BINARY",
    "WELL_SEPARATED_BINARY", 
    "OVERLAPPING_BINARY",
    "CALIBRATED_BINARY",
    "LARGE_BINARY",
    "STANDARD_MULTICLASS",
    "IMBALANCED_MULTICLASS",
    # Assertion helpers
    "assert_valid_threshold",
    "assert_valid_confusion_matrix",
    "assert_valid_metric_score",
    "assert_monotonic_increase",
    "assert_arrays_close",
    "assert_probability_matrix_valid",
    "assert_labels_valid",
    "assert_optimization_successful",
    "assert_method_consistency",
    # Hypothesis strategies
    "beta_bernoulli_calibrated",
    "labels_binary_like",
    "multiclass_labels_and_probs",
    "tied_probabilities",
    "extreme_probabilities",
    "rational_weights",
]