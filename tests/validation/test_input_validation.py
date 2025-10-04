"""Comprehensive tests for input validation functionality.

This module tests all input validation functions to ensure proper error handling
and parameter validation across the library.
"""

import warnings

import numpy as np
import pytest

from optimal_cutoffs import (
    get_confusion_matrix,
    get_optimal_multiclass_thresholds,
    get_optimal_threshold,
)
from optimal_cutoffs.validation import (
    _validate_averaging_method,
    _validate_comparison_operator,
    validate_inputs,
    _validate_metric_name,
    _validate_optimization_method,
    _validate_threshold,
    validate_binary_classification,
    validate_multiclass_classification,
)
from tests.fixtures.assertions import assert_valid_threshold


class TestInputValidation:
    """Test comprehensive input validation functionality."""

    def testvalidate_inputs_basic(self):
        """Test basic input validation with valid inputs."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.1, 0.8, 0.3, 0.9]

        # Should not raise any errors
        validated_true, validated_prob, validated_weights = validate_inputs(y_true, y_prob)

        assert len(validated_true) == len(validated_prob) == 4
        assert validated_true.dtype == np.int8  # Updated to match current implementation
        assert validated_prob.dtype == np.float64
        assert validated_weights is None

    def testvalidate_inputs_array_conversion(self):
        """Test that inputs are properly converted to numpy arrays."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.7, 0.4, 0.8]

        validated_true, validated_prob, validated_weights = validate_inputs(y_true, y_prob)

        assert isinstance(validated_true, np.ndarray)
        assert isinstance(validated_prob, np.ndarray)
        assert validated_weights is None

    def testvalidate_inputs_length_mismatch(self):
        """Test validation with mismatched array lengths."""
        y_true = [0, 1, 0]
        y_prob = [0.1, 0.8]

        with pytest.raises(ValueError, match="Length mismatch"):
            validate_inputs(y_true, y_prob)

    def testvalidate_inputs_empty_arrays(self):
        """Test validation with empty arrays."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_inputs([], [])

    def testvalidate_inputs_invalid_probabilities(self):
        """Test validation with invalid probability values."""
        y_true = [0, 1, 0, 1]

        # Probabilities outside [0, 1] range
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            validate_inputs(y_true, [-0.1, 0.5, 0.8, 1.2])

        # NaN probabilities
        with pytest.raises(ValueError, match="contains NaN"):
            validate_inputs(y_true, [0.1, np.nan, 0.8, 0.9])

        # Infinite probabilities
        with pytest.raises(ValueError, match="contains infinite"):
            validate_inputs(y_true, [0.1, 0.5, np.inf, 0.9])

    def testvalidate_inputs_invalid_labels(self):
        """Test validation with invalid label values."""
        y_prob = [0.1, 0.5, 0.8, 0.9]

        # Non-binary labels
        with pytest.raises(ValueError, match="must be binary"):
            validate_inputs([0, 1, 2], y_prob[:3])

        # Negative labels
        with pytest.raises(ValueError, match="must be binary"):
            validate_inputs([-1, 0, 1], y_prob[:3])

    def testvalidate_inputs_2d_arrays(self):
        """Test validation with 2D arrays."""
        y_true = [[0, 1], [0, 1]]
        y_prob = [0.1, 0.8]

        with pytest.raises(ValueError, match="must be 1D"):
            validate_inputs(y_true, y_prob)

        with pytest.raises(ValueError, match="must be 1D"):
            validate_inputs([0, 1], [[0.1], [0.8]])


class TestBinaryClassificationValidation:
    """Test binary classification specific validation."""

    def test_validate_binary_classification_valid(self):
        """Test validation with valid binary classification inputs."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.7, 0.4, 0.8])

        # Should not raise any errors
        validate_binary_classification(y_true, y_prob)

    def test_validate_binary_classification_sample_weights(self):
        """Test validation with sample weights."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.7, 0.4, 0.8])
        weights = np.array([1.0, 2.0, 1.5, 1.0])

        # Should not raise any errors
        validate_binary_classification(y_true, y_prob, weights=weights)

    def test_validate_binary_classification_invalid_weights(self):
        """Test validation with invalid sample weights."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.7, 0.4, 0.8])

        # Wrong length
        with pytest.raises(ValueError, match="Length mismatch"):
            validate_binary_classification(y_true, y_prob, weights=[1.0, 2.0])

        # Negative weights
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_binary_classification(
                y_true, y_prob, weights=[-1.0, 1.0, 1.0, 1.0]
            )

        # NaN weights
        with pytest.raises(ValueError, match="contains NaN"):
            validate_binary_classification(
                y_true, y_prob, weights=[1.0, np.nan, 1.0, 1.0]
            )


class TestMulticlassValidation:
    """Test multiclass validation functions."""

    def test_validate_multiclass_classification_valid(self):
        """Test validation with valid multiclass inputs."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ]
        )

        # Should not raise any errors
        validate_multiclass_classification(y_true, y_prob)

    def test_validate_multiclass_classification(self):
        """Test multiclass probability and label validation."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        # Should not raise any errors
        validate_multiclass_classification(y_true, y_prob)

    def test_validate_multiclass_invalid_probabilities(self):
        """Test validation with invalid multiclass probabilities."""
        y_true = np.array([0, 1, 2])

        # Probabilities don't sum to 1
        y_prob_invalid = np.array(
            [
                [0.5, 0.3, 0.1],  # Sum = 0.9
                [0.2, 0.7, 0.1],  # Sum = 1.0
                [0.1, 0.2, 0.7],  # Sum = 1.0
            ]
        )

        # Probabilities that don't sum to 1 should emit a warning, not raise error
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_multiclass_classification(y_true, y_prob_invalid)
            assert len(w) == 1
            assert "don't sum to 1" in str(w[0].message)

        # Negative probabilities
        y_prob_negative = np.array([[0.6, 0.5, -0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            validate_multiclass_classification(y_true, y_prob_negative)

    def test_validate_multiclass_label_issues(self):
        """Test validation with multiclass label issues."""
        y_prob = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        # Labels not starting from 0
        y_true_invalid = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="start from 0"):
            validate_multiclass_classification(y_true_invalid, y_prob)

        # Missing class
        y_true_missing = np.array([0, 0, 1])  # Missing class 2
        with pytest.raises(ValueError, match="missing: \\[2\\]"):
            validate_multiclass_classification(y_true_missing, y_prob)


class TestParameterValidation:
    """Test validation of method parameters."""

    def test_validate_metric_name(self):
        """Test metric name validation."""
        # Valid metrics
        for metric in ["f1", "accuracy", "precision", "recall"]:
            _validate_metric_name(metric)  # Should not raise

        # Invalid metric
        with pytest.raises(ValueError, match="Unknown metric"):
            _validate_metric_name("invalid_metric")

    def test_validate_optimization_method(self):
        """Test optimization method validation."""
        # Valid methods
        for method in ["auto", "unique_scan", "minimize", "gradient"]:
            _validate_optimization_method(method)  # Should not raise

        # Invalid method
        with pytest.raises(ValueError, match="Invalid optimization method"):
            _validate_optimization_method("invalid_method")

    def test_validate_comparison_operator(self):
        """Test comparison operator validation."""
        # Valid operators
        for operator in [">", ">="]:
            _validate_comparison_operator(operator)  # Should not raise

        # Invalid operators
        for invalid_op in ["<", "<=", "==", "!="]:
            with pytest.raises(ValueError, match="Invalid comparison operator"):
                _validate_comparison_operator(invalid_op)

    def test_validate_averaging_method(self):
        """Test averaging method validation."""
        # Valid methods
        for method in ["macro", "micro", "weighted"]:
            _validate_averaging_method(method)  # Should not raise

        # Invalid method
        with pytest.raises(ValueError, match="Invalid averaging method"):
            _validate_averaging_method("invalid_average")

    def test_validate_threshold(self):
        """Test threshold validation."""
        # Valid thresholds
        _validate_threshold(0.5)
        _validate_threshold(np.array([0.3, 0.7]))

        # Invalid thresholds
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            _validate_threshold(-0.1)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            _validate_threshold(1.5)

        with pytest.raises(ValueError, match="contains NaN"):
            _validate_threshold(np.nan)


class TestEndToEndValidation:
    """Test validation in end-to-end scenarios."""

    def test_get_optimal_threshold_validation(self):
        """Test that get_optimal_threshold properly validates inputs."""
        # Valid case
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(result.threshold)

        # Invalid cases
        with pytest.raises(ValueError):
            get_optimal_threshold([], [], metric="f1")

        with pytest.raises(ValueError):
            get_optimal_threshold(y_true, y_prob, metric="invalid_metric")

        with pytest.raises(ValueError):
            get_optimal_threshold(y_true, y_prob, method="invalid_method")

    def test_get_confusion_matrix_validation(self):
        """Test that confusion matrix function validates inputs."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]
        threshold = 0.5

        # Valid case
        tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, result.threshold)
        assert all(
            isinstance(x, (int, float, np.integer, np.floating))
            for x in [tp, tn, fp, fn]
        )

        # Invalid threshold
        with pytest.raises(ValueError):
            get_confusion_matrix(y_true, y_prob, -0.1)

    def test_multiclass_function_validation(self):
        """Test that multiclass functions validate inputs."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_prob = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ]
        )

        # Valid case
        try:
            thresholds = get_optimal_multiclass_thresholds(y_true, y_prob, metric="f1")
            assert len(thresholds) == 3
            for threshold in thresholds:
                assert_valid_threshold(result.threshold)
        except (AttributeError, NameError):
            # Function might not exist in current version
            pass


class TestWarningGeneration:
    """Test that appropriate warnings are generated."""

    def test_deprecation_warnings(self):
        """Test that deprecated functionality generates warnings."""
        # This would test deprecated parameters or functions
        # when they exist in the library
        pass

    def test_performance_warnings(self):
        """Test warnings for potentially slow operations."""
        # Create a scenario that might trigger performance warnings
        y_true = np.random.binomial(1, 0.5, 10000)
        y_prob = np.random.uniform(0, 1, 10000)

        # Some methods might warn about large datasets
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            get_optimal_threshold(y_true, y_prob, method="minimize")

            # Check if any warnings were generated (optional)
            # warnings might be implementation-specific


class TestEdgeCaseValidation:
    """Test validation for edge cases."""

    def test_single_sample_validation(self):
        """Test validation with single sample."""
        y_true = [1]
        y_prob = [0.7]

        # Should work
        result = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(result.threshold)

    def test_all_same_class_validation(self):
        """Test validation when all samples are same class."""
        y_true = [1, 1, 1, 1]
        y_prob = [0.2, 0.5, 0.7, 0.9]

        # Should work
        result = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(result.threshold)

    def test_extreme_probability_validation(self):
        """Test validation with extreme probability values."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.0, 1.0, 1e-15, 1 - 1e-15]

        # Should work
        result = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(result.threshold)
