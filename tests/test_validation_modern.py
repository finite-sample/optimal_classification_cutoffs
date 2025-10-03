"""Comprehensive tests for the modern validation system.

NOTE: This test file is deprecated after the validation system simplification.
The new validation system uses simple, direct functions that raise ValueError
immediately instead of collecting errors. See test_validation.py for the
tests that cover the new system.
"""

# This file is temporarily disabled as the validation system has been simplified.
# The complex ValidationResult/ValidatedData classes have been removed in favor
# of simple functions that raise errors immediately.

import pytest
pytest.skip("Test file deprecated after validation simplification", allow_module_level=True)

# import warnings
# import numpy as np

# Commented out - these classes no longer exist:
# from optimal_cutoffs.validation import (
#     ProblemType,
#     ValidatedData,
#     ValidationError,
#     ValidationResult,
#     validate_array_properties,
#     validate_binary_inputs,
#     validate_binary_labels,
#     validate_choice,
#     validate_multiclass_inputs,
#     validate_multiclass_labels,
#     validate_probabilities,
#     validate_sample_weights,
# )


class TestValidationResult:
    """Test the ValidationResult class."""

    def test_basic_validation_result(self):
        """Test basic ValidationResult functionality."""
        result = ValidationResult()
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult()
        result.add_error("field1", "error message 1")
        result.add_error("field2", "error message 2")
        
        assert not result.is_valid
        assert len(result.errors) == 2
        assert result.errors[0].field == "field1"
        assert result.errors[0].message == "error message 1"
        assert result.errors[0].severity == "error"

    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult()
        result.add_warning("field1", "warning message")
        
        assert result.is_valid  # Warnings don't make it invalid
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"

    def test_raise_if_invalid(self):
        """Test error raising."""
        result = ValidationResult()
        result.add_error("field1", "error 1")
        result.add_error("field2", "error 2")
        
        with pytest.raises(ValueError, match="Validation failed"):
            result.raise_if_invalid()

    def test_emit_warnings(self):
        """Test warning emission."""
        result = ValidationResult()
        result.add_warning("field1", "warning message")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result.emit_warnings()
            
            assert len(w) == 1
            assert "field1: warning message" in str(w[0].message)


class TestValidationFunctions:
    """Test individual validation functions."""

    def test_validate_array_properties_basic(self):
        """Test basic array validation."""
        arr = np.array([1, 2, 3, 4])
        result = validate_array_properties(arr, "test_array")
        
        assert result.is_valid
        
    def test_validate_array_properties_dimensions(self):
        """Test dimension validation."""
        arr = np.array([[1, 2], [3, 4]])
        
        # Should pass for 2D requirement
        result = validate_array_properties(arr, "test", expected_ndim=2)
        assert result.is_valid
        
        # Should fail for 1D requirement
        result = validate_array_properties(arr, "test", expected_ndim=1)
        assert not result.is_valid
        assert "must be 1D, got 2D" in result.errors[0].message

    def test_validate_array_properties_finite(self):
        """Test finite value validation."""
        # Valid finite array
        arr = np.array([1.0, 2.0, 3.0])
        result = validate_array_properties(arr, "test", require_finite=True)
        assert result.is_valid
        
        # Invalid with NaN
        arr_nan = np.array([1.0, np.nan, 3.0])
        result = validate_array_properties(arr_nan, "test", require_finite=True)
        assert not result.is_valid
        assert "contains NaN or infinite" in result.errors[0].message
        
        # Invalid with infinity
        arr_inf = np.array([1.0, np.inf, 3.0])
        result = validate_array_properties(arr_inf, "test", require_finite=True)
        assert not result.is_valid

    def test_validate_array_properties_range(self):
        """Test value range validation."""
        arr = np.array([0.2, 0.5, 0.8])
        
        # Should pass for valid range
        result = validate_array_properties(arr, "test", min_value=0.0, max_value=1.0)
        assert result.is_valid
        
        # Should fail for values outside range
        arr_invalid = np.array([-0.1, 0.5, 1.1])
        result = validate_array_properties(arr_invalid, "test", min_value=0.0, max_value=1.0)
        assert not result.is_valid
        assert len(result.errors) == 2  # Both min and max violations

    def test_validate_array_properties_empty(self):
        """Test empty array validation."""
        arr = np.array([])
        
        # Should fail when empty not allowed
        result = validate_array_properties(arr, "test", allow_empty=False)
        assert not result.is_valid
        assert "cannot be empty" in result.errors[0].message
        
        # Should pass when empty allowed
        result = validate_array_properties(arr, "test", allow_empty=True)
        assert result.is_valid

    def test_validate_binary_labels(self):
        """Test binary label validation."""
        # Valid binary labels
        labels = np.array([0, 1, 0, 1])
        result = validate_binary_labels(labels)
        assert result.is_valid
        
        # Invalid labels with values outside {0, 1}
        labels_invalid = np.array([0, 1, 2])
        result = validate_binary_labels(labels_invalid)
        assert not result.is_valid
        assert "must be binary" in result.errors[0].message

    def test_validate_multiclass_labels(self):
        """Test multiclass label validation."""
        # Valid consecutive labels
        labels = np.array([0, 1, 2, 0, 1])
        result = validate_multiclass_labels(labels, require_consecutive=True)
        assert result.is_valid
        
        # Invalid non-consecutive labels
        labels_invalid = np.array([1, 2, 1])  # Missing 0
        result = validate_multiclass_labels(labels_invalid, require_consecutive=True)
        assert not result.is_valid
        assert "consecutive integers from 0" in result.errors[0].message
        
        # Non-integer labels
        labels_float = np.array([0.0, 1.5, 2.0])
        result = validate_multiclass_labels(labels_float)
        assert not result.is_valid
        assert "must be integers" in result.errors[0].message

    def test_validate_probabilities_binary(self):
        """Test binary probability validation."""
        # Valid probabilities
        probs = np.array([0.2, 0.8, 0.3, 0.7])
        result = validate_probabilities(probs, multiclass=False)
        assert result.is_valid
        
        # Invalid probabilities outside [0, 1]
        probs_invalid = np.array([-0.1, 0.8, 1.2, 0.7])
        result = validate_probabilities(probs_invalid, multiclass=False)
        assert not result.is_valid

    def test_validate_probabilities_multiclass(self):
        """Test multiclass probability validation."""
        # Valid normalized probabilities
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        result = validate_probabilities(probs, multiclass=True)
        assert result.is_valid
        
        # Invalid probabilities that don't sum to 1
        probs_unnorm = np.array([[0.5, 0.3, 0.1], [0.2, 0.7, 0.2]])
        result = validate_probabilities(probs_unnorm, multiclass=True, check_sum=True)
        assert result.is_valid  # Should still be valid but emit warning
        assert len(result.warnings) == 1
        assert "don't sum to 1.0" in result.warnings[0].message

    def test_validate_sample_weights(self):
        """Test sample weight validation."""
        n_samples = 4
        
        # Valid weights
        weights = np.array([1.0, 2.0, 0.5, 1.5])
        result = validate_sample_weights(weights, n_samples)
        assert result.is_valid
        
        # Invalid: wrong length
        weights_wrong_length = np.array([1.0, 2.0])
        result = validate_sample_weights(weights_wrong_length, n_samples)
        assert not result.is_valid
        assert "expected length 4, got 2" in result.errors[0].message
        
        # Invalid: negative weights
        weights_negative = np.array([1.0, -1.0, 0.5, 1.5])
        result = validate_sample_weights(weights_negative, n_samples)
        assert not result.is_valid
        assert "below minimum 0.0" in result.errors[0].message
        
        # Invalid: all zero weights
        weights_zero = np.array([0.0, 0.0, 0.0, 0.0])
        result = validate_sample_weights(weights_zero, n_samples)
        assert not result.is_valid
        assert "cannot sum to zero" in result.errors[0].message


class TestProblemType:
    """Test problem type inference."""

    def test_binary_inference(self):
        """Test binary problem type inference."""
        labels = np.array([0, 1, 0, 1])
        
        # 1D predictions - binary
        predictions = np.array([0.2, 0.8, 0.3, 0.7])
        problem_type = ProblemType.infer(labels, predictions)
        assert problem_type == ProblemType.BINARY
        
        # 2D predictions with 2 classes - binary
        predictions_2d = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
        problem_type = ProblemType.infer(labels, predictions_2d)
        assert problem_type == ProblemType.BINARY

    def test_multiclass_inference(self):
        """Test multiclass problem type inference."""
        labels = np.array([0, 1, 2, 0, 1])
        predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], 
                               [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
        
        problem_type = ProblemType.infer(labels, predictions)
        assert problem_type == ProblemType.MULTICLASS


class TestValidatedData:
    """Test the ValidatedData class."""

    def test_create_binary_data(self):
        """Test creating validated binary data."""
        labels = [0, 1, 0, 1]
        predictions = [0.2, 0.8, 0.3, 0.7]
        
        data = ValidatedData.create(labels, predictions)
        
        assert data.problem_type == ProblemType.BINARY
        assert data.n_classes == 2
        assert data.n_samples == 4
        assert data.labels.dtype == np.int32
        assert data.predictions.dtype == np.float64
        assert data.weights is None

    def test_create_multiclass_data(self):
        """Test creating validated multiclass data."""
        labels = [0, 1, 2, 0, 1]
        predictions = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], 
                      [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]
        
        data = ValidatedData.create(labels, predictions)
        
        assert data.problem_type == ProblemType.MULTICLASS
        assert data.n_classes == 3
        assert data.n_samples == 5
        assert data.labels.dtype == np.int32
        assert data.predictions.dtype == np.float64

    def test_create_with_weights(self):
        """Test creating data with sample weights."""
        labels = [0, 1, 0, 1]
        predictions = [0.2, 0.8, 0.3, 0.7]
        weights = [1.0, 2.0, 0.5, 1.5]
        
        data = ValidatedData.create(labels, predictions, weights)
        
        assert data.weights is not None
        assert len(data.weights) == 4
        assert data.weights.dtype == np.float64

    def test_create_with_invalid_data(self):
        """Test that invalid data raises errors."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="length mismatch"):
            ValidatedData.create([0, 1], [0.2, 0.8, 0.3])
        
        # Invalid labels for binary
        with pytest.raises(ValueError, match="must be binary"):
            ValidatedData.create([0, 1, 2], [0.2, 0.8, 0.3])
        
        # Invalid probabilities
        with pytest.raises(ValueError, match="above maximum"):
            ValidatedData.create([0, 1], [0.2, 1.5])

    def test_create_with_custom_dtypes(self):
        """Test creating data with custom dtypes."""
        labels = [0, 1, 0, 1]
        predictions = [0.2, 0.8, 0.3, 0.7]
        
        data = ValidatedData.create(
            labels, predictions, 
            dtype_labels=np.int8,
            dtype_predictions=np.float32
        )
        
        assert data.labels.dtype == np.int8
        assert data.predictions.dtype == np.float32

    def test_create_multiclass_non_consecutive(self):
        """Test multiclass with non-consecutive labels fails by default."""
        labels = [1, 2, 1, 2]  # Missing 0
        predictions = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
        
        with pytest.raises(ValueError, match="consecutive integers from 0"):
            ValidatedData.create(labels, predictions)


class TestConvenienceFunctions:
    """Test convenience validation functions."""

    def test_validate_binary_inputs(self):
        """Test binary input validation convenience function."""
        labels = [0, 1, 0, 1]
        scores = [0.2, 0.8, 0.3, 0.7]
        weights = [1.0, 2.0, 0.5, 1.5]
        
        validated_labels, validated_scores, validated_weights = validate_binary_inputs(
            labels, scores, weights
        )
        
        assert validated_labels.dtype == np.int8
        assert validated_scores.dtype == np.float64
        assert validated_weights.dtype == np.float64
        assert len(validated_labels) == 4

    def test_validate_binary_inputs_no_weights(self):
        """Test binary validation without weights."""
        labels = [0, 1, 0, 1]
        scores = [0.2, 0.8, 0.3, 0.7]
        
        validated_labels, validated_scores, validated_weights = validate_binary_inputs(
            labels, scores
        )
        
        assert validated_weights is None

    def test_validate_binary_inputs_scores(self):
        """Test binary validation with scores outside [0,1]."""
        labels = [0, 1, 0, 1]
        scores = [-1.0, 2.0, -0.5, 1.5]
        
        # Should fail with require_proba=True (default)
        with pytest.raises(ValueError, match="above maximum"):
            validate_binary_inputs(labels, scores, require_proba=True)
        
        # The new validation system always validates probability range
        # This test needs to be updated to reflect current behavior
        # For now, test that it properly validates the range
        with pytest.raises(ValueError, match="above maximum"):
            validate_binary_inputs(labels, scores, require_proba=False)

    def test_validate_multiclass_inputs(self):
        """Test multiclass input validation convenience function."""
        labels = [0, 1, 2, 0, 1]
        predictions = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], 
                      [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]
        weights = [1.0, 2.0, 0.5, 1.5, 1.2]
        
        validated_labels, validated_preds, validated_weights = validate_multiclass_inputs(
            labels, predictions, weights
        )
        
        assert validated_labels.dtype == np.int32
        assert validated_preds.dtype == np.float64
        assert validated_weights.dtype == np.float64
        assert len(validated_labels) == 5

    def test_validate_choice(self):
        """Test choice validation function."""
        # Valid choice
        result = validate_choice("option1", {"option1", "option2", "option3"}, "test")
        assert result == "option1"
        
        # Invalid choice
        with pytest.raises(ValueError, match="Invalid test 'invalid'"):
            validate_choice("invalid", {"option1", "option2"}, "test")


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_empty_arrays(self):
        """Test validation with empty arrays."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ValidatedData.create([], [])

    def test_mismatched_shapes(self):
        """Test validation with mismatched array shapes."""
        # Wrong prediction dimensions for multiclass
        labels = [0, 1, 2]
        predictions = [0.5, 0.3, 0.8]  # 1D instead of 2D
        
        # The validation system detects this as binary (1D predictions)
        # and then fails because labels are not binary
        with pytest.raises(ValueError, match="must be binary"):
            ValidatedData.create(labels, predictions)

    def test_insufficient_classes(self):
        """Test validation with insufficient number of classes."""
        # Only one class in labels but predictions suggest more
        labels = [0, 0, 0]
        predictions = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        
        # This should actually work - the labels are consistent with binary classification
        data = ValidatedData.create(labels, predictions)
        assert data.problem_type == ProblemType.BINARY

    def test_extreme_values(self):
        """Test validation with extreme but valid values."""
        labels = [0, 1, 0, 1]
        
        # Edge case probabilities
        predictions = [0.0, 1.0, 0.0000001, 0.9999999]
        data = ValidatedData.create(labels, predictions)
        assert data.problem_type == ProblemType.BINARY

    def test_large_datasets(self):
        """Test validation doesn't break with larger datasets."""
        n_samples = 10000
        n_classes = 5
        
        labels = np.random.randint(0, n_classes, n_samples)
        predictions = np.random.rand(n_samples, n_classes)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sum warnings for this test
            data = ValidatedData.create(labels, predictions)
            
        assert data.n_samples == n_samples
        assert data.n_classes == n_classes

    def test_dtype_conversion_edge_cases(self):
        """Test edge cases in dtype conversion."""
        # Very large integers that might cause overflow
        labels = [0, 1, 0, 1]
        predictions = [0.2, 0.8, 0.3, 0.7]
        
        # Should handle conversion gracefully
        data = ValidatedData.create(labels, predictions, dtype_labels=np.int8)
        assert data.labels.dtype == np.int8

    def test_warning_emission(self):
        """Test that warnings are properly emitted."""
        labels = [0, 1, 2]
        # Probabilities that don't sum to 1
        predictions = [[0.5, 0.3, 0.1], [0.2, 0.7, 0.2], [0.8, 0.1, 0.2]]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ValidatedData.create(labels, predictions)
            
            # Should emit warning about probabilities not summing to 1
            assert len(w) == 1
            assert "don't sum to 1.0" in str(w[0].message)


class TestPerformanceCharacteristics:
    """Test performance aspects of validation."""

    def test_validation_efficiency(self):
        """Test that validation is reasonably efficient."""
        import time
        
        # Large but not huge dataset
        n_samples = 50000
        labels = np.random.randint(0, 2, n_samples)
        predictions = np.random.rand(n_samples)
        
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ValidatedData.create(labels, predictions)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0

    def test_memory_efficiency(self):
        """Test that validation doesn't create unnecessary copies."""
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        predictions = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
        
        data = ValidatedData.create(labels, predictions)
        
        # Should use the same arrays if dtypes match
        # (Note: this might create copies for validation, but final result should be efficient)
        assert data.labels.dtype == labels.dtype
        assert data.predictions.dtype == predictions.dtype


if __name__ == "__main__":
    pytest.main([__file__])