"""Comprehensive tests for input validation functionality."""

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


class TestInputValidation:
    """Test comprehensive input validation functionality."""

    def testvalidate_inputs_basic(self):
        """Test basic input validation with valid inputs."""
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7])

        validated_labels, validated_probs, validated_weights = validate_inputs(
            true_labels, pred_probs
        )

        assert np.array_equal(validated_labels, true_labels)
        assert np.array_equal(validated_probs, pred_probs)
        assert validated_weights is None

    def testvalidate_inputs_empty_arrays(self):
        """Test validation with empty arrays."""
        with pytest.raises(ValueError, match="Labels cannot be empty"):
            validate_inputs([], [0.5])

        with pytest.raises(ValueError, match="Probabilities cannot be empty"):
            validate_inputs([0], [])

    def testvalidate_inputs_dimension_mismatch(self):
        """Test validation with mismatched dimensions."""
        with pytest.raises(ValueError, match="Length mismatch"):
            validate_inputs([0, 1], [0.5])

        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_inputs([0, 1, 2], np.random.rand(2, 3))  # multiclass mismatch: 3 labels vs 2 rows

    def testvalidate_inputs_wrong_dimensions(self):
        """Test validation with wrong array dimensions."""
        with pytest.raises(ValueError, match="Labels must be 1D"):
            validate_inputs(np.array([[0, 1], [1, 0]]), [0.5, 0.8, 0.3, 0.7])

        with pytest.raises(ValueError, match="Invalid prediction array shape"):
            validate_inputs([0, 1], np.random.rand(2, 2, 2))

    def testvalidate_inputs_non_finite_values(self):
        """Test validation with NaN and infinite values."""
        # NaN in true labels
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            validate_inputs([0, np.nan, 1], [0.5, 0.6, 0.7])

        # Infinite in pred_prob
        with pytest.raises(ValueError, match="Probabilities contains infinite values"):
            validate_inputs([0, 1, 0], [0.5, np.inf, 0.7])

    def testvalidate_inputs_binary_labels_requirement(self):
        """Test binary label validation."""
        # Valid binary labels
        validate_inputs([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], require_binary=True)

        # Invalid binary labels - not in {0, 1}
        with pytest.raises(ValueError, match="Labels must be binary \\(0 or 1\\)"):
            validate_inputs([0, 1, 2], [0.1, 0.2, 0.3], require_binary=True)

        # Edge case: only zeros
        validate_inputs([0, 0, 0], [0.1, 0.2, 0.3], require_binary=True)

        # Edge case: only ones
        validate_inputs([1, 1, 1], [0.1, 0.2, 0.3], require_binary=True)

    def testvalidate_inputs_multiclass_labels(self):
        """Test multiclass label validation."""
        # Valid consecutive labels starting from 0
        true_labels = [0, 1, 2, 0, 1, 2]
        pred_probs = np.random.rand(6, 3)
        validate_inputs(true_labels, pred_probs)

        # Invalid: labels outside valid range (has label 3 for 3-class problem)
        with pytest.raises(ValueError, match="Found label 3 but n_classes=3"):
            validate_inputs(
                [0, 2, 3], np.random.rand(3, 3)
            )  # Label 3 invalid for 3 classes

        # Invalid: negative labels
        with pytest.raises(ValueError, match="Labels must be non-negative"):
            validate_inputs([-1, 0, 1], np.random.rand(3, 2))

        # Invalid: non-integer labels
        with pytest.raises(ValueError, match="Labels must be integers"):
            validate_inputs([0.5, 1.0, 1.5], np.random.rand(3, 2))

    def testvalidate_inputs_probability_range(self):
        """Test probability range validation."""
        # Valid probabilities
        validate_inputs([0, 1], [0.0, 1.0], require_proba=True)
        validate_inputs([0, 1], [0.5, 0.7], require_proba=True)

        # Invalid: below 0
        with pytest.raises(ValueError, match="Probabilities must be in \\[0, 1\\]"):
            validate_inputs([0, 1], [-0.1, 0.5], require_proba=True)

        # Invalid: above 1
        with pytest.raises(ValueError, match="Probabilities must be in \\[0, 1\\]"):
            validate_inputs([0, 1], [0.5, 1.1], require_proba=True)

    def testvalidate_inputs_multiclass_probability_sum_warning(self):
        """Test warning for multiclass probabilities that don't sum to 1."""
        true_labels = [0, 1, 2]
        # Probabilities that don't sum to 1
        pred_probs = np.array([[0.5, 0.3, 0.1], [0.2, 0.7, 0.2], [0.8, 0.1, 0.2]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_inputs(true_labels, pred_probs)

            # Should issue a warning about probabilities not summing to 1
            assert len(w) == 1
            assert "don't sum to 1.0" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def testvalidate_inputs_sample_weights(self):
        """Test sample weight validation."""
        true_labels = [0, 1, 0, 1]
        pred_probs = [0.2, 0.8, 0.3, 0.7]

        # Valid sample weights
        sample_weights = [1.0, 2.0, 1.5, 0.5]
        validated_labels, validated_probs, validated_weights = validate_inputs(
            true_labels, pred_probs, weights=sample_weights
        )
        assert validated_weights is not None
        assert np.array_equal(validated_weights, sample_weights)

        # Wrong dimension
        with pytest.raises(ValueError, match="Weights must be 1D"):
            validate_inputs(true_labels, pred_probs, weights=[[1, 2], [3, 4]])

        # Wrong length
        with pytest.raises(ValueError, match="Length mismatch"):
            validate_inputs(true_labels, pred_probs, weights=[1.0, 2.0])

        # NaN values
        with pytest.raises(ValueError, match="Sample weights contains NaN values"):
            validate_inputs(
                true_labels, pred_probs, weights=[1.0, np.nan, 1.5, 0.5]
            )

        # Negative values
        with pytest.raises(ValueError, match="Sample weights must be non-negative"):
            validate_inputs(
                true_labels, pred_probs, weights=[1.0, -1.0, 1.5, 0.5]
            )

        # All zeros
        with pytest.raises(ValueError, match="Sample weights sum to zero"):
            validate_inputs(
                true_labels, pred_probs, weights=[0.0, 0.0, 0.0, 0.0]
            )

    def test_validate_threshold(self):
        """Test threshold validation."""
        # Valid single threshold
        validated = _validate_threshold(0.5)
        assert validated == 0.5  # 0-dimensional array, but equals work

        # Valid array of thresholds
        thresholds = [0.2, 0.5, 0.8]
        validated = _validate_threshold(thresholds, n_classes=3)
        assert np.array_equal(validated, thresholds)

        # Invalid: NaN
        with pytest.raises(ValueError, match="threshold contains NaN"):
            _validate_threshold(np.nan)

        # Invalid: out of range
        with pytest.raises(ValueError, match="Threshold must be in \\[0, 1\\], got range"):
            _validate_threshold(-0.1)

        with pytest.raises(ValueError, match="Threshold must be in \\[0, 1\\], got range"):
            _validate_threshold(1.1)

        # Invalid: wrong length for multiclass
        with pytest.raises(
            ValueError, match="threshold length .* must match number of classes"
        ):
            _validate_threshold([0.5, 0.7], n_classes=3)

        # Invalid: wrong dimension for multiclass
        with pytest.raises(ValueError, match="multiclass threshold must be 1D"):
            _validate_threshold([[0.5, 0.7], [0.3, 0.9]], n_classes=2)

    def test_validate_metric_name(self):
        """Test metric name validation."""
        # Valid metric names (registered by default)
        _validate_metric_name("f1")
        _validate_metric_name("accuracy")
        _validate_metric_name("precision")
        _validate_metric_name("recall")

        # Invalid type
        with pytest.raises(TypeError, match="metric must be a string"):
            _validate_metric_name(123)

        # Unknown metric
        with pytest.raises(ValueError, match="Unknown metric 'nonexistent_metric'"):
            _validate_metric_name("nonexistent_metric")

    def test_validate_averaging_method(self):
        """Test averaging method validation."""
        # Valid averaging methods
        for method in ["macro", "micro", "weighted", "none"]:
            _validate_averaging_method(method)

        # Invalid averaging method
        with pytest.raises(ValueError, match="Invalid averaging method 'invalid'"):
            _validate_averaging_method("invalid")

    def test_validate_optimization_method(self):
        """Test optimization method validation."""
        # Valid optimization methods
        for method in ["unique_scan", "minimize", "gradient"]:
            _validate_optimization_method(method)

        # Invalid optimization method
        with pytest.raises(ValueError, match="Invalid optimization method 'invalid'"):
            _validate_optimization_method("invalid")

    def test_validate_comparison_operator(self):
        """Test comparison operator validation."""
        # Valid comparison operators
        _validate_comparison_operator(">")
        _validate_comparison_operator(">=")

        # Invalid comparison operators
        for op in ["<", "<=", "==", "!=", "invalid"]:
            with pytest.raises(ValueError, match="Invalid comparison operator"):
                _validate_comparison_operator(op)


class TestPublicFunctionValidation:
    """Test that public functions properly validate their inputs."""

    def test_get_optimal_threshold_validation(self):
        """Test that get_optimal_threshold validates inputs properly."""
        valid_labels = [0, 1, 0, 1]
        valid_probs = [0.2, 0.8, 0.3, 0.7]

        # Should work with valid inputs
        result = get_optimal_threshold(valid_labels, valid_probs)
        assert 0 <= threshold <= 1

        # Should fail with invalid metric
        with pytest.raises(ValueError, match="Unknown metric"):
            get_optimal_threshold(valid_labels, valid_probs, metric="invalid_metric")

        # Should fail with invalid method
        with pytest.raises(ValueError, match="Invalid optimization method"):
            get_optimal_threshold(valid_labels, valid_probs, method="invalid_method")

        # Should fail with invalid comparison
        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_optimal_threshold(valid_labels, valid_probs, comparison="<")

        # Should fail with invalid labels (testing binary case explicitly)
        # Need to match the length first
        with pytest.raises(ValueError, match="Labels must be binary \\(0 or 1\\)"):
            get_optimal_threshold([-1, 0, 1, 0], valid_probs)

    def test_get_confusion_matrix_validation(self):
        """Test that get_confusion_matrix validates inputs properly."""
        valid_labels = [0, 1, 0, 1]
        valid_probs = [0.2, 0.8, 0.3, 0.7]
        valid_threshold = 0.5

        # Should work with valid inputs
        tp, tn, fp, fn = get_confusion_matrix(
            valid_labels, valid_probs, valid_threshold
        )
        assert all(isinstance(x, int) for x in [tp, tn, fp, fn])

        # Should fail with invalid threshold
        with pytest.raises(ValueError, match="Threshold must be in \\[0, 1\\], got range"):
            get_confusion_matrix(valid_labels, valid_probs, -0.1)

        # Should fail with invalid comparison
        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_confusion_matrix(
                valid_labels, valid_probs, valid_threshold, comparison="<"
            )

        # Should fail with multiclass input when not allowed
        multiclass_labels = [0, 1, 2, 0, 1, 2]
        multiclass_probs = np.random.rand(6, 3)
        with pytest.raises(ValueError, match="2D pred_prob not allowed"):
            get_confusion_matrix(multiclass_labels, multiclass_probs, valid_threshold)


class TestRobustnessAndEdgeCases:
    """Test robustness and edge cases in validation."""

    def test_type_conversion(self):
        """Test that inputs are properly converted to numpy arrays."""
        # List inputs should be converted
        true_labels = [0, 1, 0, 1]  # list
        pred_probs = [0.2, 0.8, 0.3, 0.7]  # list

        validated_labels, validated_probs, _ = validate_inputs(true_labels, pred_probs)
        assert isinstance(validated_labels, np.ndarray)
        assert isinstance(validated_probs, np.ndarray)

        # Nested list for multiclass - use multiclass labels with 2D probabilities
        pred_probs_2d = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]
        multiclass_labels = [0, 1, 0, 1]  # These are valid for 2-class multiclass
        validated_labels, validated_probs, _ = validate_inputs(
            multiclass_labels, pred_probs_2d
        )
        assert validated_probs.ndim == 2

    def test_edge_case_single_sample(self):
        """Test validation with single sample."""
        validate_inputs([1], [0.7])
        # For multiclass single sample, need probability columns to match number of unique classes
        validate_inputs([0], [[0.7]])  # multiclass with 1 sample, 1 class

    def test_edge_case_single_class_multiclass(self):
        """Test multiclass with only one class (degenerate case)."""
        # This should technically fail because we need consecutive labels 0...n-1
        # But if all labels are 0, it might be valid as a single class
        true_labels = [0, 0, 0]
        pred_probs = np.random.rand(3, 1)
        validate_inputs(true_labels, pred_probs)

    def test_large_arrays_performance(self):
        """Test validation doesn't break with large arrays."""
        # Large but not huge arrays to avoid test slowdown
        n_samples = 10000
        true_labels = np.random.randint(0, 2, n_samples)
        pred_probs = np.random.rand(n_samples)

        # Should complete without error
        validate_inputs(true_labels, pred_probs)

        # Multiclass case
        n_classes = 10
        true_labels = np.random.randint(0, n_classes, n_samples)
        pred_probs = np.random.rand(n_samples, n_classes)
        validate_inputs(true_labels, pred_probs)

    def test_dtype_preservation(self):
        """Test that appropriate dtypes are preserved/converted."""
        # Integer labels should remain integers after conversion
        true_labels = np.array([0, 1, 0, 1], dtype=np.int32)
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)

        validated_labels, validated_probs, _ = validate_inputs(true_labels, pred_probs)

        # Labels should still be integers (though possibly different precision)
        assert np.issubdtype(validated_labels.dtype, np.integer)
        # Probabilities should be float
        assert np.issubdtype(validated_probs.dtype, np.floating)


class TestMulticlassValidation:
    """Test multiclass-specific validation functions."""

    def test_validate_multiclass_classification_basic(self):
        """Test basic multiclass input validation."""
        # Valid consecutive labels
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.random.rand(6, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize to probabilities

        # Should work without consecutive requirement
        validated_labels, validated_probs = validate_multiclass_classification(
            labels, probs, require_consecutive=False
        )
        assert np.array_equal(validated_labels, labels)
        assert validated_probs.shape == (6, 3)

        # Should also work with consecutive requirement
        validated_labels, validated_probs = validate_multiclass_classification(
            labels, probs, require_consecutive=True
        )
        assert np.array_equal(validated_labels, labels)

    def test_validate_multiclass_classification_non_consecutive_labels(self):
        """Test validation with non-consecutive labels."""
        # Non-consecutive labels (missing 0)
        labels = np.array([1, 2, 1, 2, 1])
        probs = np.random.rand(5, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Should work without consecutive requirement
        validated_labels, validated_probs = validate_multiclass_classification(
            labels, probs, require_consecutive=False
        )
        assert np.array_equal(validated_labels, labels)

        # Should fail with consecutive requirement
        with pytest.raises(
            ValueError, match="Labels must be consecutive integers from 0"
        ):
            validate_multiclass_classification(labels, probs, require_consecutive=True)

    def test_validate_multiclass_classification_1d_probabilities(self):
        """Test validation rejects 1D probabilities."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])  # 1D

        with pytest.raises(ValueError, match="pred_prob must be 2D for multiclass"):
            validate_multiclass_classification(labels, probs)

    def test_validate_multiclass_classification_invalid_probabilities(self):
        """Test validation with invalid probability values."""
        labels = np.array([0, 1, 2])

        # Probabilities outside [0, 1]
        probs = np.array([[-0.1, 0.5, 0.6], [0.3, 1.2, -0.1], [0.4, 0.3, 0.3]])

        with pytest.raises(ValueError, match="Probabilities must be in"):
            validate_multiclass_classification(labels, probs, require_proba=True)

        # Should work when proba requirement is disabled
        validated_labels, validated_probs = validate_multiclass_classification(
            labels, probs, require_proba=False
        )
        assert np.array_equal(validated_labels, labels)

    def test_validate_multiclass_classification_basic(self):
        """Test coordinate ascent specific validation."""
        labels = np.array([0, 1, 2, 0, 1])
        probs = np.random.rand(5, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Should work with consecutive labels
        validated_probs, validated_labels = (
            validate_multiclass_classification(probs, labels)
        )
        assert validated_probs.shape == (5, 3)
        assert np.array_equal(validated_labels, labels)

    def test_validate_multiclass_classification_non_consecutive(self):
        """Test coordinate ascent validation with non-consecutive labels."""
        labels = np.array([1, 2, 1, 2])  # Missing 0
        probs = np.random.rand(4, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Should fail with consecutive requirement (default)
        with pytest.raises(
            ValueError, match="Labels must be consecutive integers from 0"
        ):
            validate_multiclass_classification(probs, labels)

        # Should work when consecutive requirement is disabled
        validated_probs, validated_labels = (
            validate_multiclass_classification(
                probs, labels, require_consecutive=False
            )
        )
        assert validated_probs.shape == (4, 3)

    def test_validate_multiclass_classification_insufficient_classes(self):
        """Test validation with insufficient number of classes."""
        labels = np.array([0, 0, 0])  # Only one class
        probs = np.random.rand(3, 1)  # Only one class

        with pytest.raises(ValueError, match="Need at least 2 classes"):
            validate_multiclass_classification(probs, labels)

    def test_validate_multiclass_classification_dimension_mismatch(self):
        """Test validation with dimension mismatches."""
        # Wrong label dimensions
        labels = np.array([[0, 1], [1, 2]])  # 2D instead of 1D
        probs = np.random.rand(2, 3)

        with pytest.raises(ValueError, match="true_labs must be 1D"):
            validate_multiclass_classification(probs, labels)

        # Wrong probability dimensions
        labels = np.array([0, 1, 2])
        probs = np.array([0.5, 0.3, 0.2])  # 1D instead of 2D

        with pytest.raises(ValueError, match="pred_prob must be 2D for multiclass"):
            validate_multiclass_classification(probs, labels)

        # Length mismatch
        labels = np.array([0, 1, 2])
        probs = np.random.rand(5, 3)  # 5 samples vs 3 labels

        with pytest.raises(ValueError, match="Length mismatch"):
            validate_multiclass_classification(probs, labels)

    def test_multiclass_validation_consistency_across_modules(self):
        """Test that validation is consistent across different modules."""

        # Create valid test data
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.random.rand(6, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Direct API should work with valid data
        thresholds_opt = get_optimal_multiclass_thresholds(
            labels, probs, method="unique_scan"
        )

        assert len(thresholds_opt) == 3

        # Test with non-consecutive labels
        non_consecutive_labels = np.array([1, 2, 1, 2, 1, 2])  # Missing 0

        # Regular multiclass should work
        thresholds_opt = get_optimal_multiclass_thresholds(
            non_consecutive_labels, probs, method="unique_scan"
        )
        assert len(thresholds_opt) == 3

        # Coordinate ascent should fail
        with pytest.raises(ValueError, match="Labels must be consecutive"):
            get_optimal_multiclass_thresholds(
                non_consecutive_labels, probs, method="coord_ascent"
            )

        # This test was for the removed ThresholdOptimizer wrapper
        # The direct API validation is sufficient

    def test_multiclass_validation_edge_cases(self):
        """Test edge cases in multiclass validation."""
        # Single class in probability matrix
        labels = np.array([0, 0])
        probs = np.array([[1.0], [1.0]])  # Only 1 class

        with pytest.raises(ValueError, match="Need at least 2 classes"):
            validate_multiclass_classification(probs, labels)

        # Labels with gaps
        labels = np.array([0, 2, 0, 2])  # Missing class 1
        probs = np.random.rand(4, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="Labels must be consecutive"):
            validate_multiclass_classification(probs, labels)

        # Labels exceed number of classes
        labels = np.array([0, 1, 3])  # Class 3 doesn't exist in 3-class problem
        probs = np.random.rand(3, 3)

        with pytest.raises(ValueError, match="Labels.*must be within"):
            validate_multiclass_classification(probs, labels)


class TestBinaryValidation:
    """Test binary-specific validation functions."""

    def test_validate_binary_classification_basic(self):
        """Test basic binary classification validation."""
        # Valid binary input
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])

        validated_labels, validated_probs, validated_weights = (
            validate_binary_classification(labels, probs)
        )
        assert np.array_equal(validated_labels, labels)
        assert np.array_equal(validated_probs, probs)
        assert validated_weights is None

    def test_validate_binary_classification_with_weights(self):
        """Test binary validation with sample weights."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])
        weights = np.array([1.0, 2.0, 1.5, 0.5])

        validated_labels, validated_probs, validated_weights = (
            validate_binary_classification(labels, probs, weights=weights)
        )
        assert validated_weights is not None
        assert np.array_equal(validated_weights, weights)

    def test_validate_binary_classification_return_default_weights(self):
        """Test binary validation with default weights."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])

        validated_labels, validated_probs, validated_weights = (
            validate_binary_classification(labels, probs)
        )
        assert validated_weights is None  # No weights provided, should be None

    def test_validate_binary_classification_force_dtypes(self):
        """Test binary validation with forced dtypes."""
        labels = [0, 1, 0, 1]  # List input
        probs = [0.2, 0.8, 0.3, 0.7]  # List input

        validated_labels, validated_probs, _ = validate_binary_classification(
            labels, probs, force_dtypes=True
        )
        assert validated_labels.dtype == np.int8
        assert validated_probs.dtype == np.float64

    def test_validate_binary_classification_scores(self):
        """Test binary validation with scores outside [0,1]."""
        labels = np.array([0, 1, 0, 1])
        scores = np.array([-2.5, 1.3, -0.8, 3.2])

        # Should fail with require_proba=True
        with pytest.raises(ValueError, match="Probabilities must be in"):
            validate_binary_classification(labels, scores, require_proba=True)

        # Should work with require_proba=False
        validated_labels, validated_scores, _ = validate_binary_classification(
            labels, scores, require_proba=False
        )
        assert np.array_equal(validated_labels, labels)
        assert np.array_equal(validated_scores, scores)

    def test_validate_binary_classification_non_binary_labels(self):
        """Test binary validation rejects non-binary labels."""
        labels = np.array([0, 1, 2])  # Contains 2
        probs = np.array([0.2, 0.8, 0.3])

        with pytest.raises(ValueError, match="Binary labels must be from"):
            validate_binary_classification(labels, probs)

    def test_validate_binary_classification_multiclass_input(self):
        """Test binary validation rejects multiclass input."""
        labels = np.array([0, 1, 0])
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])  # 2D

        with pytest.raises(ValueError, match="2D pred_prob not allowed"):
            validate_binary_classification(labels, probs)

    def test_binary_validation_consistency_with_piecewise(self):
        """Test that binary validation is consistent with piecewise usage."""
        from optimal_cutoffs.piecewise import validate_inputs as piecewise_validate

        # Test data
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])

        # Compare results - piecewise validation (through wrapper)
        piecewise_labels, piecewise_probs = piecewise_validate(labels, probs)

        # Direct centralized validation
        central_labels, central_probs, _ = validate_binary_classification(
            labels, probs, force_dtypes=True
        )

        # Should be identical
        assert np.array_equal(piecewise_labels, central_labels)
        assert np.array_equal(piecewise_probs, central_probs)
        assert piecewise_labels.dtype == central_labels.dtype
        assert piecewise_probs.dtype == central_probs.dtype

    def test_binary_validation_consistency_with_scores(self):
        """Test consistency with score-based inputs."""
        from optimal_cutoffs.piecewise import validate_inputs as piecewise_validate

        # Test with scores outside [0,1]
        labels = np.array([0, 1, 0, 1])
        scores = np.array([-1.5, 2.3, -0.8, 1.9])

        # Piecewise validation (through wrapper)
        piecewise_labels, piecewise_scores = piecewise_validate(
            labels, scores, require_proba=False
        )

        # Direct centralized validation
        central_labels, central_scores, _ = validate_binary_classification(
            labels, scores, require_proba=False, force_dtypes=True
        )

        # Should be identical
        assert np.array_equal(piecewise_labels, central_labels)
        assert np.array_equal(piecewise_scores, central_scores)
        assert piecewise_labels.dtype == central_labels.dtype
        assert piecewise_scores.dtype == central_scores.dtype

    def test_sample_weights_consistency_with_piecewise(self):
        """Test sample weights validation consistency."""
        from optimal_cutoffs.piecewise import (
            _validate_sample_weights as piecewise_validate_weights,
        )

        n_samples = 4
        weights = np.array([1.0, 2.0, 1.5, 0.5])

        # Piecewise validation (through wrapper)
        piecewise_weights = piecewise_validate_weights(weights, n_samples)

        # Direct centralized validation
        _, _, central_weights = validate_binary_classification(
            np.zeros(n_samples),  # Dummy labels
            np.zeros(n_samples),  # Dummy probs
            weights=weights,
            require_proba=False,
        )

        # Should be identical
        assert np.array_equal(piecewise_weights, central_weights)
        assert piecewise_weights.dtype == central_weights.dtype

    def test_default_weights_consistency_with_piecewise(self):
        """Test default weights behavior consistency."""
        from optimal_cutoffs.piecewise import (
            _validate_sample_weights as piecewise_validate_weights,
        )

        n_samples = 4

        # Piecewise validation with None (returns ones array)
        piecewise_weights = piecewise_validate_weights(None, n_samples)

        # Direct centralized validation
        _, _, central_weights = validate_binary_classification(
            np.zeros(n_samples),  # Dummy labels
            np.zeros(n_samples),  # Dummy probs
            weights=None,
            require_proba=False,
        )

        # Piecewise returns default weights, central returns None
        assert central_weights is None
        assert np.array_equal(piecewise_weights, np.ones(n_samples, dtype=np.float64))
        assert piecewise_weights.dtype == np.float64
