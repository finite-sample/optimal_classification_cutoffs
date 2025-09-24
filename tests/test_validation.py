"""Comprehensive tests for input validation functionality."""

import warnings

import numpy as np
import pytest

from optimal_cutoffs import get_confusion_matrix, get_optimal_threshold
from optimal_cutoffs.validation import (
    _validate_averaging_method,
    _validate_comparison_operator,
    _validate_inputs,
    _validate_metric_name,
    _validate_optimization_method,
    _validate_threshold,
)


class TestInputValidation:
    """Test comprehensive input validation functionality."""

    def test_validate_inputs_basic(self):
        """Test basic input validation with valid inputs."""
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7])

        validated_labels, validated_probs, validated_weights = _validate_inputs(
            true_labels, pred_probs
        )

        assert np.array_equal(validated_labels, true_labels)
        assert np.array_equal(validated_probs, pred_probs)
        assert validated_weights is None

    def test_validate_inputs_empty_arrays(self):
        """Test validation with empty arrays."""
        with pytest.raises(ValueError, match="true_labs cannot be empty"):
            _validate_inputs([], [0.5])

        with pytest.raises(ValueError, match="pred_prob cannot be empty"):
            _validate_inputs([0], [])

    def test_validate_inputs_dimension_mismatch(self):
        """Test validation with mismatched dimensions."""
        with pytest.raises(ValueError, match="Length mismatch"):
            _validate_inputs([0, 1], [0.5])

        with pytest.raises(ValueError, match="Length mismatch"):
            _validate_inputs([0, 1, 0], np.random.rand(2, 3))  # multiclass mismatch

    def test_validate_inputs_wrong_dimensions(self):
        """Test validation with wrong array dimensions."""
        with pytest.raises(ValueError, match="true_labs must be 1D"):
            _validate_inputs(np.array([[0, 1], [1, 0]]), [0.5, 0.8, 0.3, 0.7])

        with pytest.raises(ValueError, match="pred_prob must be 1D or 2D"):
            _validate_inputs([0, 1], np.random.rand(2, 2, 2))

    def test_validate_inputs_non_finite_values(self):
        """Test validation with NaN and infinite values."""
        # NaN in true labels
        with pytest.raises(ValueError, match="true_labs contains NaN or infinite"):
            _validate_inputs([0, np.nan, 1], [0.5, 0.6, 0.7])

        # Infinite in pred_prob
        with pytest.raises(ValueError, match="pred_prob contains NaN or infinite"):
            _validate_inputs([0, 1, 0], [0.5, np.inf, 0.7])

    def test_validate_inputs_binary_labels_requirement(self):
        """Test binary label validation."""
        # Valid binary labels
        _validate_inputs([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], require_binary=True)

        # Invalid binary labels - not in {0, 1}
        with pytest.raises(ValueError, match="Binary labels must be from"):
            _validate_inputs([0, 1, 2], [0.1, 0.2, 0.3], require_binary=True)

        # Edge case: only zeros
        _validate_inputs([0, 0, 0], [0.1, 0.2, 0.3], require_binary=True)

        # Edge case: only ones
        _validate_inputs([1, 1, 1], [0.1, 0.2, 0.3], require_binary=True)

    def test_validate_inputs_multiclass_labels(self):
        """Test multiclass label validation."""
        # Valid consecutive labels starting from 0
        true_labels = [0, 1, 2, 0, 1, 2]
        pred_probs = np.random.rand(6, 3)
        _validate_inputs(true_labels, pred_probs)

        # Invalid: non-consecutive labels (missing class 1, has class 3)
        with pytest.raises(
            ValueError, match="must be consecutive integers starting from 0"
        ):
            _validate_inputs([0, 2, 3], np.random.rand(3, 3))  # Has 0,2,3 but missing 1

        # Invalid: negative labels
        with pytest.raises(ValueError, match="Labels must be non-negative"):
            _validate_inputs([-1, 0, 1], np.random.rand(3, 2))

        # Invalid: non-integer labels
        with pytest.raises(ValueError, match="Labels must be integers"):
            _validate_inputs([0.5, 1.0, 1.5], np.random.rand(3, 2))

    def test_validate_inputs_probability_range(self):
        """Test probability range validation."""
        # Valid probabilities
        _validate_inputs([0, 1], [0.0, 1.0], require_proba=True)
        _validate_inputs([0, 1], [0.5, 0.7], require_proba=True)

        # Invalid: below 0
        with pytest.raises(ValueError, match="Probabilities must be in \\[0, 1\\]"):
            _validate_inputs([0, 1], [-0.1, 0.5], require_proba=True)

        # Invalid: above 1
        with pytest.raises(ValueError, match="Probabilities must be in \\[0, 1\\]"):
            _validate_inputs([0, 1], [0.5, 1.1], require_proba=True)

    def test_validate_inputs_multiclass_probability_sum_warning(self):
        """Test warning for multiclass probabilities that don't sum to 1."""
        true_labels = [0, 1, 2]
        # Probabilities that don't sum to 1
        pred_probs = np.array([[0.5, 0.3, 0.1], [0.2, 0.7, 0.2], [0.8, 0.1, 0.2]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_inputs(true_labels, pred_probs)

            # Should issue a warning about probabilities not summing to 1
            assert len(w) == 1
            assert "don't sum to 1.0" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def test_validate_inputs_sample_weights(self):
        """Test sample weight validation."""
        true_labels = [0, 1, 0, 1]
        pred_probs = [0.2, 0.8, 0.3, 0.7]

        # Valid sample weights
        sample_weights = [1.0, 2.0, 1.5, 0.5]
        validated_labels, validated_probs, validated_weights = _validate_inputs(
            true_labels, pred_probs, sample_weight=sample_weights
        )
        assert validated_weights is not None
        assert np.array_equal(validated_weights, sample_weights)

        # Wrong dimension
        with pytest.raises(ValueError, match="sample_weight must be 1D"):
            _validate_inputs(true_labels, pred_probs, sample_weight=[[1, 2], [3, 4]])

        # Wrong length
        with pytest.raises(ValueError, match="Length mismatch: sample_weight"):
            _validate_inputs(true_labels, pred_probs, sample_weight=[1.0, 2.0])

        # NaN values
        with pytest.raises(ValueError, match="sample_weight contains NaN"):
            _validate_inputs(
                true_labels, pred_probs, sample_weight=[1.0, np.nan, 1.5, 0.5]
            )

        # Negative values
        with pytest.raises(ValueError, match="sample_weight must be non-negative"):
            _validate_inputs(
                true_labels, pred_probs, sample_weight=[1.0, -1.0, 1.5, 0.5]
            )

        # All zeros
        with pytest.raises(ValueError, match="sample_weight cannot sum to zero"):
            _validate_inputs(
                true_labels, pred_probs, sample_weight=[0.0, 0.0, 0.0, 0.0]
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
        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
            _validate_threshold(-0.1)

        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
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
        for method in ["smart_brute", "minimize", "gradient"]:
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
        threshold = get_optimal_threshold(valid_labels, valid_probs)
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
        with pytest.raises(ValueError, match="Labels must be non-negative"):
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
        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
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

        validated_labels, validated_probs, _ = _validate_inputs(true_labels, pred_probs)
        assert isinstance(validated_labels, np.ndarray)
        assert isinstance(validated_probs, np.ndarray)

        # Nested list for multiclass
        pred_probs_2d = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]
        validated_labels, validated_probs, _ = _validate_inputs(
            true_labels, pred_probs_2d
        )
        assert validated_probs.ndim == 2

    def test_edge_case_single_sample(self):
        """Test validation with single sample."""
        _validate_inputs([1], [0.7])
        # For multiclass single sample, need probability columns to match number of unique classes
        _validate_inputs([0], [[0.7]])  # multiclass with 1 sample, 1 class

    def test_edge_case_single_class_multiclass(self):
        """Test multiclass with only one class (degenerate case)."""
        # This should technically fail because we need consecutive labels 0...n-1
        # But if all labels are 0, it might be valid as a single class
        true_labels = [0, 0, 0]
        pred_probs = np.random.rand(3, 1)
        _validate_inputs(true_labels, pred_probs)

    def test_large_arrays_performance(self):
        """Test validation doesn't break with large arrays."""
        # Large but not huge arrays to avoid test slowdown
        n_samples = 10000
        true_labels = np.random.randint(0, 2, n_samples)
        pred_probs = np.random.rand(n_samples)

        # Should complete without error
        _validate_inputs(true_labels, pred_probs)

        # Multiclass case
        n_classes = 10
        true_labels = np.random.randint(0, n_classes, n_samples)
        pred_probs = np.random.rand(n_samples, n_classes)
        _validate_inputs(true_labels, pred_probs)

    def test_dtype_preservation(self):
        """Test that appropriate dtypes are preserved/converted."""
        # Integer labels should remain integers after conversion
        true_labels = np.array([0, 1, 0, 1], dtype=np.int32)
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)

        validated_labels, validated_probs, _ = _validate_inputs(true_labels, pred_probs)

        # Labels should still be integers (though possibly different precision)
        assert np.issubdtype(validated_labels.dtype, np.integer)
        # Probabilities should be float
        assert np.issubdtype(validated_probs.dtype, np.floating)
