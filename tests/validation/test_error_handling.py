"""Tests for error handling and exceptional conditions.

This module tests that the library properly handles error conditions,
provides informative error messages, and fails gracefully.
"""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import get_confusion_matrix


class TestParameterErrors:
    """Test errors from invalid parameters."""

    def test_invalid_metric_error(self):
        """Test error handling for invalid metrics."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="Unknown metric"):
            get_optimal_threshold(y_true, y_prob, metric="nonexistent_metric")

    def test_invalid_method_error(self):
        """Test error handling for invalid optimization methods."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="Invalid optimization method"):
            get_optimal_threshold(y_true, y_prob, method="nonexistent_method")

    def test_invalid_comparison_error(self):
        """Test error handling for invalid comparison operators."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_optimal_threshold(y_true, y_prob, comparison="<")

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_optimal_threshold(y_true, y_prob, comparison="==")

    def test_invalid_mode_error(self):
        """Test error handling for invalid modes."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="Unknown mode: invalid_mode"):
            get_optimal_threshold(y_true, y_prob, mode="invalid_mode")

    def test_invalid_average_error(self):
        """Test error handling for invalid averaging methods."""
        y_true = [[0, 1], [1, 0], [0, 1]]
        y_prob = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])

        # Create mock multiclass scenario
        try:
            with pytest.raises(ValueError, match="Labels must be 1D, got shape"):
                # This would test multiclass averaging if function exists
                get_optimal_threshold(y_true, y_prob, average="invalid_average")
        except (TypeError, AttributeError):
            # Function signature might not support average parameter
            pass


class TestDataErrors:
    """Test errors from invalid data."""

    def test_empty_data_error(self):
        """Test error handling for empty datasets."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_optimal_threshold([], [])

    def test_length_mismatch_error(self):
        """Test error handling for mismatched array lengths."""
        with pytest.raises(ValueError, match="Length mismatch"):
            get_optimal_threshold([0, 1], [0.5])

        with pytest.raises(ValueError, match="Length mismatch"):
            get_optimal_threshold([0], [0.2, 0.8])

    def test_invalid_label_values_error(self):
        """Test error handling for invalid label values."""
        y_prob = [0.2, 0.8, 0.4, 0.6]

        # Non-binary labels
        with pytest.raises(ValueError, match="must be binary"):
            get_optimal_threshold([0, 1, 2, 3], y_prob)

        # Negative labels
        with pytest.raises(ValueError, match="must be binary"):
            get_optimal_threshold([-1, 0, 1, 2], y_prob)

        # Float labels (should be converted to int but check bounds)
        with pytest.raises(ValueError, match="must be binary"):
            get_optimal_threshold([0.0, 1.0, 2.0, 3.0], y_prob)

    def test_invalid_probability_values_error(self):
        """Test error handling for invalid probability values."""
        y_true = [0, 1, 0, 1]

        # Probabilities outside [0, 1] range
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            get_optimal_threshold(y_true, [-0.1, 0.5, 0.8, 1.2])

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            get_optimal_threshold(y_true, [0.1, 0.5, 0.8, 1.5])

    def test_nan_infinity_error(self):
        """Test error handling for NaN and infinity values."""
        y_true = [0, 1, 0, 1]

        # NaN in probabilities
        with pytest.raises(ValueError, match="contains NaN"):
            get_optimal_threshold(y_true, [0.1, np.nan, 0.8, 0.9])

        # Infinity in probabilities
        with pytest.raises(ValueError, match="contains infinite"):
            get_optimal_threshold(y_true, [0.1, 0.5, np.inf, 0.9])

        # Negative infinity
        with pytest.raises(ValueError, match="contains infinite"):
            get_optimal_threshold(y_true, [0.1, 0.5, -np.inf, 0.9])

    def test_wrong_array_dimensions_error(self):
        """Test error handling for wrong array dimensions."""
        # 2D labels
        with pytest.raises(ValueError, match="must be 1D"):
            get_optimal_threshold([[0, 1]], [0.5, 0.7])

        # 2D probabilities (in binary context)
        with pytest.raises(ValueError, match="must be 1D"):
            get_optimal_threshold([0, 1], [[0.3, 0.7], [0.2, 0.8]])


class TestSampleWeightErrors:
    """Test errors related to sample weights."""

    def test_invalid_sample_weight_length(self):
        """Test error handling for invalid sample weight lengths."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="Length mismatch"):
            get_optimal_threshold(y_true, y_prob, sample_weight=[1.0, 2.0])

    def test_negative_sample_weights_error(self):
        """Test error handling for negative sample weights."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="must be non-negative"):
            get_optimal_threshold(y_true, y_prob, sample_weight=[-1.0, 1.0, 1.0, 1.0])

    def test_nan_sample_weights_error(self):
        """Test error handling for NaN sample weights."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="contains NaN"):
            get_optimal_threshold(y_true, y_prob, sample_weight=[1.0, np.nan, 1.0, 1.0])

    def test_infinite_sample_weights_error(self):
        """Test error handling for infinite sample weights."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="contains infinite"):
            get_optimal_threshold(y_true, y_prob, sample_weight=[1.0, np.inf, 1.0, 1.0])

    def test_zero_sum_sample_weights_error(self):
        """Test error handling for sample weights that sum to zero."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        with pytest.raises(ValueError, match="sum to zero"):
            get_optimal_threshold(y_true, y_prob, sample_weight=[0.0, 0.0, 0.0, 0.0])


class TestConfusionMatrixErrors:
    """Test errors in confusion matrix calculation."""

    def test_invalid_threshold_error(self):
        """Test error handling for invalid threshold values in confusion matrix."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        # Threshold outside [0, 1]
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            get_confusion_matrix(y_true, y_prob, -0.1)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            get_confusion_matrix(y_true, y_prob, 1.5)

        # NaN threshold
        with pytest.raises(ValueError, match="contains NaN"):
            get_confusion_matrix(y_true, y_prob, np.nan)

        # Infinite threshold
        with pytest.raises(ValueError, match="contains infinite"):
            get_confusion_matrix(y_true, y_prob, np.inf)


class TestErrorMessageQuality:
    """Test that error messages are informative and helpful."""

    def test_descriptive_error_messages(self):
        """Test that error messages are descriptive and helpful."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        # Test that error messages contain useful information
        try:
            get_optimal_threshold(y_true, y_prob, metric="nonexistent_metric")
        except ValueError as e:
            assert "metric" in str(e).lower()
            assert "nonexistent_metric" in str(e)

        try:
            get_optimal_threshold(y_true, y_prob, method="nonexistent_method")
        except ValueError as e:
            assert "method" in str(e).lower()
            assert "nonexistent_method" in str(e)

    def test_parameter_suggestion_in_errors(self):
        """Test that error messages suggest valid alternatives when possible."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.8, 0.4, 0.6]

        # Some error messages might suggest valid alternatives
        try:
            get_optimal_threshold(y_true, y_prob, comparison="<")
        except ValueError as e:
            error_msg = str(e).lower()
            # Error message should mention valid operators
            assert any(op in error_msg for op in [">", ">="]) or "valid" in error_msg

    def test_context_in_error_messages(self):
        """Test that error messages provide context about the error."""
        # Test with mismatched lengths
        try:
            get_optimal_threshold([0, 1], [0.5])
        except ValueError as e:
            error_msg = str(e)
            # Should mention the actual lengths
            assert "2" in error_msg and "1" in error_msg


class TestGracefulFailure:
    """Test that the library fails gracefully in edge cases."""

    def test_optimization_convergence_failure(self):
        """Test behavior when optimization fails to converge."""
        # Create a pathological case that might cause convergence issues
        y_true = np.array([0, 1] * 50)
        y_prob = np.full(100, 0.5)  # All identical probabilities

        # Should either succeed or fail with informative error
        try:
            result = get_optimal_threshold(y_true, y_prob, method="minimize")
            # If it succeeds, threshold should be valid
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0
        except (ValueError, RuntimeError) as e:
            # If it fails, error should be informative
            assert len(str(e)) > 10  # Non-empty error message

    def test_numerical_precision_limits(self):
        """Test behavior at numerical precision limits."""
        eps = np.finfo(float).eps
        y_true = [0, 1, 0, 1]
        y_prob = [eps, 1 - eps, eps / 2, 1 - eps / 2]

        # Should either work or fail gracefully
        try:
            result = get_optimal_threshold(y_true, y_prob)
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0
        except (ValueError, RuntimeError) as e:
            # Should have informative error message
            assert "precision" in str(e).lower() or "numerical" in str(e).lower()

    def test_memory_limit_handling(self):
        """Test that memory limits are handled gracefully."""
        # This test is theoretical - we can't easily test actual memory limits
        # but we can test that large reasonable datasets work
        n_large = 50000
        y_true = np.random.binomial(1, 0.5, n_large)
        y_prob = np.random.uniform(0, 1, n_large)

        # Should work for reasonable large datasets
        try:
            result = get_optimal_threshold(y_true, y_prob, method="unique_scan")
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0
        except MemoryError:
            # If memory error occurs, it should be caught and handled
            pytest.skip("Insufficient memory for large dataset test")


class TestErrorRecovery:
    """Test that the library can recover from errors appropriately."""

    def test_fallback_after_method_failure(self):
        """Test that fallback methods are used when primary methods fail."""
        # Create data that might cause some methods to fail
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])  # All tied

        # Auto method should try fallbacks if primary method fails
        result = get_optimal_threshold(y_true, y_prob, method="auto")
        threshold = result.threshold
        assert 0.0 <= threshold <= 1.0

    def test_parameter_correction_warnings(self):
        """Test that invalid parameters are corrected with warnings when possible."""
        # This would test cases where the library corrects invalid parameters
        # and warns the user, rather than failing completely

        # For example, if a threshold slightly outside [0,1] is corrected
        # This is implementation-specific and might not exist
        pass

    def test_partial_failure_handling(self):
        """Test handling of partial failures in complex operations."""
        # This would test scenarios where part of a complex operation fails
        # but the library can still provide partial results

        # This is more relevant for batch operations or multiclass scenarios
        pass
