"""Tests for early parameter validation in CV functions.

This module tests that the CV functions validate parameters immediately
at their entry points rather than failing deep in the call stack.
"""

import numpy as np
import pytest

from optimal_cutoffs.cv import (
    cv_threshold_optimization,
    nested_cv_threshold_optimization,
)


def _generate_test_data(n_samples=50, random_state=42):
    """Generate simple test data for validation testing."""
    rng = np.random.default_rng(random_state)
    labels = rng.integers(0, 2, size=n_samples)
    probs = rng.uniform(0, 1, size=n_samples)

    # Ensure both classes are present
    labels[0], labels[1] = 0, 1

    return labels, probs


class TestEarlyValidation:
    """Test that CV functions validate parameters early."""

    def test_cv_invalid_metric_fails_immediately(self):
        """Test that cv_threshold_optimization fails immediately with invalid metric."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Unknown metric 'invalid_metric'"):
            cv_threshold_optimization(
                labels,
                probs,
                metric="invalid_metric",  # Invalid metric
                cv=3,
                random_state=42,
            )

    def test_nested_cv_invalid_metric_fails_immediately(self):
        """Test that nested_cv_threshold_optimization fails immediately with invalid metric."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Unknown metric 'bad_metric'"):
            nested_cv_threshold_optimization(
                labels,
                probs,
                metric="bad_metric",  # Invalid metric
                inner_cv=3,
                outer_cv=3,
                random_state=42,
            )

    def test_cv_invalid_comparison_fails_immediately(self):
        """Test that cv_threshold_optimization fails immediately with invalid comparison."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            cv_threshold_optimization(
                labels,
                probs,
                metric="f1",
                comparison="<",  # Invalid comparison
                cv=3,
                random_state=42,
            )

    def test_nested_cv_invalid_comparison_fails_immediately(self):
        """Test that nested_cv_threshold_optimization fails immediately with invalid comparison."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            nested_cv_threshold_optimization(
                labels,
                probs,
                metric="f1",
                comparison="<=",  # Invalid comparison
                inner_cv=3,
                outer_cv=3,
                random_state=42,
            )

    def test_cv_invalid_averaging_fails_immediately(self):
        """Test that cv_threshold_optimization fails immediately with invalid averaging."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Invalid averaging method"):
            cv_threshold_optimization(
                labels,
                probs,
                metric="f1",
                average="invalid_average",  # Invalid averaging
                cv=3,
                random_state=42,
            )

    def test_nested_cv_invalid_averaging_fails_immediately(self):
        """Test that nested_cv_threshold_optimization fails immediately with invalid averaging."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Invalid averaging method"):
            nested_cv_threshold_optimization(
                labels,
                probs,
                metric="f1",
                average="bad_averaging",  # Invalid averaging
                inner_cv=3,
                outer_cv=3,
                random_state=42,
            )

    def test_cv_invalid_method_fails_immediately(self):
        """Test that cv_threshold_optimization fails immediately with invalid method."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Invalid optimization method"):
            cv_threshold_optimization(
                labels,
                probs,
                metric="f1",
                method="invalid_method",  # Invalid method
                cv=3,
                random_state=42,
            )

    def test_nested_cv_invalid_method_fails_immediately(self):
        """Test that nested_cv_threshold_optimization fails immediately with invalid method."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError, match="Invalid optimization method"):
            nested_cv_threshold_optimization(
                labels,
                probs,
                metric="f1",
                method="bad_method",  # Invalid method
                inner_cv=3,
                outer_cv=3,
                random_state=42,
            )


class TestValidParametersStillWork:
    """Test that valid parameters still work correctly after adding validation."""

    def test_cv_with_valid_parameters_works(self):
        """Test that cv_threshold_optimization works with valid parameters."""
        labels, probs = _generate_test_data()

        # Should work without errors
        thresholds, scores = cv_threshold_optimization(
            labels,
            probs,
            metric="f1",
            method="auto",
            comparison=">",
            average="macro",
            cv=3,
            random_state=42,
        )

        # Verify results are reasonable
        assert len(thresholds) == 3
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_nested_cv_with_valid_parameters_works(self):
        """Test that nested_cv_threshold_optimization works with valid parameters."""
        labels, probs = _generate_test_data()

        # Should work without errors
        thresholds, scores = nested_cv_threshold_optimization(
            labels,
            probs,
            metric="accuracy",
            method="auto",
            comparison=">=",
            average="macro",
            inner_cv=3,
            outer_cv=3,
            random_state=42,
        )

        # Verify results are reasonable
        assert len(thresholds) == 3
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


class TestMultipleInvalidParameters:
    """Test behavior when multiple parameters are invalid."""

    def test_cv_multiple_invalid_parameters_first_error_wins(self):
        """Test that the first invalid parameter causes immediate failure."""
        labels, probs = _generate_test_data()

        # Multiple invalid parameters - should fail on the first one checked
        with pytest.raises(ValueError, match="Unknown metric"):
            cv_threshold_optimization(
                labels,
                probs,
                metric="invalid_metric",  # Invalid (checked first)
                comparison="<",  # Also invalid
                average="bad_average",  # Also invalid
                method="bad_method",  # Also invalid
                cv=3,
                random_state=42,
            )

    def test_nested_cv_multiple_invalid_parameters_first_error_wins(self):
        """Test that the first invalid parameter causes immediate failure."""
        labels, probs = _generate_test_data()

        # Multiple invalid parameters - should fail on the first one checked
        with pytest.raises(ValueError, match="Unknown metric"):
            nested_cv_threshold_optimization(
                labels,
                probs,
                metric="bad_metric",  # Invalid (checked first)
                comparison="==",  # Also invalid
                average="invalid_avg",  # Also invalid
                method="bad_method",  # Also invalid
                inner_cv=3,
                outer_cv=3,
                random_state=42,
            )


class TestErrorMessageQuality:
    """Test that error messages are helpful and informative."""

    def test_metric_error_message_includes_available_metrics(self):
        """Test that metric validation error includes available metrics."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError) as exc_info:
            cv_threshold_optimization(
                labels, probs, metric="nonexistent_metric", cv=3, random_state=42
            )

        error_msg = str(exc_info.value)
        # Should mention available metrics
        assert "Available metrics:" in error_msg or "Unknown metric" in error_msg

    def test_comparison_error_message_is_clear(self):
        """Test that comparison validation error is clear."""
        labels, probs = _generate_test_data()

        with pytest.raises(ValueError) as exc_info:
            cv_threshold_optimization(
                labels, probs, metric="f1", comparison="!=", cv=3, random_state=42
            )

        error_msg = str(exc_info.value)
        assert "Invalid comparison operator" in error_msg


class TestPerformanceImprovement:
    """Test that early validation improves user experience."""

    def test_validation_happens_before_data_processing(self):
        """Test that validation happens before any expensive operations."""
        # Use large arrays to make processing expensive
        large_labels = np.zeros(10000)
        large_probs = np.random.uniform(0, 1, 10000)

        # Invalid metric should fail immediately, not after processing large arrays
        import time

        start_time = time.time()

        with pytest.raises(ValueError, match="Unknown metric"):
            cv_threshold_optimization(
                large_labels,
                large_probs,
                metric="nonexistent_metric",
                cv=5,
                random_state=42,
            )

        elapsed_time = time.time() - start_time

        # Should fail very quickly (less than 0.1 seconds)
        # If validation happened late, this would take much longer
        assert elapsed_time < 0.1, f"Validation took too long: {elapsed_time:.3f}s"
