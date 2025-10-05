"""Tests for registry flag integration and method routing."""

import numpy as np
import pytest

from optimal_cutoffs import (
    METRICS,
    get_optimal_threshold,
    get_vectorized_metric,
    has_vectorized_implementation,
    is_piecewise_metric,
)


class TestRegistryIntegration:
    """Test registry flag integration functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.random.choice([0, 1], size=100)
        self.pred_prob = np.random.random(100)

    def test_built_in_metrics_have_vectorized_implementations(self):
        """Test that built-in piecewise metrics have vectorized implementations."""
        built_in_piecewise = ["f1", "accuracy", "precision", "recall"]

        for metric in built_in_piecewise:
            assert metric in METRICS
            assert has_vectorized_implementation(metric)
            assert is_piecewise_metric(metric)

    def test_get_vectorized_metric(self):
        """Test getting vectorized metric functions."""
        # Test valid metric
        f1_vec = get_vectorized_metric("f1")
        assert callable(f1_vec)

        # Test with array inputs
        tp = np.array([10, 20])
        tn = np.array([15, 25])
        fp = np.array([5, 8])
        fn = np.array([3, 7])

        scores = f1_vec(tp, tn, fp, fn)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)

    def test_get_vectorized_metric_not_available(self):
        """Test error when requesting non-existent vectorized metric."""
        # Test should verify that unknown metrics raise errors
        with pytest.raises(ValueError, match="Unknown metric"):
            get_optimal_threshold(self.y_true, self.pred_prob, metric="custom_metric", method="sort_scan")

    def test_sample_weights_with_sort_scan(self):
        """Test sort_scan method with sample weights."""
        weights = np.random.uniform(0.5, 2.0, size=len(self.y_true))

        result = get_optimal_threshold(
            self.y_true,
            self.pred_prob,
            metric="f1",
            method="sort_scan",
            sample_weight=weights,
        )
        threshold = result.threshold
        assert 0.0 <= threshold <= 1.0

    def test_comparison_operators_with_sort_scan(self):
        """Test different comparison operators with sort_scan."""
        for comparison in [">", ">="]:
            result = get_optimal_threshold(
                self.y_true,
                self.pred_prob,
                metric="f1",
                method="sort_scan",
                comparison=comparison,
            )
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0

    def test_validation_of_new_methods(self):
        """Test validation of new optimization methods."""
        # Valid methods should work
        valid_methods = ["auto", "sort_scan", "unique_scan", "minimize", "gradient"]

        for method in valid_methods:
            # Should not raise validation error
            try:
                get_optimal_threshold(
                    self.y_true, self.pred_prob, metric="f1", method=method
                )
            except ValueError as e:
                if "Invalid optimization method" in str(e):
                    pytest.fail(
                        f"Method {method} should be valid but raised validation error"
                    )

        # Invalid method should raise error
        with pytest.raises(ValueError, match="Invalid optimization method"):
            get_optimal_threshold(
                self.y_true, self.pred_prob, metric="f1", method="invalid_method"
            )


class TestBackwardCompatibility:
    """Test that registry integration maintains backward compatibility."""

    def test_default_method_behavior(self):
        """Test that default behavior works correctly."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        pred_prob = np.random.random(100)

        # Default method should work
        result = get_optimal_threshold(y_true, pred_prob, metric="f1")
        threshold = result.threshold
        assert 0.0 <= threshold <= 1.0

        # Should be equivalent to explicit auto method
        result = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="auto"
        )
        assert threshold == threshold

    def test_existing_api_unchanged(self):
        """Test that existing API calls work unchanged."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=50)
        pred_prob = np.random.random(50)

        # All these should work as before
        result = get_optimal_threshold(y_true, pred_prob)
        result = get_optimal_threshold(y_true, pred_prob, metric="accuracy")
        result = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="minimize"
        )

        threshold = result.threshold
        assert all(0.0 <= t <= 1.0 for t in [threshold, threshold, threshold])
