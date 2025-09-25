"""Tests for registry flag integration and method routing."""

import numpy as np
import pytest

from optimal_cutoffs import (
    METRIC_REGISTRY,
    VECTORIZED_REGISTRY,
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
            assert metric in METRIC_REGISTRY
            assert metric in VECTORIZED_REGISTRY
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
        with pytest.raises(ValueError, match="Vectorized implementation not available"):
            get_vectorized_metric("nonexistent_metric")

    def test_method_auto_routing(self):
        """Test that method='auto' routes correctly."""
        # For piecewise metric with vectorized implementation -> sort_scan
        thresh_auto = get_optimal_threshold(
            self.y_true, self.pred_prob, metric="f1", method="auto"
        )
        thresh_sort_scan = get_optimal_threshold(
            self.y_true, self.pred_prob, metric="f1", method="sort_scan"
        )

        # Should be identical since auto routes to sort_scan for f1
        assert abs(thresh_auto - thresh_sort_scan) < 1e-10

    def test_sort_scan_method_direct(self):
        """Test sort_scan method directly."""
        for metric in ["f1", "accuracy", "precision", "recall"]:
            thresh = get_optimal_threshold(
                self.y_true, self.pred_prob, metric=metric, method="sort_scan"
            )
            assert 0.0 <= thresh <= 1.0

    def test_sort_scan_requires_vectorized_implementation(self):
        """Test that sort_scan requires vectorized implementation."""
        # Create a custom metric without vectorized implementation
        from optimal_cutoffs.metrics import register_metric

        @register_metric("test_metric", is_piecewise=True)
        def test_metric(tp, tn, fp, fn):
            return (tp + tn) / (tp + tn + fp + fn)

        # Should fail when trying to use sort_scan
        with pytest.raises(
            ValueError, match="sort_scan method requires vectorized implementation"
        ):
            get_optimal_threshold(
                self.y_true, self.pred_prob, metric="test_metric", method="sort_scan"
            )

        # Cleanup
        del METRIC_REGISTRY["test_metric"]

    def test_method_compatibility_across_algorithms(self):
        """Test that different methods give compatible results."""
        methods = ["auto", "sort_scan", "smart_brute"]
        thresholds = {}

        for method in methods:
            thresholds[method] = get_optimal_threshold(
                self.y_true, self.pred_prob, metric="f1", method=method
            )

        # All methods should give very similar results
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                diff = abs(thresholds[method1] - thresholds[method2])
                assert diff < 0.1, (
                    f"Large difference between {method1} and {method2}: {diff}"
                )

    def test_sample_weights_with_sort_scan(self):
        """Test sort_scan method with sample weights."""
        weights = np.random.uniform(0.5, 2.0, size=len(self.y_true))

        thresh = get_optimal_threshold(
            self.y_true,
            self.pred_prob,
            metric="f1",
            method="sort_scan",
            sample_weight=weights,
        )
        assert 0.0 <= thresh <= 1.0

    def test_comparison_operators_with_sort_scan(self):
        """Test different comparison operators with sort_scan."""
        for comparison in [">", ">="]:
            thresh = get_optimal_threshold(
                self.y_true,
                self.pred_prob,
                metric="f1",
                method="sort_scan",
                comparison=comparison,
            )
            assert 0.0 <= thresh <= 1.0

    def test_validation_of_new_methods(self):
        """Test validation of new optimization methods."""
        # Valid methods should work
        valid_methods = ["auto", "sort_scan", "smart_brute", "minimize", "gradient"]

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


class TestPerformanceComparison:
    """Test performance differences between methods."""

    def test_sort_scan_vs_smart_brute_performance(self):
        """Test that sort_scan is faster than smart_brute for large datasets."""
        import time

        # Create larger dataset
        np.random.seed(42)
        n = 5000
        y_true = np.random.choice([0, 1], size=n)
        pred_prob = np.random.random(n)

        # Time sort_scan
        start = time.time()
        thresh_sort_scan = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="sort_scan"
        )
        time_sort_scan = time.time() - start

        # Time smart_brute
        start = time.time()
        thresh_smart_brute = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="smart_brute"
        )
        time_smart_brute = time.time() - start

        print(f"Sort_scan time: {time_sort_scan:.4f}s")
        print(f"Smart_brute time: {time_smart_brute:.4f}s")
        print(f"Speedup: {time_smart_brute / time_sort_scan:.2f}x")

        # Results should be very similar
        assert abs(thresh_sort_scan - thresh_smart_brute) < 0.1

        # For small execution times, timing can be highly variable in CI environments
        # The main goal is to ensure both methods work and produce similar results
        # Performance comparison is more meaningful for much larger datasets
        if time_smart_brute > 0.01 and time_sort_scan > 0.01:
            # Only assert performance advantage when times are large enough to be meaningful
            assert time_sort_scan < time_smart_brute * 1.5  # Allow some variability
        # Otherwise, just ensure both complete successfully (which they did to get here)


class TestBackwardCompatibility:
    """Test that registry integration maintains backward compatibility."""

    def test_default_method_behavior(self):
        """Test that default behavior works correctly."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        pred_prob = np.random.random(100)

        # Default method should work
        thresh_default = get_optimal_threshold(y_true, pred_prob, metric="f1")
        assert 0.0 <= thresh_default <= 1.0

        # Should be equivalent to explicit auto method
        thresh_auto = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="auto"
        )
        assert thresh_default == thresh_auto

    def test_existing_api_unchanged(self):
        """Test that existing API calls work unchanged."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=50)
        pred_prob = np.random.random(50)

        # All these should work as before
        thresh1 = get_optimal_threshold(y_true, pred_prob)
        thresh2 = get_optimal_threshold(y_true, pred_prob, metric="accuracy")
        thresh3 = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="minimize"
        )

        assert all(0.0 <= t <= 1.0 for t in [thresh1, thresh2, thresh3])
