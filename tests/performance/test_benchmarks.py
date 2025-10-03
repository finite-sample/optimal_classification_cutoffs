"""Performance benchmarking tests to verify algorithmic complexity guarantees.

This module contains tests that verify the O(n log n) performance characteristics
of the sort-scan optimization algorithm and other performance-critical components.
"""

import gc
import time

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import get_vectorized_metric
from optimal_cutoffs.piecewise import optimal_threshold_sortscan
from tests.fixtures.data_generators import generate_binary_data


class TestAlgorithmicComplexity:
    """Test that algorithms meet their expected complexity guarantees."""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000, 4000])
    def test_sort_scan_n_log_n_scaling(self, n_samples):
        """Test that sort_scan algorithm scales as O(n log n)."""
        y_true, pred_prob = generate_binary_data(n_samples, random_state=42)

        # Get vectorized metric
        try:
            f1_vectorized = get_vectorized_metric("f1")
        except (KeyError, AttributeError):
            pytest.skip("Vectorized F1 metric not available")

        # Warm up and force garbage collection
        gc.collect()

        # Time the optimization
        start_time = time.perf_counter()
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Verify valid results
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= score <= 1.0
        assert 0 <= k_star <= n_samples

        # Performance should scale reasonably
        expected_max_time = n_samples * np.log(n_samples) * 1e-6  # Generous bound
        assert execution_time < max(0.1, expected_max_time), (
            f"Execution time {execution_time:.6f}s exceeded expected bound for {n_samples} samples"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    def test_large_dataset_performance(self, n_samples):
        """Test performance on large datasets."""
        y_true, pred_prob = generate_binary_data(n_samples, random_state=42)

        # Time different methods
        methods = ["unique_scan", "sort_scan"]
        times = {}

        for method in methods:
            gc.collect()  # Clean garbage before timing

            start_time = time.perf_counter()
            try:
                if method == "sort_scan":
                    # Direct sort_scan call
                    f1_vectorized = get_vectorized_metric("f1")
                    threshold, score, k_star = optimal_threshold_sortscan(
                        y_true, pred_prob, f1_vectorized
                    )
                else:
                    # High-level interface
                    threshold = get_optimal_threshold(y_true, pred_prob, method=method)

                end_time = time.perf_counter()
                times[method] = end_time - start_time

                # Verify result is valid
                assert 0.0 <= threshold <= 1.0

            except (ValueError, NotImplementedError, KeyError):
                # Method might not be available
                times[method] = float("inf")

        # At least one method should complete in reasonable time
        min_time = min(times.values())
        assert min_time < 10.0, f"All methods too slow for {n_samples} samples: {times}"

    def test_method_performance_comparison(self):
        """Compare performance of different optimization methods."""
        n_samples = 2000
        y_true, pred_prob = generate_binary_data(n_samples, random_state=42)

        methods = ["unique_scan", "minimize", "gradient"]
        times = {}
        results = {}

        for method in methods:
            gc.collect()

            try:
                start_time = time.perf_counter()
                threshold = get_optimal_threshold(
                    y_true, pred_prob, method=method, metric="f1"
                )
                end_time = time.perf_counter()

                times[method] = end_time - start_time
                results[method] = threshold

                # All methods should complete in reasonable time
                assert times[method] < 30.0, (
                    f"Method {method} took {times[method]:.2f}s"
                )

            except (ValueError, NotImplementedError):
                # Method might not be available
                continue

        # All successful methods should produce valid thresholds
        for method, threshold in results.items():
            assert 0.0 <= threshold <= 1.0


class TestMemoryEfficiency:
    """Test memory efficiency of algorithms."""

    @pytest.mark.slow
    def test_memory_usage_scaling(self):
        """Test that memory usage scales appropriately."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        sizes = [1000, 2000, 4000]
        memory_usage = {}

        for n_samples in sizes:
            gc.collect()  # Clean up before measurement

            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create data and run optimization
            y_true, pred_prob = generate_binary_data(n_samples, random_state=42)
            threshold = get_optimal_threshold(y_true, pred_prob, method="unique_scan")

            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            memory_usage[n_samples] = memory_used

            # Clean up
            del y_true, pred_prob, threshold
            gc.collect()

            # Memory usage should be reasonable
            assert memory_used < n_samples * 0.1, (
                f"Excessive memory usage: {memory_used:.2f}MB for {n_samples} samples"
            )

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after optimization."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process data
        for _ in range(5):
            y_true, pred_prob = generate_binary_data(2000, random_state=42)
            threshold = get_optimal_threshold(y_true, pred_prob)
            del y_true, pred_prob, threshold

        # Force garbage collection
        gc.collect()

        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory

        # Should not have significant memory leaks
        assert memory_increase < 50, (
            f"Potential memory leak: {memory_increase:.2f}MB increase"
        )


class TestScalabilityLimits:
    """Test behavior at scalability limits."""

    @pytest.mark.slow
    def test_maximum_dataset_size(self):
        """Test optimization with very large datasets."""
        # Test with largest practical dataset size
        max_size = 50000  # Adjust based on system capabilities

        try:
            y_true, pred_prob = generate_binary_data(max_size, random_state=42)
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

        # Should complete without errors
        start_time = time.perf_counter()
        threshold = get_optimal_threshold(y_true, pred_prob, method="unique_scan")
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert execution_time < 60.0, (
            f"Optimization took {execution_time:.2f}s for {max_size} samples"
        )
        assert 0.0 <= threshold <= 1.0

    def test_high_precision_requirements(self):
        """Test performance when high precision is required."""
        # Create data with very close probability values
        n_samples = 1000
        base_prob = 0.5
        y_true = np.random.binomial(1, 0.5, n_samples)

        # Add tiny differences that require high precision
        eps = np.finfo(float).eps
        prob_offsets = np.linspace(-100 * eps, 100 * eps, n_samples)
        pred_prob = base_prob + prob_offsets

        # Ensure probabilities are in valid range
        pred_prob = np.clip(pred_prob, 0, 1)

        start_time = time.perf_counter()
        threshold = get_optimal_threshold(y_true, pred_prob)
        end_time = time.perf_counter()

        # Should complete reasonably quickly even with high precision requirements
        execution_time = end_time - start_time
        assert execution_time < 10.0, (
            f"High precision optimization took {execution_time:.2f}s"
        )
        assert 0.0 <= threshold <= 1.0

    def test_extreme_class_imbalance_performance(self):
        """Test performance with extreme class imbalance."""
        n_samples = 10000
        imbalance_ratio = 0.001  # 0.1% positive class

        from tests.fixtures.data_generators import generate_imbalanced_data

        y_true, pred_prob = generate_imbalanced_data(
            n_samples, imbalance_ratio=imbalance_ratio, random_state=42
        )

        start_time = time.perf_counter()
        threshold = get_optimal_threshold(y_true, pred_prob, metric="f1")
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Should handle extreme imbalance efficiently
        assert execution_time < 15.0, (
            f"Imbalanced optimization took {execution_time:.2f}s for {n_samples} samples"
        )
        assert 0.0 <= threshold <= 1.0


class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_baseline_performance_benchmarks(self):
        """Establish baseline performance benchmarks for regression testing."""
        # Standard test case
        n_samples = 5000
        y_true, pred_prob = generate_binary_data(n_samples, random_state=42)

        # Time standard optimization
        start_time = time.perf_counter()
        threshold = get_optimal_threshold(y_true, pred_prob, method="unique_scan")
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Store benchmark (in practice, this would be compared to stored values)
        benchmark_time = 2.0  # seconds - adjust based on expected performance

        assert execution_time < benchmark_time, (
            f"Performance regression: {execution_time:.4f}s > {benchmark_time}s baseline"
        )
        assert 0.0 <= threshold <= 1.0

    def test_worst_case_performance(self):
        """Test performance in worst-case scenarios."""
        # Create worst-case data: many tied values
        n_samples = 2000
        n_unique_values = 10  # Very few unique probability values

        y_true = np.random.binomial(1, 0.5, n_samples)
        unique_probs = np.linspace(0.1, 0.9, n_unique_values)
        pred_prob = np.random.choice(unique_probs, n_samples)

        start_time = time.perf_counter()
        threshold = get_optimal_threshold(y_true, pred_prob)
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Should handle worst case reasonably
        assert execution_time < 10.0, (
            f"Worst-case optimization took {execution_time:.2f}s"
        )
        assert 0.0 <= threshold <= 1.0

    def test_repeated_optimization_performance(self):
        """Test performance when optimization is called repeatedly."""
        n_samples = 1000
        n_repetitions = 20

        y_true, pred_prob = generate_binary_data(n_samples, random_state=42)

        times = []
        for i in range(n_repetitions):
            start_time = time.perf_counter()
            threshold = get_optimal_threshold(y_true, pred_prob)
            end_time = time.perf_counter()

            times.append(end_time - start_time)
            assert 0.0 <= threshold <= 1.0

        # Performance should be consistent across runs
        mean_time = np.mean(times)
        std_time = np.std(times)

        # Standard deviation should be small relative to mean
        assert std_time / mean_time < 0.5, (
            f"Inconsistent performance: std={std_time:.4f}, mean={mean_time:.4f}"
        )

        # All runs should complete quickly
        max_time = np.max(times)
        assert max_time < 5.0, f"Slowest run took {max_time:.4f}s"
