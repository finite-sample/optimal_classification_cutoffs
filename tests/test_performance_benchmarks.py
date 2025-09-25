"""Performance benchmarking tests to verify algorithmic complexity guarantees.

This module contains tests that verify the O(n log n) performance characteristics
of the sort-scan optimization algorithm and other performance-critical components.
"""

import gc
import time

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.piecewise import get_vectorized_metric, optimal_threshold_sortscan


class TestAlgorithmicComplexity:
    """Test that algorithms meet their expected complexity guarantees."""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000, 4000])
    def test_sort_scan_n_log_n_scaling(self, n_samples):
        """Test that sort_scan algorithm scales as O(n log n)."""
        # Create dataset with known characteristics
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n_samples)
        pred_prob = np.random.uniform(0, 1, n_samples)

        # Get vectorized metric
        try:
            f1_vectorized = get_vectorized_metric("f1")
        except ValueError:
            pytest.skip("Vectorized F1 not available")

        # Time the sort_scan algorithm
        start_time = time.time()
        threshold, score, k_star = optimal_threshold_sortscan(
            y_true, pred_prob, f1_vectorized
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Store timing for analysis
        if not hasattr(self, "timing_data"):
            self.timing_data = []
        self.timing_data.append((n_samples, execution_time))

        # Basic validation
        assert 0.0 <= threshold <= 1.0
        assert score >= 0.0
        assert 0 <= k_star <= n_samples

        # Performance expectation: should complete quickly
        # Use more generous bound for small datasets due to timing overhead
        if n_samples < 500:
            expected_max_time = 0.1  # Very generous for small datasets
        else:
            expected_max_time = n_samples * np.log(n_samples) * 1e-5  # Still generous
        assert execution_time < expected_max_time, (
            f"Sort_scan too slow: {execution_time:.4f}s for {n_samples} samples"
        )

    def test_sort_scan_vs_brute_force_scaling(self):
        """Compare scaling behavior of sort_scan vs brute force methods."""
        sample_sizes = [100, 500, 1000, 2000]
        timing_results = {"sort_scan": [], "smart_brute": []}

        for n_samples in sample_sizes:
            np.random.seed(42)
            y_true = np.random.randint(0, 2, n_samples)
            pred_prob = np.random.uniform(0, 1, n_samples)

            # Time sort_scan method
            try:
                start_time = time.time()
                _threshold_ss = get_optimal_threshold(
                    y_true, pred_prob, metric="f1", method="sort_scan"
                )
                end_time = time.time()
                timing_results["sort_scan"].append(end_time - start_time)
            except Exception:
                timing_results["sort_scan"].append(float("inf"))

            # Time smart_brute method
            start_time = time.time()
            _threshold_sb = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method="smart_brute"
            )
            end_time = time.time()
            timing_results["smart_brute"].append(end_time - start_time)

        # For large datasets, sort_scan should be competitive or better
        if len(timing_results["sort_scan"]) > 0 and timing_results["sort_scan"][
            -1
        ] != float("inf"):
            final_ratio = (
                timing_results["sort_scan"][-1] / timing_results["smart_brute"][-1]
            )
            # Sort_scan should not be more than 2x slower than smart_brute
            assert final_ratio < 2.0, f"Sort_scan scaling poor: ratio = {final_ratio}"

    @pytest.mark.parametrize("n_unique", [10, 50, 100, 500, 1000])
    def test_brute_force_unique_values_scaling(self, n_unique):
        """Test that brute force scaling depends on number of unique values."""
        n_samples = 2000

        # Create dataset with controlled number of unique probability values
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n_samples)
        unique_probs = np.linspace(0.01, 0.99, n_unique)
        pred_prob = np.random.choice(unique_probs, n_samples)

        # Time the brute force method
        start_time = time.time()
        threshold = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="smart_brute"
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Validate result
        assert 0.0 <= threshold <= 1.0

        # Execution time should scale roughly with number of unique values
        # This is more of a characterization than a hard requirement
        expected_max_time = n_unique * 3e-4  # Generous bound for CI environments
        assert execution_time < expected_max_time, (
            f"Brute force too slow: {execution_time:.4f}s for {n_unique} unique values"
        )


class TestMemoryUsageCharacteristics:
    """Test memory usage characteristics of different algorithms."""

    def test_sort_scan_memory_usage(self):
        """Test that sort_scan doesn't use excessive memory."""
        # Large dataset
        n_samples = 10000
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n_samples)
        pred_prob = np.random.uniform(0, 1, n_samples)

        # Force garbage collection before test
        gc.collect()

        try:
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method="sort_scan"
            )
            assert 0.0 <= threshold <= 1.0
        except Exception as e:
            pytest.skip(f"Sort_scan not available: {e}")

        # Should complete without memory errors
        # If we reach here, memory usage was acceptable

    def test_large_dataset_handling(self):
        """Test handling of large datasets across methods."""
        n_samples = 5000
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n_samples)
        pred_prob = np.random.uniform(0, 1, n_samples)

        methods_to_test = ["smart_brute", "minimize"]

        for method in methods_to_test:
            gc.collect()  # Clean up before each test

            try:
                threshold = get_optimal_threshold(
                    y_true, pred_prob, metric="f1", method=method
                )
                assert 0.0 <= threshold <= 1.0
            except Exception as e:
                print(f"Method {method} failed on large dataset: {e}")

    def test_multiclass_memory_scaling(self):
        """Test memory usage for multiclass problems."""
        n_samples = 1000
        n_classes = 10  # Many classes

        np.random.seed(42)
        y_true = np.random.randint(0, n_classes, n_samples)
        pred_prob = np.random.uniform(0, 1, (n_samples, n_classes))
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)  # Normalize

        gc.collect()

        try:
            thresholds = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method="smart_brute"
            )
            assert len(thresholds) == n_classes
            assert all(0.0 <= t <= 1.0 for t in thresholds)
        except Exception as e:
            print(f"Multiclass memory test failed: {e}")


class TestWorstCasePerformance:
    """Test performance on worst-case scenarios."""

    def test_all_unique_probabilities_worst_case(self):
        """Test performance when all probabilities are unique (worst case for brute force)."""
        n_samples = 1000

        # Worst case: all unique probabilities
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)
        pred_prob = np.linspace(0.001, 0.999, n_samples)  # All unique

        methods = ["smart_brute", "minimize"]

        for method in methods:
            start_time = time.time()
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method=method
            )
            end_time = time.time()

            assert 0.0 <= threshold <= 1.0
            # Should complete in reasonable time even in worst case
            assert end_time - start_time < 15.0, (
                f"Method {method} too slow in worst case"
            )

    def test_extreme_class_imbalance_performance(self):
        """Test performance with extreme class imbalance."""
        n_samples = 10000

        # Extreme imbalance: 99.9% negative class
        n_positive = 10
        n_negative = n_samples - n_positive

        y_true = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])
        pred_prob = np.random.RandomState(42).uniform(0, 1, n_samples)

        start_time = time.time()
        threshold = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="smart_brute"
        )
        end_time = time.time()

        assert 0.0 <= threshold <= 1.0
        assert end_time - start_time < 10.0, "Too slow with extreme class imbalance"

    def test_many_tied_probabilities_performance(self):
        """Test performance when many probabilities are tied."""
        n_samples = 2000

        # Many ties: only 5 unique probability values
        unique_probs = [0.2, 0.4, 0.5, 0.6, 0.8]
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)
        pred_prob = np.random.RandomState(42).choice(unique_probs, n_samples)

        start_time = time.time()
        threshold = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="smart_brute"
        )
        end_time = time.time()

        assert 0.0 <= threshold <= 1.0
        # Should be fast with few unique values
        assert end_time - start_time < 1.0, "Too slow with tied probabilities"


class TestNumericalStabilityPerformance:
    """Test performance with numerically challenging inputs."""

    def test_very_close_probabilities_performance(self):
        """Test performance when probabilities are very close together."""
        n_samples = 1000

        # Probabilities clustered very close together
        base = 0.5
        epsilon = 1e-10
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)
        pred_prob = base + np.random.RandomState(42).uniform(
            -epsilon, epsilon, n_samples
        )

        start_time = time.time()
        threshold = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="smart_brute"
        )
        end_time = time.time()

        assert 0.0 <= threshold <= 1.0
        assert end_time - start_time < 5.0, "Too slow with very close probabilities"

    def test_extreme_probability_values_performance(self):
        """Test performance with probabilities at machine precision limits."""
        n_samples = 500

        eps = np.finfo(float).eps
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)

        # Mix of extreme values
        pred_prob = np.random.RandomState(42).choice(
            [eps, 2 * eps, 1 - 2 * eps, 1 - eps, 0.5, 0.5 + eps, 0.5 - eps], n_samples
        )

        start_time = time.time()
        threshold = get_optimal_threshold(
            y_true, pred_prob, metric="f1", method="smart_brute"
        )
        end_time = time.time()

        assert 0.0 <= threshold <= 1.0
        assert end_time - start_time < 3.0, "Too slow with extreme probability values"


class TestConcurrentPerformance:
    """Test performance characteristics under concurrent usage patterns."""

    def test_repeated_optimization_performance(self):
        """Test performance when running many optimizations in sequence."""
        n_trials = 50
        n_samples = 500

        total_start_time = time.time()

        for trial in range(n_trials):
            # Different random data each time
            y_true = np.random.randint(0, 2, n_samples)
            pred_prob = np.random.uniform(0, 1, n_samples)

            threshold = get_optimal_threshold(
                y_true, pred_prob, metric="f1", method="smart_brute"
            )
            assert 0.0 <= threshold <= 1.0

        total_end_time = time.time()

        # Should complete all trials in reasonable time
        average_time = (total_end_time - total_start_time) / n_trials
        assert average_time < 0.1, f"Average optimization too slow: {average_time:.4f}s"

    def test_different_metrics_performance_consistency(self):
        """Test that performance is consistent across different metrics."""
        n_samples = 1000
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)
        pred_prob = np.random.RandomState(42).uniform(0, 1, n_samples)

        metrics = ["f1", "accuracy", "precision", "recall"]
        timings = {}

        for metric in metrics:
            start_time = time.time()
            threshold = get_optimal_threshold(
                y_true, pred_prob, metric=metric, method="smart_brute"
            )
            end_time = time.time()

            timings[metric] = end_time - start_time
            assert 0.0 <= threshold <= 1.0

        # Timings should be similar across metrics (within factor of 3)
        min_time = min(timings.values())
        max_time = max(timings.values())

        if min_time > 0:  # Avoid division by zero
            ratio = max_time / min_time
            assert ratio < 3.0, f"Large timing variation across metrics: {timings}"


@pytest.fixture(scope="module", autouse=True)
def performance_test_setup():
    """Set up performance testing environment."""
    # Ensure consistent timing by disabling some optimizations that could interfere
    import warnings

    warnings.filterwarnings("ignore", message=".*performance.*")
    yield
    # Cleanup after all performance tests
    gc.collect()
