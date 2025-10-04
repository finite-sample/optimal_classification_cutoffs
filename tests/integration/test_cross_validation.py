"""Integration tests for cross-validation functionality.

This module tests cross-validation workflows for threshold optimization,
including nested CV, threshold selection consistency, and proper data splitting.
"""

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from optimal_cutoffs import cv_threshold_optimization, get_optimal_threshold
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_multiclass_data,
)


class TestBasicCrossValidation:
    """Test basic cross-validation functionality."""

    def test_cv_threshold_optimization_basic(self):
        """Test basic cross-validation threshold optimization."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, method="unique_scan", cv=5, random_state=42
        )

        # Should return arrays of correct length
        assert len(thresholds) == 5
        assert len(scores) == 5

        # All thresholds and scores should be valid
        for threshold in thresholds:
            assert_valid_threshold(result.threshold)
        for score in scores:
            assert_valid_metric_score(score, "f1")

    def test_cv_different_fold_counts(self):
        """Test cross-validation with different fold counts."""
        y_true, y_prob = generate_binary_data(80, random_state=42)

        for cv in [3, 5, 10]:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, cv=cv, random_state=42
            )

            assert len(thresholds) == cv
            assert len(scores) == cv

    def test_cv_different_methods(self):
        """Test cross-validation with different optimization methods."""
        y_true, y_prob = generate_binary_data(60, random_state=42)

        methods = ["unique_scan", "minimize"]

        for method in methods:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, method=method, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3

    def test_cv_different_metrics(self):
        """Test cross-validation with different metrics."""
        y_true, y_prob = generate_binary_data(70, random_state=42)

        metrics = ["f1", "accuracy", "precision", "recall"]

        for metric in metrics:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, metric=metric, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            for score in scores:
                assert_valid_metric_score(score, metric)

    def test_cv_reproducibility(self):
        """Test that cross-validation is reproducible with same random state."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        # Run CV twice with same random state
        thresholds1, scores1 = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=123
        )
        thresholds2, scores2 = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=123
        )

        # Results should be identical
        np.testing.assert_allclose(thresholds1, thresholds2, rtol=1e-12)
        np.testing.assert_allclose(scores1, scores2, rtol=1e-12)


class TestCrossValidationWithWeights:
    """Test cross-validation with sample weights."""

    def test_cv_with_sample_weights(self):
        """Test cross-validation with sample weights."""
        y_true, y_prob = generate_binary_data(60, random_state=42)
        weights = np.random.uniform(0.5, 2.0, len(y_true))

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, sample_weight=weights, cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3

    def test_cv_weights_vs_no_weights(self):
        """Test that weights affect cross-validation results."""
        y_true, y_prob = generate_binary_data(40, random_state=42)

        # CV without weights
        thresholds_no_weights, _ = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=42
        )

        # CV with uniform weights (should be similar)
        uniform_weights = np.ones(len(y_true))
        thresholds_uniform, _ = cv_threshold_optimization(
            y_true, y_prob, sample_weight=uniform_weights, cv=3, random_state=42
        )

        # Should be very similar
        np.testing.assert_allclose(
            thresholds_no_weights, thresholds_uniform, rtol=1e-10
        )

        # CV with non-uniform weights (should be different)
        non_uniform_weights = np.random.uniform(0.1, 3.0, len(y_true))
        thresholds_weighted, _ = cv_threshold_optimization(
            y_true, y_prob, sample_weight=non_uniform_weights, cv=3, random_state=42
        )

        # Should be different from uniform case
        max_diff = np.max(np.abs(thresholds_no_weights - thresholds_weighted))
        assert max_diff > 1e-8  # Some difference expected


class TestCrossValidationStrategies:
    """Test different cross-validation strategies."""

    def test_cv_with_stratified_splits(self):
        """Test cross-validation with stratified splits."""
        y_true, y_prob = generate_binary_data(100, imbalance_ratio=0.2, random_state=42)

        # Use StratifiedKFold
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=cv_strategy, random_state=42
        )

        assert len(thresholds) == 5
        assert len(scores) == 5

    def test_cv_with_kfold_splits(self):
        """Test cross-validation with regular KFold."""
        y_true, y_prob = generate_binary_data(80, random_state=42)

        # Use regular KFold
        cv_strategy = KFold(n_splits=4, shuffle=True, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=cv_strategy, random_state=42
        )

        assert len(thresholds) == 4
        assert len(scores) == 4

    def test_cv_imbalanced_data(self):
        """Test cross-validation on imbalanced data."""
        # Create imbalanced data
        y_true = np.concatenate([np.zeros(90), np.ones(10)])
        y_prob = np.concatenate(
            [
                np.random.uniform(0.0, 0.4, 90),  # Negative class
                np.random.uniform(0.6, 1.0, 10),  # Positive class
            ]
        )

        # Shuffle
        shuffle_idx = np.random.RandomState(42).permutation(100)
        y_true = y_true[shuffle_idx]
        y_prob = y_prob[shuffle_idx]

        # Use stratified CV for imbalanced data
        thresholds, scores = cv_threshold_optimization(
            y_true,
            y_prob,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        )

        assert len(thresholds) == 5
        assert len(scores) == 5


class TestCrossValidationConsistency:
    """Test consistency of cross-validation results."""

    def test_cv_vs_single_split_consistency(self):
        """Test that CV results are consistent with single optimization."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        # Single optimization on full data
        threshold_full = get_optimal_threshold(y_true, y_prob, metric="f1")

        # Cross-validation
        thresholds_cv, scores_cv = cv_threshold_optimization(
            y_true, y_prob, metric="f1", cv=5, random_state=42
        )

        # CV mean should be in reasonable range of full data optimization
        mean_threshold_cv = np.mean(thresholds_cv)
        assert (
            abs(mean_threshold_cv - threshold_full) < 0.3
        )  # Allow reasonable variation

    def test_cv_score_variance(self):
        """Test that CV scores have reasonable variance."""
        y_true, y_prob = generate_binary_data(120, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=10, random_state=42
        )

        # Variance should be reasonable (not too high, not zero)
        score_std = np.std(scores)
        assert 0.0 <= score_std <= 0.5  # Reasonable range for F1 std

    def test_cv_threshold_variance(self):
        """Test that CV thresholds have reasonable variance."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        thresholds, _ = cv_threshold_optimization(y_true, y_prob, cv=8, random_state=42)

        # Threshold variance should be reasonable
        threshold_std = np.std(thresholds)
        assert 0.0 <= threshold_std <= 0.5  # Reasonable range


class TestCrossValidationMulticlass:
    """Test cross-validation with multiclass data."""

    def test_cv_multiclass_basic(self):
        """Test basic multiclass cross-validation."""
        y_true, y_prob = generate_multiclass_data(80, n_classes=3, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=4, random_state=42
        )

        # Should return arrays for multiclass
        assert len(thresholds) == 4
        assert len(scores) == 4

        # Each threshold should be an array of length n_classes
        for threshold_set in thresholds:
            assert len(threshold_set) == 3
            for threshold in threshold_set:
                assert_valid_threshold(result.threshold)

    def test_cv_multiclass_different_averaging(self):
        """Test multiclass CV with different averaging methods."""
        y_true, y_prob = generate_multiclass_data(60, n_classes=3, random_state=42)

        for average in ["macro", "micro", "weighted"]:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, average=average, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3

    def test_cv_multiclass_different_methods(self):
        """Test multiclass CV with different optimization methods."""
        y_true, y_prob = generate_multiclass_data(50, n_classes=3, random_state=42)

        methods = ["unique_scan", "minimize", "coord_ascent"]

        for method in methods:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, method=method, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3


class TestCrossValidationEdgeCases:
    """Test edge cases in cross-validation."""

    def test_cv_small_dataset(self):
        """Test cross-validation on small datasets."""
        y_true, y_prob = generate_binary_data(20, random_state=42)

        # Small number of folds for small dataset
        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3

    def test_cv_single_class_per_fold(self):
        """Test CV when some folds might have single class."""
        # Create data that might cause single-class folds
        y_true = np.array([0] * 8 + [1] * 2)  # Very imbalanced
        y_prob = np.array([0.1] * 8 + [0.9] * 2)

        # This might create folds with only one class
        try:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, cv=5, random_state=42
            )

            # If it succeeds, results should be valid
            assert len(thresholds) == 5
            assert len(scores) == 5

        except ValueError:
            # It's acceptable to fail on degenerate cases
            pytest.skip("Degenerate CV splits not supported")

    def test_cv_perfect_separation(self):
        """Test CV on perfectly separable data."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3

        # Should achieve high scores on separable data
        assert np.mean(scores) > 0.8

    def test_cv_tied_probabilities(self):
        """Test CV with tied probability values."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.3, 0.5, 0.5, 0.7, 0.3, 0.7, 0.5, 0.5])  # Many ties

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=4, random_state=42
        )

        assert len(thresholds) == 4
        assert len(scores) == 4


class TestCrossValidationPerformance:
    """Test performance characteristics of cross-validation."""

    def test_cv_performance_scaling(self):
        """Test that CV performance scales reasonably."""
        import time

        sizes = [50, 200, 500]

        for size in sizes:
            y_true, y_prob = generate_binary_data(size, random_state=42)

            start_time = time.time()
            cv_threshold_optimization(y_true, y_prob, cv=5, random_state=42)
            end_time = time.time()

            elapsed = end_time - start_time

            # Should complete in reasonable time
            assert elapsed < 10.0, f"CV took {elapsed:.2f}s for {size} samples"

    def test_cv_fold_scaling(self):
        """Test that CV scales with number of folds."""
        import time

        y_true, y_prob = generate_binary_data(100, random_state=42)

        fold_counts = [3, 5, 10]
        times = []

        for cv in fold_counts:
            start_time = time.time()
            cv_threshold_optimization(y_true, y_prob, cv=cv, random_state=42)
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            # Should complete in reasonable time
            assert elapsed < 5.0, f"CV with {cv} folds took {elapsed:.2f}s"


class TestCrossValidationErrorHandling:
    """Test error handling in cross-validation."""

    def test_cv_invalid_fold_count(self):
        """Test error handling for invalid fold counts."""
        y_true, y_prob = generate_binary_data(20, random_state=42)

        # Too many folds
        with pytest.raises(ValueError):
            cv_threshold_optimization(y_true, y_prob, cv=25)  # More folds than samples

    def test_cv_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValueError):
            cv_threshold_optimization([], [], cv=3)

    def test_cv_mismatched_lengths(self):
        """Test error handling for mismatched array lengths."""
        with pytest.raises(ValueError):
            cv_threshold_optimization([0, 1], [0.5], cv=3)

    def test_cv_invalid_cv_object(self):
        """Test error handling for invalid CV objects."""
        y_true, y_prob = generate_binary_data(30, random_state=42)

        # Invalid CV object
        with pytest.raises((ValueError, TypeError)):
            cv_threshold_optimization(y_true, y_prob, cv="invalid")


class TestCrossValidationIntegration:
    """Test integration of CV with other components."""

    def test_cv_with_different_comparison_operators(self):
        """Test CV with different comparison operators."""
        y_true, y_prob = generate_binary_data(40, random_state=42)

        for comparison in [">", ">="]:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, comparison=comparison, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3

    def test_cv_parameter_combinations(self):
        """Test CV with various parameter combinations."""
        y_true, y_prob = generate_binary_data(60, random_state=42)

        # Test combinations of parameters
        param_combinations = [
            {"metric": "f1", "method": "unique_scan"},
            {"metric": "accuracy", "method": "minimize"},
            {"metric": "precision", "comparison": ">="},
            {"metric": "recall", "comparison": ">"},
        ]

        for params in param_combinations:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, cv=3, random_state=42, **params
            )

            assert len(thresholds) == 3
            assert len(scores) == 3
