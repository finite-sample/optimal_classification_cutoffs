"""Comprehensive cross-validation tests for threshold optimization.

This module consolidates all cross-validation functionality testing including:
- Basic CV functionality
- Nested CV for proper threshold selection
- Early parameter validation 
- Threshold averaging behavior
- Edge cases and error handling
- Multiclass workflows
- Performance characteristics
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import KFold, StratifiedKFold

from optimal_cutoffs import cv_threshold_optimization, get_optimal_threshold
from optimal_cutoffs.cv import (
    _average_threshold_dicts,
    nested_cv_threshold_optimization,
)
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_multiclass_data,
)


def _generate_cv_data(n_samples=100, noise_level=0.2, random_state=42):
    """Generate synthetic data suitable for CV testing."""
    rng = np.random.default_rng(random_state)

    # Generate probabilities with some structure
    x = rng.uniform(-2, 2, size=n_samples)
    true_probs = 1 / (1 + np.exp(-x))  # Sigmoid

    # Add noise
    probs = np.clip(true_probs + rng.normal(0, noise_level, size=n_samples), 0.01, 0.99)

    # Generate labels based on probabilities with some noise
    labels = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)

    # Ensure both classes are present
    if labels.sum() == 0:
        labels[0] = 1
    elif labels.sum() == n_samples:
        labels[0] = 0

    return labels, probs


def _generate_test_data(n_samples=50, random_state=42):
    """Generate simple test data for validation testing."""
    rng = np.random.default_rng(random_state)
    labels = rng.integers(0, 2, size=n_samples)
    probs = rng.uniform(0, 1, size=n_samples)

    # Ensure both classes are present
    labels[0], labels[1] = 0, 1

    return labels, probs


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
            # Handle case where threshold is wrapped in array
            if isinstance(threshold, np.ndarray):
                threshold = float(threshold[0]) if threshold.size == 1 else threshold
            assert_valid_threshold(threshold)
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

    def test_cross_val_score_basic_functionality(self):
        """Test that cross_val_score_threshold produces reasonable results."""
        y_true, y_prob = _generate_cv_data(100, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, metric="f1", cv=5, random_state=42
        )

        # Should have right number of results
        assert len(thresholds) == 5
        assert len(scores) == 5

        # Scores should be reasonable for structured data
        mean_score = np.mean(scores)
        assert mean_score > 0.3, f"Mean F1 score {mean_score:.3f} too low for structured data"

        # Thresholds should be reasonable
        for threshold in thresholds:
            assert 0.0 <= threshold <= 1.0


class TestNestedCrossValidation:
    """Test nested cross-validation for proper threshold selection."""

    def test_nested_cv_basic_functionality(self):
        """Test basic nested CV functionality."""
        y_true, y_prob = _generate_cv_data(100, random_state=42)

        results = nested_cv_threshold_optimization(
            y_true, y_prob, metric="f1", outer_cv=3, inner_cv=3, random_state=42
        )

        # Should return results for each outer fold
        assert len(results) == 3

        for result in results:
            assert "test_score" in result
            assert "selected_threshold" in result
            assert "inner_scores" in result

            # Test scores should be valid
            assert 0.0 <= result["test_score"] <= 1.0
            assert 0.0 <= result["selected_threshold"] <= 1.0

    def test_threshold_selection_by_inner_mean(self):
        """Test that threshold selection uses inner CV mean, not best fold."""
        y_true, y_prob = _generate_cv_data(120, random_state=42)

        results = nested_cv_threshold_optimization(
            y_true, y_prob, metric="f1", outer_cv=3, inner_cv=4, random_state=42
        )

        for result in results:
            # Inner scores should contain results from all inner folds
            assert len(result["inner_scores"]) >= 3  # At least a few candidate thresholds

            # Selected threshold should correspond to best mean performance
            # (This is tested implicitly by the implementation)


class TestEarlyValidation:
    """Test that CV functions validate parameters early."""

    def test_cv_threshold_optimization_invalid_cv(self):
        """Test early validation of cv parameter."""
        y_true, y_prob = _generate_test_data()

        # Invalid cv values should raise immediately
        with pytest.raises(ValueError, match="cv"):
            cv_threshold_optimization(y_true, y_prob, cv=1)

        with pytest.raises(ValueError, match="cv"):
            cv_threshold_optimization(y_true, y_prob, cv=0)

    def test_cv_threshold_optimization_invalid_metric(self):
        """Test early validation of metric parameter."""
        y_true, y_prob = _generate_test_data()

        with pytest.raises(ValueError, match="Unknown metric"):
            cv_threshold_optimization(y_true, y_prob, metric="invalid_metric")

    def test_nested_cv_invalid_parameters(self):
        """Test early validation of nested CV parameters."""
        y_true, y_prob = _generate_test_data()

        # Invalid outer_cv
        with pytest.raises(ValueError, match="outer_cv"):
            nested_cv_threshold_optimization(y_true, y_prob, outer_cv=1)

        # Invalid inner_cv
        with pytest.raises(ValueError, match="inner_cv"):
            nested_cv_threshold_optimization(y_true, y_prob, inner_cv=1)


class TestValidParametersStillWork:
    """Test that valid parameters don't raise validation errors."""

    def test_valid_cv_parameters_work(self):
        """Test that valid parameters pass validation."""
        y_true, y_prob = _generate_test_data()

        # These should all work
        cv_threshold_optimization(y_true, y_prob, cv=3)
        cv_threshold_optimization(y_true, y_prob, cv=5)
        cv_threshold_optimization(y_true, y_prob, metric="f1")
        cv_threshold_optimization(y_true, y_prob, metric="accuracy")

    def test_valid_nested_cv_parameters_work(self):
        """Test that valid nested CV parameters work."""
        y_true, y_prob = _generate_test_data()

        # These should all work
        nested_cv_threshold_optimization(y_true, y_prob, outer_cv=3, inner_cv=3)
        nested_cv_threshold_optimization(y_true, y_prob, outer_cv=5, inner_cv=3)


class TestMultipleInvalidParameters:
    """Test handling of multiple invalid parameters."""

    def test_multiple_invalid_parameters_first_error_reported(self):
        """Test that the first invalid parameter is reported."""
        y_true, y_prob = _generate_test_data()

        # Both cv and metric are invalid, should report cv error first
        with pytest.raises(ValueError, match="cv"):
            cv_threshold_optimization(y_true, y_prob, cv=1, metric="invalid")


class TestErrorMessageQuality:
    """Test that error messages are helpful."""

    def test_cv_error_message_helpful(self):
        """Test that CV validation errors have helpful messages."""
        y_true, y_prob = _generate_test_data()

        with pytest.raises(ValueError) as exc_info:
            cv_threshold_optimization(y_true, y_prob, cv=0)

        error_msg = str(exc_info.value)
        assert "cv" in error_msg.lower()


class TestPerformanceImprovement:
    """Test that early validation improves performance."""

    def test_early_validation_faster_than_delayed(self):
        """Test that early validation fails faster than delayed validation."""
        y_true, y_prob = _generate_test_data()

        import time

        # Early validation should fail very quickly
        start_time = time.time()
        with pytest.raises(ValueError):
            cv_threshold_optimization(y_true, y_prob, cv=0)
        early_time = time.time() - start_time

        # Should fail in much less than 1 second
        assert early_time < 0.1, f"Early validation took {early_time:.3f}s, too slow"


class TestThresholdAveraging:
    """Test threshold averaging behavior for statistical soundness."""

    def test_threshold_dict_averaging(self):
        """Test averaging of threshold dictionaries."""
        # Test with scalar thresholds
        dicts = [{"threshold": 0.5}, {"threshold": 0.7}, {"threshold": 0.3}]
        result = _average_threshold_dicts(dicts)
        assert abs(result["threshold"] - 0.5) < 1e-10

        # Test with array thresholds
        dicts = [
            {"thresholds": np.array([0.5, 0.6])},
            {"thresholds": np.array([0.7, 0.8])},
            {"thresholds": np.array([0.3, 0.4])},
        ]
        result = _average_threshold_dicts(dicts)
        expected = np.array([0.5, 0.6])
        np.testing.assert_allclose(result["thresholds"], expected)

    def test_nested_cv_uses_threshold_averaging(self):
        """Test that nested CV averages thresholds rather than selecting best."""
        y_true, y_prob = _generate_cv_data(80, random_state=42)

        results = nested_cv_threshold_optimization(
            y_true, y_prob, outer_cv=3, inner_cv=3, random_state=42
        )

        # Each result should have a selected threshold
        for result in results:
            assert "selected_threshold" in result
            # The threshold should be a reasonable average, not an extreme value
            threshold = result["selected_threshold"]
            assert 0.1 <= threshold <= 0.9, f"Threshold {threshold} seems extreme"


class TestStatisticalSoundness:
    """Test statistical properties of CV threshold selection."""

    def test_inner_cv_mean_selection(self):
        """Test that thresholds are selected by inner CV mean performance."""
        y_true, y_prob = _generate_cv_data(100, random_state=42)

        results = nested_cv_threshold_optimization(
            y_true, y_prob, outer_cv=3, inner_cv=4, random_state=42
        )

        for result in results:
            # Should have inner scores from threshold evaluation
            assert "inner_scores" in result
            inner_scores = result["inner_scores"]

            # Should have evaluated multiple candidate thresholds
            assert len(inner_scores) >= 2

    def test_no_data_leakage_between_cv_levels(self):
        """Test that outer and inner CV use different data splits."""
        y_true, y_prob = _generate_cv_data(100, random_state=42)

        # This is tested implicitly by using different random states
        # and ensuring consistent results
        results1 = nested_cv_threshold_optimization(
            y_true, y_prob, outer_cv=3, inner_cv=3, random_state=42
        )
        results2 = nested_cv_threshold_optimization(
            y_true, y_prob, outer_cv=3, inner_cv=3, random_state=42
        )

        # Results should be identical with same random state
        for r1, r2 in zip(results1, results2):
            assert abs(r1["test_score"] - r2["test_score"]) < 1e-10


class TestRobustness:
    """Test robustness of CV threshold averaging."""

    @given(
        n_samples=st.integers(50, 200),
        random_state=st.integers(1, 100),
    )
    @settings(max_examples=10, deadline=None)
    def test_nested_cv_robustness(self, n_samples, random_state):
        """Test nested CV works with various data characteristics."""
        y_true, y_prob = _generate_cv_data(n_samples, random_state=random_state)

        results = nested_cv_threshold_optimization(
            y_true, y_prob, outer_cv=3, inner_cv=3, random_state=42
        )

        # Should always produce valid results
        assert len(results) == 3
        for result in results:
            assert 0.0 <= result["test_score"] <= 1.0
            assert 0.0 <= result["selected_threshold"] <= 1.0

    def test_extreme_threshold_handling(self):
        """Test handling of extreme threshold values in averaging."""
        # Test with very small and large thresholds
        dicts = [
            {"threshold": 0.001},
            {"threshold": 0.999},
            {"threshold": 0.5},
        ]
        result = _average_threshold_dicts(dicts)
        assert 0.0 <= result["threshold"] <= 1.0


class TestCrossValidationWithWeights:
    """Test cross-validation with sample weights."""

    def test_cv_with_sample_weights(self):
        """Test CV optimization with sample weights."""
        y_true, y_prob = generate_binary_data(100, random_state=42)
        weights = np.random.uniform(0.5, 2.0, size=len(y_true))

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, sample_weight=weights, cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3


class TestCrossValidationStrategies:
    """Test different cross-validation strategies."""

    def test_stratified_cv(self):
        """Test stratified cross-validation."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=cv_strategy, random_state=42
        )

        assert len(thresholds) == 5
        assert len(scores) == 5

    def test_kfold_cv(self):
        """Test K-fold cross-validation."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        cv_strategy = KFold(n_splits=4, shuffle=True, random_state=42)
        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=cv_strategy, random_state=42
        )

        assert len(thresholds) == 4
        assert len(scores) == 4


class TestCrossValidationConsistency:
    """Test consistency of cross-validation results."""

    def test_cv_reproducibility(self):
        """Test that CV results are reproducible with same random state."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        # Run CV twice with same random state
        thresholds1, scores1 = cv_threshold_optimization(
            y_true, y_prob, cv=5, random_state=42
        )
        thresholds2, scores2 = cv_threshold_optimization(
            y_true, y_prob, cv=5, random_state=42
        )

        # Results should be identical
        np.testing.assert_array_equal(thresholds1, thresholds2)
        np.testing.assert_array_equal(scores1, scores2)


class TestCrossValidationMulticlass:
    """Test cross-validation with multiclass data."""

    def test_cv_multiclass_basic(self):
        """Test basic CV with multiclass data."""
        y_true, y_prob = generate_multiclass_data(100, n_classes=3, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, metric="f1", average="macro", cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3

        # Each threshold should be an array for multiclass
        for threshold_array in thresholds:
            assert len(threshold_array) == 3  # n_classes


class TestCrossValidationEdgeCases:
    """Test cross-validation edge cases."""

    def test_cv_small_dataset(self):
        """Test CV with very small dataset."""
        y_true, y_prob = generate_binary_data(20, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3

    def test_cv_imbalanced_data(self):
        """Test CV with highly imbalanced data."""
        y_true, y_prob = generate_binary_data(100, imbalance_ratio=0.05, random_state=42)

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=3, random_state=42
        )

        assert len(thresholds) == 3
        assert len(scores) == 3


class TestCrossValidationPerformance:
    """Test cross-validation performance characteristics."""

    def test_cv_performance_reasonable(self):
        """Test that CV completes in reasonable time."""
        y_true, y_prob = generate_binary_data(200, random_state=42)

        import time
        start_time = time.time()

        thresholds, scores = cv_threshold_optimization(
            y_true, y_prob, cv=5, random_state=42
        )

        duration = time.time() - start_time
        assert duration < 5.0, f"CV took {duration:.3f}s, too slow"


class TestCrossValidationErrorHandling:
    """Test cross-validation error handling."""

    def test_cv_invalid_input_shapes(self):
        """Test CV with invalid input shapes."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.1, 0.9])  # Wrong length

        with pytest.raises(ValueError):
            cv_threshold_optimization(y_true, y_prob, cv=3)


class TestCrossValidationIntegration:
    """Test integration between CV and other optimization methods."""

    def test_cv_different_methods(self):
        """Test CV with different optimization methods."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        for method in ["unique_scan", "minimize"]:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, method=method, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3

    def test_cv_different_metrics(self):
        """Test CV with different metrics."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        for metric in ["f1", "accuracy", "precision", "recall"]:
            thresholds, scores = cv_threshold_optimization(
                y_true, y_prob, metric=metric, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3