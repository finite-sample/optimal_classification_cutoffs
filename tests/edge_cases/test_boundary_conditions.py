"""Edge case tests for boundary conditions and extreme scenarios.

This module tests boundary conditions, extreme data distributions, and
numerical precision limits that could cause optimization algorithms to fail.
"""

import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import compute_metric_at_threshold
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_extreme_probabilities,
    generate_imbalanced_data,
    generate_tied_probabilities,
)


class TestLabelDistributionEdgeCases:
    """Test edge cases in label distributions."""

    def test_all_positive_labels(self):
        """Test optimization when all labels are positive."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.2, 0.5, 0.7, 0.9])

        # Should find threshold that predicts all positive
        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # Accuracy should be perfect (1.0) with appropriate threshold
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "accuracy")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_all_negative_labels(self):
        """Test optimization when all labels are negative."""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.3, 0.6, 0.8])

        # Should find threshold that predicts all negative
        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # Accuracy should be perfect (1.0) with appropriate threshold
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "accuracy")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_single_positive_sample(self):
        """Test optimization with only one positive sample."""
        y_true = np.array([0, 0, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.9])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    def test_extreme_imbalance(self):
        """Test optimization with extreme class imbalance."""
        y_true, y_prob = generate_imbalanced_data(
            1000, imbalance_ratio=0.001, random_state=42
        )

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")


class TestProbabilityDistributionEdgeCases:
    """Test edge cases in probability distributions."""

    def test_perfectly_separated_classes(self):
        """Test optimization with perfectly separated classes."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        # Should achieve perfect F1 score
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_all_identical_probabilities(self):
        """Test optimization when all probabilities are identical."""
        y_true, y_prob = generate_tied_probabilities(
            20, base_prob=0.5, tie_fraction=1.0, random_state=42
        )

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    def test_extreme_probability_values(self):
        """Test optimization with extreme probability values."""
        y_true, y_prob = generate_extreme_probabilities(30, random_state=42)

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    def test_boundary_probabilities(self):
        """Test optimization with probabilities at 0.0 and 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        # Should achieve perfect separation
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert score == pytest.approx(1.0, abs=1e-10)


class TestNumericalEdgeCases:
    """Test numerical precision and scaling edge cases."""

    def test_very_small_dataset(self):
        """Test optimization with minimal dataset size."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    def test_machine_epsilon_differences(self):
        """Test optimization with probability differences at machine epsilon."""
        eps = np.finfo(float).eps
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5 + eps, 0.5 + 2 * eps, 0.5 + 3 * eps])

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    def test_very_close_probabilities(self):
        """Test optimization with very close probability values."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array(
            [0.5000001, 0.5000002, 0.5000003, 0.5000004, 0.5000005, 0.5000006]
        )

        threshold = get_optimal_threshold(y_true, y_prob, metric="f1")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")


class TestDegenrateCaseHandling:
    """Test handling of degenerate cases."""

    def test_no_positive_predictions_possible(self):
        """Test case where optimal solution is to predict all negative."""
        # All negatives with low probabilities
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])

        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # Should predict all negative for best accuracy
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "accuracy")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_no_negative_predictions_possible(self):
        """Test case where optimal solution is to predict all positive."""
        # All positives with high probabilities
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])

        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # Should predict all positive for best accuracy
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "accuracy")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_undefined_metric_cases(self):
        """Test cases where metrics might be undefined."""
        # Case where precision might be undefined (no predicted positives)
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])  # All below typical threshold

        # Optimization should still work even if some thresholds give undefined metrics
        threshold = get_optimal_threshold(y_true, y_prob, metric="precision")
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "precision")
        assert_valid_metric_score(score, "precision", allow_nan=True)


class TestComparisonOperatorEdgeCases:
    """Test edge cases with different comparison operators."""

    def test_tied_probabilities_comparison_operators(self):
        """Test that comparison operators handle ties consistently."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])  # All tied

        # Both operators should work
        threshold_gt = get_optimal_threshold(
            y_true, y_prob, metric="f1", comparison=">"
        )
        threshold_gte = get_optimal_threshold(
            y_true, y_prob, metric="f1", comparison=">="
        )

        assert_valid_threshold(threshold_gt)
        assert_valid_threshold(threshold_gte)

        # Results should be reasonable even if different
        score_gt = compute_metric_at_threshold(
            y_true, y_prob, threshold_gt, "f1", comparison=">"
        )
        score_gte = compute_metric_at_threshold(
            y_true, y_prob, threshold_gte, "f1", comparison=">="
        )

        assert_valid_metric_score(score_gt, "f1")
        assert_valid_metric_score(score_gte, "f1")


class TestScalingLimits:
    """Test scaling behavior at extreme sizes."""

    @pytest.mark.slow
    def test_large_dataset_stability(self):
        """Test that optimization remains stable with large datasets."""
        # This test is marked slow as it uses large data
        from tests.fixtures.data_generators import generate_binary_data

        y_true, y_prob = generate_binary_data(10000, random_state=42)

        threshold = get_optimal_threshold(
            y_true, y_prob, metric="f1", method="unique_scan"
        )
        assert_valid_threshold(threshold)

        score = compute_metric_at_threshold(y_true, y_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")
        assert score > 0.1  # Should achieve reasonable performance

    def test_single_sample_edge_case(self):
        """Test optimization with single sample."""
        y_true = np.array([1])
        y_prob = np.array([0.7])

        threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
        assert_valid_threshold(threshold)

        # With single positive sample, accuracy is either 0 or 1
        score = compute_metric_at_threshold(y_true, y_prob, threshold, "accuracy")
        assert score in [0.0, 1.0]
