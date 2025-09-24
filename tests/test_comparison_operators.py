"""Test comparison operator functionality for thresholding (inclusive vs exclusive)."""

import numpy as np
import pytest

from optimal_cutoffs import (
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    get_optimal_threshold,
)
from optimal_cutoffs.wrapper import ThresholdOptimizer


class TestComparisonOperators:
    """Test inclusive vs exclusive thresholding behavior."""

    def test_binary_confusion_matrix_comparison_operators(self):
        """Test that comparison operators affect confusion matrix calculations."""
        # Simple test case where comparison operator should make a difference
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array(
            [0.3, 0.5, 0.7, 0.5]
        )  # Note: two predictions exactly at 0.5
        threshold = 0.5

        # With ">" (exclusive), prob=0.5 predictions should be negative
        tp_gt, tn_gt, fp_gt, fn_gt = get_confusion_matrix(
            true_labels, pred_probs, threshold, comparison=">"
        )

        # With ">=" (inclusive), prob=0.5 predictions should be positive
        tp_gte, tn_gte, fp_gte, fn_gte = get_confusion_matrix(
            true_labels, pred_probs, threshold, comparison=">="
        )

        # Verify the results make sense
        # For ">": predictions are [0, 0, 1, 0] -> TP=0, TN=1, FP=1, FN=2
        assert tp_gt == 0 and tn_gt == 1 and fp_gt == 1 and fn_gt == 2

        # For ">=": predictions are [0, 1, 1, 1] -> TP=2, TN=1, FP=1, FN=0
        assert tp_gte == 2 and tn_gte == 1 and fp_gte == 1 and fn_gte == 0

        # Verify they are different
        assert (tp_gt, tn_gt, fp_gt, fn_gt) != (tp_gte, tn_gte, fp_gte, fn_gte)

    def test_binary_threshold_optimization_comparison_operators(self):
        """Test that comparison operators can affect optimal threshold selection."""
        # Create data where the optimal threshold might be different
        # depending on whether we use > or >=
        np.random.seed(42)
        true_labels = np.random.randint(0, 2, 100)
        pred_probs = np.random.rand(100)

        # Get optimal thresholds with both operators
        thresh_gt = get_optimal_threshold(true_labels, pred_probs, "f1", comparison=">")
        thresh_gte = get_optimal_threshold(
            true_labels, pred_probs, "f1", comparison=">="
        )

        # Both should be valid thresholds
        assert 0 <= thresh_gt <= 1
        assert 0 <= thresh_gte <= 1

        # They might be the same or different, both are valid
        # The key is that the function accepts both operators without error

    def test_multiclass_confusion_matrix_comparison_operators(self):
        """Test comparison operators with multiclass confusion matrices."""
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        pred_probs = np.array(
            [
                [0.7, 0.2, 0.1],  # class 0
                [0.1, 0.6, 0.3],  # class 1
                [0.2, 0.3, 0.5],  # class 2
                [0.5, 0.3, 0.2],  # class 0 (tie at threshold)
                [0.3, 0.5, 0.2],  # class 1 (tie at threshold)
                [0.1, 0.4, 0.5],  # class 2 (tie at threshold)
            ]
        )
        thresholds = np.array([0.5, 0.5, 0.5])  # All thresholds at 0.5

        # Get confusion matrices with both operators
        cms_gt = get_multiclass_confusion_matrix(
            true_labels, pred_probs, thresholds, comparison=">"
        )
        cms_gte = get_multiclass_confusion_matrix(
            true_labels, pred_probs, thresholds, comparison=">="
        )

        # Should have confusion matrices for 3 classes
        assert len(cms_gt) == 3
        assert len(cms_gte) == 3

        # The results should potentially be different due to tie-breaking
        # At minimum, the function should work without errors
        for i in range(3):
            tp_gt, tn_gt, fp_gt, fn_gt = cms_gt[i]
            tp_gte, tn_gte, fp_gte, fn_gte = cms_gte[i]

            # All values should be non-negative
            assert tp_gt >= 0 and tn_gt >= 0 and fp_gt >= 0 and fn_gt >= 0
            assert tp_gte >= 0 and tn_gte >= 0 and fp_gte >= 0 and fn_gte >= 0

            # Total should equal number of samples
            assert tp_gt + tn_gt + fp_gt + fn_gt == 6
            assert tp_gte + tn_gte + fp_gte + fn_gte == 6

    def test_multiclass_threshold_optimization_comparison_operators(self):
        """Test comparison operators with multiclass threshold optimization."""
        # Simple multiclass data
        np.random.seed(123)
        n_samples = 50
        n_classes = 3
        true_labels = np.random.randint(0, n_classes, n_samples)
        pred_probs = np.random.rand(n_samples, n_classes)
        # Normalize to make them proper probabilities
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

        # Get optimal thresholds with both operators
        thresh_gt = get_optimal_threshold(true_labels, pred_probs, "f1", comparison=">")
        thresh_gte = get_optimal_threshold(
            true_labels, pred_probs, "f1", comparison=">="
        )

        # Should return arrays of thresholds
        assert isinstance(thresh_gt, np.ndarray)
        assert isinstance(thresh_gte, np.ndarray)
        assert len(thresh_gt) == n_classes
        assert len(thresh_gte) == n_classes

        # All thresholds should be valid
        assert np.all((thresh_gt >= 0) & (thresh_gt <= 1))
        assert np.all((thresh_gte >= 0) & (thresh_gte <= 1))

    def test_threshold_optimizer_comparison_operators(self):
        """Test ThresholdOptimizer class with comparison operators."""
        # Binary test case
        true_labels = np.array([0, 1, 0, 1, 0, 1])
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

        # Create optimizers with different comparison operators
        opt_gt = ThresholdOptimizer(objective="f1", comparison=">")
        opt_gte = ThresholdOptimizer(objective="f1", comparison=">=")

        # Fit both optimizers
        opt_gt.fit(true_labels, pred_probs)
        opt_gte.fit(true_labels, pred_probs)

        # Both should have learned thresholds
        assert opt_gt.threshold_ is not None
        assert opt_gte.threshold_ is not None

        # Make predictions
        pred_gt = opt_gt.predict(pred_probs)
        pred_gte = opt_gte.predict(pred_probs)

        # Predictions should be boolean arrays
        assert pred_gt.dtype == bool
        assert pred_gte.dtype == bool
        assert len(pred_gt) == len(true_labels)
        assert len(pred_gte) == len(true_labels)

    def test_comparison_operator_validation(self):
        """Test that invalid comparison operators raise appropriate errors."""
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7])

        # Invalid comparison operators should raise ValueError
        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_optimal_threshold(true_labels, pred_probs, comparison="<")

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_optimal_threshold(true_labels, pred_probs, comparison="==")

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            get_confusion_matrix(true_labels, pred_probs, 0.5, comparison="!=")

    def test_edge_cases_with_comparison_operators(self):
        """Test edge cases that might behave differently with different operators."""
        # Case where all probabilities equal the threshold
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.5, 0.5, 0.5, 0.5])  # All equal to threshold
        threshold = 0.5

        # With ">", all predictions should be negative (0)
        tp_gt, tn_gt, fp_gt, fn_gt = get_confusion_matrix(
            true_labels, pred_probs, threshold, comparison=">"
        )
        # Expected: predictions=[0,0,0,0], so TP=0, TN=2, FP=0, FN=2
        assert tp_gt == 0 and tn_gt == 2 and fp_gt == 0 and fn_gt == 2

        # With ">=", all predictions should be positive (1)
        tp_gte, tn_gte, fp_gte, fn_gte = get_confusion_matrix(
            true_labels, pred_probs, threshold, comparison=">="
        )
        # Expected: predictions=[1,1,1,1], so TP=2, TN=0, FP=2, FN=0
        assert tp_gte == 2 and tn_gte == 0 and fp_gte == 2 and fn_gte == 0

    def test_comparison_operators_with_sample_weights(self):
        """Test that comparison operators work correctly with sample weights."""
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.3, 0.5, 0.7, 0.5])  # Two at threshold 0.5
        sample_weights = np.array([1.0, 2.0, 1.0, 3.0])  # Different weights
        threshold = 0.5

        # Test with both comparison operators
        tp_gt, tn_gt, fp_gt, fn_gt = get_confusion_matrix(
            true_labels, pred_probs, threshold, sample_weights, comparison=">"
        )
        tp_gte, tn_gte, fp_gte, fn_gte = get_confusion_matrix(
            true_labels, pred_probs, threshold, sample_weights, comparison=">="
        )

        # Results should be floats when using sample weights
        assert isinstance(tp_gt, float)
        assert isinstance(tp_gte, float)

        # The weighted results should be different
        assert (tp_gt, tn_gt, fp_gt, fn_gt) != (tp_gte, tn_gte, fp_gte, fn_gte)

        # Sanity check: total weights should be preserved
        total_weight = np.sum(sample_weights)
        assert abs((tp_gt + tn_gt + fp_gt + fn_gt) - total_weight) < 1e-10
        assert abs((tp_gte + tn_gte + fp_gte + fn_gte) - total_weight) < 1e-10
