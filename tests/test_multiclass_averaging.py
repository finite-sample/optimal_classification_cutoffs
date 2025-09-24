"""Tests for multiclass averaging semantics and performance improvements."""

import numpy as np
import pytest

from optimal_cutoffs import (
    MulticlassMetricReturn,
    get_multiclass_confusion_matrix,
    get_optimal_multiclass_thresholds,
    multiclass_metric,
)


class TestMulticlassAveragingSemantics:
    """Test suite for multiclass averaging strategies."""

    def setup_method(self):
        """Set up test data for multiclass problems."""
        np.random.seed(42)

        # Simple 3-class problem with known characteristics
        self.true_labs = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        self.pred_prob = np.array(
            [
                [0.8, 0.1, 0.1],  # True: 0, likely correct
                [0.7, 0.2, 0.1],  # True: 0, likely correct
                [0.2, 0.7, 0.1],  # True: 1, likely correct
                [0.1, 0.8, 0.1],  # True: 1, likely correct
                [0.1, 0.2, 0.7],  # True: 2, likely correct
                [0.1, 0.1, 0.8],  # True: 2, likely correct
                [0.5, 0.3, 0.2],  # True: 0, uncertain
                [0.3, 0.5, 0.2],  # True: 1, uncertain
                [0.2, 0.3, 0.5],  # True: 2, uncertain
            ]
        )

        # Simple thresholds for testing
        self.thresholds = np.array([0.5, 0.5, 0.5])

        # Compute confusion matrices
        self.cms = get_multiclass_confusion_matrix(
            self.true_labs, self.pred_prob, self.thresholds
        )

    def test_average_none_returns_array(self):
        """Test that average='none' returns array of per-class scores."""
        result = multiclass_metric(self.cms, "f1", average="none")

        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # Number of classes
        assert result.dtype == np.float64

        # Each score should be between 0 and 1
        assert all(0 <= score <= 1 for score in result)

    def test_macro_micro_weighted_return_float(self):
        """Test that other averaging strategies return float."""
        for average in ["macro", "micro", "weighted"]:
            result = multiclass_metric(self.cms, "f1", average=average)

            assert isinstance(result, (float, np.floating))
            assert 0 <= result <= 1

    def test_macro_averaging_identity(self):
        """Test macro averaging identity: macro = mean(per_class_scores)."""
        # Get per-class scores
        per_class_scores = multiclass_metric(self.cms, "f1", average="none")
        macro_score = multiclass_metric(self.cms, "f1", average="macro")

        expected_macro = float(np.mean(per_class_scores))

        assert macro_score == pytest.approx(expected_macro, rel=1e-10)

    def test_micro_averaging_identity(self):
        """Test micro averaging identity: micro = metric(sum(TP), 0, sum(FP), sum(FN))."""
        # Compute pooled confusion matrix
        total_tp = sum(cm[0] for cm in self.cms)
        total_fp = sum(cm[2] for cm in self.cms)
        total_fn = sum(cm[3] for cm in self.cms)

        # Compute micro F1 manually
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        expected_micro_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        # Get micro F1 from function
        micro_f1 = multiclass_metric(self.cms, "f1", average="micro")

        assert micro_f1 == pytest.approx(expected_micro_f1, rel=1e-10)

    def test_weighted_averaging_identity(self):
        """Test weighted averaging identity: weighted = sum(score_i * support_i) / sum(support_i)."""
        # Get per-class scores and supports
        per_class_scores = multiclass_metric(self.cms, "f1", average="none")
        supports = [cm[0] + cm[3] for cm in self.cms]  # TP + FN = actual positives

        total_support = sum(supports)
        expected_weighted = (
            sum(
                score * support
                for score, support in zip(per_class_scores, supports, strict=False)
            )
            / total_support
            if total_support > 0
            else 0.0
        )

        weighted_score = multiclass_metric(self.cms, "f1", average="weighted")

        assert weighted_score == pytest.approx(expected_weighted, rel=1e-10)

    def test_all_metrics_support_all_averages(self):
        """Test that all registered metrics support all averaging strategies."""
        from optimal_cutoffs import METRIC_REGISTRY

        metrics_to_test = ["f1", "precision", "recall", "accuracy"]
        averages = ["macro", "micro", "weighted", "none"]

        for metric_name in metrics_to_test:
            assert metric_name in METRIC_REGISTRY

            for average in averages:
                result = multiclass_metric(self.cms, metric_name, average=average)

                if average == "none":
                    assert isinstance(result, np.ndarray)
                    assert len(result) == 3
                else:
                    assert isinstance(result, (float, np.floating))

    def test_invalid_average_raises_error(self):
        """Test that invalid average parameter raises appropriate error."""
        with pytest.raises(
            ValueError, match="Unknown averaging method.*invalid.*Must be one of"
        ):
            multiclass_metric(self.cms, "f1", average="invalid")

    def test_type_annotations_work(self):
        """Test that type annotations work correctly."""
        # This test mainly checks that our type hints are valid
        result_float: float = multiclass_metric(self.cms, "f1", average="macro")
        result_array: np.ndarray = multiclass_metric(self.cms, "f1", average="none")
        result_union: MulticlassMetricReturn = multiclass_metric(
            self.cms, "f1", average="macro"
        )

        assert isinstance(result_float, (float, np.floating))
        assert isinstance(result_array, np.ndarray)
        assert isinstance(result_union, (float, np.floating, np.ndarray))


class TestMulticlassOptimizationAveraging:
    """Test suite for averaging-aware optimization."""

    def setup_method(self):
        """Set up test data for optimization tests."""
        np.random.seed(42)

        # Larger dataset for optimization tests
        n_samples = 100
        self.true_labs = np.random.randint(0, 3, n_samples)
        self.pred_prob = np.random.rand(n_samples, 3)
        # Normalize probabilities
        self.pred_prob = self.pred_prob / self.pred_prob.sum(axis=1, keepdims=True)

    def test_average_parameter_affects_optimization(self):
        """Test that different averaging strategies can produce different results."""
        methods = ["minimize"]  # Use minimize for joint optimization

        for method in methods:
            # Get thresholds for different averaging strategies
            thresholds_macro = get_optimal_multiclass_thresholds(
                self.true_labs, self.pred_prob, "f1", method, average="macro"
            )
            thresholds_micro = get_optimal_multiclass_thresholds(
                self.true_labs, self.pred_prob, "f1", method, average="micro"
            )

            # Results should be valid thresholds
            assert isinstance(thresholds_macro, np.ndarray)
            assert isinstance(thresholds_micro, np.ndarray)
            assert len(thresholds_macro) == 3
            assert len(thresholds_micro) == 3
            assert all(0 <= t <= 1 for t in thresholds_macro)
            assert all(0 <= t <= 1 for t in thresholds_micro)

    def test_macro_none_weighted_equivalent(self):
        """Test that macro, none, and weighted give same results when classes balanced."""
        # Create balanced dataset
        n_per_class = 50
        true_labs_balanced = np.concatenate(
            [np.full(n_per_class, 0), np.full(n_per_class, 1), np.full(n_per_class, 2)]
        )
        np.random.shuffle(true_labs_balanced)

        pred_prob_balanced = np.random.rand(len(true_labs_balanced), 3)
        pred_prob_balanced = pred_prob_balanced / pred_prob_balanced.sum(
            axis=1, keepdims=True
        )

        # Get thresholds (should be identical for balanced data)
        thresholds_macro = get_optimal_multiclass_thresholds(
            true_labs_balanced, pred_prob_balanced, "f1", "smart_brute", average="macro"
        )
        thresholds_none = get_optimal_multiclass_thresholds(
            true_labs_balanced, pred_prob_balanced, "f1", "smart_brute", average="none"
        )
        thresholds_weighted = get_optimal_multiclass_thresholds(
            true_labs_balanced,
            pred_prob_balanced,
            "f1",
            "smart_brute",
            average="weighted",
        )

        # For balanced data, macro and weighted should be identical
        np.testing.assert_array_almost_equal(thresholds_macro, thresholds_weighted)
        np.testing.assert_array_almost_equal(thresholds_macro, thresholds_none)

    def test_vectorized_parameter_works(self):
        """Test that vectorized parameter is accepted and produces valid results."""
        thresholds_standard = get_optimal_multiclass_thresholds(
            self.true_labs,
            self.pred_prob,
            "f1",
            "smart_brute",
            average="macro",
            vectorized=False,
        )
        thresholds_vectorized = get_optimal_multiclass_thresholds(
            self.true_labs,
            self.pred_prob,
            "f1",
            "smart_brute",
            average="macro",
            vectorized=True,
        )

        # Both should produce valid thresholds
        assert isinstance(thresholds_standard, np.ndarray)
        assert isinstance(thresholds_vectorized, np.ndarray)
        assert len(thresholds_standard) == 3
        assert len(thresholds_vectorized) == 3

        # Should produce very similar results (allowing for minor optimization differences)
        np.testing.assert_array_almost_equal(
            thresholds_standard, thresholds_vectorized, decimal=3
        )

    def test_different_averaging_strategies_documented(self):
        """Test that all averaging strategies are properly documented and work."""
        averages = ["macro", "micro", "weighted", "none"]

        for average in averages:
            thresholds = get_optimal_multiclass_thresholds(
                self.true_labs, self.pred_prob, "f1", "smart_brute", average=average
            )

            assert isinstance(thresholds, np.ndarray)
            assert len(thresholds) == 3
            assert all(0 <= t <= 1 for t in thresholds)

    def test_backward_compatibility(self):
        """Test that default behavior is unchanged."""
        # Default should be macro averaging
        thresholds_default = get_optimal_multiclass_thresholds(
            self.true_labs, self.pred_prob, "f1"
        )
        thresholds_explicit = get_optimal_multiclass_thresholds(
            self.true_labs, self.pred_prob, "f1", average="macro"
        )

        np.testing.assert_array_equal(thresholds_default, thresholds_explicit)

    def test_micro_optimization_different_from_macro(self):
        """Test that micro optimization can produce different results from macro."""
        # For some datasets, micro and macro optimization should differ
        thresholds_macro = get_optimal_multiclass_thresholds(
            self.true_labs, self.pred_prob, "f1", "minimize", average="macro"
        )
        thresholds_micro = get_optimal_multiclass_thresholds(
            self.true_labs, self.pred_prob, "f1", "minimize", average="micro"
        )

        # Both should be valid
        assert isinstance(thresholds_macro, np.ndarray)
        assert isinstance(thresholds_micro, np.ndarray)
        assert len(thresholds_macro) == 3
        assert len(thresholds_micro) == 3


class TestPerformanceImprovements:
    """Test suite for performance improvements."""

    def test_vectorized_option_available(self):
        """Test that vectorized option is available and works."""
        np.random.seed(42)
        true_labs = np.random.randint(0, 3, 100)
        pred_prob = np.random.rand(100, 3)
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

        # Should work without errors
        thresholds = get_optimal_multiclass_thresholds(
            true_labs, pred_prob, "f1", "smart_brute", average="macro", vectorized=True
        )

        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == 3
        assert all(0 <= t <= 1 for t in thresholds)

    def test_large_dataset_performance(self):
        """Test performance improvements on larger datasets."""
        # This is more of a smoke test to ensure the implementation scales
        np.random.seed(42)
        n_samples = 1000
        n_classes = 5

        true_labs = np.random.randint(0, n_classes, n_samples)
        pred_prob = np.random.rand(n_samples, n_classes)
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

        # Should complete without errors on larger datasets
        thresholds = get_optimal_multiclass_thresholds(
            true_labs, pred_prob, "f1", "smart_brute", average="macro", vectorized=True
        )

        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == n_classes
        assert all(0 <= t <= 1 for t in thresholds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
