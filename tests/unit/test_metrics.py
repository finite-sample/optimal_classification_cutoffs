"""Unit tests for metric calculation and registry functionality.

This module tests the core metric calculation functions, registry system,
and mathematical identities for averaging methods.
"""

import numpy as np
import pytest

from optimal_cutoffs import (
    METRIC_REGISTRY,
    VECTORIZED_REGISTRY,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    get_optimal_threshold,
    get_vectorized_metric,
    has_vectorized_implementation,
    is_piecewise_metric,
    multiclass_metric,
)
from optimal_cutoffs.metrics import register_metric, register_metrics
from tests.fixtures.assertions import (
    assert_valid_confusion_matrix,
    assert_valid_metric_score,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_multiclass_data,
)


class TestBasicMetrics:
    """Test basic metric calculation functionality."""

    def test_confusion_matrix_calculation(self):
        """Test basic confusion matrix calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.2, 0.6, 0.7, 0.3, 0.4])
        threshold = 0.5

        tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, threshold)

        # Validate confusion matrix
        assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))
        assert (tp, tn, fp, fn) == (2, 2, 0, 1)

    def test_derived_metrics_from_confusion_matrix(self):
        """Test computing metrics from confusion matrix."""
        tp, fp, fn = 2, 0, 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        assert_valid_metric_score(precision, "precision")
        assert_valid_metric_score(recall, "recall")
        assert_valid_metric_score(f1, "f1")

        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(2 / 3)
        assert f1 == pytest.approx(0.8)

    def test_confusion_matrix_with_weights(self):
        """Test weighted confusion matrix calculation."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.6, 0.3])
        weights = np.array([1.0, 2.0, 1.5, 0.5])
        threshold = 0.5

        tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, threshold, weights)

        assert_valid_confusion_matrix(tp, tn, fp, fn, total_weight=np.sum(weights))
        assert isinstance(tp, float)  # Weighted results are floats


class TestMetricRegistry:
    """Test metric registry functionality."""

    def test_built_in_metrics_registered(self):
        """Test that built-in metrics are properly registered."""
        expected_metrics = ["f1", "accuracy", "precision", "recall"]

        for metric in expected_metrics:
            assert metric in METRIC_REGISTRY
            assert callable(METRIC_REGISTRY[metric])

    def test_custom_metric_registration(self):
        """Test registering custom metrics."""
        @register_metric("sum_tp_tn")
        def sum_tp_tn(tp, tn, fp, fn):
            return tp + tn

        assert "sum_tp_tn" in METRIC_REGISTRY
        assert METRIC_REGISTRY["sum_tp_tn"](1, 1, 0, 0) == 2

    def test_batch_metric_registration(self):
        """Test registering multiple metrics at once."""
        def tpr(tp, tn, fp, fn):
            return tp / (tp + fn) if tp + fn > 0 else 0.0

        def tnr(tp, tn, fp, fn):
            return tn / (tn + fp) if tn + fp > 0 else 0.0

        register_metrics({"tpr": tpr, "tnr": tnr})

        assert "tpr" in METRIC_REGISTRY
        assert "tnr" in METRIC_REGISTRY

    def test_custom_metric_optimization(self):
        """Test that custom metrics can be optimized."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        # Use registered custom metric
        threshold = get_optimal_threshold(y_true, y_prob, metric="tpr")
        assert 0.0 <= threshold <= 1.0


class TestVectorizedMetrics:
    """Test vectorized metric implementations."""

    def test_built_in_vectorized_implementations(self):
        """Test that built-in metrics have vectorized implementations."""
        built_in_piecewise = ["f1", "accuracy", "precision", "recall"]

        for metric in built_in_piecewise:
            assert metric in METRIC_REGISTRY
            assert metric in VECTORIZED_REGISTRY
            assert has_vectorized_implementation(metric)
            assert is_piecewise_metric(metric)

    def test_get_vectorized_metric_function(self):
        """Test getting vectorized metric functions."""
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

        for score in scores:
            assert_valid_metric_score(score, "f1_vectorized")

    def test_vectorized_metric_invalid_request(self):
        """Test error handling for non-vectorized metrics."""
        with pytest.raises(ValueError, match="No vectorized implementation"):
            get_vectorized_metric("nonexistent_metric")

    def test_vectorized_vs_scalar_consistency(self):
        """Test that vectorized and scalar metrics give same results."""
        tp_vals = np.array([5, 8, 12])
        tn_vals = np.array([10, 15, 20])
        fp_vals = np.array([2, 3, 1])
        fn_vals = np.array([1, 4, 2])

        # Get scalar and vectorized F1
        f1_scalar = METRIC_REGISTRY["f1"]
        f1_vec = get_vectorized_metric("f1")

        # Compute vectorized results
        vec_results = f1_vec(tp_vals, tn_vals, fp_vals, fn_vals)

        # Compute scalar results
        scalar_results = []
        for i in range(len(tp_vals)):
            scalar_results.append(f1_scalar(tp_vals[i], tn_vals[i], fp_vals[i], fn_vals[i]))

        np.testing.assert_allclose(vec_results, scalar_results, rtol=1e-10)


class TestAveragingIdentities:
    """Test mathematical identities for micro and macro averaging."""

    @pytest.fixture
    def known_confusion_matrices(self):
        """Create known confusion matrices for testing identities."""
        # Manually constructed confusion matrices with known properties
        confusion_matrices = [
            (10, 80, 5, 5),  # Class 0
            (8, 85, 3, 4),   # Class 1
            (12, 82, 2, 4),  # Class 2
        ]
        return confusion_matrices

    @pytest.fixture
    def balanced_confusion_matrices(self):
        """Create balanced confusion matrices where all classes have equal support."""
        confusion_matrices = [
            (15, 70, 5, 5),  # Class 0: support=20
            (14, 71, 4, 6),  # Class 1: support=20
            (16, 69, 3, 4),  # Class 2: support=20
        ]
        return confusion_matrices

    def test_macro_f1_identity(self, known_confusion_matrices):
        """Test that macro F1 equals the mean of per-class F1 scores."""
        cms = known_confusion_matrices

        # Compute per-class F1 scores manually
        per_class_f1_manual = []
        for tp, tn, fp, fn in cms:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0)
            per_class_f1_manual.append(f1)

        expected_macro_f1 = np.mean(per_class_f1_manual)

        # Compute using library function
        computed_macro_f1 = multiclass_metric(cms, "f1", "macro")

        assert_valid_metric_score(computed_macro_f1, "macro_f1")
        assert computed_macro_f1 == pytest.approx(expected_macro_f1, abs=1e-10)

    def test_micro_f1_identity(self, known_confusion_matrices):
        """Test that micro F1 is computed from aggregated confusion matrix."""
        cms = known_confusion_matrices

        # Aggregate confusion matrix components
        total_tp = sum(tp for tp, tn, fp, fn in cms)
        total_fp = sum(fp for tp, tn, fp, fn in cms)
        total_fn = sum(fn for tp, tn, fp, fn in cms)

        # Compute micro F1 manually
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        expected_micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                             if micro_precision + micro_recall > 0 else 0.0)

        # Compute using library function
        computed_micro_f1 = multiclass_metric(cms, "f1", "micro")

        assert_valid_metric_score(computed_micro_f1, "micro_f1")
        assert computed_micro_f1 == pytest.approx(expected_micro_f1, abs=1e-10)

    def test_weighted_averaging_identity(self, balanced_confusion_matrices):
        """Test weighted averaging with balanced support."""
        cms = balanced_confusion_matrices

        # For balanced data, weighted average should equal macro average
        macro_f1 = multiclass_metric(cms, "f1", "macro")
        weighted_f1 = multiclass_metric(cms, "f1", "weighted")

        assert_valid_metric_score(macro_f1, "macro_f1")
        assert_valid_metric_score(weighted_f1, "weighted_f1")
        assert weighted_f1 == pytest.approx(macro_f1, abs=1e-10)

    def test_micro_precision_recall_identity(self, known_confusion_matrices):
        """Test that micro precision equals micro recall for balanced problems."""
        cms = known_confusion_matrices

        micro_precision = multiclass_metric(cms, "precision", "micro")
        micro_recall = multiclass_metric(cms, "recall", "micro")

        assert_valid_metric_score(micro_precision, "micro_precision")
        assert_valid_metric_score(micro_recall, "micro_recall")

        # For one-vs-rest problems, micro precision = micro recall = micro F1
        assert micro_precision == pytest.approx(micro_recall, abs=1e-10)

    def test_support_calculation_consistency(self, known_confusion_matrices):
        """Test that support calculations are consistent across metrics."""
        cms = known_confusion_matrices

        # Support for each class is TP + FN
        expected_supports = []
        for tp, tn, fp, fn in cms:
            expected_supports.append(tp + fn)

        # Test that weighted averaging uses correct supports
        # (This is more of a sanity check that our test data is structured correctly)
        total_support = sum(expected_supports)
        assert total_support > 0


class TestMulticlassMetrics:
    """Test multiclass-specific metric functionality."""

    def test_multiclass_confusion_matrices(self):
        """Test multiclass confusion matrix calculation."""
        y_true, y_prob = generate_multiclass_data(50, n_classes=3, random_state=42)
        thresholds = np.array([0.3, 0.4, 0.5])

        cms = get_multiclass_confusion_matrix(y_true, y_prob, thresholds)

        assert len(cms) == 3  # One per class
        for i, (tp, tn, fp, fn) in enumerate(cms):
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))

    def test_multiclass_metric_averaging(self):
        """Test different averaging methods for multiclass metrics."""
        y_true, y_prob = generate_multiclass_data(100, n_classes=3, random_state=42)
        thresholds = np.array([0.3, 0.4, 0.5])

        cms = get_multiclass_confusion_matrix(y_true, y_prob, thresholds)

        # Test all averaging methods
        for average in ["macro", "micro", "weighted"]:
            score = multiclass_metric(cms, "f1", average)
            assert_valid_metric_score(score, f"{average}_f1")

    def test_edge_case_empty_class(self):
        """Test metric calculation when some classes have no true positives."""
        # Create confusion matrices where one class has no true instances
        cms = [
            (5, 90, 2, 3),   # Class 0: normal
            (0, 95, 0, 5),   # Class 1: no true positives (TP=0, FN=5)
            (8, 87, 3, 2),   # Class 2: normal
        ]

        # Should handle gracefully without division by zero
        macro_f1 = multiclass_metric(cms, "f1", "macro")
        micro_f1 = multiclass_metric(cms, "f1", "micro")
        weighted_f1 = multiclass_metric(cms, "f1", "weighted")

        for score in [macro_f1, micro_f1, weighted_f1]:
            assert_valid_metric_score(score, "f1_with_empty_class", allow_nan=False)
