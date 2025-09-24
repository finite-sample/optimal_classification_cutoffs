"""Identity tests for micro and macro averaging to ensure mathematical correctness."""

import numpy as np
import pytest

from optimal_cutoffs import (
    METRIC_REGISTRY,
    get_multiclass_confusion_matrix,
    multiclass_metric,
)


class TestAveragingMathematicalIdentities:
    """Rigorous tests to validate micro/macro averaging mathematical identities."""

    @pytest.fixture
    def known_confusion_matrices(self):
        """Create known confusion matrices for testing identities."""
        # Manually constructed confusion matrices with known properties
        # Class 0: TP=10, TN=80, FP=5, FN=5   -> Precision=10/15=0.667, Recall=10/15=0.667, F1=0.667
        # Class 1: TP=8,  TN=85, FP=3, FN=4   -> Precision=8/11=0.727,  Recall=8/12=0.667,  F1=0.696
        # Class 2: TP=12, TN=82, FP=2, FN=4   -> Precision=12/14=0.857, Recall=12/16=0.750, F1=0.800
        confusion_matrices = [
            (10, 80, 5, 5),  # Class 0
            (8, 85, 3, 4),  # Class 1
            (12, 82, 2, 4),  # Class 2
        ]
        return confusion_matrices

    @pytest.fixture
    def balanced_confusion_matrices(self):
        """Create balanced confusion matrices where all classes have equal support."""
        # Each class has exactly 20 true instances (TP + FN = 20)
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
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            per_class_f1_manual.append(f1)

        expected_macro_f1 = np.mean(per_class_f1_manual)

        # Compute using library functions
        per_class_f1_lib = multiclass_metric(cms, "f1", average="none")
        macro_f1_lib = multiclass_metric(cms, "f1", average="macro")

        # Test identity: macro = mean(per_class)
        assert macro_f1_lib == pytest.approx(expected_macro_f1, abs=1e-10)
        assert macro_f1_lib == pytest.approx(np.mean(per_class_f1_lib), abs=1e-10)

        # Verify per-class computations are correct
        np.testing.assert_array_almost_equal(
            per_class_f1_lib, per_class_f1_manual, decimal=10
        )

    def test_micro_f1_identity(self, known_confusion_matrices):
        """Test that micro F1 equals F1 computed on pooled confusion matrix."""
        cms = known_confusion_matrices

        # Pool confusion matrices (sum TP, FP, FN; ignore TN in OvR)
        total_tp = sum(cm[0] for cm in cms)  # 10 + 8 + 12 = 30
        total_fp = sum(cm[2] for cm in cms)  # 5 + 3 + 2 = 10
        total_fn = sum(cm[3] for cm in cms)  # 5 + 4 + 4 = 13

        # Compute micro F1 manually
        micro_precision = (
            total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        )
        expected_micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall > 0
            else 0.0
        )

        # Compute using library function
        micro_f1_lib = multiclass_metric(cms, "f1", average="micro")

        # Test identity: micro F1 = F1(sum(TP), sum(FP), sum(FN))
        assert micro_f1_lib == pytest.approx(expected_micro_f1, abs=1e-10)

        # Double check with exact values
        assert micro_precision == pytest.approx(30 / 40, abs=1e-10)  # 0.75
        assert micro_recall == pytest.approx(30 / 43, abs=1e-10)  # ~0.698
        expected_exact = 2 * (30 / 40) * (30 / 43) / ((30 / 40) + (30 / 43))
        assert micro_f1_lib == pytest.approx(expected_exact, abs=1e-10)

    def test_weighted_f1_identity(self, known_confusion_matrices):
        """Test that weighted F1 equals support-weighted mean of per-class F1 scores."""
        cms = known_confusion_matrices

        # Compute per-class F1 scores and supports
        per_class_f1 = []
        supports = []
        for tp, tn, fp, fn in cms:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            support = tp + fn  # True positives for this class
            per_class_f1.append(f1)
            supports.append(support)

        # Compute weighted average manually
        total_support = sum(supports)
        expected_weighted_f1 = (
            sum(
                f1 * support
                for f1, support in zip(per_class_f1, supports, strict=False)
            )
            / total_support
            if total_support > 0
            else 0.0
        )

        # Compute using library function
        weighted_f1_lib = multiclass_metric(cms, "f1", average="weighted")

        # Test identity: weighted = sum(f1_i * support_i) / sum(support_i)
        assert weighted_f1_lib == pytest.approx(expected_weighted_f1, abs=1e-10)

    def test_balanced_data_weighted_equals_macro(self, balanced_confusion_matrices):
        """Test that for balanced data, weighted average equals macro average."""
        cms = balanced_confusion_matrices

        # Verify data is actually balanced
        supports = [tp + fn for tp, tn, fp, fn in cms]
        assert all(support == supports[0] for support in supports), (
            "Test data should be balanced"
        )

        # Compute both averages
        macro_f1 = multiclass_metric(cms, "f1", average="macro")
        weighted_f1 = multiclass_metric(cms, "f1", average="weighted")

        # For balanced data: weighted = macro
        assert weighted_f1 == pytest.approx(macro_f1, abs=1e-10)

    def test_all_metrics_satisfy_identities(self, known_confusion_matrices):
        """Test that all registered metrics satisfy averaging identities."""
        cms = known_confusion_matrices
        metrics_to_test = ["f1", "precision", "recall", "accuracy"]

        for metric_name in metrics_to_test:
            assert metric_name in METRIC_REGISTRY

            # Test macro identity
            per_class_scores = multiclass_metric(cms, metric_name, average="none")
            macro_score = multiclass_metric(cms, metric_name, average="macro")
            expected_macro = np.mean(per_class_scores)

            assert macro_score == pytest.approx(expected_macro, abs=1e-10), (
                f"Macro identity failed for {metric_name}"
            )

            # Test weighted identity
            supports = [tp + fn for tp, tn, fp, fn in cms]
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
            weighted_score = multiclass_metric(cms, metric_name, average="weighted")

            assert weighted_score == pytest.approx(expected_weighted, abs=1e-10), (
                f"Weighted identity failed for {metric_name}"
            )

    def test_micro_precision_recall_identity(self, known_confusion_matrices):
        """Test micro-averaging identities for precision and recall specifically."""
        cms = known_confusion_matrices

        # Pool confusion matrices
        total_tp = sum(cm[0] for cm in cms)
        total_fp = sum(cm[2] for cm in cms)
        total_fn = sum(cm[3] for cm in cms)

        # Test micro precision identity
        expected_micro_precision = (
            total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        )
        micro_precision = multiclass_metric(cms, "precision", average="micro")
        assert micro_precision == pytest.approx(expected_micro_precision, abs=1e-10)

        # Test micro recall identity
        expected_micro_recall = (
            total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        )
        micro_recall = multiclass_metric(cms, "recall", average="micro")
        assert micro_recall == pytest.approx(expected_micro_recall, abs=1e-10)

    def test_micro_accuracy_identity(self, known_confusion_matrices):
        """Test micro-averaging identity for accuracy in OvR setting."""
        cms = known_confusion_matrices

        # For accuracy in OvR, micro = total_correct / total_predictions
        total_tp = sum(cm[0] for cm in cms)
        total_fp = sum(cm[2] for cm in cms)
        total_fn = sum(cm[3] for cm in cms)
        total_predictions = total_tp + total_fp + total_fn

        expected_micro_accuracy = (
            total_tp / total_predictions if total_predictions > 0 else 0.0
        )
        micro_accuracy = multiclass_metric(cms, "accuracy", average="micro")

        assert micro_accuracy == pytest.approx(expected_micro_accuracy, abs=1e-10)

    def test_edge_case_all_zeros(self):
        """Test averaging identities with edge case of all-zero confusion matrices."""
        # Edge case: no predictions or all wrong
        cms = [(0, 100, 0, 0), (0, 100, 0, 0), (0, 100, 0, 0)]

        for average in ["macro", "micro", "weighted", "none"]:
            result = multiclass_metric(cms, "f1", average=average)

            if average == "none":
                assert isinstance(result, np.ndarray)
                assert all(score == 0.0 for score in result)
            else:
                assert result == 0.0

    def test_edge_case_perfect_classification(self):
        """Test averaging identities with perfect classification."""
        # Perfect classification: all TP, no FP or FN
        cms = [(20, 80, 0, 0), (15, 85, 0, 0), (25, 75, 0, 0)]

        for average in ["macro", "micro", "weighted"]:
            f1_score = multiclass_metric(cms, "f1", average=average)
            precision_score = multiclass_metric(cms, "precision", average=average)
            recall_score = multiclass_metric(cms, "recall", average=average)

            # Perfect classification should give 1.0 for all metrics and averages
            assert f1_score == pytest.approx(1.0, abs=1e-10)
            assert precision_score == pytest.approx(1.0, abs=1e-10)
            assert recall_score == pytest.approx(1.0, abs=1e-10)

    def test_real_world_confusion_matrices(self):
        """Test identities on realistic confusion matrices from actual classification."""
        # Simulate realistic confusion matrices that might come from actual predictions
        np.random.seed(42)
        n_samples = 300
        n_classes = 4

        # Generate synthetic true labels and predictions
        true_labels = np.random.randint(0, n_classes, n_samples)
        pred_probs = np.random.rand(n_samples, n_classes)
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

        # Use fixed thresholds to get confusion matrices
        thresholds = np.full(n_classes, 0.25)  # 1/n_classes
        cms = get_multiclass_confusion_matrix(true_labels, pred_probs, thresholds)

        # Test identities on this realistic data
        for metric_name in ["f1", "precision", "recall"]:
            per_class_scores = multiclass_metric(cms, metric_name, average="none")
            macro_score = multiclass_metric(cms, metric_name, average="macro")

            # Macro identity should hold
            expected_macro = np.mean(per_class_scores)
            assert macro_score == pytest.approx(expected_macro, abs=1e-10), (
                f"Macro identity failed for {metric_name} on realistic data"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
