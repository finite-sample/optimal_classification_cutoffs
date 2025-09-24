"""Regression tests for the minimize_scalar fallback mechanism.

This tests the specific fix where minimize_scalar can return suboptimal thresholds
for piecewise-constant metrics, and the fallback mechanism ensures we get the
best threshold from the discrete candidate set.
"""

import numpy as np
from scipy import optimize

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.optimizers import _metric_score


class TestMinimizeFallbackRegression:
    """Test the minimize_scalar fallback mechanism that was implemented to fix suboptimal results."""

    def test_f1_minimize_scalar_fallback_case(self):
        """Test a specific case where minimize_scalar returns suboptimal F1 threshold.

        This reproduces the exact scenario that the fallback mechanism was designed to fix.
        """
        # Carefully crafted case where minimize_scalar fails
        # F1 is piecewise-constant, so the optimum is at one of the probability values
        true_labels = np.array([0, 0, 1, 1, 0, 1, 0])
        pred_probs = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        # First, find what minimize_scalar alone would return
        minimize_result = optimize.minimize_scalar(
            lambda t: -_metric_score(true_labels, pred_probs, t, "f1"),
            bounds=(0, 1),
            method="bounded",
        )
        minimize_threshold = minimize_result.x
        minimize_score = _metric_score(
            true_labels, pred_probs, minimize_threshold, "f1"
        )

        # Now find the best threshold from discrete candidates (what fallback does)
        candidates = np.unique(pred_probs)
        candidate_scores = [
            _metric_score(true_labels, pred_probs, t, "f1") for t in candidates
        ]
        best_candidate_idx = np.argmax(candidate_scores)
        best_candidate_threshold = candidates[best_candidate_idx]  # noqa: F841
        best_candidate_score = candidate_scores[best_candidate_idx]

        # The fallback mechanism should choose the better of the two
        fallback_threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )
        fallback_score = _metric_score(
            true_labels, pred_probs, fallback_threshold, "f1"
        )

        # The fallback should be at least as good as both minimize and best candidate
        assert fallback_score >= minimize_score - 1e-10, (
            f"Fallback score {fallback_score} worse than minimize score {minimize_score}"
        )
        assert fallback_score >= best_candidate_score - 1e-10, (
            f"Fallback score {fallback_score} worse than best candidate score {best_candidate_score}"
        )

        # With the enhanced minimize method, the fallback may use piecewise optimization
        # which can return midpoints or other optimal thresholds not in the original candidate set.
        # The key requirement is that the fallback score is optimal.
        # Verify that the fallback gives at least as good a score as both approaches
        assert fallback_score >= minimize_score - 1e-10
        assert fallback_score >= best_candidate_score - 1e-10

    def test_precision_minimize_scalar_fallback(self):
        """Test fallback mechanism with precision metric."""
        # Create a case where precision optimization might benefit from fallback
        true_labels = np.array([0, 0, 0, 1, 1, 0, 1])
        pred_probs = np.array([0.2, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85])

        # Test minimize method (with fallback)
        threshold_minimize = get_optimal_threshold(
            true_labels, pred_probs, "precision", method="minimize"
        )
        score_minimize = _metric_score(
            true_labels, pred_probs, threshold_minimize, "precision"
        )

        # Test smart_brute (our reference implementation)
        threshold_brute = get_optimal_threshold(
            true_labels, pred_probs, "precision", method="smart_brute"
        )
        score_brute = _metric_score(
            true_labels, pred_probs, threshold_brute, "precision"
        )

        # The minimize method (with fallback) should be at least as good as brute force
        assert score_minimize >= score_brute - 1e-10, (
            f"Minimize fallback score {score_minimize} worse than brute force {score_brute}"
        )

    def test_recall_minimize_scalar_fallback(self):
        """Test fallback mechanism with recall metric."""
        true_labels = np.array([1, 0, 1, 0, 1, 1, 0])
        pred_probs = np.array([0.15, 0.3, 0.45, 0.5, 0.6, 0.8, 0.95])

        # Test minimize method (with fallback)
        threshold_minimize = get_optimal_threshold(
            true_labels, pred_probs, "recall", method="minimize"
        )
        score_minimize = _metric_score(
            true_labels, pred_probs, threshold_minimize, "recall"
        )

        # Test smart_brute (our reference)
        threshold_brute = get_optimal_threshold(
            true_labels, pred_probs, "recall", method="smart_brute"
        )
        score_brute = _metric_score(true_labels, pred_probs, threshold_brute, "recall")

        # Minimize should perform at least as well
        assert score_minimize >= score_brute - 1e-10, (
            f"Minimize fallback score {score_minimize} worse than brute force {score_brute}"
        )

    def test_accuracy_minimize_scalar_fallback(self):
        """Test fallback mechanism with accuracy metric."""
        true_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pred_probs = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Test minimize method
        threshold_minimize = get_optimal_threshold(
            true_labels, pred_probs, "accuracy", method="minimize"
        )
        score_minimize = _metric_score(
            true_labels, pred_probs, threshold_minimize, "accuracy"
        )

        # Test reference method
        threshold_brute = get_optimal_threshold(
            true_labels, pred_probs, "accuracy", method="smart_brute"
        )
        score_brute = _metric_score(
            true_labels, pred_probs, threshold_brute, "accuracy"
        )

        # Should be equivalent (accuracy is piecewise-constant)
        assert score_minimize >= score_brute - 1e-10

    def test_fallback_mechanism_implementation_details(self):
        """Test that the fallback mechanism works as documented in the code."""
        true_labels = np.array([0, 0, 1, 1, 0, 1])
        pred_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.9])

        # Manually implement what the fallback should do
        # 1. Run minimize_scalar
        minimize_result = optimize.minimize_scalar(
            lambda t: -_metric_score(true_labels, pred_probs, t, "f1"),
            bounds=(0, 1),
            method="bounded",
        )

        # 2. Get all candidates (unique probabilities + minimize result)
        candidates = np.unique(np.append(pred_probs, minimize_result.x))

        # 3. Evaluate all candidates and pick best
        scores = [_metric_score(true_labels, pred_probs, t, "f1") for t in candidates]
        expected_best_threshold = candidates[np.argmax(scores)]

        # 4. Compare with actual implementation
        actual_threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )

        # With the enhanced minimize method, the implementation now uses piecewise optimization
        # for F1 metric, which can return midpoints and other optimal thresholds.
        # The key requirement is that the actual result should be at least as good as the
        # old fallback mechanism.
        actual_score = _metric_score(true_labels, pred_probs, actual_threshold, "f1")
        expected_score = _metric_score(
            true_labels, pred_probs, expected_best_threshold, "f1"
        )

        assert actual_score >= expected_score - 1e-10, (
            f"Enhanced minimize method score {actual_score} worse than expected {expected_score}"
        )

    def test_fallback_doesnt_hurt_when_minimize_is_optimal(self):
        """Test that fallback doesn't harm performance when minimize_scalar is already optimal."""
        # Create a case where minimize_scalar should work well
        # Use a smooth, non-piecewise metric or a case where the optimum aligns
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # Clean separation

        # For this clean case, minimize_scalar should find a good solution
        minimize_result = optimize.minimize_scalar(  # noqa: F841
            lambda t: -_metric_score(true_labels, pred_probs, t, "f1"),
            bounds=(0, 1),
            method="bounded",
        )

        fallback_threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )

        # The fallback should still produce a good result
        fallback_score = _metric_score(
            true_labels, pred_probs, fallback_threshold, "f1"
        )

        # Should achieve high performance on this well-separated case
        assert fallback_score >= 0.8, (
            f"Low score {fallback_score} on well-separated case"
        )

    def test_fallback_with_edge_cases(self):
        """Test that fallback mechanism handles edge cases gracefully."""
        # Case with very few samples
        true_labels = np.array([0, 1])
        pred_probs = np.array([0.3, 0.7])

        threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )
        assert 0 <= threshold <= 1

        # Should achieve perfect or near-perfect score
        score = _metric_score(true_labels, pred_probs, threshold, "f1")
        assert score >= 0.9  # Should get high score with perfect separation

    def test_multiple_metrics_fallback_consistency(self):
        """Test that fallback works consistently across different metrics."""
        true_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        pred_probs = np.array([0.1, 0.25, 0.35, 0.5, 0.6, 0.75, 0.85, 0.95])

        metrics = ["f1", "precision", "recall", "accuracy"]

        for metric in metrics:
            # Test that minimize method works without error
            threshold_minimize = get_optimal_threshold(
                true_labels, pred_probs, metric, method="minimize"
            )

            # Test that smart_brute works as reference
            threshold_brute = get_optimal_threshold(
                true_labels, pred_probs, metric, method="smart_brute"
            )

            # Both should produce valid thresholds
            assert 0 <= threshold_minimize <= 1, (
                f"Invalid threshold for {metric}: {threshold_minimize}"
            )
            assert 0 <= threshold_brute <= 1, (
                f"Invalid threshold for {metric}: {threshold_brute}"
            )

            # Scores should be reasonable
            score_minimize = _metric_score(
                true_labels, pred_probs, threshold_minimize, metric
            )
            score_brute = _metric_score(
                true_labels, pred_probs, threshold_brute, metric
            )

            # Minimize (with fallback) should be at least as good as brute force
            assert score_minimize >= score_brute - 1e-10, (
                f"Minimize fallback underperformed brute force for {metric}: "
                f"{score_minimize} vs {score_brute}"
            )

    def test_gradient_method_consistency(self):
        """Test that gradient method also works consistently (though it doesn't have fallback)."""
        true_labels = np.array([0, 1, 0, 1, 0, 1])
        pred_probs = np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Gradient method should work without errors
        threshold_gradient = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="gradient"
        )

        # Should produce valid threshold
        assert 0 <= threshold_gradient <= 1

        # Should produce reasonable score (gradient method may not be as precise)
        if 0 <= threshold_gradient <= 1:  # Only test if threshold is valid
            score = _metric_score(true_labels, pred_probs, threshold_gradient, "f1")
            assert score >= 0, f"Negative score from gradient method: {score}"


class TestFallbackEdgeCases:
    """Test edge cases specific to the fallback mechanism."""

    def test_fallback_with_duplicate_probabilities(self):
        """Test fallback when many probabilities are duplicated."""
        true_labels = np.array([0, 1, 0, 1, 0, 1])
        pred_probs = np.array([0.3, 0.3, 0.7, 0.7, 0.3, 0.7])  # Only two unique values

        threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )
        assert 0 <= threshold <= 1

        # Should achieve reasonable performance with only two values
        score = _metric_score(true_labels, pred_probs, threshold, "f1")
        assert score >= 0.5  # Should achieve reasonable score

    def test_fallback_with_extreme_probabilities(self):
        """Test fallback with probabilities at boundaries."""
        true_labels = np.array([0, 0, 1, 1])
        pred_probs = np.array([0.0, 0.1, 0.9, 1.0])

        threshold = get_optimal_threshold(
            true_labels, pred_probs, "accuracy", method="minimize"
        )

        # Should achieve perfect accuracy
        score = _metric_score(true_labels, pred_probs, threshold, "accuracy")
        assert abs(score - 1.0) < 1e-10, f"Expected perfect accuracy, got {score}"

    def test_fallback_numerical_precision(self):
        """Test that fallback handles numerical precision issues."""
        true_labels = np.array([0, 1, 0, 1])

        # Probabilities that differ by tiny amounts
        eps = 1e-14
        pred_probs = np.array([0.5 - eps, 0.5 + eps, 0.5 - 2 * eps, 0.5 + 2 * eps])

        # Should handle without numerical issues
        threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )
        assert 0 <= threshold <= 1

        # Should produce valid confusion matrix
        from optimal_cutoffs import get_confusion_matrix

        tp, tn, fp, fn = get_confusion_matrix(true_labels, pred_probs, threshold)
        assert tp + tn + fp + fn == len(true_labels)

    def test_fallback_performance_characteristics(self):
        """Test that fallback doesn't significantly hurt performance."""
        # Larger dataset to test performance
        n_samples = 1000
        np.random.seed(42)
        true_labels = np.random.randint(0, 2, n_samples)
        pred_probs = np.random.beta(
            2, 2, n_samples
        )  # Reasonable probability distribution

        import time

        # Time the minimize method (with fallback)
        start_time = time.time()
        threshold = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="minimize"
        )
        minimize_time = time.time() - start_time

        # Time the smart_brute method
        start_time = time.time()
        threshold_brute = get_optimal_threshold(
            true_labels, pred_probs, "f1", method="smart_brute"
        )
        brute_time = time.time() - start_time

        # Minimize should complete in reasonable time (allowing for scipy overhead)
        assert minimize_time < max(brute_time * 50, 1.0), (
            f"Minimize method too slow: {minimize_time:.4f}s vs {brute_time:.4f}s"
        )

        # Both should produce good results
        score_minimize = _metric_score(true_labels, pred_probs, threshold, "f1")
        score_brute = _metric_score(true_labels, pred_probs, threshold_brute, "f1")

        assert score_minimize >= score_brute - 1e-10
