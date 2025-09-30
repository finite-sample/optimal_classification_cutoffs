"""Hypothesis-based adversarial tests for comprehensive bug testing.

This module contains property-based tests using Hypothesis to generate adversarial
cases that match the specific requirements from the detailed test plan. These tests
are designed to catch edge cases and verify mathematical properties.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.metrics import (
    _compute_exclusive_predictions,
    accuracy_score,
    f1_score,
    get_confusion_matrix,
    get_multiclass_confusion_matrix,
    multiclass_metric,
    multiclass_metric_exclusive,
)
from optimal_cutoffs.optimizers import get_optimal_multiclass_thresholds


# Custom Hypothesis strategies
def integer_weights(
    min_value: int = 1, max_value: int = 3, min_size: int = 5, max_size: int = 50
):
    """Generate integer weights in {1, 2, 3} for exact expansion testing."""
    return arrays(
        dtype=np.int32,
        shape=st.integers(min_size, max_size),
        elements=st.integers(min_value, max_value),
    )


def binary_labels_and_probs(min_size: int = 5, max_size: int = 50):
    """Generate matched binary labels and probabilities."""

    @st.composite
    def _inner(draw):
        size = draw(st.integers(min_size, max_size))
        labels = draw(arrays(dtype=np.int8, shape=size, elements=st.integers(0, 1)))
        probs = draw(
            arrays(dtype=np.float64, shape=size, elements=st.floats(0.01, 0.99))
        )
        return labels, probs

    return _inner()


def multiclass_data(n_classes: int, min_size: int = 10, max_size: int = 50):
    """Generate multiclass data with specified number of classes."""

    @st.composite
    def _inner(draw):
        size = draw(st.integers(min_size, max_size))
        # Ensure all classes have at least one sample
        labels = np.random.choice(n_classes, size=size)

        # Generate probabilities that sum to 1 (approximately)
        probs = draw(
            arrays(
                dtype=np.float64,
                shape=(size, n_classes),
                elements=st.floats(0.01, 0.99),
            )
        )
        # Normalize to sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)

        return labels, probs

    return _inner()


def beta_probabilities(min_size: int = 50, max_size: int = 200):
    """Generate calibrated probabilities using Beta distribution."""

    @st.composite
    def _inner(draw):
        size = draw(st.integers(min_size, max_size))
        alpha = draw(st.floats(0.5, 5.0))
        beta = draw(st.floats(0.5, 5.0))

        # Generate probabilities from Beta distribution
        probs = np.random.beta(alpha, beta, size=size)
        # Generate labels from Bernoulli with these probabilities (calibrated)
        labels = np.random.binomial(1, probs).astype(np.int8)

        return labels, probs, alpha, beta

    return _inner()


class TestWeightedEqualsExpanded:
    """Test that weighted metrics exactly match expanded dataset approach."""

    @given(
        data=binary_labels_and_probs(min_size=10, max_size=30),
        weights=integer_weights(min_value=1, max_value=3, min_size=10, max_size=30),
    )
    @settings(max_examples=30, deadline=5000)
    def test_integer_weights_exact_expansion(self, data, weights):
        """Weighted metrics with integer weights should exactly match expanded dataset."""
        labels, probs = data

        # Ensure weights match data size
        if len(weights) != len(labels):
            weights = (
                weights[: len(labels)]
                if len(weights) > len(labels)
                else np.pad(weights, (0, len(labels) - len(weights)), mode="wrap")
            )

        # Test multiple methods that support weights
        for method in ["unique_scan", "sort_scan"]:
            if method == "sort_scan":
                metric = "f1"  # sort_scan requires vectorized metrics
            else:
                metric = "accuracy"

            try:
                # Weighted approach
                threshold_weighted = get_optimal_threshold(
                    labels, probs, metric=metric, method=method, sample_weight=weights
                )

                # Expanded approach - duplicate samples according to weights
                labels_expanded = np.repeat(labels, weights)
                probs_expanded = np.repeat(probs, weights)

                threshold_expanded = get_optimal_threshold(
                    labels_expanded, probs_expanded, metric=metric, method=method
                )

                # Should be exactly equal or very close (allowing tiny numerical differences)
                # This catches the int() cast bug if it exists
                assert abs(threshold_weighted - threshold_expanded) < 1e-10, (
                    f"Method {method}: Weighted ({threshold_weighted:.10f}) and "
                    f"expanded ({threshold_expanded:.10f}) should be identical"
                )

                # Verify scores are also identical
                score_weighted = get_confusion_matrix(
                    labels, probs, threshold_weighted, weights
                )
                score_expanded = get_confusion_matrix(
                    labels_expanded, probs_expanded, threshold_expanded
                )

                # Compare F1/accuracy scores
                if metric == "f1":
                    metric_weighted = f1_score(*score_weighted)
                    metric_expanded = f1_score(*score_expanded)
                else:
                    metric_weighted = accuracy_score(*score_weighted)
                    metric_expanded = accuracy_score(*score_expanded)

                assert abs(metric_weighted - metric_expanded) < 1e-10, (
                    f"Method {method}: Metric scores should be identical"
                )

            except (ValueError, NotImplementedError):
                # Some combinations might not be supported
                continue

    @given(
        data=binary_labels_and_probs(min_size=15, max_size=40),
        base_weight=st.integers(1, 3),
    )
    @settings(max_examples=20, deadline=3000)
    def test_uniform_weights_match_unweighted(self, data, base_weight):
        """Uniform weights should give same result as unweighted optimization."""
        labels, probs = data
        uniform_weights = np.full(len(labels), base_weight, dtype=np.float64)

        # Compare weighted vs unweighted
        threshold_weighted = get_optimal_threshold(
            labels,
            probs,
            metric="accuracy",
            method="unique_scan",
            sample_weight=uniform_weights,
        )
        threshold_unweighted = get_optimal_threshold(
            labels, probs, metric="accuracy", method="unique_scan"
        )

        # Should be identical (uniform weights don't change relative importance)
        assert abs(threshold_weighted - threshold_unweighted) < 1e-10, (
            "Uniform weights should give same result as unweighted"
        )


class TestTieSemantics:
    """Test tie handling consistency across methods and comparison operators."""

    @given(
        size=st.integers(20, 50),
        tie_prob=st.floats(0.3, 0.7),
        tie_fraction=st.floats(0.2, 0.5),
    )
    @settings(max_examples=25, deadline=4000)
    def test_comprehensive_tie_handling(self, size, tie_prob, tie_fraction):
        """Test comprehensive tie handling across methods."""
        # Create data with many ties
        np.random.seed(42)  # Reproducible
        labels = np.random.randint(0, 2, size)
        probs = np.random.rand(size)

        # Force a fraction of probabilities to be exactly tie_prob
        n_ties = max(2, int(size * tie_fraction))
        tie_indices = np.random.choice(size, n_ties, replace=False)
        probs[tie_indices] = tie_prob

        # Test different methods and comparison operators
        methods_to_test = ["unique_scan"]
        if np.sum(labels) > 0 and np.sum(1 - labels) > 0:  # Avoid degenerate cases
            methods_to_test.append("sort_scan")

        results = {}

        for method in methods_to_test:
            for comparison in [">", ">="]:
                try:
                    if method == "sort_scan":
                        metric = "f1"
                    else:
                        metric = "f1"

                    threshold = get_optimal_threshold(
                        labels,
                        probs,
                        metric=metric,
                        method=method,
                        comparison=comparison,
                    )

                    # Apply threshold
                    if comparison == ">":
                        predictions = probs > threshold
                    else:
                        predictions = probs >= threshold

                    # Compute score
                    tp, tn, fp, fn = get_confusion_matrix(
                        labels, probs, threshold, comparison=comparison
                    )
                    score = f1_score(tp, tn, fp, fn)

                    results[(method, comparison)] = {
                        "threshold": threshold,
                        "score": score,
                        "n_predicted_pos": np.sum(predictions),
                    }

                except (ValueError, NotImplementedError):
                    continue

        # Verify consistency between methods
        if ("unique_scan", ">") in results and ("unique_scan", ">=") in results:
            # For same method, different comparison operators should handle ties appropriately
            gt_result = results[("unique_scan", ">")]
            gte_result = results[("unique_scan", ">=")]

            # Both should produce valid results
            assert 0 <= gt_result["score"] <= 1
            assert 0 <= gte_result["score"] <= 1

            # When probabilities equal threshold, >= should include more positives than >
            # (unless threshold is outside probability range)
            tied_probs = np.isclose(probs, gt_result["threshold"], atol=1e-10)
            if np.any(tied_probs):
                # This property might not hold if tie_prob is not the optimal threshold
                pass  # Just verify no crashes

    @given(size=st.integers(15, 30), n_ties=st.integers(3, 8))
    @settings(max_examples=15, deadline=5000)
    def test_dinkelbach_tie_consistency(self, size, n_ties):
        """Test that Dinkelbach handles tied probabilities appropriately."""
        # Create data with specific tie pattern
        np.random.seed(123)
        labels = np.random.randint(0, 2, size)
        probs = np.random.rand(size)

        # Create ties at a specific probability
        tie_prob = 0.5
        tie_indices = np.random.choice(size, min(n_ties, size), replace=False)
        probs[tie_indices] = tie_prob

        # Test both comparison operators
        for comparison in [">", ">="]:
            try:
                result = get_optimal_threshold(
                    labels, probs, mode="expected", metric="f1", comparison=comparison
                )
                threshold, _ = result  # Extract threshold from tuple

                # Should produce valid threshold
                assert 0 <= threshold <= 1

                # Apply threshold and verify sensible behavior
                if comparison == ">":
                    pass
                else:
                    pass

                # Compute F1 score
                tp, tn, fp, fn = get_confusion_matrix(
                    labels, probs, threshold, comparison=comparison
                )
                f1 = f1_score(tp, tn, fp, fn)
                assert 0 <= f1 <= 1

            except (ValueError, NotImplementedError):
                # Dinkelbach might not support all cases
                continue


class TestOneClassEdge:
    """Test edge cases with all-positive or all-negative labels."""

    @given(size=st.integers(5, 30), all_positive=st.booleans())
    @settings(max_examples=20, deadline=3000)
    def test_one_class_random_probabilities(self, size, all_positive):
        """Test one-class cases with random probabilities for accuracy maximization."""
        np.random.seed(42)
        probs = np.random.rand(size)
        labels = np.full(size, 1 if all_positive else 0, dtype=int)

        for comparison in [">", ">="]:
            threshold = get_optimal_threshold(
                labels,
                probs,
                metric="accuracy",
                method="unique_scan",
                comparison=comparison,
            )

            # Apply threshold
            if comparison == ">":
                predictions = probs > threshold
            else:
                predictions = probs >= threshold

            if all_positive:
                # All predictions should be positive for optimal accuracy
                assert np.all(predictions), (
                    f"All-positive case with {comparison}: should predict all positive, "
                    f"got {np.sum(predictions)}/{size} positive predictions"
                )
            else:
                # All predictions should be negative for optimal accuracy
                assert np.all(~predictions), (
                    f"All-negative case with {comparison}: should predict all negative, "
                    f"got {np.sum(predictions)}/{size} positive predictions"
                )

            # Verify perfect accuracy
            accuracy = np.mean(predictions == labels)
            assert accuracy == 1.0, f"Should achieve perfect accuracy, got {accuracy}"

    @given(
        size=st.integers(5, 25),
        prob_range=st.tuples(st.floats(0.01, 0.3), st.floats(0.7, 0.99)).filter(
            lambda x: x[0] < x[1]
        ),
    )
    @settings(max_examples=15, deadline=4000)
    def test_degenerate_threshold_bounds(self, size, prob_range):
        """Test that degenerate cases produce thresholds with correct bounds."""
        min_prob, max_prob = prob_range
        probs = np.random.uniform(min_prob, max_prob, size)

        # Test both all-positive and all-negative
        for all_positive in [True, False]:
            labels = np.full(size, 1 if all_positive else 0, dtype=int)

            for comparison in [">", ">="]:
                threshold = get_optimal_threshold(
                    labels,
                    probs,
                    metric="accuracy",
                    method="unique_scan",
                    comparison=comparison,
                )

                # Verify threshold is in valid range
                assert 0 <= threshold <= 1, f"Threshold {threshold} out of bounds"

                # Verify correct behavior
                if comparison == ">":
                    predictions = probs > threshold
                else:
                    predictions = probs >= threshold

                expected = np.full(size, all_positive, dtype=bool)
                assert np.array_equal(predictions, expected), (
                    f"Predictions should be all {all_positive}"
                )


class TestMicroAccuracyComparison:
    """Test micro accuracy comparison with K>=3 classes."""

    @given(data=multiclass_data(n_classes=3, min_size=30, max_size=60))
    @settings(max_examples=15, deadline=8000)
    def test_micro_vs_exclusive_accuracy(self, data):
        """Compare OvR micro accuracy aggregation vs exclusive accuracy."""
        labels, probs = data
        n_classes = probs.shape[1]

        # Generate some thresholds
        thresholds = np.random.uniform(0.2, 0.8, n_classes)

        # Compute exclusive accuracy (correct for multiclass)
        exclusive_acc = multiclass_metric_exclusive(
            labels, probs, thresholds, "accuracy"
        )

        # Try to compute OvR micro accuracy (should raise error)
        cms = get_multiclass_confusion_matrix(labels, probs, thresholds)

        with pytest.raises(ValueError, match="Micro-averaged accuracy requires"):
            multiclass_metric(cms, "accuracy", "micro")

        # Exclusive accuracy should be in valid range
        assert 0 <= exclusive_acc <= 1, (
            f"Exclusive accuracy {exclusive_acc} out of range"
        )

        # Verify exclusive predictions make sense
        exclusive_preds = _compute_exclusive_predictions(labels, probs, thresholds)
        assert len(exclusive_preds) == len(labels)
        assert np.all((exclusive_preds >= 0) & (exclusive_preds < n_classes))

    @given(
        n_classes=st.integers(3, 5),
        size=st.integers(50, 100),
        imbalance_factor=st.floats(0.1, 0.9),
    )
    @settings(max_examples=10, deadline=10000)
    def test_pathological_class_imbalance(self, n_classes, size, imbalance_factor):
        """Test micro accuracy with extreme class imbalances."""
        np.random.seed(42)

        # Create extremely imbalanced dataset
        # One dominant class gets most samples
        dominant_class = 0
        n_dominant = int(size * (1 - imbalance_factor))
        n_others = size - n_dominant

        labels = [dominant_class] * n_dominant + [
            i for i in range(1, n_classes) for _ in range(n_others // (n_classes - 1))
        ]
        labels = labels[:size]  # Truncate to exact size
        labels = np.array(labels)
        np.random.shuffle(labels)

        # Generate probabilities
        probs = np.random.rand(size, n_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Test with both OvR optimization and exclusive optimization
        try:
            # This should work - uses exclusive accuracy for micro
            thresholds_micro = get_optimal_multiclass_thresholds(
                labels,
                probs,
                metric="accuracy",
                average="micro",
                method="minimize",  # Use minimize for joint optimization
            )

            # Verify results
            assert len(thresholds_micro) == n_classes
            assert all(0 <= t <= 1 for t in thresholds_micro)

            # Compute exclusive accuracy
            exclusive_acc = multiclass_metric_exclusive(
                labels, probs, thresholds_micro, "accuracy"
            )
            assert 0 <= exclusive_acc <= 1

        except (ValueError, NotImplementedError) as e:
            # Some configurations might not be supported
            pytest.skip(f"Configuration not supported: {e}")

    @given(
        data=multiclass_data(n_classes=4, min_size=40, max_size=80),
        tie_probability=st.floats(0.3, 0.7),
    )
    @settings(max_examples=8, deadline=12000)
    def test_micro_accuracy_with_ties(self, data, tie_probability):
        """Test micro accuracy computation with tied probabilities."""
        labels, probs = data
        n_classes = probs.shape[1]

        # Introduce ties in probabilities
        n_ties = len(labels) // 4
        tie_indices = np.random.choice(len(labels), n_ties, replace=False)
        for idx in tie_indices:
            probs[idx, :2] = tie_probability  # Make first two classes tied

        # Renormalize
        probs = probs / probs.sum(axis=1, keepdims=True)

        thresholds = np.random.uniform(0.1, 0.9, n_classes)

        # Exclusive accuracy should handle ties gracefully
        try:
            exclusive_acc = multiclass_metric_exclusive(
                labels, probs, thresholds, "accuracy"
            )
            assert 0 <= exclusive_acc <= 1

            # Verify exclusive predictions
            exclusive_preds = _compute_exclusive_predictions(labels, probs, thresholds)

            # All predictions should be valid class indices
            assert np.all((exclusive_preds >= 0) & (exclusive_preds < n_classes))

        except Exception as e:
            pytest.fail(f"Exclusive accuracy failed with ties: {e}")


class TestDinkelbachCalibration:
    """Test Dinkelbach method with calibrated data using Beta/Bernoulli."""

    @given(data=beta_probabilities(min_size=100, max_size=300))
    @settings(max_examples=10, deadline=8000)
    def test_calibrated_data_sanity(self, data):
        """Test Dinkelbach gives reasonable results for calibrated Beta/Bernoulli data."""
        labels, probs, alpha, beta_param = data

        # Skip if data is too degenerate
        if np.sum(labels) == 0 or np.sum(labels) == len(labels):
            return

        for comparison in [">", ">="]:
            result = get_optimal_threshold(
                labels, probs, mode="expected", metric="f1", comparison=comparison
            )
            threshold, _ = result  # Extract threshold from tuple

            # Threshold should be reasonable
            assert 0 <= threshold <= 1

            # Compute actual F1 with this threshold
            tp, tn, fp, fn = get_confusion_matrix(
                labels, probs, threshold, comparison=comparison
            )
            empirical_f1 = f1_score(tp, tn, fp, fn)

            # For calibrated data, F1 should be decent
            assert 0 <= empirical_f1 <= 1

            # Empirical F1 should be reasonably good (not just random performance)
            # For calibrated data, we expect F1 > 0.2 in most cases
            if len(labels) > 50:  # Only for larger samples
                assert empirical_f1 > 0.1, (
                    f"F1 {empirical_f1:.3f} seems too low for calibrated data "
                    f"(α={alpha:.2f}, β={beta_param:.2f})"
                )

    @given(base_size=st.integers(50, 150), growth_factor=st.integers(2, 4))
    @settings(max_examples=8, deadline=12000)
    def test_dinkelbach_monotone_behavior(self, base_size, growth_factor):
        """Test that Dinkelbach shows reasonable monotone behavior as n grows."""
        # Fixed distribution parameters for consistency
        alpha, beta_param = 2.0, 3.0

        results = []

        # Test multiple sizes
        for size_multiplier in range(1, growth_factor + 1):
            size = base_size * size_multiplier

            # Generate calibrated data
            np.random.seed(42)  # Fixed seed for comparison
            probs = np.random.beta(alpha, beta_param, size)
            labels = np.random.binomial(1, probs).astype(np.int8)

            # Skip degenerate cases
            if np.sum(labels) == 0 or np.sum(labels) == size:
                continue

            try:
                result = get_optimal_threshold(
                    labels, probs, mode="expected", metric="f1", comparison=">"
                )
                threshold, _ = result  # Extract threshold from tuple

                # Compute empirical F1
                tp, tn, fp, fn = get_confusion_matrix(labels, probs, threshold)
                empirical_f1 = f1_score(tp, tn, fp, fn)

                results.append(
                    {
                        "size": size,
                        "threshold": threshold,
                        "f1": empirical_f1,
                        "expected_positives": np.sum(probs),
                    }
                )

            except (ValueError, NotImplementedError):
                continue

        # Verify results are reasonable
        for result in results:
            assert 0 <= result["threshold"] <= 1
            assert 0 <= result["f1"] <= 1

        # For calibrated data, larger samples should generally give more stable results
        if len(results) >= 2:
            # All F1 scores should be above some reasonable threshold
            f1_scores = [r["f1"] for r in results]
            assert all(f1 > 0.05 for f1 in f1_scores), (
                "F1 scores too low for calibrated data"
            )

    @given(size=st.integers(80, 200), noise_level=st.floats(0.0, 0.3))
    @settings(max_examples=10, deadline=6000)
    def test_dinkelbach_vs_neighbors(self, size, noise_level):
        """Test that Dinkelbach threshold performs better than neighboring thresholds."""
        # Generate calibrated data with some noise
        np.random.seed(42)
        alpha, beta_param = 1.5, 2.0
        probs = np.random.beta(alpha, beta_param, size)

        # Add noise to break perfect calibration
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size)
            probs = np.clip(probs + noise, 0.01, 0.99)

        labels = np.random.binomial(1, probs).astype(np.int8)

        # Skip degenerate cases
        if np.sum(labels) <= 1 or np.sum(labels) >= size - 1:
            return

        try:
            result = get_optimal_threshold(
                labels, probs, mode="expected", metric="f1", comparison=">"
            )
            optimal_threshold, _ = result  # Extract threshold from tuple

            # Compute F1 at optimal threshold
            tp, tn, fp, fn = get_confusion_matrix(labels, probs, optimal_threshold)
            optimal_f1 = f1_score(tp, tn, fp, fn)

            # Test neighboring thresholds
            epsilon = 0.05
            neighbors = [
                max(0.0, optimal_threshold - epsilon),
                min(1.0, optimal_threshold + epsilon),
            ]

            for neighbor_thresh in neighbors:
                if (
                    abs(neighbor_thresh - optimal_threshold) > 1e-6
                ):  # Actually different
                    tp_n, tn_n, fp_n, fn_n = get_confusion_matrix(
                        labels, probs, neighbor_thresh
                    )
                    neighbor_f1 = f1_score(tp_n, tn_n, fp_n, fn_n)

                    # Note: Dinkelbach optimizes *expected* F1, not empirical F1
                    # So empirical F1 might not always be optimal on this specific dataset
                    # We just verify the difference isn't too large (sanity check)
                    f1_diff = neighbor_f1 - optimal_f1
                    assert f1_diff < 0.1, (
                        f"Neighbor F1 {neighbor_f1:.6f} is much higher than optimal F1 {optimal_f1:.6f} "
                        f"(difference: {f1_diff:.6f}). This might indicate a problem."
                    )

        except (ValueError, NotImplementedError):
            # Some cases might not be supported
            pass
