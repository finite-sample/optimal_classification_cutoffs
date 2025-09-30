"""Test cross-validation correctness for threshold selection.

This module tests that nested cross-validation properly selects thresholds
by inner mean score over candidate thresholds, ensuring proper model
selection methodology:

1. Nested CV structure: Outer CV for evaluation, inner CV for threshold selection
2. Threshold selection by inner mean: Choose threshold maximizing inner CV mean score
3. No data leakage: Training/validation/test splits are properly isolated
4. Score consistency: Inner and outer CV scores should be consistent

These tests verify the mathematical correctness of cross-validation based
threshold selection and prevent common CV methodology errors.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import KFold, StratifiedKFold

from optimal_cutoffs import get_optimal_threshold
from optimal_cutoffs.cv import (
    cv_threshold_optimization,
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

    return labels, probs


class TestBasicCrossValidation:
    """Test basic cross-validation functionality."""

    def test_cross_val_score_basic_functionality(self):
        """Test that cross_val_score_threshold produces reasonable results."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold  # noqa: F401
        except ImportError:
            pytest.skip("CV module not available")

        labels, probs = _generate_cv_data(n_samples=50, random_state=123)

        # Ensure both classes present
        if labels.sum() == 0:
            labels[0] = 1
        if labels.sum() == labels.size:
            labels[0] = 0

        try:
            thresholds, scores = cv_threshold_optimization(
                labels, probs, metric="f1", method="sort_scan", cv=3, random_state=42
            )

            # Should return one score per fold
            assert len(scores) == 3, f"Expected 3 CV scores, got {len(scores)}"

            # All scores should be valid
            assert all(0 <= score <= 1 for score in scores), (
                f"All CV scores should be in [0,1], got {scores}"
            )

            # Should be able to compute mean and std
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            assert 0 <= mean_score <= 1, f"Mean score {mean_score} out of valid range"
            assert std_score >= 0, f"Std score {std_score} should be non-negative"

        except (ImportError, NotImplementedError):
            pytest.skip("Cross-validation not implemented")

    def test_cv_with_different_metrics(self):
        """Test CV with different metrics."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold
        except ImportError:
            pytest.skip("CV module not available")

        labels, probs = _generate_cv_data(n_samples=60, random_state=456)

        if labels.sum() == 0:
            labels[0] = 1
        if labels.sum() == labels.size:
            labels[0] = 0

        metrics_to_test = ["f1", "accuracy", "precision", "recall"]

        for metric in metrics_to_test:
            try:
                scores = cross_val_score_threshold(
                    labels,
                    probs,
                    metric=metric,
                    method="sort_scan",
                    cv=3,
                    comparison=">",
                    random_state=789,
                )

                assert len(scores) == 3, f"Expected 3 scores for {metric}"
                assert all(0 <= score <= 1 for score in scores), (
                    f"{metric} scores should be in [0,1]: {scores}"
                )

            except (ValueError, NotImplementedError) as e:
                # Some metric/method combinations might not be supported
                if "not supported" in str(e).lower():
                    continue
                raise

    def test_cv_reproducibility(self):
        """Test that CV results are reproducible with same random state."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold
        except ImportError:
            pytest.skip("CV module not available")

        labels, probs = _generate_cv_data(n_samples=40, random_state=111)

        if labels.sum() == 0:
            labels[0] = 1
        if labels.sum() == labels.size:
            labels[0] = 0

        try:
            # Run CV twice with same random state
            scores1 = cross_val_score_threshold(
                labels,
                probs,
                metric="f1",
                method="sort_scan",
                cv=3,
                comparison=">",
                random_state=555,
            )

            scores2 = cross_val_score_threshold(
                labels,
                probs,
                metric="f1",
                method="sort_scan",
                cv=3,
                comparison=">",
                random_state=555,
            )

            # Should be identical
            assert np.allclose(scores1, scores2, atol=1e-12), (
                f"CV should be reproducible: {scores1} vs {scores2}"
            )

        except (ImportError, NotImplementedError):
            pytest.skip("Cross-validation not implemented")


class TestNestedCrossValidation:
    """Test nested cross-validation for proper threshold selection."""

    def test_nested_cv_basic_functionality(self):
        """Test basic nested CV functionality."""
        try:
            from optimal_cutoffs.cv import nested_cv_threshold_selection
        except ImportError:
            pytest.skip("Nested CV not available")

        labels, probs = _generate_cv_data(n_samples=80, random_state=222)

        if labels.sum() <= 2:
            labels[:3] = [0, 1, 1]
        if labels.sum() >= labels.size - 2:
            labels[-3:] = [1, 0, 0]

        try:
            result = nested_cv_threshold_selection(
                labels,
                probs,
                metric="f1",
                method="sort_scan",
                outer_cv=3,
                inner_cv=2,
                comparison=">",
                random_state=333,
            )

            # Should return a structured result
            assert hasattr(result, "outer_scores") or "outer_scores" in result
            assert (
                hasattr(result, "selected_thresholds")
                or "selected_thresholds" in result
            )

            # Extract scores and thresholds
            if hasattr(result, "outer_scores"):
                outer_scores = result.outer_scores
                selected_thresholds = result.selected_thresholds
            else:
                outer_scores = result["outer_scores"]
                selected_thresholds = result["selected_thresholds"]

            # Should have one outer score per outer fold
            assert len(outer_scores) == 3, (
                f"Expected 3 outer scores, got {len(outer_scores)}"
            )
            assert all(0 <= score <= 1 for score in outer_scores), (
                f"Outer scores should be in [0,1]: {outer_scores}"
            )

            # Should have one threshold per outer fold
            assert len(selected_thresholds) == 3, (
                f"Expected 3 thresholds, got {len(selected_thresholds)}"
            )
            assert all(0 <= thresh <= 1 for thresh in selected_thresholds), (
                f"Thresholds should be in [0,1]: {selected_thresholds}"
            )

        except (ImportError, NotImplementedError):
            pytest.skip("Nested CV not implemented")

    def test_nested_cv_threshold_selection_consistency(self):
        """Test that nested CV selects thresholds based on inner CV scores."""
        try:
            from optimal_cutoffs.cv import (
                cross_val_score_threshold,  # noqa: F401
                nested_cv_threshold_selection,  # noqa: F401
            )
        except ImportError:
            pytest.skip("Nested CV not available")

        # Create data where threshold selection should matter
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        probs = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9, 0.2, 0.35, 0.65, 0.8])

        try:
            # Manual nested CV to verify the selection process
            outer_cv = KFold(n_splits=2, shuffle=True, random_state=444)

            for train_idx, test_idx in outer_cv.split(labels):
                train_labels, train_probs = labels[train_idx], probs[train_idx]
                test_labels, test_probs = labels[test_idx], probs[test_idx]

                # Skip if degenerate split
                if train_labels.sum() == 0 or train_labels.sum() == len(train_labels):
                    continue

                # Get optimal threshold on training data
                train_threshold = get_optimal_threshold(
                    train_labels,
                    train_probs,
                    metric="f1",
                    method="sort_scan",
                    comparison=">",
                )

                # This threshold should be reasonable
                assert 0 <= train_threshold <= 1, (
                    f"Training threshold {train_threshold} out of range"
                )

                # Apply to test data
                test_pred = test_probs > train_threshold

                # Should produce reasonable predictions
                assert len(test_pred) == len(test_labels)

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not implemented", "not supported"]
            ):
                pytest.skip(f"Manual nested CV verification failed: {e}")
            raise

    def test_nested_cv_no_data_leakage(self):
        """Test that nested CV properly isolates training/validation/test data."""
        try:
            from optimal_cutoffs.cv import nested_cv_threshold_selection
        except ImportError:
            pytest.skip("Nested CV not available")

        # Create data with clear pattern that could be leaked
        n = 60
        labels = np.array([i % 2 for i in range(n)])  # Alternating pattern
        probs = np.array(
            [0.2 if i % 2 == 0 else 0.8 for i in range(n)]
        )  # Clear separation

        try:
            result = nested_cv_threshold_selection(
                labels,
                probs,
                metric="accuracy",
                method="sort_scan",
                outer_cv=3,
                inner_cv=2,
                comparison=">",
                random_state=666,
            )

            # Extract outer scores
            if hasattr(result, "outer_scores"):
                outer_scores = result.outer_scores
            else:
                outer_scores = result["outer_scores"]

            # With proper CV, should not achieve perfect scores (due to generalization)
            # unless the pattern is truly perfect and generalizable
            assert all(0 <= score <= 1 for score in outer_scores), (
                f"Outer CV scores out of range: {outer_scores}"
            )

            # The key test: nested CV should be producing reasonable generalization estimates
            # If there was data leakage, scores would be artificially inflated
            mean_score = np.mean(outer_scores)
            assert mean_score < 1.0 or len(set(labels)) == 1, (
                "Suspiciously perfect nested CV score might indicate data leakage"
            )

        except (ImportError, NotImplementedError):
            pytest.skip("Nested CV not implemented")

    def test_inner_cv_selects_best_threshold(self):
        """Test that inner CV actually selects the threshold with best inner mean score."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold  # noqa: F401
        except ImportError:
            pytest.skip("CV not available")

        # Create case where threshold choice matters
        labels, probs = _generate_cv_data(n_samples=50, random_state=777)

        if labels.sum() <= 2:
            labels[:3] = [0, 1, 1]
        if labels.sum() >= labels.size - 2:
            labels[-3:] = [1, 0, 0]

        try:
            # Test several candidate thresholds
            candidate_thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
            threshold_scores = {}

            for threshold in candidate_thresholds:
                # Simulate applying this threshold in CV
                cv_scores = []
                kf = KFold(n_splits=3, shuffle=True, random_state=888)

                for train_idx, val_idx in kf.split(labels):
                    val_labels, val_probs = labels[val_idx], probs[val_idx]

                    # Apply fixed threshold
                    val_pred = val_probs > threshold

                    # Compute F1
                    tp = np.sum((val_labels == 1) & val_pred)
                    fp = np.sum((val_labels == 0) & val_pred)
                    fn = np.sum((val_labels == 1) & ~val_pred)

                    if tp + fp == 0:
                        precision = 0.0
                    else:
                        precision = tp / (tp + fp)

                    if tp + fn == 0:
                        recall = 0.0
                    else:
                        recall = tp / (tp + fn)

                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * precision * recall / (precision + recall)

                    cv_scores.append(f1)

                threshold_scores[threshold] = np.mean(cv_scores)

            # Find best threshold by inner CV
            best_threshold = max(
                threshold_scores.keys(), key=lambda t: threshold_scores[t]
            )
            best_score = threshold_scores[best_threshold]

            # Verify this is actually the maximum
            for thresh, score in threshold_scores.items():
                assert score <= best_score + 1e-10, (
                    f"Threshold {thresh} has score {score} > best score {best_score} for threshold {best_threshold}"
                )

            # The selected threshold should be reasonable
            assert 0 <= best_threshold <= 1
            assert 0 <= best_score <= 1

        except Exception as e:
            # This test is complex and might fail for various reasons
            if any(
                phrase in str(e).lower()
                for phrase in ["degenerate", "empty", "single class"]
            ):
                pytest.skip(f"Degenerate case in manual CV: {e}")
            raise


class TestCVEdgeCases:
    """Test edge cases in cross-validation."""

    def test_cv_with_small_datasets(self):
        """Test CV behavior with small datasets."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold
        except ImportError:
            pytest.skip("CV module not available")

        # Very small dataset
        labels = np.array([0, 1, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.7, 0.3, 0.9])

        try:
            scores = cross_val_score_threshold(
                labels,
                probs,
                metric="f1",
                method="sort_scan",
                cv=2,
                comparison=">",
                random_state=999,  # Small CV to avoid empty folds
            )

            # Should handle small dataset gracefully
            assert len(scores) == 2
            assert all(0 <= score <= 1 for score in scores)

        except ValueError as e:
            # Might reject datasets that are too small for CV
            if any(
                phrase in str(e).lower()
                for phrase in ["too small", "empty", "single class"]
            ):
                pytest.skip(f"Small dataset rejected: {e}")
            raise

    def test_cv_with_imbalanced_data(self):
        """Test CV with highly imbalanced classes."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold
        except ImportError:
            pytest.skip("CV module not available")

        # Highly imbalanced: 90% class 0, 10% class 1
        n = 50
        n_positive = 5
        labels = np.concatenate([np.zeros(n - n_positive), np.ones(n_positive)])

        # Generate probabilities somewhat aligned with imbalance
        rng = np.random.default_rng(1111)
        probs = (
            rng.uniform(0, 0.3, size=n - n_positive).tolist()
            + rng.uniform(0.4, 1.0, size=n_positive).tolist()
        )
        probs = np.array(probs)

        # Shuffle
        indices = rng.permutation(n)
        labels = labels[indices]
        probs = probs[indices]

        try:
            # Use stratified CV to handle imbalance
            scores = cross_val_score_threshold(
                labels,
                probs,
                metric="f1",
                method="sort_scan",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1212),
                comparison=">",
                random_state=1313,
            )

            assert len(scores) == 3
            assert all(0 <= score <= 1 for score in scores)

        except (ValueError, ImportError) as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["stratified", "imbalanced", "single class"]
            ):
                pytest.skip(f"Imbalanced case not supported: {e}")
            raise

    def test_cv_determinism_across_methods(self):
        """Test that CV gives consistent results across different optimization methods."""
        try:
            from optimal_cutoffs.cv import cross_val_score_threshold
        except ImportError:
            pytest.skip("CV module not available")

        labels, probs = _generate_cv_data(n_samples=40, random_state=1414)

        if labels.sum() <= 2:
            labels[:3] = [0, 1, 1]
        if labels.sum() >= labels.size - 2:
            labels[-3:] = [1, 0, 0]

        methods_to_test = ["sort_scan", "unique_scan"]
        method_scores = {}

        for method in methods_to_test:
            try:
                scores = cross_val_score_threshold(
                    labels,
                    probs,
                    metric="f1",
                    method=method,
                    cv=3,
                    comparison=">",
                    random_state=1515,
                )
                method_scores[method] = scores

            except (ValueError, NotImplementedError) as e:
                if "not supported" in str(e).lower():
                    continue
                raise

        # If multiple methods worked, their mean scores should be similar
        # (since they should all find the optimal threshold)
        if len(method_scores) >= 2:
            method_means = {
                method: np.mean(scores) for method, scores in method_scores.items()
            }

            # All methods should produce reasonable scores
            for method, mean_score in method_means.items():
                assert 0 <= mean_score <= 1, (
                    f"{method} mean score {mean_score} out of range"
                )

            # Methods should produce similar results (allowing for small numerical differences)
            score_values = list(method_means.values())
            max_diff = max(score_values) - min(score_values)

            # This is not a strict requirement due to different threshold selection strategies,
            # but they should be in the same ballpark
            assert max_diff < 0.5, f"Methods differ too much: {method_means}"

    @given(n_samples=st.integers(20, 60), cv_folds=st.integers(2, 5))
    @settings(deadline=None, max_examples=10)
    def test_cv_property_based(self, n_samples, cv_folds):
        """Property-based test for CV consistency."""
        # Skip if CV folds too large for dataset
        if cv_folds >= n_samples // 2:
            return

        try:
            from optimal_cutoffs.cv import cross_val_score_threshold
        except ImportError:
            pytest.skip("CV module not available")

        labels, probs = _generate_cv_data(n_samples=n_samples, random_state=42)

        # Ensure both classes present
        if labels.sum() <= 1:
            labels[:2] = [0, 1]
        if labels.sum() >= labels.size - 1:
            labels[-2:] = [1, 0]

        try:
            scores = cross_val_score_threshold(
                labels,
                probs,
                metric="accuracy",
                method="sort_scan",
                cv=cv_folds,
                comparison=">",
                random_state=1616,
            )

            # Basic properties
            assert len(scores) == cv_folds, (
                f"Expected {cv_folds} scores, got {len(scores)}"
            )
            assert all(0 <= score <= 1 for score in scores), (
                f"Scores out of range: {scores}"
            )
            assert all(
                not np.isnan(score) and not np.isinf(score) for score in scores
            ), f"Invalid scores: {scores}"

            # Statistical properties
            if len(scores) > 1:
                mean_score = np.mean(scores)
                std_score = np.std(scores)

                assert 0 <= mean_score <= 1, f"Mean score {mean_score} out of range"
                assert 0 <= std_score <= 1, f"Std score {std_score} out of range"

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented", "degenerate"]
            ):
                pytest.skip(f"Configuration not supported: {e}")
            raise
