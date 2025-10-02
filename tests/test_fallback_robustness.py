"""Comprehensive tests for fallback robustness in binary optimization."""

import numpy as np
import pytest
import warnings

from optimal_cutoffs.binary_optimization import (
    optimal_threshold_piecewise,
    optimal_threshold_minimize,
    optimal_threshold_gradient,
    _optimal_threshold_piecewise_fallback,
)
from optimal_cutoffs.metrics import compute_metric_at_threshold


class TestFallbackTieHandling:
    """Test that fallback methods handle ties correctly."""

    def test_ties_with_inclusive_semantics(self):
        """Test ties with >= comparison are handled correctly."""
        # All samples have same probability - pure tie scenario
        y_true = np.array([1, 0, 1, 0])
        p_tied = np.array([0.5, 0.5, 0.5, 0.5])

        # Should not crash and return a valid threshold
        threshold = _optimal_threshold_piecewise_fallback(
            y_true, p_tied, metric="f1", comparison=">="
        )
        assert isinstance(threshold, float)
        assert np.isfinite(threshold)
        
        # Test with main piecewise function too
        threshold_main = optimal_threshold_piecewise(
            y_true, p_tied, metric="f1", comparison=">="
        )
        assert isinstance(threshold_main, float)
        assert np.isfinite(threshold_main)

    def test_ties_with_exclusive_semantics(self):
        """Test ties with > comparison are handled correctly."""
        y_true = np.array([1, 0, 1, 0])
        p_tied = np.array([0.5, 0.5, 0.5, 0.5])

        threshold = _optimal_threshold_piecewise_fallback(
            y_true, p_tied, metric="f1", comparison=">"
        )
        assert isinstance(threshold, float)
        assert np.isfinite(threshold)

    def test_fallback_vs_main_consistency(self):
        """Test that fallback gives same result as main when fast path unavailable."""
        y_true = np.array([0, 1, 0, 1, 0])
        p_test = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
        
        # Create a custom non-vectorized metric 
        from optimal_cutoffs.metrics import register_metric
        
        def custom_metric(tp, tn, fp, fn):
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)  # precision-like
        
        # Register without vectorized version
        register_metric("test_nonvectorized", custom_metric, is_piecewise=True)
        
        try:
            # Force fallback by using non-vectorized metric
            threshold_fallback = _optimal_threshold_piecewise_fallback(
                y_true, p_test, metric="test_nonvectorized", comparison=">"
            )
            
            # Should get same result from main function when it falls back
            threshold_main = optimal_threshold_piecewise(
                y_true, p_test, metric="test_nonvectorized", comparison=">"
            )
            
            assert abs(threshold_fallback - threshold_main) < 1e-10
        finally:
            # Clean up
            from optimal_cutoffs.metrics import METRIC_REGISTRY, METRIC_PROPERTIES
            if "test_nonvectorized" in METRIC_REGISTRY:
                del METRIC_REGISTRY["test_nonvectorized"]
            if "test_nonvectorized" in METRIC_PROPERTIES:
                del METRIC_PROPERTIES["test_nonvectorized"]


class TestScoreHandling:
    """Test that methods handle arbitrary scores (require_proba=False) correctly."""

    def test_fallback_with_scores(self):
        """Test fallback works with arbitrary score ranges."""
        y_true = np.array([0, 1, 0, 1])
        scores = np.array([-2.1, 0.3, 0.1, 3.7])

        threshold = _optimal_threshold_piecewise_fallback(
            y_true, scores, metric="f1", require_proba=False
        )
        
        # Should be in extended score range (nextafter can go slightly beyond)
        min_bound = np.nextafter(np.min(scores), -np.inf)
        max_bound = np.nextafter(np.max(scores), np.inf)
        assert min_bound <= threshold <= max_bound

    def test_piecewise_with_scores(self):
        """Test main piecewise function with scores."""
        y_true = np.array([0, 1, 0, 1])
        scores = np.array([-2.1, 0.3, 0.1, 3.7])

        threshold = optimal_threshold_piecewise(
            y_true, scores, metric="f1", require_proba=False
        )
        
        # Should work and give reasonable threshold
        assert isinstance(threshold, float)
        assert np.isfinite(threshold)

    def test_minimize_with_scores(self):
        """Test minimize function with scores."""
        y_true = np.array([0, 0, 1, 1, 0])
        scores = np.array([-3.0, -2.0, 0.0, 2.0, 3.0])

        threshold = optimal_threshold_minimize(
            y_true, scores, metric="f1", require_proba=False
        )
        
        # Should be within reasonable bounds of the score range
        assert -4.0 <= threshold <= 4.0  # Allow some tolerance beyond strict bounds

    def test_gradient_with_scores(self):
        """Test gradient function with scores."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([-2.0, -1.0, 1.0, 2.0])

        # Should warn about piecewise metric but still work
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            threshold = optimal_threshold_gradient(
                y_true, scores, metric="f1", require_proba=False, max_iter=5
            )
            
            # Should issue warning about piecewise metric
            assert len(w) > 0
            assert "piecewise metric" in str(w[0].message).lower()
        
        # Should return finite result
        assert isinstance(threshold, float)
        assert np.isfinite(threshold)


class TestEdgeCases:
    """Test edge cases that could break fallback methods."""

    def test_all_positives(self):
        """Test when all samples are positive."""
        y_all_pos = np.array([1, 1, 1, 1])
        p_test = np.array([0.2, 0.5, 0.7, 0.9])

        threshold_gt = _optimal_threshold_piecewise_fallback(
            y_all_pos, p_test, metric="f1", comparison=">"
        )
        threshold_gte = _optimal_threshold_piecewise_fallback(
            y_all_pos, p_test, metric="f1", comparison=">="
        )
        
        assert isinstance(threshold_gt, float)
        assert isinstance(threshold_gte, float)
        
        # For all positives, optimal should predict all as positive
        # So threshold should be low
        assert threshold_gt <= np.min(p_test)
        assert threshold_gte <= np.min(p_test)

    def test_all_negatives(self):
        """Test when all samples are negative."""
        y_all_neg = np.array([0, 0, 0, 0])
        p_test = np.array([0.2, 0.5, 0.7, 0.9])

        # Use accuracy instead of F1 since F1 is undefined with all negatives
        threshold_gt = _optimal_threshold_piecewise_fallback(
            y_all_neg, p_test, metric="accuracy", comparison=">"
        )
        threshold_gte = _optimal_threshold_piecewise_fallback(
            y_all_neg, p_test, metric="accuracy", comparison=">="
        )
        
        assert isinstance(threshold_gt, float)
        assert isinstance(threshold_gte, float)
        
        # For all negatives with accuracy, optimal should predict all as negative
        # So threshold should be high (at or beyond max)
        assert threshold_gt >= np.max(p_test)
        assert threshold_gte > np.max(p_test)

    def test_single_sample(self):
        """Test with single sample."""
        y_single = np.array([1])
        p_single = np.array([0.7])

        threshold = _optimal_threshold_piecewise_fallback(
            y_single, p_single, metric="f1", comparison=">"
        )
        
        assert isinstance(threshold, float)
        assert np.isfinite(threshold)

    def test_empty_arrays(self):
        """Test that empty arrays raise appropriate errors."""
        y_empty = np.array([])
        p_empty = np.array([])

        with pytest.raises(ValueError):
            _optimal_threshold_piecewise_fallback(
                y_empty, p_empty, metric="f1"
            )


class TestCandidateGeneration:
    """Test that candidate generation is correct and tie-safe."""

    def test_midpoint_generation(self):
        """Test that midpoints are generated correctly."""
        y_true = np.array([0, 1, 0, 1])
        # Test with distinct probabilities
        p_distinct = np.array([0.2, 0.4, 0.6, 0.8])

        # Directly test the fallback to see its behavior
        threshold = _optimal_threshold_piecewise_fallback(
            y_true, p_distinct, metric="f1", comparison=">"
        )
        
        # Should find a good threshold, not just boundary values
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_nextafter_boundaries(self):
        """Test that boundary conditions use nextafter correctly."""
        y_true = np.array([0, 1])
        p_test = np.array([0.3, 0.7])

        # The fallback should consider nextafter values for extremes
        threshold_gt = _optimal_threshold_piecewise_fallback(
            y_true, p_test, metric="accuracy", comparison=">"
        )
        threshold_gte = _optimal_threshold_piecewise_fallback(
            y_true, p_test, metric="accuracy", comparison=">="
        )
        
        assert isinstance(threshold_gt, float)
        assert isinstance(threshold_gte, float)


class TestConsistencyAcrossMethods:
    """Test consistency between different optimization methods."""

    def test_methods_agree_on_simple_case(self):
        """Test that different methods agree on a simple case."""
        y_true = np.array([0, 0, 1, 1])
        p_test = np.array([0.1, 0.3, 0.7, 0.9])

        # All methods should find similar thresholds for this clear case
        thresh_piecewise = optimal_threshold_piecewise(
            y_true, p_test, metric="f1", comparison=">"
        )
        thresh_minimize = optimal_threshold_minimize(
            y_true, p_test, metric="f1", comparison=">"
        )
        
        # Compute F1 scores at both thresholds - they should be close
        score_piecewise = compute_metric_at_threshold(
            y_true, p_test, thresh_piecewise, "f1", comparison=">"
        )
        score_minimize = compute_metric_at_threshold(
            y_true, p_test, thresh_minimize, "f1", comparison=">"
        )
        
        # Scores should be very close (allowing for numerical differences)
        assert abs(score_piecewise - score_minimize) < 0.01

    def test_fallback_consistency(self):
        """Test that fallback gives consistent results."""
        y_true = np.array([0, 1, 0, 1, 0])
        p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Run fallback multiple times - should be deterministic
        thresh1 = _optimal_threshold_piecewise_fallback(
            y_true, p_test, metric="f1", comparison=">"
        )
        thresh2 = _optimal_threshold_piecewise_fallback(
            y_true, p_test, metric="f1", comparison=">"
        )
        
        assert thresh1 == thresh2


class TestGradientWarnings:
    """Test that gradient method issues appropriate warnings."""

    def test_gradient_warns_on_piecewise_metrics(self):
        """Test that gradient method warns for piecewise metrics."""
        y_true = np.array([0, 1, 0, 1])
        p_test = np.array([0.2, 0.4, 0.6, 0.8])

        piecewise_metrics = ["f1", "accuracy", "precision", "recall"]
        
        for metric in piecewise_metrics:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                optimal_threshold_gradient(
                    y_true, p_test, metric=metric, max_iter=5
                )
                
                # Should issue warning
                assert len(w) > 0
                warning_msg = str(w[0].message).lower()
                assert "gradient ascent is ineffective" in warning_msg
                assert "piecewise" in warning_msg

    def test_gradient_no_warn_on_smooth_metrics(self):
        """Test that gradient method doesn't warn for smooth metrics."""
        y_true = np.array([0, 1, 0, 1])
        p_test = np.array([0.2, 0.4, 0.6, 0.8])

        # Create a mock smooth metric by registering one
        from optimal_cutoffs.metrics import register_metric
        
        def smooth_metric(tp, tn, fp, fn):
            # A smooth, non-piecewise metric
            return tp / (tp + fp + 0.001)  # Smoothed precision
        
        # Register temporarily
        register_metric("test_smooth", smooth_metric, is_piecewise=False)
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                optimal_threshold_gradient(
                    y_true, p_test, metric="test_smooth", max_iter=5
                )
                    
                # Should not issue piecewise warning
                piecewise_warnings = [
                    warning for warning in w 
                    if "piecewise" in str(warning.message).lower()
                ]
                assert len(piecewise_warnings) == 0
        finally:
            # Clean up the registered metric
            from optimal_cutoffs.metrics import METRIC_REGISTRY, METRIC_PROPERTIES
            if "test_smooth" in METRIC_REGISTRY:
                del METRIC_REGISTRY["test_smooth"]
            if "test_smooth" in METRIC_PROPERTIES:
                del METRIC_PROPERTIES["test_smooth"]


if __name__ == "__main__":
    pytest.main([__file__])