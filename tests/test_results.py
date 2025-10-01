"""Tests for ThresholdResult unified result wrapper."""

import numpy as np
import pytest

from optimal_cutoffs.results import ThresholdResult, create_result


class TestThresholdResult:
    """Test ThresholdResult class functionality."""

    def test_basic_creation(self):
        """Test basic ThresholdResult creation."""
        result = ThresholdResult(threshold=0.5, score=0.8)
        assert result.threshold == 0.5
        assert result.score == 0.8
        assert not result.is_multiclass

    def test_scalar_ndarray_normalization(self):
        """Test that scalar ndarray thresholds become floats."""
        result = ThresholdResult(threshold=np.array(0.5))
        assert isinstance(result.threshold, float)
        assert result.threshold == 0.5
        assert not result.is_multiclass

    def test_multiclass_detection(self):
        """Test is_multiclass detection based on array size > 1."""
        # Single-element array should not be multiclass
        result = ThresholdResult(threshold=np.array([0.5]))
        assert not result.is_multiclass

        # Multi-element array should be multiclass
        result = ThresholdResult(threshold=np.array([0.3, 0.6]))
        assert result.is_multiclass

        # Float should not be multiclass
        result = ThresholdResult(threshold=0.5)
        assert not result.is_multiclass

    def test_array_shape_validation(self):
        """Test that bad array shapes are rejected."""
        # 2D threshold array should raise error
        with pytest.raises(ValueError, match="threshold array must be 1D"):
            ThresholdResult(threshold=np.ones((2, 2)))

        # 2D per_class_scores should raise error
        with pytest.raises(ValueError, match="per_class_scores must be 1D"):
            ThresholdResult(threshold=0.5, per_class_scores=np.ones((2, 2)))

        # 2D decisions should raise error
        with pytest.raises(ValueError, match="decisions must be 1D"):
            ThresholdResult(threshold=0.5, decisions=np.ones((2, 2)))

    def test_dtype_normalization(self):
        """Test that arrays are normalized to appropriate dtypes."""
        result = ThresholdResult(
            threshold=np.array([0.3, 0.6], dtype=np.float32),
            per_class_scores=np.array([0.8, 0.9], dtype=np.float32),
        )
        assert result.threshold.dtype == np.float64
        assert result.per_class_scores.dtype == np.float64

    def test_decisions_field(self):
        """Test decisions field for Bayes results."""
        decisions = np.array([1, 0, 2])
        result = ThresholdResult(threshold=None, decisions=decisions)
        assert np.array_equal(result.decisions, decisions)
        assert result.threshold is None


class TestCreateResult:
    """Test create_result factory function."""

    def test_threshold_required_validation(self):
        """Test that either threshold or decisions must be provided."""
        # Both None should raise error
        with pytest.raises(
            ValueError, match="either 'threshold' or 'decisions' must be provided"
        ):
            create_result()

        # Threshold only should work
        result = create_result(threshold=0.5)
        assert result.threshold == 0.5

        # Decisions only should work
        result = create_result(decisions=np.array([1, 0, 2]))
        assert np.array_equal(result.decisions, [1, 0, 2])

    def test_metadata_key_consistency(self):
        """Test that average key is used consistently."""
        # Test direct average parameter
        result = create_result(threshold=0.5, average="micro")
        assert result.metadata["average"] == "micro"

        # Test legacy averaging_method parameter
        result = create_result(threshold=0.5, averaging_method="macro")
        assert result.metadata["average"] == "macro"

        # Test that average takes precedence over averaging_method
        result = create_result(threshold=0.5, average="micro", averaging_method="macro")
        assert result.metadata["average"] == "micro"

    def test_metric_metadata(self):
        """Test that metric is stored in metadata."""
        result = create_result(threshold=0.5, metric="F1")
        assert result.metadata["metric"] == "f1"  # lowercased

    def test_score_array_handling(self):
        """Test handling of array scores."""
        # Array score should become per_class_scores
        scores = np.array([0.8, 0.9, 0.7])
        result = create_result(threshold=np.array([0.3, 0.4, 0.5]), score=scores)
        assert np.array_equal(result.per_class_scores, scores)
        assert abs(result.score - np.mean(scores)) < 1e-10

        # Test with NaN values (should use nanmean)
        scores_with_nan = np.array([0.8, np.nan, 0.7])
        result = create_result(
            threshold=np.array([0.3, 0.4, 0.5]), score=scores_with_nan
        )
        assert result.score == 0.75  # mean of 0.8 and 0.7


class TestLegacyFormat:
    """Test to_legacy_format method."""

    def test_empirical_mode(self):
        """Test empirical mode legacy format."""
        # Binary empirical should return float
        result = create_result(threshold=0.5, mode="empirical")
        assert result.to_legacy_format() == 0.5

        # Multiclass empirical should return array
        thresholds = np.array([0.3, 0.6])
        result = create_result(threshold=thresholds, mode="empirical")
        np.testing.assert_array_equal(result.to_legacy_format(), thresholds)

    def test_expected_micro_format(self):
        """Test expected micro averaging legacy format."""
        result = create_result(
            threshold=0.42, score=0.88, mode="expected", average="micro", metric="f1"
        )
        legacy = result.to_legacy_format()

        expected = {"threshold": 0.42, "score": 0.88, "f_beta": 0.88}
        assert legacy == expected

    def test_expected_macro_format(self):
        """Test expected macro averaging legacy format."""
        result = create_result(
            threshold=np.array([0.3, 0.6]),
            per_class_scores=np.array([0.8, 0.6]),
            score=0.7,
            mode="expected",
            average="macro",
            metric="fbeta",
        )
        legacy = result.to_legacy_format()

        assert "thresholds" in legacy
        assert "per_class" in legacy
        assert "score" in legacy
        assert legacy["f_beta"] == legacy["score"]  # legacy alias
        assert legacy["f_beta_per_class"] is result.per_class_scores

    def test_expected_micro_with_array_threshold(self):
        """Test expected micro with single-element array threshold."""
        result = create_result(
            threshold=np.array([0.42]),  # length-1 array
            score=0.88,
            mode="expected",
            average="micro",
        )
        legacy = result.to_legacy_format()

        assert legacy["threshold"] == 0.42  # should be extracted as float

    def test_bayes_decisions_format(self):
        """Test Bayes decisions legacy format."""
        decisions = np.array([1, 0, 2])
        result = create_result(
            decisions=decisions, mode="bayes", method="utility_matrix"
        )
        legacy = result.to_legacy_format()

        np.testing.assert_array_equal(legacy, decisions)

    def test_bayes_thresholds_format(self):
        """Test Bayes thresholds legacy format."""
        thresholds = np.array([0.3, 0.6])
        result = create_result(threshold=thresholds, mode="bayes", method="ovr")
        legacy = result.to_legacy_format()

        np.testing.assert_array_equal(legacy, thresholds)

    def test_legacy_aliases_fbeta_only(self):
        """Test that legacy aliases only appear for F-beta metrics."""
        # F-beta metric should have aliases
        result = create_result(
            threshold=0.5, score=0.8, mode="expected", average="micro", metric="f1"
        )
        legacy = result.to_legacy_format()
        assert "f_beta" in legacy

        # Non F-beta metric should not have aliases
        result = create_result(
            threshold=0.5,
            score=0.8,
            mode="expected",
            average="micro",
            metric="precision",
        )
        legacy = result.to_legacy_format()
        assert "f_beta" not in legacy
        assert "score" in legacy


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_threshold_with_decisions(self):
        """Test that None threshold works with decisions."""
        result = create_result(decisions=[1, 0, 2], mode="bayes")
        assert result.threshold is None
        assert result.decisions is not None

    def test_directions_field(self):
        """Test directions field handling."""
        result = create_result(
            threshold=np.array([0.3, 0.6]), directions=np.array([">", "<"])
        )
        assert result.directions is not None

    def test_nan_score_handling(self):
        """Test NaN score handling."""
        # Test per-class scores with NaN
        per_class = np.array([0.8, np.nan, 0.7])
        result = create_result(
            threshold=np.array([0.3, 0.4, 0.5]), per_class_scores=per_class
        )
        # Should compute nanmean for overall score
        assert abs(result.score - 0.75) < 1e-10

    def test_empty_per_class_scores(self):
        """Test empty per-class scores."""
        result = create_result(
            threshold=0.5,
            score=np.array([]),  # empty array
        )
        assert result.score == 0.0

    def test_metadata_preservation(self):
        """Test that additional metadata is preserved."""
        result = create_result(
            threshold=0.5, custom_field="test_value", another_field=123
        )
        assert result.metadata["custom_field"] == "test_value"
        assert result.metadata["another_field"] == 123
