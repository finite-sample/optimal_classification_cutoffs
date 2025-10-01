"""Comprehensive tests for the ThresholdOptimizer wrapper."""

import numpy as np
import pytest

from optimal_cutoffs.wrapper import ThresholdOptimizer


class TestBinaryVsMulticlassDetection:
    """Test proper detection of binary vs multiclass inputs."""

    def test_binary_1d_input(self):
        """Test that 1D input is correctly detected as binary."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.3, 0.9, 0.7])

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_prob)

        assert optimizer.is_multiclass_ is False
        assert optimizer.n_classes_ == 2

        # Test prediction
        predictions = optimizer.predict(y_prob)
        assert predictions.dtype == np.int64  # Returns int, not bool
        assert all(pred in [0, 1] for pred in predictions)

    def test_binary_n_1_input(self):
        """Test that (n, 1) input is correctly detected as binary."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([[0.1], [0.8], [0.3], [0.9], [0.7]])

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_prob)

        assert optimizer.is_multiclass_ is False
        assert optimizer.n_classes_ == 2

        # Test prediction
        predictions = optimizer.predict(y_prob)
        assert predictions.dtype == np.int64
        assert all(pred in [0, 1] for pred in predictions)

    def test_multiclass_input(self):
        """Test that (n, k) with k > 1 is detected as multiclass."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_prob)

        assert optimizer.is_multiclass_ is True
        assert optimizer.n_classes_ == 3

        # Test prediction
        predictions = optimizer.predict(y_prob)
        assert all(pred in [0, 1, 2] for pred in predictions)


class TestAverageParameter:
    """Test that the average parameter is properly handled."""

    def test_average_parameter_forwarding(self):
        """Test that average parameter is forwarded to get_optimal_threshold."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        # Test macro averaging
        optimizer_macro = ThresholdOptimizer(metric="f1", average="macro")
        optimizer_macro.fit(y_true, y_prob)
        assert len(optimizer_macro.threshold_) == 3  # Per-class thresholds

        # Test micro averaging
        optimizer_micro = ThresholdOptimizer(metric="f1", average="micro")
        optimizer_micro.fit(y_true, y_prob)
        # Micro can return either single threshold or per-class thresholds
        if isinstance(optimizer_micro.threshold_, float):
            assert isinstance(optimizer_micro.threshold_, float)
        else:
            assert len(optimizer_micro.threshold_) == 3


class TestExpectedModeResults:
    """Test proper handling of expected mode return values."""

    def test_expected_binary_tuple_result(self):
        """Test that binary expected mode returns are normalized properly."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.3, 0.9, 0.7])

        optimizer = ThresholdOptimizer(metric="f1", mode="expected")
        optimizer.fit(y_true, y_prob)

        # Should have both threshold and expected score
        assert isinstance(optimizer.threshold_, float)
        assert isinstance(optimizer.expected_score_, float)
        assert 0.0 <= optimizer.threshold_ <= 1.0

    def test_expected_multiclass_dict_result(self):
        """Test that multiclass expected mode returns are normalized properly."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        # Test macro averaging (should return per-class thresholds)
        optimizer_macro = ThresholdOptimizer(
            metric="f1", mode="expected", average="macro"
        )
        optimizer_macro.fit(y_true, y_prob)

        assert isinstance(optimizer_macro.threshold_, np.ndarray)
        assert len(optimizer_macro.threshold_) == 3
        assert isinstance(optimizer_macro.expected_score_, float)

        # Test micro averaging (should return single threshold)
        optimizer_micro = ThresholdOptimizer(
            metric="f1", mode="expected", average="micro"
        )
        optimizer_micro.fit(y_true, y_prob)

        # Micro mode returns a dict with single threshold
        assert isinstance(optimizer_micro.threshold_, float)
        assert isinstance(optimizer_micro.expected_score_, float)


class TestBayesUtilityMatrix:
    """Test Bayes mode with utility matrix implementation."""

    def test_bayes_utility_matrix_fit_predict(self):
        """Test that utility matrix mode works correctly."""
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])

        # Standard identity matrix (correct=1, incorrect=0)
        U = np.eye(3)

        optimizer = ThresholdOptimizer(mode="bayes", utility_matrix=U)
        # fit() should be a no-op for utility_matrix mode
        optimizer.fit(None, y_prob)  # true_labs not used for bayes mode

        assert optimizer.threshold_ == "bayes/utility_matrix"
        assert optimizer.expected_score_ is None
        assert optimizer._use_utility_matrix_in_predict is True

        # Predict should use utility matrix directly
        predictions = optimizer.predict(y_prob)
        expected = np.array([0, 1, 2])  # argmax of each row
        np.testing.assert_array_equal(predictions, expected)

    def test_bayes_utility_matrix_abstain(self):
        """Test utility matrix with abstain option."""
        y_prob = np.array(
            [
                [0.4, 0.3, 0.3],  # Uncertain case
                [0.1, 0.8, 0.1],  # Clear case
            ]
        )

        # Utility matrix with abstain option (decision 3)
        U = np.array(
            [
                [1, 0, 0],  # Predict class 0
                [0, 1, 0],  # Predict class 1
                [0, 0, 1],  # Predict class 2
                [0.5, 0.5, 0.5],  # Abstain (conservative option)
            ]
        )

        optimizer = ThresholdOptimizer(mode="bayes", utility_matrix=U)
        optimizer.fit(None, y_prob)

        predictions = optimizer.predict(y_prob)

        # First sample: uncertain, might choose abstain (decision 3)
        # Second sample: clear class 1, should choose decision 1
        assert predictions[1] == 1  # Clear class 1 case
        assert 0 <= predictions[0] <= 3  # Valid decision


class TestPredictionValidation:
    """Test robust prediction input validation."""

    def test_shape_mismatch_error(self):
        """Test that shape mismatches raise clear errors."""
        y_true = np.array([0, 1, 2])
        y_prob_train = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_prob_train)

        # Try to predict with wrong number of classes
        y_prob_wrong = np.array([[0.5, 0.5], [0.3, 0.7]])  # 2 classes instead of 3

        with pytest.raises(ValueError, match="Expected 3 classes, got 2"):
            optimizer.predict(y_prob_wrong)

    def test_binary_multiclass_mismatch(self):
        """Test error when switching between binary and multiclass."""
        # Train on binary
        y_true_binary = np.array([0, 1, 0, 1])
        y_prob_binary = np.array([0.2, 0.8, 0.3, 0.9])

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true_binary, y_prob_binary)

        # Try to predict multiclass
        y_prob_multi = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])

        with pytest.raises(ValueError, match="dimensionality does not match"):
            optimizer.predict(y_prob_multi)

    def test_probability_validation_bayes_mode(self):
        """Test that probabilities are validated for bayes mode."""
        y_prob_invalid = np.array([0.1, 1.5, 0.3, -0.1])  # Outside [0,1]

        optimizer = ThresholdOptimizer(mode="bayes", utility={"fp": -1, "fn": -5})

        with pytest.raises(ValueError, match="finite probabilities in \\[0,1\\]"):
            optimizer.fit(None, y_prob_invalid)

    def test_probability_validation_expected_mode(self):
        """Test that probabilities are validated for expected mode."""
        y_true = np.array([0, 1, 0, 1])
        y_prob_invalid = np.array([0.1, 1.5, 0.3, -0.1])  # Outside [0,1]

        optimizer = ThresholdOptimizer(mode="expected")

        with pytest.raises(ValueError, match="finite probabilities in \\[0,1\\]"):
            optimizer.fit(y_true, y_prob_invalid)


class TestVerboseOutput:
    """Test that verbose parameter produces output."""

    def test_verbose_output(self, capsys):
        """Test that verbose=True prints optimization path."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])

        optimizer = ThresholdOptimizer(metric="f1", verbose=True)
        optimizer.fit(y_true, y_prob)

        captured = capsys.readouterr()
        assert "[ThresholdOptimizer]" in captured.out
        assert "simple binary optimizer" in captured.out

    def test_verbose_general_path(self, capsys):
        """Test verbose output for general optimizer path."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        optimizer = ThresholdOptimizer(metric="f1", verbose=True)
        optimizer.fit(y_true, y_prob)

        captured = capsys.readouterr()
        assert "[ThresholdOptimizer]" in captured.out
        assert "general optimizer" in captured.out


class TestSklearnCompatibility:
    """Test sklearn-compatible parameter interface."""

    def test_get_params(self):
        """Test get_params method."""
        optimizer = ThresholdOptimizer(
            metric="f1", method="sort_scan", average="micro", beta=2.0
        )

        params = optimizer.get_params()

        assert params["metric"] == "f1"
        assert params["method"] == "sort_scan"
        assert params["average"] == "micro"
        assert params["beta"] == 2.0
        assert "verbose" in params
        assert "mode" in params

    def test_set_params(self):
        """Test set_params method."""
        optimizer = ThresholdOptimizer(metric="accuracy")

        # Change parameters
        optimizer.set_params(metric="f1", method="minimize", average="weighted")

        assert optimizer.metric == "f1"
        assert optimizer.method == "minimize"
        assert optimizer.average == "weighted"

    def test_set_invalid_param(self):
        """Test that setting invalid parameter raises error."""
        optimizer = ThresholdOptimizer()

        with pytest.raises(ValueError, match="Invalid parameter 'invalid_param'"):
            optimizer.set_params(invalid_param="value")

    def test_utility_matrix_flag_update(self):
        """Test that utility matrix flag updates correctly in set_params."""
        optimizer = ThresholdOptimizer()
        assert optimizer._use_utility_matrix_in_predict is False

        U = np.eye(3)
        optimizer.set_params(mode="bayes", utility_matrix=U)
        assert optimizer._use_utility_matrix_in_predict is True

        optimizer.set_params(mode="empirical")
        assert optimizer._use_utility_matrix_in_predict is False


class TestThresholdTypes:
    """Test handling of different threshold types."""

    def test_scalar_threshold_multiclass(self):
        """Test multiclass prediction with scalar threshold (micro averaging)."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        optimizer = ThresholdOptimizer(metric="f1", average="micro")
        optimizer.fit(y_true, y_prob)

        # Should handle scalar threshold correctly
        predictions = optimizer.predict(y_prob)
        assert len(predictions) == len(y_true)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_array_threshold_multiclass(self):
        """Test multiclass prediction with per-class thresholds."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        optimizer = ThresholdOptimizer(metric="f1", average="macro")
        optimizer.fit(y_true, y_prob)

        # Should handle per-class thresholds correctly
        predictions = optimizer.predict(y_prob)
        assert len(predictions) == len(y_true)
        assert all(pred in [0, 1, 2] for pred in predictions)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_unfitted_estimator(self):
        """Test error when predicting before fitting."""
        optimizer = ThresholdOptimizer()
        y_prob = np.array([0.2, 0.8, 0.3])

        with pytest.raises(RuntimeError, match="has not been fitted"):
            optimizer.predict(y_prob)

    def test_wrong_threshold_shape(self):
        """Test error for wrong threshold array shape."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_prob)

        # Manually corrupt threshold shape to test validation
        optimizer.threshold_ = np.array([0.5, 0.5])  # Wrong shape for 3 classes

        with pytest.raises(ValueError, match="Per-class threshold shape must be"):
            optimizer.predict(y_prob)

    def test_comparison_operators(self):
        """Test both comparison operators work correctly."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])

        # Test exclusive comparison
        optimizer_gt = ThresholdOptimizer(metric="f1", comparison=">")
        optimizer_gt.fit(y_true, y_prob)
        pred_gt = optimizer_gt.predict(y_prob)

        # Test inclusive comparison
        optimizer_gte = ThresholdOptimizer(metric="f1", comparison=">=")
        optimizer_gte.fit(y_true, y_prob)
        pred_gte = optimizer_gte.predict(y_prob)

        # Both should return valid predictions
        assert all(pred in [0, 1] for pred in pred_gt)
        assert all(pred in [0, 1] for pred in pred_gte)
