"""Tests for coordinate-ascent multiclass threshold optimization."""

import numpy as np
import pytest

from optimal_cutoffs import ThresholdOptimizer, get_optimal_multiclass_thresholds
from optimal_cutoffs.metrics import get_vectorized_metric
from optimal_cutoffs.multiclass_coord import (
    _assign_labels_shifted,
    _macro_f1_from_assignments,
    optimal_multiclass_thresholds_coord_ascent,
)
from optimal_cutoffs.piecewise import optimal_threshold_sortscan


class TestCoordinateAscentCore:
    """Test core coordinate ascent functionality."""

    def test_assign_labels_shifted(self):
        """Test label assignment using argmax(P - tau)."""
        # Simple 3-class example
        P = np.array(
            [
                [0.7, 0.2, 0.1],  # Sample 0
                [0.1, 0.8, 0.1],  # Sample 1
                [0.2, 0.1, 0.7],  # Sample 2
            ]
        )

        # Zero thresholds -> standard argmax
        tau = np.array([0.0, 0.0, 0.0])
        y_pred = _assign_labels_shifted(P, tau)
        np.testing.assert_array_equal(y_pred, [0, 1, 2])

        # Shift class 0 threshold up -> fewer class 0 predictions
        tau = np.array([0.5, 0.0, 0.0])
        y_pred = _assign_labels_shifted(P, tau)
        # Sample 0: [0.7-0.5, 0.2-0.0, 0.1-0.0] = [0.2, 0.2, 0.1] -> argmax = 0 or 1 (tie)
        # Since argmax picks first in ties, still class 0
        assert y_pred[0] in [0, 1]  # Allow for tie-breaking
        assert y_pred[1] == 1  # Class 1 clearly best
        assert y_pred[2] == 2  # Class 2 clearly best

    def test_macro_f1_from_assignments(self):
        """Test macro-F1 computation from class assignments."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1])
        n_classes = 3

        macro_f1 = _macro_f1_from_assignments(y_true, y_pred, n_classes)

        # Manual calculation:
        # Class 0: TP=1, FP=1, FN=1 -> F1 = 2*1/(2*1+1+1) = 2/4 = 0.5
        # Class 1: TP=1, FP=2, FN=1 -> F1 = 2*1/(2*1+2+1) = 2/5 = 0.4
        # Class 2: TP=1, FP=0, FN=1 -> F1 = 2*1/(2*1+0+1) = 2/3 ≈ 0.667
        # Macro-F1 = (0.5 + 0.4 + 0.667) / 3 ≈ 0.522

        assert 0.52 <= macro_f1 <= 0.53

    def test_coord_ascent_monotone_macro_f1(self):
        """Test that coordinate ascent produces monotone increasing macro-F1."""
        rng = np.random.default_rng(42)
        n, C = 300, 4
        P = rng.dirichlet(alpha=np.ones(C), size=n)  # Valid probabilities per row
        y_true = rng.integers(0, C, size=n)

        f1_metric = get_vectorized_metric("f1")
        tau, best_macro, history = optimal_multiclass_thresholds_coord_ascent(
            y_true,
            P,
            sortscan_metric_fn=f1_metric,
            sortscan_kernel=optimal_threshold_sortscan,
            max_iter=10,
        )

        # Check monotone ascent in history
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-12, (
                f"Non-monotone at step {i}: {history[i - 1]} -> {history[i]}"
            )

        # Verify final result
        assert len(tau) == C
        assert 0.0 <= best_macro <= 1.0
        assert best_macro == history[-1]

    def test_coord_ascent_vs_ovr_baseline(self):
        """Test that coordinate ascent improves over One-vs-Rest baseline."""
        rng = np.random.default_rng(42)
        n, C = 200, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        f1_metric = get_vectorized_metric("f1")

        # Compare coordinate ascent to OvR baseline
        tau0 = np.zeros(C)  # Initialize OvR thresholds
        for c in range(C):
            y_c = (y_true == c).astype(int)
            t, _, _ = optimal_threshold_sortscan(y_c, P[:, c], f1_metric)
            tau0[c] = t

        y_pred0 = _assign_labels_shifted(P, tau0)
        macro0 = _macro_f1_from_assignments(y_true, y_pred0, C)

        # Coordinate ascent
        tau, best_macro, _ = optimal_multiclass_thresholds_coord_ascent(
            y_true,
            P,
            sortscan_metric_fn=f1_metric,
            sortscan_kernel=optimal_threshold_sortscan,
            max_iter=10,
            init="ovr_sortscan",
        )

        # Coordinate ascent should be >= OvR baseline
        assert best_macro >= macro0 - 1e-12

    def test_coord_ascent_edge_cases(self):
        """Test coordinate ascent with edge cases."""
        f1_metric = get_vectorized_metric("f1")

        # Very simple case: 2 samples, 2 classes
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        y_true = np.array([0, 1])

        tau, best_macro, history = optimal_multiclass_thresholds_coord_ascent(
            y_true,
            P,
            sortscan_metric_fn=f1_metric,
            sortscan_kernel=optimal_threshold_sortscan,
            max_iter=5,
        )

        assert len(tau) == 2
        assert 0.0 <= best_macro <= 1.0
        assert len(history) >= 1

    def test_coord_ascent_initialization_strategies(self):
        """Test different initialization strategies."""
        rng = np.random.default_rng(123)
        n, C = 100, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        f1_metric = get_vectorized_metric("f1")

        # Test different initialization methods
        for init in ["ovr_sortscan", "zeros"]:
            tau, best_macro, _ = optimal_multiclass_thresholds_coord_ascent(
                y_true,
                P,
                sortscan_metric_fn=f1_metric,
                sortscan_kernel=optimal_threshold_sortscan,
                max_iter=5,
                init=init,
            )
            assert len(tau) == C
            assert 0.0 <= best_macro <= 1.0

        # Test invalid initialization
        with pytest.raises(ValueError, match="Unknown init"):
            optimal_multiclass_thresholds_coord_ascent(
                y_true,
                P,
                sortscan_metric_fn=f1_metric,
                sortscan_kernel=optimal_threshold_sortscan,
                init="invalid_init",
            )


class TestCoordinateAscentIntegration:
    """Test integration with main API."""

    def test_get_optimal_multiclass_thresholds_coord_ascent(self):
        """Test coordinate ascent through main API."""
        rng = np.random.default_rng(42)
        n, C = 150, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Test through main API
        tau = get_optimal_multiclass_thresholds(
            y_true, P, metric="f1", method="coord_ascent"
        )

        assert len(tau) == C
        assert all(isinstance(t, (float, np.floating)) for t in tau)

    def test_coord_ascent_unsupported_features(self):
        """Test that unsupported features raise appropriate errors."""
        rng = np.random.default_rng(42)
        n, C = 50, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Sample weights not supported
        with pytest.raises(NotImplementedError, match="sample weights"):
            get_optimal_multiclass_thresholds(
                y_true, P, metric="f1", method="coord_ascent", sample_weight=np.ones(n)
            )

        # Only '>' comparison supported
        with pytest.raises(NotImplementedError, match="comparison"):
            get_optimal_multiclass_thresholds(
                y_true, P, metric="f1", method="coord_ascent", comparison=">="
            )

        # Only F1 metric supported currently
        with pytest.raises(NotImplementedError, match="F1 metric"):
            get_optimal_multiclass_thresholds(
                y_true, P, metric="accuracy", method="coord_ascent"
            )

    def test_threshold_optimizer_coord_ascent(self):
        """Test ThresholdOptimizer wrapper with coordinate ascent."""
        rng = np.random.default_rng(42)
        n, C = 100, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Test through wrapper
        optimizer = ThresholdOptimizer(objective="f1", method="coord_ascent")
        optimizer.fit(y_true, P)

        # Check that thresholds were learned
        assert optimizer.threshold_ is not None
        assert len(optimizer.threshold_) == C
        assert optimizer.is_multiclass_

        # Test prediction
        y_pred = optimizer.predict(P)
        assert len(y_pred) == n
        assert all(0 <= pred < C for pred in y_pred)

    def test_coord_ascent_prediction_consistency(self):
        """Test that coord_ascent predictions match argmax(P - tau)."""
        rng = np.random.default_rng(42)
        n, C = 80, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Get thresholds via coordinate ascent
        tau = get_optimal_multiclass_thresholds(
            y_true, P, metric="f1", method="coord_ascent"
        )

        # Manual prediction using argmax(P - tau)
        y_pred_manual = _assign_labels_shifted(P, tau)

        # Prediction via wrapper
        optimizer = ThresholdOptimizer(objective="f1", method="coord_ascent")
        optimizer.fit(y_true, P)
        y_pred_wrapper = optimizer.predict(P)

        # Should be identical
        np.testing.assert_array_equal(y_pred_manual, y_pred_wrapper)


class TestCoordinateAscentPerformance:
    """Test performance characteristics of coordinate ascent."""

    def test_coord_ascent_imbalanced_data(self):
        """Test coordinate ascent performance on imbalanced datasets."""
        rng = np.random.default_rng(42)
        n = 300

        # Create imbalanced dataset (class 0 dominant)
        class_probs = [0.7, 0.2, 0.1]  # Imbalanced
        y_true = rng.choice(3, size=n, p=class_probs)

        # Generate probabilities with some correlation to true labels
        P = rng.dirichlet(alpha=np.ones(3), size=n)
        # Add bias toward correct class
        for i in range(n):
            P[i, y_true[i]] += 0.3
            P[i] /= P[i].sum()  # Renormalize

        # Compare OvR vs coordinate ascent
        tau_ovr = get_optimal_multiclass_thresholds(
            y_true, P, metric="f1", method="smart_brute"
        )
        y_pred_ovr = _assign_labels_shifted(P, tau_ovr)
        macro_f1_ovr = _macro_f1_from_assignments(y_true, y_pred_ovr, 3)

        tau_coord = get_optimal_multiclass_thresholds(
            y_true, P, metric="f1", method="coord_ascent"
        )
        y_pred_coord = _assign_labels_shifted(P, tau_coord)
        macro_f1_coord = _macro_f1_from_assignments(y_true, y_pred_coord, 3)

        # Coordinate ascent should perform at least as well as OvR
        assert macro_f1_coord >= macro_f1_ovr - 1e-10

        # On imbalanced data, coordinate ascent often improves significantly
        print(f"OvR macro-F1: {macro_f1_ovr:.4f}")
        print(f"Coord-ascent macro-F1: {macro_f1_coord:.4f}")
        print(f"Improvement: {macro_f1_coord - macro_f1_ovr:.4f}")

    def test_coord_ascent_convergence(self):
        """Test convergence properties of coordinate ascent."""
        rng = np.random.default_rng(42)
        n, C = 200, 4
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        f1_metric = get_vectorized_metric("f1")

        # Test with different stopping tolerances
        for tol_stops in [1, 2, 3]:
            tau, best_macro, history = optimal_multiclass_thresholds_coord_ascent(
                y_true,
                P,
                sortscan_metric_fn=f1_metric,
                sortscan_kernel=optimal_threshold_sortscan,
                max_iter=20,
                tol_stops=tol_stops,
            )

            # Should terminate before max_iter due to convergence
            assert len(history) <= 20
            assert len(tau) == C
            assert 0.0 <= best_macro <= 1.0
