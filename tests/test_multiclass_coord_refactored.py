"""Tests for next-generation multiclass coordinate ascent optimization."""

import json
import pickle
import time
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from optimal_cutoffs.optimize import (
    _assign_labels_shifted,
    coordinate_ascent_kernel,
    get_performance_info,
)
from optimal_cutoffs.numba_utils import NUMBA_AVAILABLE


class TestThresholdSolution:
    """Test the immutable ThresholdSolution dataclass."""

    def test_threshold_solution_creation(self):
        """Test basic creation and properties."""
        thresholds = np.array([0.1, 0.2, 0.3])
        history = np.array([0.5, 0.6, 0.7])
        
        solution = ThresholdSolution(
            thresholds=thresholds,
            score=0.7,
            converged=True,
            iterations=3,
            history=history
        )
        
        assert np.array_equal(solution.thresholds, thresholds)
        assert solution.score == 0.7
        assert solution.converged
        assert solution.iterations == 3
        assert np.array_equal(solution.history, history)

    def test_per_class_thresholds_property(self):
        """Test the cached per_class_thresholds property."""
        thresholds = np.array([0.1, 0.2, 0.3])
        solution = ThresholdSolution(
            thresholds=thresholds,
            score=0.7,
            converged=True,
            iterations=3,
            history=np.array([0.7])
        )
        
        expected = {0: 0.1, 1: 0.2, 2: 0.3}
        assert solution.per_class_thresholds == expected

    def test_predict_method(self):
        """Test prediction with thresholds."""
        thresholds = np.array([0.0, 0.1, 0.2])
        solution = ThresholdSolution(
            thresholds=thresholds,
            score=0.7,
            converged=True,
            iterations=3,
            history=np.array([0.7])
        )
        
        # Test probabilities
        probabilities = np.array([
            [0.8, 0.1, 0.1],  # Class 0 (0.8 > 0.0 - 0.1 and > 0.1 - 0.2)
            [0.1, 0.8, 0.1],  # Class 1 (0.8 - 0.1 = 0.7 > others)
            [0.1, 0.1, 0.8],  # Class 2 (0.8 - 0.2 = 0.6 > others)
        ])
        
        predictions = solution.predict(probabilities)
        assert predictions.shape == (3,)
        assert all(0 <= p <= 2 for p in predictions)

    def test_predict_validation(self):
        """Test prediction input validation."""
        solution = ThresholdSolution(
            thresholds=np.array([0.1, 0.2]),
            score=0.7,
            converged=True,
            iterations=3,
            history=np.array([0.7])
        )
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D probabilities"):
            solution.predict(np.array([0.5, 0.3]))
        
        # Wrong number of classes
        with pytest.raises(ValueError, match="has 3 classes.*2 thresholds"):
            solution.predict(np.array([[0.3, 0.3, 0.4]]))

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        thresholds = np.array([0.1, 0.2, 0.3])
        history = np.array([0.5, 0.6, 0.7])
        metadata = {'method': 'test', 'n_samples': 100}
        
        solution = ThresholdSolution(
            thresholds=thresholds,
            score=0.7,
            converged=True,
            iterations=3,
            history=history,
            metadata=metadata
        )
        
        # Test to_json
        json_data = solution.to_json()
        assert json_data['score'] == 0.7
        assert json_data['converged'] is True
        assert json_data['thresholds'] == [0.1, 0.2, 0.3]
        assert json_data['metadata'] == metadata
        
        # Test from_json
        reconstructed = ThresholdSolution.from_json(json_data)
        assert np.array_equal(reconstructed.thresholds, solution.thresholds)
        assert reconstructed.score == solution.score
        assert reconstructed.converged == solution.converged
        assert np.array_equal(reconstructed.history, solution.history)
        assert reconstructed.metadata == solution.metadata

    def test_immutability(self):
        """Test that ThresholdSolution is truly immutable."""
        solution = ThresholdSolution(
            thresholds=np.array([0.1, 0.2]),
            score=0.7,
            converged=True,
            iterations=3,
            history=np.array([0.7])
        )
        
        # Should not be able to modify attributes
        with pytest.raises((AttributeError, TypeError)):
            solution.score = 0.8


class TestCoordinateAscentKernel:
    """Test the core Numba-optimized coordinate ascent kernel."""

    def test_basic_functionality(self):
        """Test basic coordinate ascent functionality."""
        # Simple 3-class problem
        np.random.seed(42)
        n_samples, n_classes = 100, 3
        
        # Create simple test data
        y_true = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        probs = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float64)
        
        thresholds, score, history = coordinate_ascent_kernel(
            y_true, probs, max_iter=10, tol=1e-12
        )
        
        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == n_classes
        assert isinstance(score, (float, np.floating))
        assert isinstance(history, np.ndarray)
        assert len(history) > 0
        assert 0.0 <= score <= 1.0

    def test_convergence_behavior(self):
        """Test that the algorithm converges and improves."""
        np.random.seed(42)
        n_samples, n_classes = 200, 4
        
        y_true = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        probs = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float64)
        
        thresholds, score, history = coordinate_ascent_kernel(
            y_true, probs, max_iter=20, tol=1e-10
        )
        
        # Score should improve or stay the same
        for i in range(1, len(history)):
            assert history[i] >= history[i-1] - 1e-10, "Score should not decrease"
        
        # Final score should be the best
        assert abs(score - history[-1]) < 1e-10

    def test_perfect_classification_case(self):
        """Test behavior when perfect classification is possible."""
        # Create perfect case where each sample has probability 1.0 for correct class
        n_samples, n_classes = 60, 3
        samples_per_class = n_samples // n_classes
        
        y_true = np.repeat(np.arange(n_classes), samples_per_class).astype(np.int32)
        probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        
        for i in range(n_samples):
            probs[i, y_true[i]] = 0.99
            # Add small noise to other classes
            for j in range(n_classes):
                if j != y_true[i]:
                    probs[i, j] = 0.01 / (n_classes - 1)
        
        thresholds, score, history = coordinate_ascent_kernel(
            y_true, probs, max_iter=10, tol=1e-12
        )
        
        # Should achieve very high score
        assert score > 0.95, f"Expected high score for perfect case, got {score}"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimum case: 2 classes, 2 samples
        y_true = np.array([0, 1], dtype=np.int32)
        probs = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        
        thresholds, score, history = coordinate_ascent_kernel(
            y_true, probs, max_iter=5, tol=1e-12
        )
        
        assert len(thresholds) == 2
        assert 0.0 <= score <= 1.0
        assert len(history) > 0

    def test_deterministic_behavior(self):
        """Test that the algorithm is deterministic."""
        np.random.seed(12345)
        n_samples, n_classes = 100, 3
        
        y_true = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        probs = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float64)
        
        # Run twice with same inputs
        result1 = coordinate_ascent_kernel(y_true, probs, max_iter=10, tol=1e-12)
        result2 = coordinate_ascent_kernel(y_true, probs, max_iter=10, tol=1e-12)
        
        # Should get identical results
        np.testing.assert_array_almost_equal(result1[0], result2[0])
        assert abs(result1[1] - result2[1]) < 1e-10
        np.testing.assert_array_almost_equal(result1[2], result2[2])


class TestCoordinateAscentOptimizer:
    """Test the scikit-learn compatible CoordinateAscentOptimizer."""

    def test_basic_fit_predict_workflow(self):
        """Test basic fit/predict workflow."""
        # Generate test data
        X, y = make_classification(
            n_samples=200, n_classes=3, n_features=10, 
            n_informative=8, random_state=42
        )
        
        # Simulate probability predictions (from a pre-trained classifier)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        # Fit threshold optimizer
        optimizer = CoordinateAscentOptimizer(max_iter=15, verbose=1)
        optimizer.fit(probs, y)
        
        # Check fitted attributes
        assert hasattr(optimizer, 'solution_')
        assert isinstance(optimizer.solution_, ThresholdSolution)
        assert len(optimizer.solution_.thresholds) == 3
        
        # Test predictions
        predictions = optimizer.predict(probs)
        assert len(predictions) == len(y)
        assert all(0 <= p <= 2 for p in predictions)
        
        # Test scoring
        score = optimizer.score(probs, y)
        assert 0.0 <= score <= 1.0

    def test_input_validation(self):
        """Test input validation."""
        optimizer = CoordinateAscentOptimizer()
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="X must be 2D"):
            optimizer.fit(np.array([0.5, 0.3]), np.array([0, 1]))
        
        with pytest.raises(ValueError, match="y must be 1D"):
            optimizer.fit(np.array([[0.5, 0.3]]), np.array([[0, 1]]))
        
        # Length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            optimizer.fit(np.array([[0.5, 0.3]]), np.array([0, 1]))
        
        # Too few classes
        with pytest.raises(ValueError, match="Need at least 2 classes"):
            optimizer.fit(np.array([[0.5]]), np.array([0]))
        
        # Invalid labels
        with pytest.raises(ValueError, match="consecutive integers from 0"):
            optimizer.fit(np.array([[0.5, 0.3, 0.2]]), np.array([1, 2]))  # Missing 0
        
        with pytest.raises(ValueError, match="Label.*>= n_classes"):
            optimizer.fit(np.array([[0.5, 0.3]]), np.array([0, 2]))  # Label 2 >= 2 classes

    def test_predict_without_fitting(self):
        """Test that predict raises error when not fitted."""
        optimizer = CoordinateAscentOptimizer()
        with pytest.raises(ValueError, match="must be fitted"):
            optimizer.predict(np.array([[0.5, 0.3, 0.2]]))

    def test_sklearn_compatibility(self):
        """Test compatibility with scikit-learn utilities."""
        # Generate test data
        X, y = make_classification(
            n_samples=150, n_classes=3, n_features=8, 
            n_informative=6, random_state=42
        )
        
        # Get probabilities
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        # Test with Pipeline
        pipeline = Pipeline([
            ('optimizer', CoordinateAscentOptimizer(max_iter=10))
        ])
        
        pipeline.fit(probs, y)
        predictions = pipeline.predict(probs)
        score = pipeline.score(probs, y)
        
        assert len(predictions) == len(y)
        assert 0.0 <= score <= 1.0
        
        # Test with GridSearchCV
        param_grid = {'max_iter': [5, 10], 'tol': [1e-10, 1e-8]}
        grid_search = GridSearchCV(
            CoordinateAscentOptimizer(), param_grid, cv=3, scoring='f1_macro'
        )
        
        grid_search.fit(probs, y)
        assert hasattr(grid_search, 'best_params_')
        assert hasattr(grid_search, 'best_score_')

    def test_serialization(self):
        """Test that fitted optimizers can be serialized."""
        # Generate test data
        X, y = make_classification(
            n_samples=100, n_classes=3, n_features=5, random_state=42
        )
        
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        # Fit optimizer
        optimizer = CoordinateAscentOptimizer(max_iter=10)
        optimizer.fit(probs, y)
        
        # Test pickle serialization
        serialized = pickle.dumps(optimizer)
        deserialized = pickle.loads(serialized)
        
        # Should produce same predictions
        pred1 = optimizer.predict(probs)
        pred2 = deserialized.predict(probs)
        np.testing.assert_array_equal(pred1, pred2)


class TestAdaptiveThresholdOptimizer:
    """Test the adaptive optimizer with auto-tuning."""

    def test_without_skopt(self):
        """Test behavior when scikit-optimize is not available."""
        with patch('optimal_cutoffs.multiclass_coord.gp_minimize', side_effect=ImportError):
            X, y = make_classification(n_samples=100, n_classes=3, random_state=42)
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X, y)
            probs = clf.predict_proba(X)
            
            optimizer = AdaptiveCoordinateAscentOptimizer(auto_tune=True, verbose=1)
            optimizer.fit(probs, y)  # Should work without crashing
            
            assert hasattr(optimizer, 'solution_')

    @pytest.mark.skipif(True, reason="scikit-optimize not available in test environment")
    def test_with_skopt(self):
        """Test adaptive tuning when scikit-optimize is available."""
        # This test would run if scikit-optimize was available
        X, y = make_classification(n_samples=100, n_classes=3, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        optimizer = AdaptiveCoordinateAscentOptimizer(
            auto_tune=True, tuning_calls=5, verbose=1
        )
        optimizer.fit(probs, y)
        
        assert hasattr(optimizer, 'solution_')
        # The tolerance should have been auto-tuned
        assert hasattr(optimizer, 'tol')

    def test_no_auto_tune(self):
        """Test that auto_tune=False works normally."""
        X, y = make_classification(n_samples=100, n_classes=3, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        optimizer = AdaptiveCoordinateAscentOptimizer(auto_tune=False)
        optimizer.fit(probs, y)
        
        assert hasattr(optimizer, 'solution_')


class TestOnlineThresholdOptimizer:
    """Test the online learning optimizer."""

    def test_basic_online_workflow(self):
        """Test basic online learning workflow."""
        n_classes = 3
        optimizer = OnlineCoordinateAscentOptimizer(n_classes=n_classes)
        
        # Initial state
        assert len(optimizer.thresholds) == n_classes
        assert optimizer.n_updates == 0
        
        # Simulate streaming data
        np.random.seed(42)
        for batch in range(5):
            # Generate batch
            batch_size = 20
            X_batch = np.random.dirichlet([1] * n_classes, batch_size)
            y_batch = np.random.randint(0, n_classes, batch_size)
            
            # Update
            optimizer.partial_fit(X_batch, y_batch)
            
            # Test predictions
            predictions = optimizer.predict(X_batch)
            assert len(predictions) == batch_size
            assert all(0 <= p < n_classes for p in predictions)
        
        # Should have made updates
        assert optimizer.n_updates == 5

    def test_online_convergence(self):
        """Test that online optimizer can improve over time."""
        n_classes = 3
        n_samples = 1000
        
        # Create data where each class has distinct probability pattern
        np.random.seed(42)
        X = np.zeros((n_samples, n_classes))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            true_class = i % n_classes
            y[i] = true_class
            # Give higher probability to true class
            X[i, true_class] = 0.7
            for j in range(n_classes):
                if j != true_class:
                    X[i, j] = 0.3 / (n_classes - 1)
        
        optimizer = OnlineCoordinateAscentOptimizer(
            n_classes=n_classes, learning_rate=0.1
        )
        
        # Track initial performance
        initial_predictions = optimizer.predict(X[:100])
        initial_f1 = f1_score(y[:100], initial_predictions, average='macro')
        
        # Train on batches
        batch_size = 50
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            optimizer.partial_fit(X[start:end], y[start:end])
        
        # Test final performance
        final_predictions = optimizer.predict(X[:100])
        final_f1 = f1_score(y[:100], final_predictions, average='macro')
        
        # Performance should improve (or at least not get much worse)
        assert final_f1 >= initial_f1 - 0.1

    def test_get_params(self):
        """Test parameter retrieval."""
        optimizer = OnlineCoordinateAscentOptimizer(
            n_classes=4, learning_rate=0.05, momentum=0.8
        )
        
        params = optimizer.get_params()
        assert 'thresholds' in params
        assert 'n_updates' in params
        assert 'learning_rate' in params
        assert 'momentum' in params
        assert params['learning_rate'] == 0.05
        assert params['momentum'] == 0.8
        assert len(params['thresholds']) == 4


class TestFunctionalAPI:
    """Test the functional API."""

    def test_optimize_thresholds_fast(self):
        """Test fast optimization method."""
        # Generate test data
        X, y = make_classification(n_samples=100, n_classes=3, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        solution = optimize_thresholds(probs, y, method='fast', max_iter=10)
        
        assert isinstance(solution, ThresholdSolution)
        assert len(solution.thresholds) == 3
        assert 0.0 <= solution.score <= 1.0
        assert solution.metadata['method'] == 'fast'

    def test_optimize_thresholds_adaptive(self):
        """Test adaptive optimization method."""
        X, y = make_classification(n_samples=100, n_classes=3, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        
        solution = optimize_thresholds(probs, y, method='adaptive', auto_tune=False)
        
        assert isinstance(solution, ThresholdSolution)
        assert len(solution.thresholds) == 3

    def test_optimize_thresholds_invalid_method(self):
        """Test error for invalid method."""
        probs = np.array([[0.5, 0.3, 0.2]])
        labels = np.array([0])
        
        with pytest.raises(ValueError, match="Unknown method"):
            optimize_thresholds(probs, labels, method='invalid')

    def test_optimize_thresholds_online_error(self):
        """Test that online method raises appropriate error."""
        probs = np.array([[0.5, 0.3, 0.2]])
        labels = np.array([0])
        
        with pytest.raises(ValueError, match="Use OnlineCoordinateAscentOptimizer directly"):
            optimize_thresholds(probs, labels, method='online')


class TestPerformanceAndCompatibility:
    """Test performance characteristics and compatibility."""

    def test_performance_info(self):
        """Test performance information reporting."""
        info = get_performance_info()
        
        assert 'numba_available' in info
        assert 'expected_speedup' in info
        assert 'parallel_processing' in info
        assert 'fastmath_enabled' in info
        
        if NUMBA_AVAILABLE:
            assert '10-100x' in info['expected_speedup']
        else:
            assert '1x' in info['expected_speedup']

    def test_legacy_compatibility(self):
        """Test legacy compatibility functions."""
        P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        tau = np.array([0.0, 0.1, 0.2])
        
        predictions = _assign_labels_shifted(P, tau)
        assert len(predictions) == 2
        assert all(0 <= p <= 2 for p in predictions)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_performance_benchmark(self):
        """Benchmark Numba vs Python performance (when available)."""
        # Generate larger dataset for meaningful benchmark
        n_samples, n_classes = 1000, 5
        np.random.seed(42)
        
        y_true = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        probs = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float64)
        
        # Time the Numba version
        start_time = time.time()
        thresholds_numba, score_numba, history_numba = coordinate_ascent_kernel(
            y_true, probs, max_iter=10, tol=1e-12
        )
        numba_time = time.time() - start_time
        
        # For comparison, if we had a pure Python version, we could time it here
        # This test mainly verifies that Numba version runs without errors
        assert len(thresholds_numba) == n_classes
        assert 0.0 <= score_numba <= 1.0
        assert len(history_numba) > 0
        
        print(f"Numba optimization time: {numba_time:.4f} seconds")


class TestRobustness:
    """Test robustness and edge cases."""

    def test_extreme_imbalanced_data(self):
        """Test with extremely imbalanced data."""
        n_samples = 300
        n_classes = 3
        
        # Create highly imbalanced dataset
        y = np.zeros(n_samples, dtype=np.int32)
        y[:10] = 1  # Only 10 samples of class 1
        y[:2] = 2   # Only 2 samples of class 2
        # Rest are class 0
        
        # Create probabilities that reflect the imbalance
        probs = np.random.dirichlet([10, 1, 0.1], n_samples).astype(np.float64)
        
        optimizer = CoordinateAscentOptimizer(max_iter=15)
        optimizer.fit(probs, y)
        
        # Should complete without errors
        assert hasattr(optimizer, 'solution_')
        predictions = optimizer.predict(probs)
        assert len(predictions) == n_samples

    def test_nearly_perfect_probabilities(self):
        """Test with nearly deterministic probabilities."""
        n_samples = 120
        n_classes = 3
        samples_per_class = n_samples // n_classes
        
        y = np.repeat(np.arange(n_classes), samples_per_class).astype(np.int32)
        probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        
        # Make probabilities very close to 1.0 for correct class
        for i in range(n_samples):
            probs[i, y[i]] = 0.999
            for j in range(n_classes):
                if j != y[i]:
                    probs[i, j] = 0.001 / (n_classes - 1)
        
        optimizer = CoordinateAscentOptimizer(max_iter=10)
        optimizer.fit(probs, y)
        
        # Should achieve very high score
        assert optimizer.solution_.score > 0.95

    def test_tied_probabilities(self):
        """Test behavior with tied probabilities."""
        # Create case where many samples have tied probabilities
        n_samples = 100
        n_classes = 3
        
        y = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        probs = np.full((n_samples, n_classes), 1.0 / n_classes, dtype=np.float64)
        
        # Add small random noise to break ties slightly
        noise = np.random.normal(0, 0.01, (n_samples, n_classes))
        probs += noise
        probs = np.abs(probs)  # Ensure positive
        
        # Renormalize to valid probabilities
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        optimizer = CoordinateAscentOptimizer(max_iter=10)
        optimizer.fit(probs, y)
        
        # Should complete without errors
        assert hasattr(optimizer, 'solution_')

    def test_numerical_precision(self):
        """Test numerical precision and stability."""
        # Test with very small probability differences
        n_samples = 50
        n_classes = 3
        
        y = np.random.randint(0, n_classes, n_samples).astype(np.int32)
        
        # Create probabilities with very small differences (numerical precision test)
        base_prob = 1.0 / n_classes
        probs = np.full((n_samples, n_classes), base_prob, dtype=np.float64)
        
        # Add tiny differences
        for i in range(n_samples):
            probs[i, y[i]] += 1e-10
            probs[i] = probs[i] / probs[i].sum()  # Renormalize
        
        optimizer = CoordinateAscentOptimizer(max_iter=10, tol=1e-15)
        optimizer.fit(probs, y)
        
        # Should handle numerical precision gracefully
        assert hasattr(optimizer, 'solution_')
        assert np.all(np.isfinite(optimizer.solution_.thresholds))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])