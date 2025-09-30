"""Basic example demonstrating the optimal classification cutoff utilities."""

import numpy as np

from optimal_cutoffs import (
    ThresholdOptimizer,
    get_confusion_matrix,
    get_optimal_threshold,
)

# Simulated data
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6, 0.3, 0.5])

# Compute optimal threshold for accuracy
best_threshold = get_optimal_threshold(y_true, y_prob, metric="accuracy")
print(f"Optimal threshold (accuracy): {best_threshold:.2f}")

tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, best_threshold)
print(f"Confusion matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Using ThresholdOptimizer class
optimizer = ThresholdOptimizer(metric="accuracy")
optimizer.fit(y_true, y_prob)
pred_labels = optimizer.predict(y_prob)
print("Predictions:", pred_labels.astype(int))
