"""Basic example demonstrating the optimal classification cutoff utilities."""

import numpy as np

from optimal_cutoffs import (
    confusion_matrix_at_threshold,
    get_optimal_threshold,
)

# Simulated data
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6, 0.3, 0.5])

# Compute optimal threshold for accuracy
result = get_optimal_threshold(y_true, y_prob, metric="accuracy")
best_threshold = result.threshold  # Extract the threshold value
print(f"Optimal threshold (accuracy): {best_threshold:.2f}")

tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, best_threshold)
print(f"Confusion matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Using the result object's predict method (recommended)
predictions = result.predict(y_prob)
print("Predictions:", predictions)
