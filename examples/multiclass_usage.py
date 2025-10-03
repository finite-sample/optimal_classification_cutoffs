"""Example demonstrating multiclass classification threshold optimization."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from optimal_cutoffs import (
    get_multiclass_confusion_matrix,
    get_optimal_multiclass_thresholds,
    get_optimal_threshold,
    multiclass_metric,
)

# Generate synthetic multiclass data
print("=== Multiclass Classification Threshold Optimization ===\n")

X, y = make_classification(
    n_samples=300,
    n_features=10,
    n_classes=3,
    n_informative=8,
    n_redundant=2,
    random_state=42,
)

# Train a logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)

# Get predicted probabilities (shape: n_samples x n_classes)
y_prob = model.predict_proba(X)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Probability matrix shape: {y_prob.shape}\n")

# Method 1: Using get_optimal_threshold for multiclass
print("=== Method 1: get_optimal_threshold ===")

# Get optimal thresholds for multiclass
thresholds = get_optimal_threshold(y, y_prob, metric="f1", method="unique_scan")
print(f"Optimal thresholds per class: {thresholds}")

# Apply thresholds to make predictions (simple approach)
# For each sample, predict class with highest probability among those above threshold
y_pred_optimized = []
for i in range(len(y_prob)):
    above_threshold = y_prob[i] > thresholds
    if np.any(above_threshold):
        # Among classes above threshold, pick highest probability
        candidates = np.where(above_threshold)[0]
        y_pred_optimized.append(candidates[np.argmax(y_prob[i, candidates])])
    else:
        # No class above threshold, pick highest probability
        y_pred_optimized.append(np.argmax(y_prob[i]))

y_pred_optimized = np.array(y_pred_optimized)
print(f"Optimized predictions: {y_pred_optimized[:10]}...")

# Compare with default argmax predictions
y_pred_default = np.argmax(y_prob, axis=1)
print(f"Default predictions:   {y_pred_default[:10]}...")

print("\n=== Performance Comparison ===")
print("Default (argmax) performance:")
print(classification_report(y, y_pred_default, digits=3))

print("Optimized thresholds performance:")
print(classification_report(y, y_pred_optimized, digits=3))

# Method 2: Using lower-level functions
print("\n=== Method 2: Lower-level functions ===")

# Get optimal thresholds for different metrics
thresholds_f1 = get_optimal_multiclass_thresholds(y, y_prob, metric="f1")
thresholds_precision = get_optimal_multiclass_thresholds(y, y_prob, metric="precision")
thresholds_recall = get_optimal_multiclass_thresholds(y, y_prob, metric="recall")

print(f"F1-optimized thresholds:        {thresholds_f1}")
print(f"Precision-optimized thresholds: {thresholds_precision}")
print(f"Recall-optimized thresholds:    {thresholds_recall}")

# Compute confusion matrices and multiclass metrics
cms = get_multiclass_confusion_matrix(y, y_prob, thresholds_f1)
print("\nPer-class confusion matrices (F1-optimized):")
for i, cm in enumerate(cms):
    tp, tn, fp, fn = cm
    print(f"  Class {i}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# Compute multiclass metrics with different averaging strategies
f1_macro = multiclass_metric(cms, "f1", "macro")
f1_micro = multiclass_metric(cms, "f1", "micro")
f1_weighted = multiclass_metric(cms, "f1", "weighted")

print("\nMulticlass F1 scores:")
print(f"  Macro average:    {f1_macro:.3f}")
print(f"  Micro average:    {f1_micro:.3f}")
print(f"  Weighted average: {f1_weighted:.3f}")

# Method 3: Class-specific analysis
print("\n=== Method 3: Class-specific Analysis ===")

for class_idx in range(len(np.unique(y))):
    # Convert to binary problem for this class
    y_binary = (y == class_idx).astype(int)
    y_prob_binary = y_prob[:, class_idx]

    # Optimize threshold for this specific class
    class_optimizer = ThresholdOptimizer(metric="f1")
    class_optimizer.fit(y_binary, y_prob_binary)

    print(f"Class {class_idx}:")
    print(f"  Optimal threshold: {class_optimizer.threshold_:.3f}")
    print(
        f"  Class frequency:   {np.sum(y_binary)}/{len(y_binary)} "
        f"({np.mean(y_binary):.1%})"
    )

# Method 4: Different optimization methods
print("\n=== Method 4: Different Optimization Methods ===")

methods = ["unique_scan", "minimize", "gradient"]
for method in methods:
    thresholds = get_optimal_multiclass_thresholds(
        y, y_prob, metric="f1", method=method
    )
    print(f"{method:12s}: {thresholds}")

print("\n=== Summary ===")
print("Key advantages of multiclass threshold optimization:")
print("1. Handles class imbalance better than simple argmax")
print("2. Optimizes specific metrics (F1, precision, recall) per class")
print("3. Uses One-vs-Rest strategy for independent class thresholds")
print("4. Maintains backward compatibility with binary classification")
