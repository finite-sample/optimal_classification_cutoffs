"""Example demonstrating multiclass classification threshold optimization."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from optimal_cutoffs import get_optimal_threshold

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
result = get_optimal_threshold(y, y_prob, metric="f1")
thresholds = result.thresholds
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

# Method 2: Different metrics optimization
print("\n=== Method 2: Different Metrics ===")

# Get optimal thresholds for different metrics
result_f1 = get_optimal_threshold(y, y_prob, metric="f1")
result_precision = get_optimal_threshold(y, y_prob, metric="precision")
result_recall = get_optimal_threshold(y, y_prob, metric="recall")

print(f"F1-optimized thresholds:        {result_f1.thresholds}")
print(f"Precision-optimized thresholds: {result_precision.thresholds}")
print(f"Recall-optimized thresholds:    {result_recall.thresholds}")

# Make predictions using different threshold sets
y_pred_f1 = result_f1.predict(y_prob)
y_pred_precision = result_precision.predict(y_prob)
y_pred_recall = result_recall.predict(y_prob)

print("\nPerformance with different metrics:")
print("F1-optimized:")
print(classification_report(y, y_pred_f1, digits=3))
print("Precision-optimized:")
print(classification_report(y, y_pred_precision, digits=3))
print("Recall-optimized:")
print(classification_report(y, y_pred_recall, digits=3))

# Method 3: Class-specific analysis
print("\n=== Method 3: Class-specific Analysis ===")

for class_idx in range(len(np.unique(y))):
    # Convert to binary problem for this class
    y_binary = (y == class_idx).astype(int)
    y_prob_binary = y_prob[:, class_idx]

    # Optimize threshold for this specific class
    result_binary = get_optimal_threshold(y_binary, y_prob_binary, metric="f1")

    print(f"Class {class_idx}:")
    print(f"  Optimal threshold: {result_binary.threshold:.3f}")
    print(
        f"  Class frequency:   {np.sum(y_binary)}/{len(y_binary)} "
        f"({np.mean(y_binary):.1%})"
    )

# Method 4: Different optimization methods
print("\n=== Method 4: Different Optimization Methods ===")

methods = ["auto", "minimize", "gradient"]
for method in methods:
    try:
        result_method = get_optimal_threshold(
            y, y_prob, metric="f1", method=method
        )
        print(f"{method:12s}: {result_method.thresholds}")
    except Exception as e:
        print(f"{method:12s}: Error - {e}")

print("\n=== Summary ===")
print("Key advantages of multiclass threshold optimization:")
print("1. Handles class imbalance better than simple argmax")
print("2. Optimizes specific metrics (F1, precision, recall) per class")
print("3. Uses One-vs-Rest strategy for independent class thresholds")
print("4. Maintains backward compatibility with binary classification")
