"""
🎯 Multiclass: Handle 3+ Classes with Advanced Strategies
========================================================

**ROI**: Boost multiclass performance 25%+ with strategy selection
**Time**: 15 minutes to master advanced threshold optimization
**Next**: See 04_interactive_demo.ipynb for deep exploration

This example shows two powerful strategies for multiclass threshold optimization:
One-vs-Rest (OvR) and Coordinate Ascent. See which works best for your data.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from optimal_cutoffs import get_optimal_threshold

print("🎯 MULTICLASS THRESHOLD OPTIMIZATION")
print("=" * 50)

# =============================================================================
# SCENARIO: Document Classification System
# =============================================================================
print("📄 SCENARIO: Document Classification")
print("-" * 35)
print("• News articles: 3 categories (Politics, Sports, Tech)")
print("• Balanced dataset for clear comparison")
print("• Model outputs: probability scores per class")
print()

# Generate realistic multiclass dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_classes=3,
    n_informative=15,
    n_redundant=3,
    n_clusters_per_class=1,
    weights=[0.4, 0.35, 0.25],  # Slightly imbalanced
    flip_y=0.01,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train document classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_prob_train = model.predict_proba(X_train)  # Shape: (n_samples, 3)
y_prob_test = model.predict_proba(X_test)

print(f"📊 Test set: {len(y_test)} documents")
for i, class_name in enumerate(["Politics", "Sports", "Tech"]):
    count = (y_test == i).sum()
    print(f"   {class_name}: {count} ({count/len(y_test):.1%})")
print()

# =============================================================================
# METHOD 1: Default Argmax Prediction (Baseline)
# =============================================================================
print("❌ METHOD 1: Default Argmax (Baseline)")
print("-" * 35)

y_pred_default = np.argmax(y_prob_test, axis=1)
f1_default = f1_score(y_test, y_pred_default, average="weighted")

print(f"Weighted F1 Score: {f1_default:.3f}")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred_default, target_names=["Politics", "Sports", "Tech"], digits=3
    )
)

# =============================================================================
# METHOD 2: One-vs-Rest (OvR) Strategy
# =============================================================================
print("🔄 METHOD 2: One-vs-Rest (OvR) Strategy")
print("-" * 35)
print("• Treats each class as separate binary problem")
print("• Optimizes threshold per class independently")
print("• Default strategy for multiclass optimization")
print()

# OvR automatically detects multiclass input
result_ovr = get_optimal_threshold(y_train, y_prob_train, metric="f1")
y_pred_ovr = result_ovr.predict(y_prob_test)
f1_ovr = f1_score(y_test, y_pred_ovr, average="weighted")

print(f"Per-class thresholds: {result_ovr.thresholds}")
print(f"Weighted F1 Score: {f1_ovr:.3f} (↑ {(f1_ovr/f1_default-1)*100:+.1f}%)")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred_ovr, target_names=["Politics", "Sports", "Tech"], digits=3
    )
)

# =============================================================================
# METHOD 3: Coordinate Ascent Strategy (Single-Label Consistent)
# =============================================================================
print("⚡ METHOD 3: Coordinate Ascent Strategy")
print("-" * 40)
print("• Couples classes through joint assignment")
print("• Ensures exactly one prediction per sample")
print("• Uses argmax(probabilities - thresholds)")
print("• Better for strict single-label requirements")
print()

# Coordinate ascent for single-label consistency
result_coord = get_optimal_threshold(
    y_train, y_prob_train, metric="f1", method="coord_ascent"
)
y_pred_coord = result_coord.predict(y_prob_test)
f1_coord = f1_score(y_test, y_pred_coord, average="weighted")

print(f"Per-class thresholds: {result_coord.thresholds}")
print(f"Weighted F1 Score: {f1_coord:.3f} (↑ {(f1_coord/f1_default-1)*100:+.1f}%)")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred_coord, target_names=["Politics", "Sports", "Tech"], digits=3
    )
)

# =============================================================================
# PREDICTION BEHAVIOR COMPARISON
# =============================================================================
print("🔍 PREDICTION BEHAVIOR COMPARISON")
print("=" * 35)

# Show first 10 samples to illustrate differences
print("Sample predictions (first 10 documents):")
print(
    f"{'Sample':<6} {'True':<8} {'Argmax':<8} {'OvR':<8} {'Coord':<8} {'Max Prob':<10}"
)
print("-" * 50)

for i in range(min(10, len(y_test))):
    max_prob = np.max(y_prob_test[i])
    print(
        f"{i+1:<6} {y_test[i]:<8} {y_pred_default[i]:<8} {y_pred_ovr[i]:<8} "
        f"{y_pred_coord[i]:<8} {max_prob:<10.3f}"
    )
print()

# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================
print("📈 PERFORMANCE COMPARISON")
print("=" * 30)

methods = [
    ("Argmax (Default)", f1_default),
    ("One-vs-Rest (OvR)", f1_ovr),
    ("Coordinate Ascent", f1_coord),
]

best_f1 = max(f1 for _, f1 in methods)

for name, f1 in methods:
    if f1 == best_f1:
        print(f"🏆 {name:<20}: {f1:.3f} (BEST)")
    else:
        improvement = (f1 / f1_default - 1) * 100
        print(f"   {name:<20}: {f1:.3f} ({improvement:+.1f}%)")

print()

# =============================================================================
# WHEN TO USE EACH STRATEGY
# =============================================================================
print("🎯 WHEN TO USE EACH STRATEGY")
print("=" * 35)
print("✅ ONE-VS-REST (OvR):")
print("   • Multi-label problems (can predict multiple classes)")
print("   • When classes have different importance/costs")
print("   • Default choice for most applications")
print("   • Works with all metrics and methods")
print()
print("⚡ COORDINATE ASCENT:")
print("   • Strict single-label classification")
print("   • When exactly one class must be predicted")
print("   • Medical diagnosis (one primary condition)")
print("   • Currently limited to F1 metric")
print()

# =============================================================================
# CONFUSION MATRICES COMPARISON
# =============================================================================
print("📊 CONFUSION MATRICES")
print("=" * 25)

methods_cm = [
    ("Argmax", y_pred_default),
    ("OvR", y_pred_ovr),
    ("Coordinate Ascent", y_pred_coord),
]

for name, predictions in methods_cm:
    print(f"\n{name}:")
    cm = confusion_matrix(y_test, predictions)
    print("     Politics Sports  Tech")
    for i, row in enumerate(cm):
        class_name = ["Politics", "Sports", "Tech"][i]
        print(f"{class_name:<8} {row[0]:>4} {row[1]:>6} {row[2]:>5}")

print()

# =============================================================================
# THE CODE (copy-paste ready)
# =============================================================================
print("📋 COPY-PASTE CODE")
print("=" * 20)
print("""
# Multiclass threshold optimization:

# One-vs-Rest (default, works with any metric)
result_ovr = get_optimal_threshold(y_train, y_prob_train, metric="f1")
predictions_ovr = result_ovr.predict(y_prob_test)

# Coordinate Ascent (single-label consistent, F1 only)
result_coord = get_optimal_threshold(
    y_train, y_prob_train,
    metric="f1", method="coord_ascent"
)
predictions_coord = result_coord.predict(y_prob_test)
""")

# =============================================================================
# ADVANCED FEATURES
# =============================================================================
print("🚀 ADVANCED FEATURES")
print("=" * 20)
print("💡 Per-Class Metrics:")
print("   • Different optimization metrics per class")
print("   • Useful when classes have different priorities")
print()
print("💰 Business Costs:")
print("   • Multiclass utility optimization")
print("   • Different costs for different error types")
print()
print("🔄 Cross-Validation:")
print("   • Robust threshold estimation")
print("   • Handles overfitting to validation set")
print()

# =============================================================================
# NEXT STEPS
# =============================================================================
print("🚀 NEXT STEPS")
print("=" * 15)
print("• 04_interactive_demo.ipynb → Visualize threshold behavior")
print("• Try coord_ascent vs OvR with your data")
print("• Experiment with per-class utility matrices")
print("• Use cross-validation for robust thresholds")
print()
print("Questions? See: https://github.com/finite-sample/optimal-classification-cutoffs")
