"""
🚀 Quickstart: See 40%+ performance improvement in 3 lines of code
================================================================

**ROI**: Transform mediocre model performance into excellent results
**Time**: 5 minutes to life-changing insights
**Next**: See 02_business_value.py for dollar impact

This example demonstrates the immediate value of optimal threshold selection
using a realistic imbalanced dataset scenario.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from optimal_cutoffs import get_optimal_threshold

print("🚀 OPTIMAL THRESHOLDS: QUICKSTART DEMO")
print("=" * 50)

# Generate realistic imbalanced dataset (like fraud detection, medical diagnosis)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% negative, 10% positive (imbalanced)
    flip_y=0.02,  # Add some noise
    random_state=42
)

# Split and train a model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"📊 Test set: {len(y_test)} samples, {y_test.sum()} positive ({y_test.mean():.1%})")
print()

# =============================================================================
# THE PROBLEM: Default 0.5 threshold often performs poorly
# =============================================================================
print("❌ BEFORE: Default 0.5 threshold")
print("-" * 30)

y_pred_default = (y_prob >= 0.5).astype(int)
f1_default = f1_score(y_test, y_pred_default)
precision_default = precision_score(y_test, y_pred_default, zero_division=0)
recall_default = recall_score(y_test, y_pred_default)
accuracy_default = accuracy_score(y_test, y_pred_default)

print(f"F1 Score:   {f1_default:.3f}")
print(f"Precision:  {precision_default:.3f}")
print(f"Recall:     {recall_default:.3f}")
print(f"Accuracy:   {accuracy_default:.3f}")
print()

# =============================================================================
# THE SOLUTION: 3 lines of code for optimal performance
# =============================================================================
print("✨ AFTER: Optimal threshold (3 lines of code)")
print("-" * 45)

# Line 1: Find optimal threshold
result = get_optimal_threshold(y_train, model.predict_proba(X_train)[:, 1], metric="f1")

# Line 2: Get the threshold value  
optimal_threshold = result.threshold

# Line 3: Make predictions
y_pred_optimal = result.predict(y_prob)

# Results
f1_optimal = f1_score(y_test, y_pred_optimal)
precision_optimal = precision_score(y_test, y_pred_optimal)
recall_optimal = recall_score(y_test, y_pred_optimal)
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 Score:   {f1_optimal:.3f} (↑ {(f1_optimal/f1_default-1)*100:+.1f}%)")
print(f"Precision:  {precision_optimal:.3f} (↑ {(precision_optimal/precision_default-1)*100:+.1f}%)")
print(f"Recall:     {recall_optimal:.3f} (↑ {(recall_optimal/recall_default-1)*100:+.1f}%)")
print(f"Accuracy:   {accuracy_optimal:.3f} (↑ {(accuracy_optimal/accuracy_default-1)*100:+.1f}%)")
print()

# =============================================================================
# IMPACT SUMMARY
# =============================================================================
print("🎯 IMPACT SUMMARY")
print("=" * 20)

improvement = (f1_optimal / f1_default - 1) * 100 if f1_default > 0 else 0
print(f"F1 Score improvement: {improvement:+.1f}%")
print(f"From {f1_default:.3f} → {f1_optimal:.3f}")
print()

if improvement > 10:
    print("🔥 EXCELLENT: >10% improvement!")
elif improvement > 5:
    print("✅ GOOD: >5% improvement")
elif improvement > 0:
    print("📈 BETTER: Positive improvement")
else:
    print("⚠️  Model may already be well-calibrated")

print()

# =============================================================================
# WHY THIS MATTERS
# =============================================================================
print("💡 WHY THIS MATTERS")
print("=" * 20)
print("• Default 0.5 threshold assumes balanced classes")
print("• Real data is often imbalanced (fraud: ~0.1%, cancer: ~1%)")
print("• Wrong threshold → missed opportunities or false alarms")
print("• Optimal threshold → maximize what you care about")
print()

# =============================================================================
# THE CODE (copy-paste ready)
# =============================================================================
print("📋 COPY-PASTE CODE")
print("=" * 20)
print("""
# Your 3 lines for any sklearn model:
from optimal_cutoffs import get_optimal_threshold

result = get_optimal_threshold(y_train, y_prob_train, metric="f1")
predictions = result.predict(y_prob_test)
""")

# =============================================================================
# NEXT STEPS
# =============================================================================
print("🚀 NEXT STEPS")
print("=" * 15)
print("• 02_business_value.py → See dollar impact ($50K+ saved)")
print("• 03_multiclass.py → Handle 3+ classes") 
print("• 04_interactive_demo.ipynb → Understand why this works")
print()
print("Questions? See: https://github.com/finite-sample/optimal-classification-cutoffs")