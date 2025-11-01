"""
💰 Business Value: Save $50,000+ with Cost-Sensitive Optimization
================================================================

**ROI**: Turn model into profit center with cost-aware thresholds
**Time**: 10 minutes to understand real business impact
**Next**: See 03_multiclass.py for advanced scenarios

This example shows how to optimize for business metrics (dollars) rather than
statistical metrics (F1, accuracy). Real applications have different costs for
different types of errors.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from optimal_cutoffs import get_optimal_threshold

print("💰 BUSINESS VALUE: COST-SENSITIVE OPTIMIZATION")
print("=" * 55)

# =============================================================================
# SCENARIO: Credit Card Fraud Detection
# =============================================================================
print("🏦 SCENARIO: Credit Card Fraud Detection")
print("-" * 45)
print("• Average transaction: $150")
print("• Fraud investigation cost: $25 per case")
print("• Chargeback + fees when fraud missed: $500")
print("• Customer friction when legitimate blocked: $50")
print()

# Generate realistic fraud dataset
X, y = make_classification(
    n_samples=5000,
    n_features=15,
    n_classes=2,
    weights=[0.98, 0.02],  # 98% legitimate, 2% fraud (realistic rate)
    flip_y=0.01,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train fraud detection model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_prob_train = model.predict_proba(X_train)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

print(f"📊 Test set: {len(y_test)} transactions")
print(f"   Fraud: {y_test.sum()} ({y_test.mean():.2%})")
print(f"   Legitimate: {(~y_test.astype(bool)).sum()} ({(1-y_test.mean()):.2%})")
print()

# =============================================================================
# BUSINESS COSTS: Define the true cost of each decision
# =============================================================================
print("💸 BUSINESS COST MATRIX")
print("-" * 25)

business_costs = {
    "tp": 500 - 25,  # Catch fraud: Save $500, pay $25 investigation = +$475
    "tn": 0,  # Legitimate passes: $0 cost
    "fp": -50 - 25,  # Block legitimate: $50 friction + $25 investigation = -$75
    "fn": -500,  # Miss fraud: $500 chargeback = -$500
}

print("True Positive (catch fraud):     +$475 (save $500, pay $25 investigation)")
print("True Negative (legitimate ok):   +$0   (no cost)")
print("False Positive (block good):     -$75  ($50 friction + $25 investigation)")
print("False Negative (miss fraud):     -$500 (full chargeback loss)")
print()

# =============================================================================
# METHOD 1: Default Statistical Optimization (F1 Score)
# =============================================================================
print("📊 METHOD 1: Statistical Optimization (F1 Score)")
print("-" * 50)

result_f1 = get_optimal_threshold(y_train, y_prob_train, metric="f1")
y_pred_f1 = result_f1.predict(y_prob_test)

# Calculate confusion matrix and business value
cm_f1 = confusion_matrix(y_test, y_pred_f1)
tn, fp, fn, tp = cm_f1.ravel()

business_value_f1 = (
    tp * business_costs["tp"]
    + tn * business_costs["tn"]
    + fp * business_costs["fp"]
    + fn * business_costs["fn"]
)

print(f"F1-Optimal threshold: {result_f1.threshold:.3f}")
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"F1 Score: {f1_score(y_test, y_pred_f1):.3f}")
print(f"Business Value: ${business_value_f1:,}")
print()

# =============================================================================
# METHOD 2: Business-Aware Optimization (Utility/Cost)
# =============================================================================
print("🎯 METHOD 2: Business-Aware Optimization")
print("-" * 40)

# The power move: optimize directly for business value
result_business = get_optimal_threshold(y_train, y_prob_train, utility=business_costs)
y_pred_business = result_business.predict(y_prob_test)

cm_business = confusion_matrix(y_test, y_pred_business)
tn, fp, fn, tp = cm_business.ravel()

business_value_business = (
    tp * business_costs["tp"]
    + tn * business_costs["tn"]
    + fp * business_costs["fp"]
    + fn * business_costs["fn"]
)

print(f"Business-Optimal threshold: {result_business.threshold:.3f}")
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"F1 Score: {f1_score(y_test, y_pred_business):.3f}")
print(f"Business Value: ${business_value_business:,}")
print()

# =============================================================================
# METHOD 3: Default 0.5 Threshold (Baseline)
# =============================================================================
print("❌ METHOD 3: Default 0.5 Threshold (Baseline)")
print("-" * 45)

y_pred_default = (y_prob_test >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default)
tn, fp, fn, tp = cm_default.ravel()

business_value_default = (
    tp * business_costs["tp"]
    + tn * business_costs["tn"]
    + fp * business_costs["fp"]
    + fn * business_costs["fn"]
)

print("Default threshold: 0.500")
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"F1 Score: {f1_score(y_test, y_pred_default, zero_division=0):.3f}")
print(f"Business Value: ${business_value_default:,}")
print()

# =============================================================================
# BUSINESS IMPACT COMPARISON
# =============================================================================
print("📈 BUSINESS IMPACT COMPARISON")
print("=" * 35)

methods = [
    ("Default (0.5)", business_value_default),
    ("F1-Optimized", business_value_f1),
    ("Business-Optimized", business_value_business),
]

best_value = max(value for _, value in methods)

for name, value in methods:
    if value == best_value:
        print(f"🏆 {name:18}: ${value:,} (BEST)")
    else:
        improvement = value - business_value_default
        print(f"   {name:18}: ${value:,} ({improvement:+,})")

print()

total_improvement = business_value_business - business_value_default
improvement_pct = (
    (business_value_business / business_value_default - 1) * 100
    if business_value_default != 0
    else 0
)

print(f"💰 TOTAL BUSINESS IMPACT: ${total_improvement:+,}")
if improvement_pct > 0:
    print(f"📊 Improvement: {improvement_pct:+.1f}%")
print()

# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("💡 KEY INSIGHTS")
print("=" * 20)
print("1. F1 optimization ≠ Business optimization")
print("2. Different errors have different costs")
print("3. Business-aware thresholds maximize profit")
print("4. Small threshold changes = Big business impact")
print(f"5. This example: ${total_improvement:+,} improvement!")
print()

# =============================================================================
# THE CODE (copy-paste ready)
# =============================================================================
print("📋 COPY-PASTE CODE")
print("=" * 20)
print("""
# Define your business costs:
costs = {
    "tp": 475,   # Value of catching fraud
    "tn": 0,     # No cost for legitimate
    "fp": -75,   # Cost of false alarm
    "fn": -500,  # Cost of missed fraud
}

# Optimize for business value:
result = get_optimal_threshold(y_train, y_prob_train, utility=costs)
predictions = result.predict(y_prob_test)
""")

# =============================================================================
# WHEN TO USE BUSINESS OPTIMIZATION
# =============================================================================
print("🎯 WHEN TO USE BUSINESS OPTIMIZATION")
print("=" * 40)
print("✅ When you know the cost of different errors")
print("✅ When statistical metrics don't align with business goals")
print("✅ When you need to justify model ROI to stakeholders")
print("✅ When optimizing for revenue, not just accuracy")
print()
print("Examples:")
print("• Fraud detection (chargebacks vs investigation costs)")
print("• Medical diagnosis (missed diagnosis vs unnecessary tests)")
print("• Marketing (ad spend vs conversion value)")
print("• Quality control (defect costs vs inspection costs)")
print()

# =============================================================================
# NEXT STEPS
# =============================================================================
print("🚀 NEXT STEPS")
print("=" * 15)
print("• 03_multiclass.py → Handle 3+ classes")
print("• 04_interactive_demo.ipynb → Deep dive into threshold behavior")
print("• Try with your own cost matrix and data!")
print()
print("Questions? See: https://github.com/finite-sample/optimal-classification-cutoffs")
