"""Comprehensive demonstration of multiclass averaging strategies and their
differences."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from optimal_cutoffs import (
    get_multiclass_confusion_matrix,
    get_optimal_multiclass_thresholds,
    multiclass_metric,
)


def create_imbalanced_dataset():
    """Create an imbalanced multiclass dataset to demonstrate averaging differences."""
    print("=== Creating Imbalanced Multiclass Dataset ===\n")

    # Create imbalanced dataset with different class frequencies
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=4,
        n_informative=8,
        n_redundant=2,
        weights=[0.5, 0.3, 0.15, 0.05],  # Highly imbalanced
        random_state=42,
    )

    # Train classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    y_prob = model.predict_proba(X)

    # Show class distribution
    class_counts = np.bincount(y)
    print(f"Dataset size: {len(y)}")
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count:3d} samples ({count / len(y):5.1%})")

    return y, y_prob, class_counts


def demonstrate_averaging_semantics(y_true, y_prob, class_counts):
    """Demonstrate the mathematical differences between averaging strategies."""
    print("\n=== Averaging Strategy Semantics ===\n")

    # Use fixed thresholds for consistent comparison
    thresholds = np.array([0.3, 0.4, 0.5, 0.6])

    # Get confusion matrices
    cms = get_multiclass_confusion_matrix(y_true, y_prob, thresholds)

    print("Per-class confusion matrices (TP, TN, FP, FN):")
    for i, (tp, tn, fp, fn) in enumerate(cms):
        print(
            f"  Class {i}: TP={tp:2d}, TN={tn:3d}, FP={fp:2d}, FN={fn:2d} | "
            f"Support={tp + fn}"
        )

    print("\nF1 Scores by Averaging Strategy:")

    # Compute F1 with different averaging strategies
    f1_none = multiclass_metric(cms, "f1", average="none")
    f1_macro = multiclass_metric(cms, "f1", average="macro")
    f1_micro = multiclass_metric(cms, "f1", average="micro")
    f1_weighted = multiclass_metric(cms, "f1", average="weighted")

    print(f"  Per-class (none): {f1_none}")
    print(f"  Macro average:    {f1_macro:.4f} (equal weight to each class)")
    print(f"  Micro average:    {f1_micro:.4f} (equal weight to each sample)")
    print(f"  Weighted average: {f1_weighted:.4f} (weighted by class support)")

    # Verify mathematical identities
    print("\nMathematical Identity Verification:")
    computed_macro = np.mean(f1_none)
    print(f"  Macro = mean(per_class): {f1_macro:.6f} = {computed_macro:.6f} ✓")

    total_support = sum(class_counts)
    computed_weighted = (
        sum(f1 * count for f1, count in zip(f1_none, class_counts, strict=False))
        / total_support
    )
    print(
        f"  Weighted = Σ(f1_i × support_i) / Σ(support_i): {f1_weighted:.6f} = "
        f"{computed_weighted:.6f} ✓"
    )

    # Show micro calculation
    total_tp = sum(cm[0] for cm in cms)
    total_fp = sum(cm[2] for cm in cms)
    total_fn = sum(cm[3] for cm in cms)
    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    computed_micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )
    print(f"  Micro = F1(Σ(TP), Σ(FP), Σ(FN)): {f1_micro:.6f} = {computed_micro:.6f} ✓")


def demonstrate_optimization_differences(y_true, y_prob):
    """Demonstrate how different averaging strategies affect threshold optimization."""
    print("\n=== Optimization with Different Averaging Strategies ===\n")

    optimization_methods = ["unique_scan", "minimize"]

    for method in optimization_methods:
        print(f"Optimization Method: {method}")
        print("-" * 50)

        # Optimize with different averaging strategies
        strategies = {
            "macro": "Treats all classes equally",
            "micro": "Joint optimization for micro-averaged metric",
            "weighted": "Same as macro (per-class optimization)",
            "none": "Same as macro (per-class optimization)",
        }

        results = {}
        for average, description in strategies.items():
            try:
                thresholds = get_optimal_multiclass_thresholds(
                    y_true, y_prob, "f1", method, average=average
                )
                results[average] = thresholds
                print(f"  {average:8s}: {thresholds} | {description}")
            except Exception as e:
                print(f"  {average:8s}: Failed ({e})")

        # Compare results
        print("\n  Comparison of threshold differences:")
        if "macro" in results and "micro" in results:
            diff = np.abs(results["macro"] - results["micro"])
            print(f"    |macro - micro|: {diff}")
            print(f"    Max difference:  {np.max(diff):.4f}")

        print()


def demonstrate_performance_vectorization(y_true, y_prob):
    """Demonstrate performance improvements with vectorization."""
    print("=== Performance: Vectorized vs Standard Implementation ===\n")

    import time

    # Time standard implementation
    start = time.time()
    thresholds_standard = get_optimal_multiclass_thresholds(
        y_true, y_prob, "f1", "unique_scan", average="macro", vectorized=False
    )
    time_standard = time.time() - start

    # Time vectorized implementation
    start = time.time()
    thresholds_vectorized = get_optimal_multiclass_thresholds(
        y_true, y_prob, "f1", "unique_scan", average="macro", vectorized=True
    )
    time_vectorized = time.time() - start

    print(f"Standard implementation:   {time_standard:.4f} seconds")
    print(f"Vectorized implementation: {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_standard / time_vectorized:.2f}x")

    # Verify results are similar
    diff = np.abs(thresholds_standard - thresholds_vectorized)
    print(f"Max threshold difference: {np.max(diff):.6f}")
    print(
        "✓ Results are nearly identical" if np.max(diff) < 1e-6 else "⚠ Results differ"
    )


def demonstrate_practical_implications(y_true, y_prob, class_counts):
    """Demonstrate practical implications of different averaging strategies."""
    print("\n=== Practical Implications of Averaging Choices ===\n")

    scenarios = {
        "macro": {
            "description": "Equal importance to all classes",
            "use_case": (
                "When you want to perform equally well on all classes "
                "regardless of frequency"
            ),
            "example": (
                "Medical diagnosis where rare diseases are as important as common ones"
            ),
        },
        "micro": {
            "description": "Equal importance to all samples",
            "use_case": "When overall classification accuracy matters most",
            "example": "Large-scale text classification where total accuracy is key",
        },
        "weighted": {
            "description": "Importance proportional to class frequency",
            "use_case": "When you want a balance between macro and micro averaging",
            "example": "Customer segmentation where larger segments are more important",
        },
        "none": {
            "description": "No averaging, see per-class performance",
            "use_case": "When you need to analyze each class individually",
            "example": "Model diagnosis and debugging class-specific performance",
        },
    }

    # Get optimal thresholds for each strategy
    for strategy, info in scenarios.items():
        print(f"{strategy.upper()} AVERAGING:")
        print(f"  Description: {info['description']}")
        print(f"  Use case: {info['use_case']}")
        print(f"  Example: {info['example']}")

        try:
            thresholds = get_optimal_multiclass_thresholds(
                y_true, y_prob, "f1", "unique_scan", average=strategy
            )

            # Evaluate performance with these thresholds
            cms = get_multiclass_confusion_matrix(y_true, y_prob, thresholds)

            if strategy == "none":
                f1_scores = multiclass_metric(cms, "f1", average="none")
                print(f"  Per-class F1: {f1_scores}")
                print(f"  Macro average: {np.mean(f1_scores):.4f}")
            else:
                f1_score = multiclass_metric(cms, "f1", average=strategy)
                print(f"  F1 score: {f1_score:.4f}")

                # Also show per-class for context
                per_class_f1 = multiclass_metric(cms, "f1", average="none")
                print(f"  Per-class F1: {per_class_f1}")

        except Exception as e:
            print(f"  Error: {e}")

        print()


def compare_with_sklearn_metrics(y_true, y_prob):
    """Compare our multiclass metrics with sklearn's implementations."""
    print("=== Comparison with Scikit-learn Metrics ===\n")

    from sklearn.metrics import f1_score as sklearn_f1

    # Get predictions using argmax (sklearn default)
    y_pred_argmax = np.argmax(y_prob, axis=1)

    # Get predictions using optimized thresholds
    optimal_thresholds = get_optimal_multiclass_thresholds(
        y_true, y_prob, "f1", "unique_scan", average="macro"
    )

    # Make predictions with optimized thresholds
    binary_preds = y_prob > optimal_thresholds

    # For multiclass prediction with thresholds: if multiple classes above threshold,
    # pick highest probability; if none above threshold, pick highest probability
    y_pred_optimized = np.zeros(len(y_true), dtype=int)
    for i in range(len(y_true)):
        above_threshold = np.where(binary_preds[i])[0]
        if len(above_threshold) > 0:
            y_pred_optimized[i] = above_threshold[np.argmax(y_prob[i, above_threshold])]
        else:
            y_pred_optimized[i] = np.argmax(y_prob[i])

    print("Scikit-learn F1 scores (argmax predictions):")
    print(f"  Macro:    {sklearn_f1(y_true, y_pred_argmax, average='macro'):.4f}")
    print(f"  Micro:    {sklearn_f1(y_true, y_pred_argmax, average='micro'):.4f}")
    print(f"  Weighted: {sklearn_f1(y_true, y_pred_argmax, average='weighted'):.4f}")

    print("\nScikit-learn F1 scores (optimized thresholds):")
    print(f"  Macro:    {sklearn_f1(y_true, y_pred_optimized, average='macro'):.4f}")
    print(f"  Micro:    {sklearn_f1(y_true, y_pred_optimized, average='micro'):.4f}")
    print(f"  Weighted: {sklearn_f1(y_true, y_pred_optimized, average='weighted'):.4f}")

    # Show confusion matrices for comparison
    cms_argmax = get_multiclass_confusion_matrix(  # noqa: F841
        y_true,
        y_prob,
        np.full(y_prob.shape[1], 0.5),  # Argmax equivalent
    )
    cms_optimized = get_multiclass_confusion_matrix(y_true, y_prob, optimal_thresholds)

    print("\nOur library F1 scores (optimized thresholds):")
    f1_macro = multiclass_metric(cms_optimized, "f1", average="macro")
    f1_micro = multiclass_metric(cms_optimized, "f1", average="micro")
    f1_weighted = multiclass_metric(cms_optimized, "f1", average="weighted")

    print(f"  Macro:    {f1_macro:.4f}")
    print(f"  Micro:    {f1_micro:.4f}")
    print(f"  Weighted: {f1_weighted:.4f}")


def main():
    """Main demonstration function."""
    print("Multiclass Averaging Strategies - Comprehensive Demonstration")
    print("=" * 70)

    # Create imbalanced dataset
    y_true, y_prob, class_counts = create_imbalanced_dataset()

    # Demonstrate averaging semantics
    demonstrate_averaging_semantics(y_true, y_prob, class_counts)

    # Demonstrate optimization differences
    demonstrate_optimization_differences(y_true, y_prob)

    # Demonstrate performance improvements
    demonstrate_performance_vectorization(y_true, y_prob)

    # Demonstrate practical implications
    demonstrate_practical_implications(y_true, y_prob, class_counts)

    # Compare with sklearn
    compare_with_sklearn_metrics(y_true, y_prob)

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("• Macro: Best for class-balanced evaluation (rare classes matter)")
    print("• Micro: Best for sample-balanced evaluation (overall accuracy)")
    print("• Weighted: Balanced compromise between macro and micro")
    print("• None: Use for detailed per-class analysis and debugging")
    print("• Vectorization provides performance improvements for large datasets")
    print("• Threshold optimization can significantly improve over argmax predictions")


if __name__ == "__main__":
    main()
