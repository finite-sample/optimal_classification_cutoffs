"""Generate plots demonstrating piecewise-constant nature of classification metrics.

This script creates visualizations that illustrate why continuous optimizers
can miss the global optimum for metrics like F1 score, and demonstrates the
effectiveness of the fallback mechanism.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Add the parent directory to sys.path to import optimal_cutoffs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimal_cutoffs.metrics import get_confusion_matrix, METRIC_REGISTRY


def compute_metric_at_threshold(y_true, y_prob, threshold, metric="f1"):
    """Helper function to compute metric at threshold using new API."""
    tp, tn, fp, fn = get_confusion_matrix(y_true, y_prob, threshold)
    return METRIC_REGISTRY[metric](tp, tn, fp, fn)


def plot_piecewise_f1_demonstration():
    """Create a comprehensive plot showing piecewise-constant F1 behavior."""

    # Example data that demonstrates the piecewise nature clearly
    y_true = np.array([0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    # Generate a dense grid of thresholds for plotting
    thresholds = np.linspace(0.05, 0.95, 1000)
    f1_scores = [
        compute_metric_at_threshold(y_true, y_prob, t, "f1") for t in thresholds
    ]

    # Find unique probabilities (the breakpoints)
    unique_probs = np.unique(y_prob)
    unique_f1s = [
        compute_metric_at_threshold(y_true, y_prob, t, "f1") for t in unique_probs
    ]

    # Run minimize_scalar to show potential suboptimal convergence
    result = optimize.minimize_scalar(
        lambda t: -compute_metric_at_threshold(y_true, y_prob, t, "f1"),
        bounds=(0, 1),
        method="bounded",
    )
    minimize_threshold = result.x
    minimize_f1 = compute_metric_at_threshold(y_true, y_prob, minimize_threshold, "f1")

    # Find the true optimal
    optimal_idx = np.argmax(unique_f1s)
    optimal_threshold = unique_probs[optimal_idx]
    optimal_f1 = unique_f1s[optimal_idx]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main F1 vs Threshold plot
    ax1.plot(thresholds, f1_scores, "b-", linewidth=2, label="F1 Score")

    # Mark the unique probability breakpoints
    ax1.scatter(
        unique_probs,
        unique_f1s,
        color="red",
        s=100,
        zorder=5,
        label="Candidate thresholds",
    )

    # Mark the optimal threshold
    ax1.scatter(
        [optimal_threshold],
        [optimal_f1],
        color="green",
        s=150,
        marker="*",
        zorder=6,
        label=f"Optimal (F1={optimal_f1:.3f})",
    )

    # Mark where minimize_scalar converged
    ax1.scatter(
        [minimize_threshold],
        [minimize_f1],
        color="orange",
        s=120,
        marker="x",
        zorder=6,
        label=f"minimize_scalar result\n(t={minimize_threshold:.3f})",
    )

    # Add vertical lines at breakpoints to emphasize piecewise nature
    for prob in unique_probs:
        ax1.axvline(x=prob, color="red", linestyle="--", alpha=0.3)

    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("F1 Score")
    ax1.set_title(
        "Piecewise-Constant Nature of F1 Score\n"
        + "F1 only changes at unique probability values"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Zoom-in view showing constant regions
    zoom_start, zoom_end = 0.35, 0.65
    zoom_mask = (thresholds >= zoom_start) & (thresholds <= zoom_end)

    ax2.plot(thresholds[zoom_mask], np.array(f1_scores)[zoom_mask], "b-", linewidth=3)
    ax2.axvline(
        x=0.4, color="red", linestyle="--", linewidth=2, label="Breakpoint at p=0.4"
    )
    ax2.axvline(
        x=0.6, color="red", linestyle="--", linewidth=2, label="Breakpoint at p=0.6"
    )

    # Highlight the constant regions
    constant_region_1 = (thresholds >= zoom_start) & (thresholds < 0.4)
    constant_region_2 = (thresholds >= 0.4) & (thresholds < 0.6)
    constant_region_3 = (thresholds >= 0.6) & (thresholds <= zoom_end)

    if np.any(constant_region_1):
        ax2.fill_between(
            thresholds[constant_region_1],
            0,
            np.array(f1_scores)[constant_region_1],
            alpha=0.2,
            color="blue",
            label="Constant region 1",
        )
    if np.any(constant_region_2):
        ax2.fill_between(
            thresholds[constant_region_2],
            0,
            np.array(f1_scores)[constant_region_2],
            alpha=0.2,
            color="green",
            label="Constant region 2",
        )
    if np.any(constant_region_3):
        ax2.fill_between(
            thresholds[constant_region_3],
            0,
            np.array(f1_scores)[constant_region_3],
            alpha=0.2,
            color="orange",
            label="Constant region 3",
        )

    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Zoom: F1 Score is Constant Between Breakpoints")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        "/Users/soodoku/Documents/GitHub/optimal_classification_cutoffs/docs/piecewise_f1_demo.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "unique_probs": unique_probs,
        "unique_f1s": unique_f1s,
        "optimal_threshold": optimal_threshold,
        "optimal_f1": optimal_f1,
        "minimize_threshold": minimize_threshold,
        "minimize_f1": minimize_f1,
    }


def plot_optimization_methods_comparison():
    """Compare different optimization methods on the same data."""

    # Use more complex data to show clearer differences
    np.random.seed(42)
    n_samples = 50
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.beta(2, 2, n_samples)  # Bell-shaped distribution

    thresholds = np.linspace(0.01, 0.99, 500)
    f1_scores = [
        compute_metric_at_threshold(y_true, y_prob, t, "f1") for t in thresholds
    ]

    # Smart brute force approach
    unique_probs = np.unique(y_prob)
    unique_f1s = [
        compute_metric_at_threshold(y_true, y_prob, t, "f1") for t in unique_probs
    ]
    brute_optimal_idx = np.argmax(unique_f1s)
    brute_threshold = unique_probs[brute_optimal_idx]
    brute_f1 = unique_f1s[brute_optimal_idx]

    # Minimize_scalar approach
    result = optimize.minimize_scalar(
        lambda t: -compute_metric_at_threshold(y_true, y_prob, t, "f1"),
        bounds=(0, 1),
        method="bounded",
    )
    minimize_threshold = result.x
    minimize_f1 = compute_metric_at_threshold(y_true, y_prob, minimize_threshold, "f1")

    # Fallback mechanism (what the library actually does)
    candidates = np.unique(np.append(y_prob, minimize_threshold))
    candidate_scores = [
        compute_metric_at_threshold(y_true, y_prob, t, "f1") for t in candidates
    ]
    fallback_idx = np.argmax(candidate_scores)
    fallback_threshold = candidates[fallback_idx]
    fallback_f1 = candidate_scores[fallback_idx]

    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(thresholds, f1_scores, "b-", linewidth=1.5, alpha=0.7, label="F1 Score")

    # Show all unique probabilities as candidates
    ax.scatter(
        unique_probs,
        unique_f1s,
        color="lightcoral",
        s=20,
        alpha=0.6,
        label=f"All candidates ({len(unique_probs)} points)",
    )

    # Smart brute force result
    ax.scatter(
        [brute_threshold],
        [brute_f1],
        color="green",
        s=150,
        marker="*",
        zorder=5,
        label=f"Smart brute force\n(F1={brute_f1:.3f})",
    )

    # Minimize_scalar result
    ax.scatter(
        [minimize_threshold],
        [minimize_f1],
        color="red",
        s=120,
        marker="x",
        zorder=5,
        label=f"minimize_scalar only\n(F1={minimize_f1:.3f})",
    )

    # Fallback result
    ax.scatter(
        [fallback_threshold],
        [fallback_f1],
        color="blue",
        s=150,
        marker="D",
        zorder=5,
        label=f"With fallback\n(F1={fallback_f1:.3f})",
    )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(
        "Comparison of Optimization Methods\n"
        + f"Data: {n_samples} samples, {len(unique_probs)} unique probabilities"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        "/Users/soodoku/Documents/GitHub/optimal_classification_cutoffs/docs/optimization_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "n_unique": len(unique_probs),
        "brute_f1": brute_f1,
        "minimize_f1": minimize_f1,
        "fallback_f1": fallback_f1,
        "improvement": fallback_f1 - minimize_f1,
    }


def plot_multiple_metrics_comparison():
    """Show how different metrics have different optimal thresholds."""

    # Create data with clear separation
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])

    thresholds = np.linspace(0.05, 0.95, 200)

    metrics = ["accuracy", "f1", "precision", "recall"]
    colors = ["blue", "red", "green", "orange"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    optimal_thresholds = {}

    for metric, color in zip(metrics, colors, strict=False):
        scores = [
            compute_metric_at_threshold(y_true, y_prob, t, metric) for t in thresholds
        ]
        ax.plot(thresholds, scores, color=color, linewidth=2, label=metric.capitalize())

        # Find optimal for this metric
        unique_probs = np.unique(y_prob)
        unique_scores = [
            compute_metric_at_threshold(y_true, y_prob, t, metric) for t in unique_probs
        ]
        optimal_idx = np.argmax(unique_scores)
        optimal_threshold = unique_probs[optimal_idx]
        optimal_score = unique_scores[optimal_idx]

        ax.scatter(
            [optimal_threshold],
            [optimal_score],
            color=color,
            s=150,
            marker="*",
            zorder=5,
            edgecolors="black",
            linewidth=1,
        )

        optimal_thresholds[metric] = optimal_threshold

    # Add vertical lines at unique probabilities
    unique_probs = np.unique(y_prob)
    for prob in unique_probs:
        ax.axvline(x=prob, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_title(
        "Different Metrics Have Different Optimal Thresholds\n"
        + "Stars show optimal thresholds for each metric"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(
        "/Users/soodoku/Documents/GitHub/optimal_classification_cutoffs/docs/multiple_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return optimal_thresholds


if __name__ == "__main__":
    print("Generating piecewise F1 demonstration plot...")
    demo_results = plot_piecewise_f1_demonstration()
    print(f"✓ Optimal threshold: {demo_results['optimal_threshold']:.3f}")
    print(f"✓ minimize_scalar found: {demo_results['minimize_threshold']:.3f}")

    print("\nGenerating optimization methods comparison...")
    comp_results = plot_optimization_methods_comparison()
    print(f"✓ Smart brute force F1: {comp_results['brute_f1']:.3f}")
    print(f"✓ minimize_scalar F1: {comp_results['minimize_f1']:.3f}")
    print(f"✓ With fallback F1: {comp_results['fallback_f1']:.3f}")
    print(f"✓ Improvement from fallback: {comp_results['improvement']:.3f}")

    print("\nGenerating multiple metrics comparison...")
    metric_results = plot_multiple_metrics_comparison()
    print("✓ Optimal thresholds by metric:")
    for metric, threshold in metric_results.items():
        print(f"   {metric}: {threshold:.3f}")

    print("\n✓ All plots saved to docs/ directory")
