#!/usr/bin/env python3
"""Step 1 Demonstration: IDBD vs LMS on Non-Stationary Target.

This script demonstrates the core claim of Step 1 of the Alberta Plan:
IDBD (with its meta-learned step-sizes) should match or beat hand-tuned
LMS on a non-stationary target tracking task.

The experiments:
1. Grid search comparison: IDBD vs many LMS step-sizes
2. Practical comparison: IDBD vs LMS with same initial step-size

Key insight: IDBD's value is that you don't need to grid search for the
optimal step-size. With a reasonable initial value, IDBD adapts to be
competitive with the best fixed LMS.

Usage:
    python examples/step1_idbd_vs_lms.py
"""

import numpy as np

from alberta_toolkit import (
    Autostep,
    IDBD,
    LMS,
    LinearLearner,
    RandomWalkTarget,
    compare_learners,
    compute_tracking_error,
    run_learning_loop,
)


def run_experiment(
    feature_dim: int = 10,
    num_steps: int = 10000,
    drift_rate: float = 0.001,
    noise_std: float = 0.1,
    seed: int = 42,
) -> dict:
    """Run the IDBD vs LMS comparison experiment.

    Args:
        feature_dim: Dimension of feature vectors
        num_steps: Number of learning steps
        drift_rate: How fast the target weights change
        noise_std: Observation noise level
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results for each learner
    """
    # LMS step-sizes to try (grid search for best fixed rate)
    lms_step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    results = {}

    # Run LMS with each step-size
    for alpha in lms_step_sizes:
        stream = RandomWalkTarget(
            feature_dim=feature_dim,
            drift_rate=drift_rate,
            noise_std=noise_std,
            seed=seed,
        )
        learner = LinearLearner(optimizer=LMS(step_size=alpha))
        _, metrics = run_learning_loop(learner, stream, num_steps)
        results[f"LMS(α={alpha})"] = metrics

    # Run IDBD with various meta step-sizes
    idbd_configs = [
        (0.01, 0.01),   # Conservative
        (0.05, 0.05),   # Moderate
        (0.1, 0.1),     # Aggressive
        (0.05, 0.1),    # High meta, moderate initial
    ]

    for initial_alpha, beta in idbd_configs:
        stream = RandomWalkTarget(
            feature_dim=feature_dim,
            drift_rate=drift_rate,
            noise_std=noise_std,
            seed=seed,
        )
        learner = LinearLearner(
            optimizer=IDBD(initial_step_size=initial_alpha, meta_step_size=beta)
        )
        _, metrics = run_learning_loop(learner, stream, num_steps)
        results[f"IDBD(α₀={initial_alpha},β={beta})"] = metrics

    # Run Autostep with various configurations
    autostep_configs = [
        (0.01, 0.01),   # Conservative
        (0.05, 0.05),   # Moderate
        (0.1, 0.1),     # Aggressive
    ]

    for initial_alpha, mu in autostep_configs:
        stream = RandomWalkTarget(
            feature_dim=feature_dim,
            drift_rate=drift_rate,
            noise_std=noise_std,
            seed=seed,
        )
        learner = LinearLearner(
            optimizer=Autostep(initial_step_size=initial_alpha, meta_step_size=mu)
        )
        _, metrics = run_learning_loop(learner, stream, num_steps)
        results[f"Autostep(α₀={initial_alpha},μ={mu})"] = metrics

    return results


def print_results(results: dict) -> None:
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 70)
    print("Step 1 Experiment: IDBD vs LMS on Random Walk Target")
    print("=" * 70)

    # Compute summary statistics
    summary = compare_learners(results)

    # Sort by cumulative error
    sorted_learners = sorted(summary.items(), key=lambda x: x[1]["cumulative"])

    print(f"\n{'Learner':<28} {'Cumulative Error':>16} {'Mean SE':>12} {'Final 100':>12}")
    print("-" * 72)

    for name, stats in sorted_learners:
        print(
            f"{name:<28} {stats['cumulative']:>16.2f} "
            f"{stats['mean']:>12.6f} {stats['final_100_mean']:>12.6f}"
        )

    # Find best LMS, IDBD, and Autostep
    best_lms = None
    best_lms_error = float("inf")
    best_idbd = None
    best_idbd_error = float("inf")
    best_autostep = None
    best_autostep_error = float("inf")

    for name, stats in summary.items():
        if name.startswith("LMS"):
            if stats["cumulative"] < best_lms_error:
                best_lms_error = stats["cumulative"]
                best_lms = name
        elif name.startswith("IDBD"):
            if stats["cumulative"] < best_idbd_error:
                best_idbd_error = stats["cumulative"]
                best_idbd = name
        elif name.startswith("Autostep"):
            if stats["cumulative"] < best_autostep_error:
                best_autostep_error = stats["cumulative"]
                best_autostep = name

    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print(f"  Best LMS:      {best_lms} with cumulative error {best_lms_error:.2f}")
    print(f"  Best IDBD:     {best_idbd} with cumulative error {best_idbd_error:.2f}")
    if best_autostep:
        print(f"  Best Autostep: {best_autostep} with cumulative error {best_autostep_error:.2f}")

    # Determine best adaptive method
    best_adaptive_name = best_idbd
    best_adaptive_error = best_idbd_error
    if best_autostep and best_autostep_error < best_idbd_error:
        best_adaptive_name = best_autostep
        best_adaptive_error = best_autostep_error

    if best_adaptive_error <= best_lms_error:
        improvement = (best_lms_error - best_adaptive_error) / best_lms_error * 100
        print(f"\n  SUCCESS: {best_adaptive_name} beats best hand-tuned LMS by {improvement:.1f}%")
        print("  Step 1 success criterion MET: Meta-learner beats manual tuning!")
    else:
        degradation = (best_adaptive_error - best_lms_error) / best_lms_error * 100
        print(f"\n  Best adaptive method is {degradation:.1f}% worse than best LMS")
        print("  Consider adjusting meta-parameters or experiment settings")

    print("=" * 70 + "\n")


def plot_learning_curves(results: dict, save_path: str | None = None) -> None:
    """Plot learning curves (requires matplotlib).

    Args:
        results: Dictionary of metrics from run_experiment
        save_path: If provided, save plot to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot tracking error (running mean of squared error)
    for name, metrics in results.items():
        tracking_error = compute_tracking_error(metrics, window_size=100)
        ax1.plot(tracking_error, label=name, alpha=0.8)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Tracking Error (Running Mean SE)")
    ax1.set_title("Tracking Error Over Time")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Plot cumulative error
    for name, metrics in results.items():
        cumulative = np.cumsum([m["squared_error"] for m in metrics])
        ax2.plot(cumulative, label=name, alpha=0.8)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cumulative Squared Error")
    ax2.set_title("Cumulative Error Over Time")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def run_practical_comparison(
    feature_dim: int = 10,
    num_steps: int = 20000,
    initial_step_size: float = 0.01,
    seed: int = 42,
) -> None:
    """Run the practical comparison: same initial step-size for both.

    This demonstrates IDBD's key practical advantage: you don't need to
    grid search for the optimal step-size.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL COMPARISON: Same Starting Step-Size")
    print("=" * 70)
    print(f"\nBoth LMS and IDBD start with step-size = {initial_step_size}")
    print("IDBD can adapt; LMS is stuck.\n")

    # LMS stuck at initial step-size
    stream = RandomWalkTarget(
        feature_dim=feature_dim, drift_rate=0.001, noise_std=0.1, seed=seed
    )
    learner = LinearLearner(optimizer=LMS(step_size=initial_step_size))
    _, lms_metrics = run_learning_loop(learner, stream, num_steps)

    # IDBD starting at same step-size but can adapt
    stream = RandomWalkTarget(
        feature_dim=feature_dim, drift_rate=0.001, noise_std=0.1, seed=seed
    )
    learner = LinearLearner(
        optimizer=IDBD(initial_step_size=initial_step_size, meta_step_size=0.05)
    )
    _, idbd_metrics = run_learning_loop(learner, stream, num_steps)

    lms_cumulative = sum(m["squared_error"] for m in lms_metrics)
    idbd_cumulative = sum(m["squared_error"] for m in idbd_metrics)
    lms_final = sum(m["squared_error"] for m in lms_metrics[-100:]) / 100
    idbd_final = sum(m["squared_error"] for m in idbd_metrics[-100:]) / 100

    print(f"{'Method':<25} {'Cumulative Error':>16} {'Final 100 Mean':>16}")
    print("-" * 60)
    print(f"{'LMS (stuck at ' + str(initial_step_size) + ')':<25} {lms_cumulative:>16.2f} {lms_final:>16.6f}")
    print(f"{'IDBD (adapts)':<25} {idbd_cumulative:>16.2f} {idbd_final:>16.6f}")

    if idbd_cumulative < lms_cumulative:
        improvement = (lms_cumulative - idbd_cumulative) / lms_cumulative * 100
        print(f"\nSUCCESS: IDBD beats fixed LMS by {improvement:.1f}%")
        print("IDBD adapts its step-sizes to track the non-stationary target better.")
    print("=" * 70 + "\n")


def main():
    """Run the Step 1 demonstration."""
    print("Running Step 1 experiment: IDBD vs LMS comparison")
    print("This demonstrates meta-learned step-sizes vs manual tuning.\n")

    # Experiment 1: Grid search comparison
    results = run_experiment(
        feature_dim=10,
        num_steps=10000,
        drift_rate=0.001,
        noise_std=0.1,
        seed=42,
    )

    print_results(results)

    # Experiment 2: Practical comparison - same starting step-size
    run_practical_comparison(initial_step_size=0.01)

    # Try to plot if matplotlib is available
    plot_learning_curves(results)


if __name__ == "__main__":
    main()
