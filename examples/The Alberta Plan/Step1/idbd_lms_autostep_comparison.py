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
    python examples/step1_idbd_vs_lms.py --output-dir output/
"""

import argparse
from pathlib import Path

import jax.random as jr
import numpy as np

from alberta_framework import (
    AbruptChangeStream,
    Autostep,
    IDBD,
    LMS,
    LinearLearner,
    RandomWalkStream,
    Timer,
    compare_learners,
    compute_tracking_error,
    metrics_to_dicts,
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
        stream = RandomWalkStream(
            feature_dim=feature_dim,
            drift_rate=drift_rate,
            noise_std=noise_std,
        )
        learner = LinearLearner(optimizer=LMS(step_size=alpha))
        key = jr.key(seed)
        _, metrics = run_learning_loop(learner, stream, num_steps, key)
        results[f"LMS(α={alpha})"] = metrics_to_dicts(metrics)

    # Run IDBD with various meta step-sizes
    idbd_configs = [
        (0.01, 0.01),   # Conservative
        (0.05, 0.05),   # Moderate
        (0.1, 0.1),     # Aggressive
        (0.05, 0.1),    # High meta, moderate initial
    ]

    for initial_alpha, beta in idbd_configs:
        stream = RandomWalkStream(
            feature_dim=feature_dim,
            drift_rate=drift_rate,
            noise_std=noise_std,
        )
        learner = LinearLearner(
            optimizer=IDBD(initial_step_size=initial_alpha, meta_step_size=beta)
        )
        key = jr.key(seed)
        _, metrics = run_learning_loop(learner, stream, num_steps, key)
        results[f"IDBD(α₀={initial_alpha},β={beta})"] = metrics_to_dicts(metrics)

    # Run Autostep with various configurations
    autostep_configs = [
        (0.01, 0.01),   # Conservative
        (0.05, 0.05),   # Moderate
        (0.1, 0.1),     # Aggressive
    ]

    for initial_alpha, mu in autostep_configs:
        stream = RandomWalkStream(
            feature_dim=feature_dim,
            drift_rate=drift_rate,
            noise_std=noise_std,
        )
        learner = LinearLearner(
            optimizer=Autostep(initial_step_size=initial_alpha, meta_step_size=mu)
        )
        key = jr.key(seed)
        _, metrics = run_learning_loop(learner, stream, num_steps, key)
        results[f"Autostep(α₀={initial_alpha},μ={mu})"] = metrics_to_dicts(metrics)

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
    stream = RandomWalkStream(
        feature_dim=feature_dim, drift_rate=0.001, noise_std=0.1
    )
    learner = LinearLearner(optimizer=LMS(step_size=initial_step_size))
    key = jr.key(seed)
    _, lms_metrics = run_learning_loop(learner, stream, num_steps, key)
    lms_metrics = metrics_to_dicts(lms_metrics)

    # IDBD starting at same step-size but can adapt
    stream = RandomWalkStream(
        feature_dim=feature_dim, drift_rate=0.001, noise_std=0.1
    )
    learner = LinearLearner(
        optimizer=IDBD(initial_step_size=initial_step_size, meta_step_size=0.05)
    )
    key = jr.key(seed)
    _, idbd_metrics = run_learning_loop(learner, stream, num_steps, key)
    idbd_metrics = metrics_to_dicts(idbd_metrics)

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


def run_abrupt_change_experiment(
    feature_dim: int = 10,
    num_steps: int = 2100,
    change_interval: int = 2000,
    noise_std: float = 0.1,
    seed: int = 42,
) -> dict:
    """Run experiment with abrupt target change to compare adaptation speed.

    The experiment is configured so an abrupt change occurs near the end,
    allowing us to visualize recovery in the final steps.

    Args:
        feature_dim: Dimension of feature vectors
        num_steps: Number of learning steps
        change_interval: Steps between abrupt weight changes
        noise_std: Observation noise level
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results for each learner
    """
    results = {}

    # Shared configuration
    stream_kwargs = {
        "feature_dim": feature_dim,
        "change_interval": change_interval,
        "noise_std": noise_std,
    }

    # LMS with various step-sizes
    for alpha in [0.01, 0.02, 0.05]:
        stream = AbruptChangeStream(**stream_kwargs)
        learner = LinearLearner(optimizer=LMS(step_size=alpha))
        key = jr.key(seed)
        _, metrics = run_learning_loop(learner, stream, num_steps, key)
        results[f"LMS(α={alpha})"] = metrics_to_dicts(metrics)

    # IDBD with moderate meta step-size
    stream = AbruptChangeStream(**stream_kwargs)
    learner = LinearLearner(
        optimizer=IDBD(initial_step_size=0.02, meta_step_size=0.05)
    )
    key = jr.key(seed)
    _, metrics = run_learning_loop(learner, stream, num_steps, key)
    results["IDBD(α₀=0.02,β=0.05)"] = metrics_to_dicts(metrics)

    # Autostep with moderate settings
    stream = AbruptChangeStream(**stream_kwargs)
    learner = LinearLearner(
        optimizer=Autostep(initial_step_size=0.02, meta_step_size=0.05)
    )
    key = jr.key(seed)
    _, metrics = run_learning_loop(learner, stream, num_steps, key)
    results["Autostep(α₀=0.02,μ=0.05)"] = metrics_to_dicts(metrics)

    return results


def plot_abrupt_change_recovery(
    results: dict,
    last_n_steps: int = 100,
    save_path: str | None = None,
) -> None:
    """Plot the last N steps after an abrupt change to show recovery.

    Args:
        results: Dictionary of metrics from run_abrupt_change_experiment
        last_n_steps: Number of final steps to plot
        save_path: If provided, save plot to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    colors = {"LMS": "blue", "IDBD": "green", "Autostep": "red"}

    # Plot 1: Squared error over the last N steps
    for name, metrics in results.items():
        errors = [m["squared_error"] for m in metrics[-last_n_steps:]]
        color = next((c for key, c in colors.items() if key in name), "gray")
        ax1.plot(range(last_n_steps), errors, label=name, alpha=0.8, color=color)

    ax1.set_xlabel("Steps After Abrupt Change")
    ax1.set_ylabel("Squared Error")
    ax1.set_title(f"Recovery After Abrupt Target Change (Last {last_n_steps} Steps)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Cumulative error over the last N steps (shows total cost of recovery)
    for name, metrics in results.items():
        errors = [m["squared_error"] for m in metrics[-last_n_steps:]]
        cumulative = np.cumsum(errors)
        color = next((c for key, c in colors.items() if key in name), "gray")
        ax2.plot(range(last_n_steps), cumulative, label=name, alpha=0.8, color=color)

    ax2.set_xlabel("Steps After Abrupt Change")
    ax2.set_ylabel("Cumulative Squared Error")
    ax2.set_title(f"Cumulative Error During Recovery (Last {last_n_steps} Steps)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Add title
    fig.suptitle(
        "Recovery After Abrupt Target Change",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Abrupt change plot saved to {save_path}")
    else:
        plt.show()


def main(output_dir: str | None = None):
    """Run the Step 1 demonstration.

    Args:
        output_dir: If provided, save plots to this directory instead of showing.
    """
    with Timer("Total experiment runtime"):
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

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
        save_path = str(output_path / "idbd_vs_lms.png") if output_dir else None
        plot_learning_curves(results, save_path=save_path)

        # Experiment 3: Abrupt change recovery comparison
        print("\n" + "=" * 70)
        print("ABRUPT CHANGE EXPERIMENT: Recovery After Target Shift")
        print("=" * 70)
        print("\nThis experiment shows how each method recovers after an abrupt")
        print("target change at step 2000. The last 100 steps show recovery behavior.\n")

        abrupt_results = run_abrupt_change_experiment(
            feature_dim=10,
            num_steps=2100,
            change_interval=2000,
            noise_std=0.1,
            seed=42,
        )

        # Print recovery statistics
        print(f"{'Method':<28} {'Last 100 Mean SE':>16} {'Last 100 Cumulative':>20}")
        print("-" * 68)
        stats = {}
        for name, metrics in abrupt_results.items():
            last_100 = [m["squared_error"] for m in metrics[-100:]]
            mean_se = np.mean(last_100)
            cumulative = np.sum(last_100)
            stats[name] = cumulative
            print(f"{name:<28} {mean_se:>16.4f} {cumulative:>20.2f}")

        # Analyze results
        print("\n" + "-" * 68)
        best_method = min(stats, key=stats.get)
        print(f"Best recovery: {best_method}")

        # Compare adaptive methods to best LMS
        lms_methods = {k: v for k, v in stats.items() if k.startswith("LMS")}
        best_lms = min(lms_methods, key=lms_methods.get)
        best_lms_error = lms_methods[best_lms]

        for name, error in stats.items():
            if not name.startswith("LMS"):
                if error < best_lms_error:
                    pct = (best_lms_error - error) / best_lms_error * 100
                    print(f"{name} beats {best_lms} by {pct:.1f}%")
                else:
                    pct = (error - best_lms_error) / best_lms_error * 100
                    print(f"{name} is {pct:.1f}% worse than {best_lms}")

        print("=" * 70 + "\n")

        # Plot abrupt change recovery
        abrupt_save_path = str(output_path / "abrupt_change_recovery.png") if output_dir else None
        plot_abrupt_change_recovery(abrupt_results, last_n_steps=100, save_path=abrupt_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1 Demonstration: IDBD vs LMS"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: show interactively)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
