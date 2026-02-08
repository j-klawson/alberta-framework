#!/usr/bin/env python3
"""Step 2 Demonstration: Linear vs MLP on Non-Stationary Targets.

This script compares LinearLearner+Autostep (Step 1) against MLPLearner+ObGD (Step 2)
on various non-stationary supervised learning problems. The goal is to demonstrate that
nonlinear function approximation with ObGD's overshooting prevention can work in the
streaming setting.

The experiments:
1. Random walk drift: Continuous target drift
2. Abrupt changes: Sudden target shifts
3. Dynamic scale shifts: Features with time-varying scales

Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning Finally Works"

Usage:
    python "examples/The Alberta Plan/Step2/linear_vs_mlp_comparison.py"
    python "examples/The Alberta Plan/Step2/linear_vs_mlp_comparison.py" --output-dir output/
"""

import argparse
from pathlib import Path

import jax.random as jr
import numpy as np

from alberta_framework import (
    AbruptChangeStream,
    Autostep,
    DynamicScaleShiftStream,
    LinearLearner,
    MLPLearner,
    ObGD,
    ObGDBounding,
    RandomWalkStream,
    Timer,
    metrics_to_dicts,
    run_learning_loop,
    run_mlp_learning_loop,
)


def run_comparison(
    stream_name: str,
    stream,
    num_steps: int = 10000,
    num_seeds: int = 5,
) -> dict:
    """Run Linear vs MLP comparison on a given stream.

    Args:
        stream_name: Name for display
        stream: Experience stream to use
        num_steps: Number of training steps
        num_seeds: Number of random seeds for averaging

    Returns:
        Dictionary mapping learner name to list of per-seed metrics
    """
    configs = {
        "Linear+Autostep": lambda: LinearLearner(
            optimizer=Autostep(initial_step_size=0.1, meta_step_size=0.05)
        ),
        "Linear+ObGD": lambda: LinearLearner(
            optimizer=ObGD(step_size=1.0, kappa=2.0)
        ),
        "MLP(128)+ObGD": lambda: MLPLearner(
            hidden_sizes=(128,), step_size=1.0, bounder=ObGDBounding(kappa=2.0), sparsity=0.9
        ),
        "MLP(128,128)+ObGD": lambda: MLPLearner(
            hidden_sizes=(128, 128), step_size=1.0, bounder=ObGDBounding(kappa=2.0), sparsity=0.9
        ),
    }

    results = {}
    for name, make_learner in configs.items():
        seed_metrics = []
        for seed in range(num_seeds):
            key = jr.key(seed)
            learner = make_learner()

            if isinstance(learner, MLPLearner):
                _, metrics = run_mlp_learning_loop(
                    learner, stream, num_steps, key
                )
            else:
                _, metrics = run_learning_loop(
                    learner, stream, num_steps, key
                )

            seed_metrics.append(metrics_to_dicts(metrics))

        results[name] = seed_metrics

    return results


def print_comparison(stream_name: str, results: dict) -> None:
    """Print comparison table for a stream."""
    print(f"\n{'=' * 70}")
    print(f"Stream: {stream_name}")
    print(f"{'=' * 70}")

    print(f"\n{'Learner':<25} {'Mean SE':>12} {'Std SE':>12} {'Final 500 Mean':>16}")
    print("-" * 68)

    for name, seed_metrics_list in results.items():
        # Compute mean squared error across seeds
        all_se = []
        all_final = []
        for metrics in seed_metrics_list:
            mean_se = np.mean([m["squared_error"] for m in metrics])
            final_se = np.mean([m["squared_error"] for m in metrics[-500:]])
            all_se.append(mean_se)
            all_final.append(final_se)

        mean = np.mean(all_se)
        std = np.std(all_se)
        final_mean = np.mean(all_final)

        print(f"{name:<25} {mean:>12.6f} {std:>12.6f} {final_mean:>16.6f}")


def plot_comparison(
    stream_name: str,
    results: dict,
    save_path: str | None = None,
) -> None:
    """Plot learning curves with confidence intervals.

    Args:
        stream_name: Name for plot title
        results: Dict mapping learner names to list of per-seed metrics
        save_path: If provided, save plot instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Linear+Autostep": "blue",
        "Linear+ObGD": "orange",
        "MLP(128)+ObGD": "green",
        "MLP(128,128)+ObGD": "red",
    }

    window = 200

    for name, seed_metrics_list in results.items():
        # Compute running mean of squared error per seed
        all_curves = []
        for metrics in seed_metrics_list:
            se = np.array([m["squared_error"] for m in metrics])
            # Running mean
            cumsum = np.cumsum(se)
            running = np.zeros_like(se)
            for i in range(len(se)):
                start = max(0, i - window + 1)
                running[i] = (cumsum[i] - (cumsum[start - 1] if start > 0 else 0)) / (i - start + 1)
            all_curves.append(running)

        curves = np.array(all_curves)
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        color = colors.get(name, "gray")
        ax.plot(mean_curve, label=name, color=color, alpha=0.9)
        ax.fill_between(
            range(len(mean_curve)),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Tracking Error (Running Mean SE)")
    ax.set_title(f"Linear vs MLP: {stream_name}")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main(output_dir: str | None = None):
    """Run the Step 2 demonstration."""
    with Timer("Total Step 2 experiment runtime"):
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        print("Step 2: Linear vs MLP Comparison")
        print("Comparing LinearLearner+Autostep vs MLPLearner+ObGD")
        print("on non-stationary supervised learning problems.\n")

        feature_dim = 10
        num_steps = 10000
        num_seeds = 5

        # Experiment 1: Random Walk
        print("Running Experiment 1: Random Walk Drift...")
        stream1 = RandomWalkStream(
            feature_dim=feature_dim, drift_rate=0.001, noise_std=0.1
        )
        results1 = run_comparison("Random Walk", stream1, num_steps, num_seeds)
        print_comparison("Random Walk (drift_rate=0.001)", results1)
        save1 = str(output_path / "step2_random_walk.png") if output_dir else None
        plot_comparison("Random Walk", results1, save_path=save1)

        # Experiment 2: Abrupt Changes
        print("\nRunning Experiment 2: Abrupt Changes...")
        stream2 = AbruptChangeStream(
            feature_dim=feature_dim, change_interval=2000, noise_std=0.1
        )
        results2 = run_comparison("Abrupt Change", stream2, num_steps, num_seeds)
        print_comparison("Abrupt Changes (interval=2000)", results2)
        save2 = str(output_path / "step2_abrupt_change.png") if output_dir else None
        plot_comparison("Abrupt Changes", results2, save_path=save2)

        # Experiment 3: Dynamic Scale Shifts
        print("\nRunning Experiment 3: Dynamic Scale Shifts...")
        stream3 = DynamicScaleShiftStream(
            feature_dim=feature_dim,
            scale_change_interval=3000,
            weight_change_interval=2000,
            min_scale=0.01,
            max_scale=100.0,
        )
        results3 = run_comparison("Dynamic Scale Shift", stream3, num_steps, num_seeds)
        print_comparison("Dynamic Scale Shifts", results3)
        save3 = str(output_path / "step2_dynamic_scale.png") if output_dir else None
        plot_comparison("Dynamic Scale Shifts", results3, save_path=save3)

        print("\n" + "=" * 70)
        print("Step 2 experiments complete.")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2 Demonstration: Linear vs MLP"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: show interactively)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
