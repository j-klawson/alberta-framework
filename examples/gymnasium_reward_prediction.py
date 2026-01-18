#!/usr/bin/env python3
"""Gymnasium Reward Prediction Example: IDBD vs LMS on CartPole.

This example demonstrates using Gymnasium environments as experience streams
for the Alberta Framework's learners. We compare IDBD's adaptive step-sizes
against fixed LMS on predicting immediate rewards in CartPole.

The task: Given the current state and action, predict the immediate reward.

Usage:
    pip install gymnasium
    python examples/gymnasium_reward_prediction.py
"""

import numpy as np

try:
    import gymnasium
except ImportError:
    raise ImportError(
        "gymnasium is required for this example. "
        "Install it with: pip install gymnasium"
    )

from alberta_framework import IDBD, LMS, LinearLearner, compare_learners
from alberta_framework.streams.gymnasium import (
    GymnasiumStream,
    PredictionMode,
    collect_trajectory,
    learn_from_trajectory,
    make_gymnasium_stream,
)


def run_reward_prediction_experiment(
    env_id: str = "CartPole-v1",
    num_steps: int = 10000,
    seed: int = 42,
) -> dict:
    """Run reward prediction experiment comparing IDBD vs LMS.

    Uses trajectory collection for efficient scan-based learning.

    Args:
        env_id: Gymnasium environment ID
        num_steps: Number of learning steps
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results for each learner
    """
    results = {}

    # Collect trajectory once (reused for all learners)
    env = gymnasium.make(env_id)
    observations, targets = collect_trajectory(
        env=env,
        policy=None,  # Random policy
        num_steps=num_steps,
        mode=PredictionMode.REWARD,
        include_action_in_features=True,
        seed=seed,
    )
    env.close()

    # LMS with various step-sizes
    lms_step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]

    for alpha in lms_step_sizes:
        learner = LinearLearner(optimizer=LMS(step_size=alpha))
        _, metrics = learn_from_trajectory(learner, observations, targets)
        # Convert metrics array to list of dicts
        metrics_list = [
            {"squared_error": float(metrics[i, 0]), "error": float(metrics[i, 1])}
            for i in range(metrics.shape[0])
        ]
        results[f"LMS(α={alpha})"] = metrics_list

    # IDBD with various configurations
    idbd_configs = [
        (0.01, 0.01),   # Conservative
        (0.01, 0.05),   # Higher meta learning rate
        (0.05, 0.05),   # Moderate
    ]

    for initial_alpha, beta in idbd_configs:
        learner = LinearLearner(
            optimizer=IDBD(initial_step_size=initial_alpha, meta_step_size=beta)
        )
        _, metrics = learn_from_trajectory(learner, observations, targets)
        metrics_list = [
            {"squared_error": float(metrics[i, 0]), "error": float(metrics[i, 1])}
            for i in range(metrics.shape[0])
        ]
        results[f"IDBD(α₀={initial_alpha},β={beta})"] = metrics_list

    return results


def print_results(results: dict, env_id: str) -> None:
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"Reward Prediction Experiment: {env_id}")
    print("=" * 70)
    print("Task: Predict immediate reward from (state, action)")
    print()

    # Compute summary statistics
    summary = compare_learners(results)

    # Sort by cumulative error
    sorted_learners = sorted(summary.items(), key=lambda x: x[1]["cumulative"])

    print(f"{'Learner':<28} {'Cumulative Error':>16} {'Mean SE':>12} {'Final 100':>12}")
    print("-" * 72)

    for name, stats in sorted_learners:
        print(
            f"{name:<28} {stats['cumulative']:>16.2f} "
            f"{stats['mean']:>12.6f} {stats['final_100_mean']:>12.6f}"
        )

    # Find best LMS and IDBD
    best_lms = None
    best_lms_error = float("inf")
    best_idbd = None
    best_idbd_error = float("inf")

    for name, stats in summary.items():
        if name.startswith("LMS"):
            if stats["cumulative"] < best_lms_error:
                best_lms_error = stats["cumulative"]
                best_lms = name
        elif name.startswith("IDBD"):
            if stats["cumulative"] < best_idbd_error:
                best_idbd_error = stats["cumulative"]
                best_idbd = name

    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print(f"  Best LMS:  {best_lms} with cumulative error {best_lms_error:.2f}")
    print(f"  Best IDBD: {best_idbd} with cumulative error {best_idbd_error:.2f}")

    if best_idbd_error <= best_lms_error:
        improvement = (best_lms_error - best_idbd_error) / best_lms_error * 100
        print(f"\n  IDBD beats best hand-tuned LMS by {improvement:.1f}%")
    else:
        degradation = (best_idbd_error - best_lms_error) / best_lms_error * 100
        print(f"\n  Best IDBD is {degradation:.1f}% worse than best LMS")

    print("=" * 70 + "\n")


def plot_learning_curves(results: dict, env_id: str, save_path: str | None = None) -> None:
    """Plot learning curves (requires matplotlib).

    Args:
        results: Dictionary of metrics from run_experiment
        env_id: Environment ID for plot title
        save_path: If provided, save plot to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    from alberta_framework import compute_tracking_error

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot tracking error (running mean of squared error)
    for name, metrics in results.items():
        tracking_error = compute_tracking_error(metrics, window_size=100)
        ax1.plot(tracking_error, label=name, alpha=0.8)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Tracking Error (Running Mean SE)")
    ax1.set_title(f"Tracking Error: {env_id} Reward Prediction")
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


def demo_stream_usage():
    """Demonstrate basic GymnasiumStream usage."""
    print("=" * 70)
    print("Demo: Basic GymnasiumStream Usage")
    print("=" * 70)

    # Create a stream from CartPole
    env = gymnasium.make("CartPole-v1")
    stream = GymnasiumStream(
        env,
        mode=PredictionMode.REWARD,
        include_action_in_features=True,
        seed=42,
    )

    print(f"\nEnvironment: CartPole-v1")
    print(f"Feature dimension: {stream.feature_dim}")
    print(f"Target dimension: {stream.target_dim}")
    print(f"Prediction mode: {stream.mode}")

    # Generate a few timesteps
    print("\nFirst 5 timesteps:")
    for i, timestep in enumerate(stream):
        if i >= 5:
            break
        print(f"  Step {i}: obs shape={timestep.observation.shape}, "
              f"target={float(timestep.target[0]):.1f}")

    print(f"\nAfter 5 steps: {stream.episode_count} episodes completed")
    print("=" * 70 + "\n")


def main():
    """Run the Gymnasium reward prediction example."""
    print("Gymnasium Reward Prediction Example")
    print("Comparing IDBD vs LMS on CartPole reward prediction\n")

    # Demo basic stream usage
    demo_stream_usage()

    # Run main experiment
    env_id = "CartPole-v1"
    results = run_reward_prediction_experiment(
        env_id=env_id,
        num_steps=10000,
        seed=42,
    )

    print_results(results, env_id)

    # Try to plot if matplotlib is available
    plot_learning_curves(results, env_id)


if __name__ == "__main__":
    main()
