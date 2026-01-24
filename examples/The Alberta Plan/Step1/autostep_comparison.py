#!/usr/bin/env python3
"""Step 1 Autostep Comparison: IDBD vs Autostep vs LMS.

This script provides a comprehensive comparison of the three step-size
adaptation strategies available in the Alberta Framework:

1. LMS: Fixed step-size (requires manual tuning)
2. IDBD: Meta-learned step-sizes via gradient correlation
3. Autostep: Tuning-free adaptation with gradient normalization

The key difference between IDBD and Autostep:
- IDBD adapts step-sizes based on gradient correlation
- Autostep additionally normalizes gradients, making it more robust

This script generates publication-quality plots including:
- Learning curves (tracking error over time)
- Step-size evolution for adaptive methods
- Stream visualization (target weight behavior)
- Final performance comparison bar charts

References:
- Sutton 1992, "Adapting Bias by Gradient Descent"
- Mahmood et al. 2012, "Tuning-free step-size adaptation"

Usage:
    python autostep_comparison.py
    python autostep_comparison.py --output-dir output/
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

from alberta_framework import (
    Autostep,
    IDBD,
    LMS,
    LinearLearner,
    AbruptChangeStream,
    CyclicStream,
    RandomWalkStream,
    StepSizeTrackingConfig,
    Timer,
    compare_learners,
    compute_tracking_error,
    metrics_to_dicts,
    run_learning_loop,
)


def generate_stream_trajectory(
    stream_class,
    feature_dim: int = 10,
    num_steps: int = 10000,
    seed: int = 42,
    **stream_kwargs,
) -> np.ndarray:
    """Generate trajectory of true weights for stream visualization.

    Returns:
        Array of shape (num_steps, feature_dim) with true weight values over time.
    """
    key = jr.key(seed)
    stream = stream_class(feature_dim=feature_dim, **stream_kwargs)
    state = stream.init(key)

    # Collect true weights at each step
    weights_history = []

    def step_fn(state, idx):
        _, new_state = stream.step(state, idx)
        # Extract true_weights from state (different streams have different state structures)
        if hasattr(new_state, "true_weights"):
            weights = new_state.true_weights
        elif hasattr(new_state, "signs"):
            # For SuttonExperiment1Stream, construct weights from signs
            weights = jnp.concatenate([new_state.signs, jnp.zeros(15)])
        elif hasattr(new_state, "configurations"):
            # For CyclicStream, get current configuration
            config_idx = (new_state.step_count // stream._cycle_length) % stream._num_configurations
            weights = new_state.configurations[config_idx]
        else:
            weights = jnp.zeros(feature_dim)
        return new_state, weights

    _, weights_history = lax.scan(step_fn, state, jnp.arange(num_steps))
    return np.array(weights_history)


def run_comparison_with_tracking(
    stream_class,
    stream_name: str,
    feature_dim: int = 10,
    num_steps: int = 10000,
    seed: int = 42,
    tracking_interval: int = 100,
    **stream_kwargs,
) -> dict:
    """Run comparison across all optimizers with step-size tracking.

    Returns:
        Dictionary with results including metrics and step-size history.
    """
    results = {}
    tracking_config = StepSizeTrackingConfig(interval=tracking_interval)

    # Best LMS configuration for comparison
    stream = stream_class(feature_dim=feature_dim, **stream_kwargs)
    learner = LinearLearner(optimizer=LMS(step_size=0.05))
    key = jr.key(seed)
    state, metrics, history = run_learning_loop(
        learner, stream, num_steps, key, step_size_tracking=tracking_config
    )
    results["LMS(α=0.05)"] = {
        "metrics": metrics_to_dicts(metrics),
        "step_sizes": np.array(history.step_sizes),
        "recording_indices": np.array(history.recording_indices),
        "optimizer_type": "LMS",
    }

    # IDBD - best configuration
    stream = stream_class(feature_dim=feature_dim, **stream_kwargs)
    learner = LinearLearner(
        optimizer=IDBD(initial_step_size=0.05, meta_step_size=0.05)
    )
    key = jr.key(seed)
    state, metrics, history = run_learning_loop(
        learner, stream, num_steps, key, step_size_tracking=tracking_config
    )
    results["IDBD(α₀=0.05,β=0.05)"] = {
        "metrics": metrics_to_dicts(metrics),
        "step_sizes": np.array(history.step_sizes),
        "recording_indices": np.array(history.recording_indices),
        "optimizer_type": "IDBD",
    }

    # Autostep - best configuration
    stream = stream_class(feature_dim=feature_dim, **stream_kwargs)
    learner = LinearLearner(
        optimizer=Autostep(initial_step_size=0.05, meta_step_size=0.05)
    )
    key = jr.key(seed)
    state, metrics, history = run_learning_loop(
        learner, stream, num_steps, key, step_size_tracking=tracking_config
    )
    results["Autostep(α₀=0.05,μ=0.05)"] = {
        "metrics": metrics_to_dicts(metrics),
        "step_sizes": np.array(history.step_sizes),
        "recording_indices": np.array(history.recording_indices),
        "optimizer_type": "Autostep",
    }

    return results


def print_comparison_results(results: dict, stream_name: str) -> dict:
    """Print and analyze comparison results."""
    print(f"\n{'='*70}")
    print(f"Results on {stream_name}")
    print("=" * 70)

    # Extract just metrics for comparison
    metrics_only = {name: data["metrics"] for name, data in results.items()}
    summary = compare_learners(metrics_only)
    sorted_learners = sorted(summary.items(), key=lambda x: x[1]["cumulative"])

    print(f"\n{'Optimizer':<30} {'Cumulative':>14} {'Mean SE':>12} {'Final 100':>12}")
    print("-" * 70)

    for name, stats in sorted_learners:
        print(
            f"{name:<30} {stats['cumulative']:>14.2f} "
            f"{stats['mean']:>12.6f} {stats['final_100_mean']:>12.6f}"
        )

    return summary


def plot_comparison_figure(
    results: dict,
    stream_name: str,
    weights_trajectory: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Create a multi-panel comparison figure.

    Panels:
    1. Stream visualization (target weight evolution)
    2. Learning curves (tracking error)
    3. Step-size evolution for IDBD and Autostep
    4. Final performance bar chart
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    # Set up publication style
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    })

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    colors = {
        "LMS": "#1f77b4",      # Blue
        "IDBD": "#ff7f0e",     # Orange
        "Autostep": "#2ca02c", # Green
    }

    # Panel 1: Stream visualization (target weight evolution)
    ax1 = fig.add_subplot(gs[0, 0])
    num_weights_to_show = min(5, weights_trajectory.shape[1])
    for i in range(num_weights_to_show):
        ax1.plot(weights_trajectory[:, i], alpha=0.7, linewidth=0.8, label=f"w_{i+1}")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("True Weight Value")
    ax1.set_title(f"Target Weight Evolution: {stream_name}")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Learning curves (tracking error)
    ax2 = fig.add_subplot(gs[0, 1])
    for name, data in results.items():
        opt_type = data["optimizer_type"]
        tracking_error = compute_tracking_error(data["metrics"], window_size=100)
        ax2.plot(tracking_error, label=name, color=colors[opt_type], alpha=0.8)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Tracking Error (Running Mean SE)")
    ax2.set_title("Learning Curves: Tracking Error Over Time")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Step-size evolution
    ax3 = fig.add_subplot(gs[1, 0])

    for name, data in results.items():
        opt_type = data["optimizer_type"]
        step_sizes = data["step_sizes"]  # Shape: (num_recordings, num_weights)
        indices = data["recording_indices"]

        # Plot mean step-size with shaded region for min/max
        mean_ss = np.mean(step_sizes, axis=1)
        min_ss = np.min(step_sizes, axis=1)
        max_ss = np.max(step_sizes, axis=1)

        ax3.plot(indices, mean_ss, label=f"{name} (mean)", color=colors[opt_type], linewidth=1.5)
        ax3.fill_between(indices, min_ss, max_ss, color=colors[opt_type], alpha=0.2)

    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Step Size (α)")
    ax3.set_title("Step-Size Evolution (Mean ± Range)")
    ax3.set_yscale("log")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Final performance bar chart
    ax4 = fig.add_subplot(gs[1, 1])

    # Extract metrics for comparison
    metrics_only = {name: data["metrics"] for name, data in results.items()}
    summary = compare_learners(metrics_only)

    names = list(summary.keys())
    cumulative_errors = [summary[n]["cumulative"] for n in names]
    bar_colors = [colors[results[n]["optimizer_type"]] for n in names]

    bars = ax4.bar(names, cumulative_errors, color=bar_colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, cumulative_errors):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cumulative_errors) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax4.set_ylabel("Cumulative Squared Error")
    ax4.set_title("Final Performance Comparison")
    ax4.tick_params(axis="x", rotation=15)

    # Add improvement annotations
    best_adaptive = min(
        (n for n in names if results[n]["optimizer_type"] != "LMS"),
        key=lambda n: summary[n]["cumulative"],
    )
    lms_name = [n for n in names if results[n]["optimizer_type"] == "LMS"][0]

    if summary[best_adaptive]["cumulative"] < summary[lms_name]["cumulative"]:
        improvement = (
            (summary[lms_name]["cumulative"] - summary[best_adaptive]["cumulative"])
            / summary[lms_name]["cumulative"]
            * 100
        )
        ax4.annotate(
            f"Best adaptive beats LMS by {improvement:.1f}%",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=9,
            color="green",
            fontweight="bold",
        )

    plt.suptitle(f"Algorithm Comparison: {stream_name}", fontsize=13, fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_step_size_adaptation_detail(
    results: dict,
    stream_name: str,
    save_path: str | None = None,
) -> None:
    """Create detailed step-size adaptation plot (Sutton 1992 style).

    Shows per-weight step-size evolution to demonstrate how IDBD and Autostep
    adapt differently to different features.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (name, data) in zip(axes, [(n, d) for n, d in results.items() if d["optimizer_type"] != "LMS"]):
        step_sizes = data["step_sizes"]
        indices = data["recording_indices"]

        # Plot first 5 weight step-sizes
        num_to_show = min(5, step_sizes.shape[1])
        for i in range(num_to_show):
            ax.plot(indices, step_sizes[:, i], color=colors[i], label=f"α_{i+1}", linewidth=1.2)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Step Size (α)")
        ax.set_title(f"{name}: Per-Weight Step-Size Adaptation")
        ax.set_yscale("log")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Per-Weight Step-Size Adaptation: {stream_name}\n(Demonstrates meta-learning behavior)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Detailed step-size plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cumulative_error_curves(
    all_results: dict,
    save_path: str | None = None,
) -> None:
    """Plot cumulative error curves for all stream types (Sutton 1992 Figure 2 style)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {
        "LMS": "#1f77b4",
        "IDBD": "#ff7f0e",
        "Autostep": "#2ca02c",
    }

    for ax, (stream_name, results) in zip(axes, all_results.items()):
        for name, data in results.items():
            opt_type = data["optimizer_type"]
            metrics = data["metrics"]
            cumulative = np.cumsum([m["squared_error"] for m in metrics])
            ax.plot(cumulative, label=name, color=colors[opt_type], linewidth=1.5)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cumulative Squared Error")
        ax.set_title(stream_name)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Cumulative Error Over Time (Lower is Better)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Cumulative error plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_summary_bar_chart(
    all_results: dict,
    save_path: str | None = None,
) -> None:
    """Create summary bar chart comparing all methods across all streams."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "LMS": "#1f77b4",
        "IDBD": "#ff7f0e",
        "Autostep": "#2ca02c",
    }

    stream_names = list(all_results.keys())
    x = np.arange(len(stream_names))
    width = 0.25

    # Collect data for each optimizer type
    lms_errors = []
    idbd_errors = []
    autostep_errors = []

    for stream_name in stream_names:
        results = all_results[stream_name]
        for name, data in results.items():
            opt_type = data["optimizer_type"]
            metrics = data["metrics"]
            cumulative = sum(m["squared_error"] for m in metrics)
            if opt_type == "LMS":
                lms_errors.append(cumulative)
            elif opt_type == "IDBD":
                idbd_errors.append(cumulative)
            else:
                autostep_errors.append(cumulative)

    # Plot bars
    ax.bar(x - width, lms_errors, width, label="LMS", color=colors["LMS"], alpha=0.8)
    ax.bar(x, idbd_errors, width, label="IDBD", color=colors["IDBD"], alpha=0.8)
    ax.bar(x + width, autostep_errors, width, label="Autostep", color=colors["Autostep"], alpha=0.8)

    ax.set_xlabel("Stream Type")
    ax.set_ylabel("Cumulative Squared Error")
    ax.set_title("Performance Comparison Across Non-Stationarity Types")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("(")[0].strip() for s in stream_names], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Summary bar chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main(output_dir: str | None = None):
    """Run comprehensive Autostep comparison with plots."""
    with Timer("Total experiment runtime"):
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None

        print("=" * 70)
        print("Step 1: IDBD vs Autostep vs LMS Comparison")
        print("=" * 70)
        print("\nComparing three step-size strategies across different non-stationarity types.")
        print("This replicates key results from Sutton 1992 and Mahmood et al. 2012.")

        # Test configurations
        stream_configs = [
            (RandomWalkStream, "Random Walk (gradual drift)", {"drift_rate": 0.001}),
            (AbruptChangeStream, "Abrupt Changes (sudden shifts)", {"change_interval": 1000}),
            (CyclicStream, "Cyclic (repeating patterns)", {"cycle_length": 500}),
        ]

        all_results = {}
        all_trajectories = {}

        for stream_class, stream_name, stream_kwargs in stream_configs:
            print(f"\n{'-'*70}")
            print(f"Running: {stream_name}")
            print("-" * 70)

            # Generate weight trajectory for visualization
            trajectory = generate_stream_trajectory(
                stream_class,
                feature_dim=10,
                num_steps=10000,
                seed=42,
                **stream_kwargs,
            )
            all_trajectories[stream_name] = trajectory

            # Run comparison with step-size tracking
            results = run_comparison_with_tracking(
                stream_class,
                stream_name,
                feature_dim=10,
                num_steps=10000,
                seed=42,
                tracking_interval=50,
                **stream_kwargs,
            )
            all_results[stream_name] = results

            # Print results
            print_comparison_results(results, stream_name)

            # Generate plots for this stream
            if output_path:
                safe_name = stream_name.split("(")[0].strip().lower().replace(" ", "_")
                save_path = str(output_path / f"comparison_{safe_name}.png")
                detail_path = str(output_path / f"stepsize_detail_{safe_name}.png")
            else:
                save_path = None
                detail_path = None

            plot_comparison_figure(results, stream_name, trajectory, save_path=save_path)
            plot_step_size_adaptation_detail(results, stream_name, save_path=detail_path)

        # Generate summary plots across all streams
        if output_path:
            cumulative_path = str(output_path / "cumulative_error_all_streams.png")
            summary_path = str(output_path / "summary_comparison.png")
        else:
            cumulative_path = None
            summary_path = None

        plot_cumulative_error_curves(all_results, save_path=cumulative_path)
        plot_summary_bar_chart(all_results, save_path=summary_path)

        # Overall analysis
        print("\n" + "=" * 70)
        print("OVERALL ANALYSIS")
        print("=" * 70)

        print("\nMethod performance across different non-stationarity types:")
        print("-" * 70)

        for stream_name, results in all_results.items():
            metrics_only = {name: data["metrics"] for name, data in results.items()}
            summary = compare_learners(metrics_only)
            best_name = min(summary.items(), key=lambda x: x[1]["cumulative"])[0]
            best_type = results[best_name]["optimizer_type"]
            print(f"  {stream_name}: Winner = {best_type}")

        print("\n" + "-" * 70)
        print("KEY INSIGHTS:")
        print("  - LMS: Best when optimal step-size is known a priori")
        print("  - IDBD: Adapts to non-stationarity via gradient correlation")
        print("  - Autostep: More robust due to gradient normalization")
        print("  - Adaptive methods shine when optimal step-size varies over time")
        print("=" * 70 + "\n")

        if output_path:
            print(f"\nAll plots saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: IDBD vs Autostep vs LMS Comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: show interactively)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
