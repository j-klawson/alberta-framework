#!/usr/bin/env python3
"""Replication of Experiment 2 from Sutton 1992 IDBD paper.

This script replicates Experiment 2 from:
    Sutton, R.S. (1992). "Adapting Bias by Gradient Descent:
    An Incremental Version of Delta-Bar-Delta"

Experiment 2 Design: "Does IDBD find the optimal alpha_i?"
This experiment has two parts:

Part 1 (Figure 4): Track IDBD Learning Rate Evolution
- Same task as Experiment 1 (20 inputs, 5 relevant with +/-1, 15 irrelevant with 0)
- Run IDBD with theta=0.001 (small meta-learning rate) for 250,000 steps
- Initial alpha=0.05 for all weights
- Track learning rates over time
- Expected: Relevant inputs converge to alpha~0.13, irrelevant to <0.007

Part 2 (Figure 5): Verify Optimality via Grid Search
- Fix irrelevant input learning rates to 0
- Vary relevant input learning rates from 0.05 to 0.25
- For each alpha: 20,000 burn-in, measure MSE over 10,000 steps
- Expected: Minimum error at alpha~0.13, confirming IDBD found optimal

Usage:
    python "examples/The Alberta Plan/Step1/sutton1992_experiment2.py"
"""

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework import IDBD, LinearLearner, run_learning_loop
from alberta_framework.core.types import IDBDState, TimeStep
from alberta_framework.streams.synthetic import SuttonExperiment1Stream


def track_learning_rate_evolution(
    theta: float = 0.001,
    initial_alpha: float = 0.05,
    num_steps: int = 250000,
    record_interval: int = 1000,
    seed: int = 0,
) -> dict[str, list]:
    """Track IDBD learning rate evolution over time (Figure 4).

    Args:
        theta: Meta-step-size for IDBD
        initial_alpha: Initial step-size for all weights
        num_steps: Total steps to run
        record_interval: Record learning rates every this many steps
        seed: Random seed

    Returns:
        Dictionary with 'steps', 'relevant_alphas', 'irrelevant_alphas' lists
    """
    stream = SuttonExperiment1Stream(
        num_relevant=5,
        num_irrelevant=15,
        change_interval=20,
    )
    learner = LinearLearner(
        optimizer=IDBD(initial_step_size=initial_alpha, meta_step_size=theta)
    )

    # Initialize stream and learner state
    key = jr.key(seed)
    stream_state = stream.init(key)
    learner_state = learner.init(stream.feature_dim)

    history: dict[str, list] = {
        "steps": [],
        "relevant_alphas": [],  # Mean of first 5 weights
        "irrelevant_alphas": [],  # Mean of last 15 weights
        "relevant_alpha_0": [],  # First relevant weight (for plotting)
        "irrelevant_alpha_5": [],  # First irrelevant weight (for plotting)
    }

    # Record initial values
    optimizer_state = learner_state.optimizer_state
    assert isinstance(optimizer_state, IDBDState)
    alphas = jnp.exp(optimizer_state.log_step_sizes)
    history["steps"].append(0)
    history["relevant_alphas"].append(float(jnp.mean(alphas[:5])))
    history["irrelevant_alphas"].append(float(jnp.mean(alphas[5:])))
    history["relevant_alpha_0"].append(float(alphas[0]))
    history["irrelevant_alpha_5"].append(float(alphas[5]))

    for step in range(num_steps):
        # Get next time step from stream
        time_step, stream_state = stream.step(stream_state, jnp.array(step))

        result = learner.update(learner_state, time_step.observation, time_step.target)
        learner_state = result.state

        # Record at intervals
        if (step + 1) % record_interval == 0:
            optimizer_state = learner_state.optimizer_state
            assert isinstance(optimizer_state, IDBDState)
            alphas = jnp.exp(optimizer_state.log_step_sizes)

            history["steps"].append(step + 1)
            history["relevant_alphas"].append(float(jnp.mean(alphas[:5])))
            history["irrelevant_alphas"].append(float(jnp.mean(alphas[5:])))
            history["relevant_alpha_0"].append(float(alphas[0]))
            history["irrelevant_alpha_5"].append(float(alphas[5]))

    return history


def run_per_weight_lms(
    stream: SuttonExperiment1Stream,
    stream_state: "SuttonExperiment1State",  # type: ignore  # noqa: F821
    relevant_alpha: float,
    irrelevant_alpha: float,
    num_steps: int,
    initial_weights: Array | None = None,
) -> tuple[Array, list[float], "SuttonExperiment1State"]:  # type: ignore  # noqa: F821
    """Run LMS with different learning rates for relevant vs irrelevant inputs.

    This implements a simple manual learning loop since the standard LMS
    optimizer uses a single global step-size.

    Args:
        stream: Experience stream
        stream_state: Current stream state
        relevant_alpha: Learning rate for first 5 (relevant) weights
        irrelevant_alpha: Learning rate for last 15 (irrelevant) weights
        num_steps: Number of steps to run
        initial_weights: Optional initial weights (defaults to zeros)

    Returns:
        Tuple of (final_weights, list of squared errors, final stream state)
    """
    feature_dim = stream.feature_dim
    weights = initial_weights if initial_weights is not None else jnp.zeros(feature_dim, dtype=jnp.float32)

    # Per-weight learning rates
    alphas = jnp.concatenate([
        jnp.full(5, relevant_alpha, dtype=jnp.float32),
        jnp.full(15, irrelevant_alpha, dtype=jnp.float32),
    ])

    squared_errors = []

    for step in range(num_steps):
        time_step, stream_state = stream.step(stream_state, jnp.array(step))

        x = time_step.observation
        y_star = jnp.squeeze(time_step.target)

        # Predict
        y = jnp.dot(weights, x)
        error = y_star - y
        squared_errors.append(float(error ** 2))

        # Update: w_i += alpha_i * error * x_i
        weights = weights + alphas * error * x

    return weights, squared_errors, stream_state


def run_optimal_alpha_search(
    alphas: list[float],
    burn_in_steps: int = 20000,
    measurement_steps: int = 10000,
    seed: int = 0,
) -> dict[float, float]:
    """Run grid search over relevant input learning rates (Figure 5).

    Fixes irrelevant input learning rates to 0 and varies the relevant
    input learning rates.

    Args:
        alphas: List of learning rates to try for relevant inputs
        burn_in_steps: Steps before measuring
        measurement_steps: Steps to measure MSE over
        seed: Random seed

    Returns:
        Dictionary mapping alpha to asymptotic MSE
    """
    results = {}

    for alpha in alphas:
        # Create fresh stream for each experiment
        stream = SuttonExperiment1Stream(
            num_relevant=5,
            num_irrelevant=15,
            change_interval=20,
        )

        # Initialize stream state
        key = jr.key(seed)
        stream_state = stream.init(key)

        feature_dim = stream.feature_dim
        weights = jnp.zeros(feature_dim, dtype=jnp.float32)

        # Per-weight learning rates
        step_sizes = jnp.concatenate([
            jnp.full(5, alpha, dtype=jnp.float32),
            jnp.full(15, 0.0, dtype=jnp.float32),  # irrelevant fixed to 0
        ])

        measurement_errors = []

        for step in range(burn_in_steps + measurement_steps):
            time_step, stream_state = stream.step(stream_state, jnp.array(step))

            x = time_step.observation
            y_star = jnp.squeeze(time_step.target)

            # Predict
            y = jnp.dot(weights, x)
            error = y_star - y

            # Record error during measurement phase
            if step >= burn_in_steps:
                measurement_errors.append(float(error ** 2))

            # Update: w_i += alpha_i * error * x_i
            weights = weights + step_sizes * error * x

        mse = sum(measurement_errors) / len(measurement_errors)
        results[alpha] = mse
        print(f"  alpha={alpha:.3f}: MSE = {mse:.4f}")

    return results


def plot_figure4(history: dict[str, list], save_path: str | None = None) -> None:
    """Plot learning rate evolution (Figure 4 from paper).

    Args:
        history: Dictionary from track_learning_rate_evolution
        save_path: If provided, save to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = [s / 1000 for s in history["steps"]]  # Convert to thousands

    ax.plot(steps, history["relevant_alpha_0"], "b-", label="Relevant input (x1)", linewidth=2)
    ax.plot(steps, history["irrelevant_alpha_5"], "r-", label="Irrelevant input (x6)", linewidth=2)

    # Add horizontal reference lines
    ax.axhline(y=0.13, color="blue", linestyle="--", alpha=0.5, label="Expected optimal (~0.13)")
    ax.axhline(y=0.007, color="red", linestyle="--", alpha=0.5, label="Expected irrelevant (<0.007)")

    ax.set_xlabel("Examples (thousands)", fontsize=12)
    ax.set_ylabel("Learning Rate (alpha)", fontsize=12)
    ax.set_title(
        "Replication of Sutton 1992, Figure 4:\n"
        "Time Course of Learning Rates with IDBD",
        fontsize=14,
    )
    ax.legend(loc="right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure 4 saved to {save_path}")
    else:
        plt.show()


def plot_figure5(results: dict[float, float], optimal_alpha: float | None = None, save_path: str | None = None) -> None:
    """Plot asymptotic error vs learning rate (Figure 5 from paper).

    Args:
        results: Dictionary mapping alpha to MSE
        optimal_alpha: Alpha found by IDBD (for vertical line)
        save_path: If provided, save to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    alphas = sorted(results.keys())
    mses = [results[a] for a in alphas]

    ax.plot(alphas, mses, "bo-", markersize=8, linewidth=2)

    # Find minimum
    min_idx = mses.index(min(mses))
    min_alpha = alphas[min_idx]
    min_mse = mses[min_idx]

    ax.axvline(x=min_alpha, color="green", linestyle="--", alpha=0.7,
               label=f"Grid search minimum (alpha={min_alpha:.3f})")

    if optimal_alpha is not None:
        ax.axvline(x=optimal_alpha, color="blue", linestyle=":", alpha=0.7,
                   label=f"IDBD converged value (alpha={optimal_alpha:.3f})")

    ax.scatter([min_alpha], [min_mse], color="green", s=150, zorder=5, marker="*")

    ax.set_xlabel("Learning Rate of Relevant Inputs (alpha)", fontsize=12)
    ax.set_ylabel("Asymptotic Error (MSE)", fontsize=12)
    ax.set_title(
        "Replication of Sutton 1992, Figure 5:\n"
        "Asymptotic Error vs Learning Rate",
        fontsize=14,
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure 5 saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    """Run the Sutton 1992 Experiment 2 replication."""
    print("=" * 70)
    print("Replication of Sutton (1992) Experiment 2:")
    print("Does IDBD Find the Optimal alpha_i?")
    print("=" * 70)

    # ========================================================================
    # Part 1: Track learning rate evolution (Figure 4)
    # ========================================================================
    print("\n" + "-" * 70)
    print("Part 1: Learning Rate Evolution (Figure 4)")
    print("-" * 70)
    print("Running IDBD with theta=0.001 for 250,000 steps...")
    print("This demonstrates how IDBD adapts per-weight learning rates.")
    print()

    history = track_learning_rate_evolution(
        theta=0.001,
        initial_alpha=0.05,
        num_steps=250000,
        record_interval=1000,
        seed=42,
    )

    final_relevant = history["relevant_alphas"][-1]
    final_irrelevant = history["irrelevant_alphas"][-1]

    print(f"Initial learning rate: 0.05 (all weights)")
    print(f"\nFinal learning rates after 250,000 steps:")
    print(f"  Relevant inputs (mean):   {final_relevant:.4f}")
    print(f"  Irrelevant inputs (mean): {final_irrelevant:.6f}")
    print(f"\nPaper reports:")
    print(f"  Relevant inputs:   ~0.13")
    print(f"  Irrelevant inputs: <0.007 (heading towards 0)")

    if final_relevant > 0.1 and final_irrelevant < 0.01:
        print("\nSUCCESS: Learning rates evolved as expected!")
    else:
        print("\nNote: Results differ from paper - may need parameter tuning")

    # ========================================================================
    # Part 2: Verify optimality via grid search (Figure 5)
    # ========================================================================
    print("\n" + "-" * 70)
    print("Part 2: Optimal Learning Rate Search (Figure 5)")
    print("-" * 70)
    print("Running grid search over learning rates for relevant inputs...")
    print("(Irrelevant input learning rates fixed to 0)")
    print()

    # Grid of learning rates similar to paper's range
    alphas = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]

    results = run_optimal_alpha_search(
        alphas=alphas,
        burn_in_steps=20000,
        measurement_steps=10000,
        seed=42,
    )

    # Find optimal
    min_alpha = min(results, key=results.get)  # type: ignore
    min_mse = results[min_alpha]

    print(f"\nGrid search results:")
    print(f"  Optimal alpha: {min_alpha:.3f}")
    print(f"  Minimum MSE:   {min_mse:.4f}")
    print(f"\nIDDB converged to: {final_relevant:.3f}")
    print(f"Paper reports optimal: ~0.13")

    # Check if IDBD found near-optimal
    if abs(final_relevant - min_alpha) < 0.03:
        print("\nSUCCESS: IDBD found near-optimal learning rate!")
    else:
        print(f"\nNote: IDBD alpha ({final_relevant:.3f}) differs from grid search "
              f"optimal ({min_alpha:.3f})")

    # ========================================================================
    # Generate plots
    # ========================================================================
    print("\n" + "-" * 70)
    print("Generating Figures")
    print("-" * 70)

    plot_figure4(history)
    plot_figure5(results, optimal_alpha=final_relevant)


if __name__ == "__main__":
    main()
