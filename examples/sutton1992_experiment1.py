#!/usr/bin/env python3
"""Replication of Experiment 1 from Sutton 1992 IDBD paper.

This script replicates the key experiment from:
    Sutton, R.S. (1992). "Adapting Bias by Gradient Descent:
    An Incremental Version of Delta-Bar-Delta"

Experiment 1 Design:
- 20 real-valued inputs, 1 output
- Inputs drawn independently from N(0, 1)
- Target: y* = s1*x1 + s2*x2 + s3*x3 + s4*x4 + s5*x5 + 0*x6 + ... + 0*x20
  - Only first 5 inputs are relevant (weights are +1 or -1)
  - Last 15 inputs are irrelevant (weights are 0)
- Non-stationarity: Every 20 examples, one of the five signs is randomly flipped

Procedure:
- Run each algorithm for 20,000 examples (burn-in to get past transients)
- Then run another 10,000 examples
- Measure average MSE over those 10,000 examples as asymptotic performance

Expected Results (from paper):
- LMS with best alpha (~0.04): MSE ~3.5
- IDBD over broad theta range: MSE ~1.5
- IDBD achieves less than half the error of best-tuned LMS

Usage:
    python examples/sutton1992_experiment1.py
"""

import jax.numpy as jnp

from alberta_framework import IDBD, LMS, LinearLearner, run_learning_loop
from alberta_framework.streams.synthetic import SuttonExperiment1Stream


def run_single_experiment(
    optimizer: LMS | IDBD,
    burn_in_steps: int = 20000,
    measurement_steps: int = 10000,
    seed: int = 0,
) -> float:
    """Run a single experiment and return asymptotic MSE.

    Args:
        optimizer: LMS or IDBD optimizer
        burn_in_steps: Steps to run before measuring (default 20,000)
        measurement_steps: Steps to measure performance (default 10,000)
        seed: Random seed

    Returns:
        Average mean squared error over measurement period
    """
    stream = SuttonExperiment1Stream(
        num_relevant=5,
        num_irrelevant=15,
        change_interval=20,
        seed=seed,
    )
    learner = LinearLearner(optimizer=optimizer)

    # Burn-in phase
    state, _ = run_learning_loop(learner, stream, burn_in_steps)

    # Measurement phase
    _, metrics = run_learning_loop(learner, stream, measurement_steps, state=state)

    # Compute average MSE
    avg_mse = sum(m["squared_error"] for m in metrics) / len(metrics)
    return avg_mse


def run_lms_sweep(
    alphas: list[float],
    burn_in: int = 20000,
    measurement: int = 10000,
    seed: int = 0,
) -> dict[float, float]:
    """Run LMS with different step-sizes.

    Args:
        alphas: List of step-sizes to try
        burn_in: Burn-in steps
        measurement: Measurement steps
        seed: Random seed

    Returns:
        Dictionary mapping alpha to asymptotic MSE
    """
    results = {}
    for alpha in alphas:
        mse = run_single_experiment(
            LMS(step_size=alpha),
            burn_in_steps=burn_in,
            measurement_steps=measurement,
            seed=seed,
        )
        results[alpha] = mse
        print(f"  LMS(alpha={alpha:.4f}): MSE = {mse:.4f}")
    return results


def run_idbd_sweep(
    thetas: list[float],
    initial_alpha: float = 0.05,
    burn_in: int = 20000,
    measurement: int = 10000,
    seed: int = 0,
) -> dict[float, float]:
    """Run IDBD with different meta-step-sizes.

    Args:
        thetas: List of meta-step-sizes to try
        initial_alpha: Initial step-size for all weights (paper uses 0.05)
        burn_in: Burn-in steps
        measurement: Measurement steps
        seed: Random seed

    Returns:
        Dictionary mapping theta to asymptotic MSE
    """
    results = {}
    for theta in thetas:
        mse = run_single_experiment(
            IDBD(initial_step_size=initial_alpha, meta_step_size=theta),
            burn_in_steps=burn_in,
            measurement_steps=measurement,
            seed=seed,
        )
        results[theta] = mse
        print(f"  IDBD(theta={theta:.4f}): MSE = {mse:.4f}")
    return results


def plot_figure3(
    lms_results: dict[float, float],
    idbd_results: dict[float, float],
    save_path: str | None = None,
) -> None:
    """Replicate Figure 3 from the paper.

    Figure 3 shows asymptotic error vs step-size parameter for both LMS and IDBD,
    with LMS using alpha (upper axis) and IDBD using theta (lower axis).

    Args:
        lms_results: Dictionary of alpha -> MSE for LMS
        idbd_results: Dictionary of theta -> MSE for IDBD
        save_path: If provided, save to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    _, ax1 = plt.subplots(figsize=(8, 6))

    # Sort results by parameter value
    lms_alphas = sorted(lms_results.keys())
    lms_mses = [lms_results[a] for a in lms_alphas]
    idbd_thetas = sorted(idbd_results.keys())
    idbd_mses = [idbd_results[t] for t in idbd_thetas]

    # Plot IDBD on primary x-axis (theta, bottom)
    (line1,) = ax1.plot(
        idbd_thetas, idbd_mses, "s-", color="blue", label="IDBD(theta)", markersize=8
    )
    ax1.set_xlabel("theta (IDBD meta-step-size)", fontsize=12)
    ax1.set_ylabel("Asymptotic Error (MSE)", fontsize=12)
    ax1.tick_params(axis="x", labelcolor="blue")

    # Create secondary x-axis for LMS (alpha, top)
    ax2 = ax1.twiny()
    (line2,) = ax2.plot(
        lms_alphas, lms_mses, "o-", color="red", label="LMS(alpha)", markersize=8
    )
    ax2.set_xlabel("alpha (LMS step-size)", fontsize=12)
    ax2.tick_params(axis="x", labelcolor="red")

    # Add legend
    ax1.legend([line1, line2], ["IDBD(theta)", "LMS(alpha)"], loc="upper right")

    # Add grid and title
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "Replication of Sutton 1992, Figure 3:\n"
        "Asymptotic Performance of IDBD vs LMS",
        fontsize=14,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def analyze_learning_rate_evolution(
    theta: float = 0.001,
    initial_alpha: float = 0.05,
    num_steps: int = 250000,
    seed: int = 0,
) -> None:
    """Analyze how IDBD learning rates evolve over time (Figure 4 from paper).

    The paper shows that after 250,000 steps:
    - Relevant input learning rates converge to ~0.13
    - Irrelevant input learning rates converge to ~0 (< 0.007)

    Args:
        theta: Meta-step-size (paper uses 0.001 for this experiment)
        initial_alpha: Initial step-size
        num_steps: Number of steps to run
        seed: Random seed
    """
    print("\n" + "=" * 70)
    print("Learning Rate Evolution Analysis (cf. Figure 4)")
    print("=" * 70)
    print(f"Running IDBD with theta={theta} for {num_steps:,} steps...")

    stream = SuttonExperiment1Stream(
        num_relevant=5,
        num_irrelevant=15,
        change_interval=20,
        seed=seed,
    )
    learner = LinearLearner(
        optimizer=IDBD(initial_step_size=initial_alpha, meta_step_size=theta)
    )

    state, _ = run_learning_loop(learner, stream, num_steps)

    # Extract final learning rates from IDBD state
    # IDBD stores log_step_sizes, so we need to exponentiate
    from alberta_framework.core.types import IDBDState

    optimizer_state = state.optimizer_state
    assert isinstance(optimizer_state, IDBDState), "Expected IDBD optimizer state"
    log_alphas = optimizer_state.log_step_sizes
    alphas = jnp.exp(log_alphas)

    relevant_alphas = alphas[:5]
    irrelevant_alphas = alphas[5:]

    print(f"\nFinal learning rates after {num_steps:,} steps:")
    print(f"  Relevant inputs (1-5):   mean={float(jnp.mean(relevant_alphas)):.4f}, "
          f"range=[{float(jnp.min(relevant_alphas)):.4f}, {float(jnp.max(relevant_alphas)):.4f}]")
    print(f"  Irrelevant inputs (6-20): mean={float(jnp.mean(irrelevant_alphas)):.6f}, "
          f"max={float(jnp.max(irrelevant_alphas)):.6f}")

    print("\nPaper reports after 250,000 steps:")
    print("  Relevant inputs: ~0.13")
    print("  Irrelevant inputs: < 0.007 (heading towards 0)")


def main() -> None:
    """Run the Sutton 1992 Experiment 1 replication."""
    print("=" * 70)
    print("Replication of Sutton (1992) Experiment 1: Does IDBD Help?")
    print("=" * 70)
    print("\nExperiment setup:")
    print("  - 20 inputs: 5 relevant (weights +/-1), 15 irrelevant (weights 0)")
    print("  - Sign flip every 20 examples")
    print("  - 20,000 burn-in steps, 10,000 measurement steps")
    print()

    # LMS step-sizes to sweep (similar to paper's range)
    # Paper shows alpha from ~0 to ~0.08
    lms_alphas = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    # IDBD meta-step-sizes to sweep (similar to paper's range)
    # Paper shows theta from ~0 to ~0.02
    idbd_thetas = [0.001, 0.002, 0.005, 0.007, 0.01, 0.012, 0.015, 0.017, 0.02]

    print("Running LMS sweep...")
    lms_results = run_lms_sweep(lms_alphas, seed=42)

    print("\nRunning IDBD sweep...")
    idbd_results = run_idbd_sweep(idbd_thetas, initial_alpha=0.05, seed=42)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Find best parameters by sorting by MSE
    lms_sorted = sorted(lms_results.items(), key=lambda x: x[1])
    best_lms_alpha, best_lms_mse = lms_sorted[0]

    idbd_sorted = sorted(idbd_results.items(), key=lambda x: x[1])
    best_idbd_theta, best_idbd_mse = idbd_sorted[0]

    print(f"\nBest LMS:  alpha={best_lms_alpha:.4f}, MSE={best_lms_mse:.4f}")
    print(f"Best IDBD: theta={best_idbd_theta:.4f}, MSE={best_idbd_mse:.4f}")

    if best_idbd_mse < best_lms_mse:
        improvement = (best_lms_mse - best_idbd_mse) / best_lms_mse * 100
        print(f"\nIDDB outperforms best LMS by {improvement:.1f}%")

        if best_idbd_mse < best_lms_mse * 0.5:
            print("SUCCESS: IDBD achieves less than half the error of best LMS!")
            print("(Paper reports IDBD MSE ~1.5 vs LMS MSE ~3.5)")
    else:
        print("\nNote: IDBD did not outperform best LMS in this run")
        print("Consider adjusting parameters or running longer")

    # Run learning rate evolution analysis
    analyze_learning_rate_evolution(theta=0.001, num_steps=50000, seed=42)

    # Try to plot Figure 3
    print("\n" + "=" * 70)
    print("Generating Figure 3 replication...")
    plot_figure3(lms_results, idbd_results)


if __name__ == "__main__":
    main()
