#!/usr/bin/env python3
"""External Normalization Study: EMANormalizer vs Autostep/IDBD v_i.

Tests whether external feature normalization provides benefits beyond
Autostep's internal gradient normalization when feature scales change.

Research Gap:
- Mahmood's 2012 paper pre-normalized all robot sensor data
- The scale invariance claim from v_i was never tested against dynamically
  changing feature scales

Hypotheses:
- H0 (Redundancy): Autostep's v_i effectively tracks scale², making
  EMANormalizer redundant
- H1 (Complementary): EMANormalizer adapts faster to scale changes,
  providing additional benefit during transients
- H2 (IDBD-specific): IDBD benefits more from EMANormalizer than Autostep
  (since IDBD lacks v_i normalization)

Experimental Design (2x2x3 factorial):
- Optimizer: IDBD, Autostep
- Normalization: None, EMANormalizer
- Scale Type: Static, Abrupt, Drift
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    IDBD,
    Autostep,
    EMANormalizer,
    LinearLearner,
    RandomWalkStream,
    ScaledStreamWrapper,
    Timer,
    make_scale_range,
    metrics_to_dicts,
    run_learning_loop,
)
from alberta_framework.streams.synthetic import (
    DynamicScaleShiftStream,
    ScaleDriftStream,
)

# === Configuration ===
FEATURE_DIM = 20
NUM_STEPS = 50000
NUM_SEEDS = 30
SCALE_RANGE = (0.01, 100.0)  # 10^4 range

# Optimizer hyperparameters
IDBD_PARAMS = {
    "initial_step_size": 0.05,
    "meta_step_size": 1e-3,
}

AUTOSTEP_PARAMS = {
    "initial_step_size": 0.05,
    "meta_step_size": 0.05,
    "normalizer_decay": 0.99,
}

# EMANormalizer hyperparameters
NORMALIZER_PARAMS = {
    "decay": 0.99,
    "epsilon": 1e-8,
}


# === Stream Factories ===


def make_static_scale_stream():
    """Create stream with static multi-scale features."""
    scales = make_scale_range(FEATURE_DIM, *SCALE_RANGE)
    inner = RandomWalkStream(feature_dim=FEATURE_DIM, drift_rate=0.001)
    return ScaledStreamWrapper(inner, feature_scales=scales)


def make_abrupt_scale_stream():
    """Create stream with abrupt scale shifts."""
    return DynamicScaleShiftStream(
        feature_dim=FEATURE_DIM,
        scale_change_interval=5000,
        weight_change_interval=2000,
        min_scale=SCALE_RANGE[0],
        max_scale=SCALE_RANGE[1],
    )


def make_drift_scale_stream():
    """Create stream with continuous scale drift."""
    return ScaleDriftStream(
        feature_dim=FEATURE_DIM,
        weight_drift_rate=0.001,
        scale_drift_rate=0.02,
        min_log_scale=jnp.log(SCALE_RANGE[0]),
        max_log_scale=jnp.log(SCALE_RANGE[1]),
    )


# === Learner Factories ===


def make_idbd_learner():
    """Create IDBD learner without external normalization."""
    return LinearLearner(optimizer=IDBD(**IDBD_PARAMS))


def make_idbd_normalized_learner():
    """Create IDBD learner with external normalization."""
    return LinearLearner(
        optimizer=IDBD(**IDBD_PARAMS),
        normalizer=EMANormalizer(**NORMALIZER_PARAMS),
    )


def make_autostep_learner():
    """Create Autostep learner without external normalization."""
    return LinearLearner(optimizer=Autostep(**AUTOSTEP_PARAMS))


def make_autostep_normalized_learner():
    """Create Autostep learner with external normalization."""
    return LinearLearner(
        optimizer=Autostep(**AUTOSTEP_PARAMS),
        normalizer=EMANormalizer(**NORMALIZER_PARAMS),
    )


# === Experiment Utilities ===


def run_single_experiment(
    learner_factory,
    stream_factory,
    num_steps: int,
    seed: int,
) -> dict:
    """Run a single experiment and return metrics."""
    key = jr.key(seed)
    learner = learner_factory()
    stream = stream_factory()

    _, metrics = run_learning_loop(learner, stream, num_steps, key)

    metrics_list = metrics_to_dicts(metrics)
    squared_errors = np.array([m["squared_error"] for m in metrics_list])

    return {
        "squared_errors": squared_errors,
        "final_mse": np.mean(squared_errors[-1000:]),
        "total_mse": np.mean(squared_errors),
    }


def run_multi_seed(
    learner_factory,
    stream_factory,
    num_steps: int,
    seeds: list[int],
    verbose: bool = True,
) -> dict:
    """Run experiment over multiple seeds."""
    all_squared_errors = []
    final_mses = []
    total_mses = []

    for seed in seeds:
        result = run_single_experiment(learner_factory, stream_factory, num_steps, seed)
        all_squared_errors.append(result["squared_errors"])
        final_mses.append(result["final_mse"])
        total_mses.append(result["total_mse"])
        if verbose:
            print(".", end="", flush=True)

    if verbose:
        print()

    return {
        "squared_errors": np.array(all_squared_errors),  # Shape: (n_seeds, n_steps)
        "final_mse_mean": np.mean(final_mses),
        "final_mse_std": np.std(final_mses),
        "total_mse_mean": np.mean(total_mses),
        "total_mse_std": np.std(total_mses),
    }


def compute_improvement(base_mean: float, improved_mean: float) -> tuple[float, str]:
    """Compute improvement percentage and direction."""
    if improved_mean < base_mean:
        pct = (base_mean - improved_mean) / base_mean * 100
        return pct, "better"
    else:
        pct = (improved_mean - base_mean) / base_mean * 100
        return pct, "worse"


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)


def main(output_dir: str | None = None, num_seeds: int = 30):
    """Run the external normalization study."""
    with Timer("Total experiment runtime"):
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("External Normalization Study")
        print("Testing: Does EMANormalizer help IDBD/Autostep with dynamic scales?")
        print("=" * 80)
        print(f"\nConfig: {FEATURE_DIM} features, {NUM_STEPS} steps, {num_seeds} seeds")
        print(f"Scale range: {SCALE_RANGE[0]} to {SCALE_RANGE[1]}")

        # Define experimental conditions
        learner_configs = [
            ("IDBD", make_idbd_learner),
            ("IDBD+Norm", make_idbd_normalized_learner),
            ("Autostep", make_autostep_learner),
            ("Autostep+Norm", make_autostep_normalized_learner),
        ]

        stream_configs = [
            ("Static", make_static_scale_stream),
            ("Abrupt", make_abrupt_scale_stream),
            ("Drift", make_drift_scale_stream),
        ]

        seeds = list(range(num_seeds))

        # Run all experiments
        results = {}
        for stream_name, stream_factory in stream_configs:
            print(f"\n--- {stream_name} Scale Condition ---")
            for learner_name, learner_factory in learner_configs:
                print(f"  Running {learner_name}", end="", flush=True)
                key = f"{learner_name}_{stream_name}"
                results[key] = run_multi_seed(
                    learner_factory, stream_factory, NUM_STEPS, seeds
                )
                print(
                    f" MSE: {results[key]['final_mse_mean']:.6f} "
                    f"+/- {results[key]['final_mse_std']:.6f}"
                )

        # Print results summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        print("\nFinal MSE (mean +/- std over last 1000 steps):")
        print("-" * 80)
        header = f"{'Condition':<20} {'IDBD':>15} {'IDBD+Norm':>15}"
        header += f" {'Autostep':>15} {'Autostep+Norm':>15}"
        print(header)
        print("-" * 80)

        for stream_name, _ in stream_configs:
            row = f"{stream_name:<20}"
            for learner_name, _ in learner_configs:
                key = f"{learner_name}_{stream_name}"
                r = results[key]
                row += f" {r['final_mse_mean']:>13.4f}"
            print(row)

        # Compute normalization benefits
        print("\n" + "=" * 80)
        print("NORMALIZATION BENEFIT ANALYSIS")
        print("=" * 80)

        print("\nImprovement from adding EMANormalizer:")
        print("-" * 80)

        for stream_name, _ in stream_configs:
            print(f"\n{stream_name} Scales:")

            # IDBD improvement
            idbd_base = results[f"IDBD_{stream_name}"]["final_mse_mean"]
            idbd_norm = results[f"IDBD+Norm_{stream_name}"]["final_mse_mean"]
            pct, direction = compute_improvement(idbd_base, idbd_norm)
            print(f"  IDBD:     {pct:>6.1f}% {direction}")

            # Autostep improvement
            auto_base = results[f"Autostep_{stream_name}"]["final_mse_mean"]
            auto_norm = results[f"Autostep+Norm_{stream_name}"]["final_mse_mean"]
            pct, direction = compute_improvement(auto_base, auto_norm)
            print(f"  Autostep: {pct:>6.1f}% {direction}")

        # Test H2: Does IDBD benefit more than Autostep?
        print("\n" + "=" * 80)
        print("HYPOTHESIS TESTING (H2: IDBD benefits more from normalization)")
        print("=" * 80)

        for stream_name, _ in stream_configs:
            idbd_delta = (
                results[f"IDBD_{stream_name}"]["final_mse_mean"]
                - results[f"IDBD+Norm_{stream_name}"]["final_mse_mean"]
            )
            auto_delta = (
                results[f"Autostep_{stream_name}"]["final_mse_mean"]
                - results[f"Autostep+Norm_{stream_name}"]["final_mse_mean"]
            )

            print(f"\n{stream_name} Scales:")
            print(f"  IDBD improvement (delta MSE):     {idbd_delta:>10.6f}")
            print(f"  Autostep improvement (delta MSE): {auto_delta:>10.6f}")

            if idbd_delta > auto_delta:
                print("  --> IDBD benefits MORE from normalization (supports H2)")
            else:
                print("  --> Autostep benefits MORE (contradicts H2)")

        # Conclusions
        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)

        # Check H0/H1 for Autostep
        auto_static_base = results["Autostep_Static"]["final_mse_mean"]
        _norm_key = (
            "Autostep_Static+Norm"
            if "Autostep_Static+Norm" in results
            else "Autostep+Norm_Static"
        )
        auto_static_norm = results[_norm_key]["final_mse_mean"]  # noqa: F841

        auto_abrupt_base = results["Autostep_Abrupt"]["final_mse_mean"]
        auto_abrupt_norm = results["Autostep+Norm_Abrupt"]["final_mse_mean"]

        print("\nFor Autostep:")
        static_pct, static_dir = compute_improvement(auto_static_base, auto_abrupt_norm)
        abrupt_pct, abrupt_dir = compute_improvement(auto_abrupt_base, auto_abrupt_norm)

        if abs((auto_abrupt_norm - auto_abrupt_base) / auto_abrupt_base) < 0.05:
            print("  H0 supported: EMANormalizer provides minimal additional benefit")
            print("  (Autostep's v_i provides sufficient scale invariance)")
        else:
            if auto_abrupt_norm < auto_abrupt_base:
                print("  H1 supported: EMANormalizer provides additional benefit")
                print("  (faster adaptation during scale transients)")
            else:
                print("  H0 partially supported: EMANormalizer may interfere with v_i")

        # Save plots if output_dir provided and matplotlib available
        if output_dir:
            try:
                import matplotlib.pyplot as plt

                # Create learning curve plots
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                for idx, (stream_name, _) in enumerate(stream_configs):
                    ax = axes[idx]
                    ax.set_title(f"{stream_name} Scale Condition")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Squared Error (smoothed)")

                    for learner_name, _ in learner_configs:
                        key = f"{learner_name}_{stream_name}"
                        errors = results[key]["squared_errors"]
                        mean_error = np.mean(errors, axis=0)

                        # Smooth with running average
                        window = 500
                        smoothed = np.convolve(
                            mean_error, np.ones(window) / window, mode="valid"
                        )
                        ax.plot(smoothed, label=learner_name, alpha=0.8)

                    ax.legend()
                    ax.set_yscale("log")

                plt.tight_layout()
                plt.savefig(output_path / "learning_curves.png", dpi=150)
                plt.savefig(output_path / "learning_curves.pdf")
                plt.close()

                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))

                x = np.arange(len(stream_configs))
                width = 0.2
                offsets = [-1.5, -0.5, 0.5, 1.5]

                for i, (learner_name, _) in enumerate(learner_configs):
                    means = [
                        results[f"{learner_name}_{sn}"]["final_mse_mean"]
                        for sn, _ in stream_configs
                    ]
                    stds = [
                        results[f"{learner_name}_{sn}"]["final_mse_std"]
                        for sn, _ in stream_configs
                    ]
                    ax.bar(
                        x + offsets[i] * width,
                        means,
                        width,
                        label=learner_name,
                        yerr=stds,
                        capsize=3,
                    )

                ax.set_xlabel("Scale Condition")
                ax.set_ylabel("Final MSE")
                ax.set_title("Final Performance by Condition")
                ax.set_xticks(x)
                ax.set_xticklabels([sn for sn, _ in stream_configs])
                ax.legend()
                ax.set_yscale("log")

                plt.tight_layout()
                plt.savefig(output_path / "final_performance.png", dpi=150)
                plt.savefig(output_path / "final_performance.pdf")
                plt.close()

                print(f"\nPlots saved to {output_path}")

            except ImportError:
                print("\nmatplotlib not available, skipping plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External Normalization Study: EMANormalizer vs IDBD/Autostep"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (optional)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of random seeds (default: 30)",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir, num_seeds=args.seeds)
