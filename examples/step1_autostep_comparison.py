#!/usr/bin/env python3
"""Step 1 Autostep Comparison: IDBD vs Autostep vs LMS.

This script provides a comprehensive comparison of the three step-size
adaptation strategies available in the Alberta Toolkit:

1. LMS: Fixed step-size (requires manual tuning)
2. IDBD: Meta-learned step-sizes via gradient correlation
3. Autostep: Tuning-free adaptation with gradient normalization

The key difference between IDBD and Autostep:
- IDBD adapts step-sizes based on gradient correlation
- Autostep additionally normalizes gradients, making it more robust

Usage:
    python examples/step1_autostep_comparison.py
"""

import numpy as np

from alberta_toolkit import (
    Autostep,
    IDBD,
    LMS,
    LinearLearner,
    AbruptChangeTarget,
    CyclicTarget,
    RandomWalkTarget,
    compare_learners,
    compute_tracking_error,
    run_learning_loop,
)


def run_comparison(
    stream_class,
    stream_name: str,
    feature_dim: int = 10,
    num_steps: int = 10000,
    seed: int = 42,
    **stream_kwargs,
) -> dict:
    """Run comparison across all optimizers on a given stream type.

    Args:
        stream_class: Stream class to use
        stream_name: Human-readable stream name
        feature_dim: Dimension of features
        num_steps: Number of steps to run
        seed: Random seed
        **stream_kwargs: Additional arguments for stream

    Returns:
        Dictionary of results per optimizer
    """
    results = {}

    # LMS configurations
    lms_configs = [0.01, 0.05, 0.1]
    for alpha in lms_configs:
        stream = stream_class(feature_dim=feature_dim, seed=seed, **stream_kwargs)
        learner = LinearLearner(optimizer=LMS(step_size=alpha))
        _, metrics = run_learning_loop(learner, stream, num_steps)
        results[f"LMS(α={alpha})"] = metrics

    # IDBD configurations
    idbd_configs = [(0.05, 0.05), (0.1, 0.1)]
    for initial_alpha, beta in idbd_configs:
        stream = stream_class(feature_dim=feature_dim, seed=seed, **stream_kwargs)
        learner = LinearLearner(
            optimizer=IDBD(initial_step_size=initial_alpha, meta_step_size=beta)
        )
        _, metrics = run_learning_loop(learner, stream, num_steps)
        results[f"IDBD(α₀={initial_alpha},β={beta})"] = metrics

    # Autostep configurations
    autostep_configs = [(0.05, 0.05), (0.1, 0.1)]
    for initial_alpha, mu in autostep_configs:
        stream = stream_class(feature_dim=feature_dim, seed=seed, **stream_kwargs)
        learner = LinearLearner(
            optimizer=Autostep(initial_step_size=initial_alpha, meta_step_size=mu)
        )
        _, metrics = run_learning_loop(learner, stream, num_steps)
        results[f"Autostep(α₀={initial_alpha},μ={mu})"] = metrics

    return results


def print_comparison_results(results: dict, stream_name: str) -> dict:
    """Print and analyze comparison results.

    Returns summary statistics dictionary.
    """
    print(f"\n{'='*70}")
    print(f"Results on {stream_name}")
    print("=" * 70)

    summary = compare_learners(results)
    sorted_learners = sorted(summary.items(), key=lambda x: x[1]["cumulative"])

    print(f"\n{'Optimizer':<30} {'Cumulative':>14} {'Mean SE':>12} {'Final 100':>12}")
    print("-" * 70)

    for name, stats in sorted_learners:
        print(
            f"{name:<30} {stats['cumulative']:>14.2f} "
            f"{stats['mean']:>12.6f} {stats['final_100_mean']:>12.6f}"
        )

    # Find best of each type
    best = {"LMS": None, "IDBD": None, "Autostep": None}
    best_error = {"LMS": float("inf"), "IDBD": float("inf"), "Autostep": float("inf")}

    for name, stats in summary.items():
        for opt_type in best:
            if name.startswith(opt_type):
                if stats["cumulative"] < best_error[opt_type]:
                    best_error[opt_type] = stats["cumulative"]
                    best[opt_type] = name

    print("\n" + "-" * 70)
    print("Best of each type:")
    for opt_type in ["LMS", "IDBD", "Autostep"]:
        if best[opt_type]:
            print(f"  {opt_type:10s}: {best[opt_type]} (error: {best_error[opt_type]:.2f})")

    return summary


def main():
    """Run comprehensive Autostep comparison."""
    print("=" * 70)
    print("Step 1: IDBD vs Autostep vs LMS Comparison")
    print("=" * 70)
    print("\nComparing three step-size strategies across different non-stationarity types.")

    # Test on different stream types
    stream_configs = [
        (RandomWalkTarget, "Random Walk (gradual drift)", {"drift_rate": 0.001}),
        (AbruptChangeTarget, "Abrupt Changes (sudden shifts)", {"change_interval": 1000}),
        (CyclicTarget, "Cyclic (repeating patterns)", {"cycle_length": 500}),
    ]

    all_summaries = {}

    for stream_class, stream_name, stream_kwargs in stream_configs:
        results = run_comparison(
            stream_class,
            stream_name,
            feature_dim=10,
            num_steps=10000,
            seed=42,
            **stream_kwargs,
        )
        summary = print_comparison_results(results, stream_name)
        all_summaries[stream_name] = summary

    # Overall analysis
    print("\n" + "=" * 70)
    print("OVERALL ANALYSIS")
    print("=" * 70)

    print("\nMethod performance across different non-stationarity types:")
    print("-" * 70)

    for stream_name, summary in all_summaries.items():
        # Find winner
        best_name = min(summary.items(), key=lambda x: x[1]["cumulative"])[0]
        best_type = "LMS" if best_name.startswith("LMS") else \
                    "IDBD" if best_name.startswith("IDBD") else "Autostep"
        print(f"  {stream_name}: Winner = {best_type}")

    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("  - LMS: Best when optimal step-size is known a priori")
    print("  - IDBD: Adapts to non-stationarity via gradient correlation")
    print("  - Autostep: More robust due to gradient normalization")
    print("  - Adaptive methods shine when optimal step-size varies over time")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
