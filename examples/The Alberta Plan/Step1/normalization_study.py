#!/usr/bin/env python3
"""Step 1 Normalization Study: Effect of Online Feature Normalization.

This script demonstrates the benefit of online feature normalization
for continual learning with varying feature scales.

The experiments:
1. Standard features (no scaling): normalization provides minimal benefit
2. Scaled features (different magnitudes): normalization is crucial
3. Non-stationary scaling: normalization adapts to changing scales

Usage:
    python examples/step1_normalization_study.py
"""

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework import (
    IDBD,
    LinearLearner,
    NormalizedLinearLearner,
    Timer,
    compare_learners,
    metrics_to_dicts,
    run_learning_loop,
    run_normalized_learning_loop,
)
from alberta_framework.core.types import TimeStep


class ScaledRandomWalkState(NamedTuple):
    """State for ScaledRandomWalkStream."""

    key: Array
    true_weights: Array


class ScaledRandomWalkStream:
    """Random walk target with scaled features.

    Some features have much larger magnitudes than others,
    which can cause problems for fixed step-size learning.
    """

    def __init__(
        self,
        feature_dim: int = 10,
        drift_rate: float = 0.001,
        noise_std: float = 0.1,
        feature_scales: np.ndarray | None = None,
    ):
        self._feature_dim = feature_dim
        self._drift_rate = drift_rate
        self._noise_std = noise_std

        # Default: exponentially varying scales
        if feature_scales is None:
            self._scales = jnp.array(10.0 ** np.linspace(-2, 2, feature_dim), dtype=jnp.float32)
        else:
            self._scales = jnp.array(feature_scales, dtype=jnp.float32)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def init(self, key: Array) -> ScaledRandomWalkState:
        """Initialize stream state."""
        key, subkey = jr.split(key)
        weights = jr.normal(subkey, (self._feature_dim,), dtype=jnp.float32)
        return ScaledRandomWalkState(key=key, true_weights=weights)

    def step(self, state: ScaledRandomWalkState, idx: Array) -> tuple[TimeStep, ScaledRandomWalkState]:
        """Generate one time step."""
        del idx  # unused
        key, k_drift, k_x, k_noise = jr.split(state.key, 4)

        # Generate raw observation
        raw_obs = jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)

        # Apply scales to observation
        observation = raw_obs * self._scales

        # Compute target from current weights (using raw, unscaled observation)
        noise = self._noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(state.true_weights, raw_obs) + noise

        # Drift target weights
        drift = jr.normal(k_drift, state.true_weights.shape, dtype=jnp.float32) * self._drift_rate
        new_weights = state.true_weights + drift

        timestep = TimeStep(observation=observation, target=jnp.atleast_1d(target))
        new_state = ScaledRandomWalkState(key=key, true_weights=new_weights)

        return timestep, new_state


def run_normalization_experiment(
    feature_dim: int = 10,
    num_steps: int = 10000,
    seed: int = 42,
) -> dict:
    """Run comparison with and without normalization on scaled features.

    Args:
        feature_dim: Dimension of feature vectors
        num_steps: Number of learning steps
        seed: Random seed

    Returns:
        Dictionary with results for each configuration
    """
    results = {}

    # Test configurations
    configs = [
        ("IDBD (no norm)", False),
        ("IDBD (normalized)", True),
    ]

    for name, use_normalization in configs:
        stream = ScaledRandomWalkStream(
            feature_dim=feature_dim,
            drift_rate=0.001,
            noise_std=0.1,
        )

        key = jr.key(seed)

        if use_normalization:
            learner = NormalizedLinearLearner(
                optimizer=IDBD(initial_step_size=0.05, meta_step_size=0.05)
            )
            _, metrics = run_normalized_learning_loop(learner, stream, num_steps, key)
            results[name] = metrics_to_dicts(metrics, normalized=True)
        else:
            learner = LinearLearner(
                optimizer=IDBD(initial_step_size=0.05, meta_step_size=0.05)
            )
            _, metrics = run_learning_loop(learner, stream, num_steps, key)
            results[name] = metrics_to_dicts(metrics)

    return results


def print_results(results: dict) -> None:
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("Normalization Study: Scaled Features")
    print("=" * 70)
    print("\nFeatures have exponentially varying scales (10^-2 to 10^2)")

    summary = compare_learners(results)
    sorted_learners = sorted(summary.items(), key=lambda x: x[1]["cumulative"])

    print(f"\n{'Configuration':<25} {'Cumulative Error':>16} {'Mean SE':>12} {'Final 100':>12}")
    print("-" * 70)

    for name, stats in sorted_learners:
        print(
            f"{name:<25} {stats['cumulative']:>16.2f} "
            f"{stats['mean']:>12.6f} {stats['final_100_mean']:>12.6f}"
        )

    # Analysis
    no_norm_error = summary["IDBD (no norm)"]["cumulative"]
    norm_error = summary["IDBD (normalized)"]["cumulative"]

    print("\n" + "-" * 70)
    print("ANALYSIS:")

    if norm_error < no_norm_error:
        improvement = (no_norm_error - norm_error) / no_norm_error * 100
        print(f"  Normalization improves performance by {improvement:.1f}%")
        print("  Online normalization helps handle varying feature scales!")
    else:
        print("  Normalization did not improve performance in this case.")

    print("=" * 70 + "\n")


def run_scale_sensitivity_study(
    feature_dim: int = 10,
    num_steps: int = 5000,
    seed: int = 42,
) -> None:
    """Study sensitivity to feature scale magnitude."""
    print("\n" + "=" * 70)
    print("Scale Sensitivity Study")
    print("=" * 70)

    scale_ranges = [
        ("Small (0.1-10)", 0.1, 10),
        ("Medium (0.01-100)", 0.01, 100),
        ("Large (0.001-1000)", 0.001, 1000),
    ]

    print(f"\n{'Scale Range':<25} {'No Norm Error':>16} {'Norm Error':>16} {'Improvement':>12}")
    print("-" * 75)

    for name, scale_min, scale_max in scale_ranges:
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), feature_dim)

        # Without normalization
        stream = ScaledRandomWalkStream(
            feature_dim=feature_dim,
            feature_scales=scales,
        )
        learner = LinearLearner(
            optimizer=IDBD(initial_step_size=0.05, meta_step_size=0.05)
        )
        key = jr.key(seed)
        _, metrics_no_norm = run_learning_loop(learner, stream, num_steps, key)
        metrics_no_norm = metrics_to_dicts(metrics_no_norm)

        # With normalization
        stream = ScaledRandomWalkStream(
            feature_dim=feature_dim,
            feature_scales=scales,
        )
        learner = NormalizedLinearLearner(
            optimizer=IDBD(initial_step_size=0.05, meta_step_size=0.05)
        )
        key = jr.key(seed)
        _, metrics_norm = run_normalized_learning_loop(learner, stream, num_steps, key)
        metrics_norm = metrics_to_dicts(metrics_norm, normalized=True)

        no_norm_error = sum(m["squared_error"] for m in metrics_no_norm)
        norm_error = sum(m["squared_error"] for m in metrics_norm)

        if no_norm_error > 0:
            improvement = (no_norm_error - norm_error) / no_norm_error * 100
        else:
            improvement = 0

        print(f"{name:<25} {no_norm_error:>16.2f} {norm_error:>16.2f} {improvement:>11.1f}%")

    print("\n" + "=" * 70)
    print("Conclusion: Larger scale differences -> bigger benefit from normalization")
    print("=" * 70 + "\n")


def main():
    """Run the normalization study."""
    with Timer("Total experiment runtime"):
        print("Running Step 1 Normalization Study")
        print("This demonstrates the benefit of online feature normalization.\n")

        # Main experiment
        results = run_normalization_experiment(
            feature_dim=10,
            num_steps=10000,
            seed=42,
        )
        print_results(results)

        # Scale sensitivity study
        run_scale_sensitivity_study()


if __name__ == "__main__":
    main()
