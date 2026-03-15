"""Horde on Random Walk: temporal GVF predictions on synthetic streams.

Demonstrates HordeLearner with a mix of gamma=0 (single-step) and
gamma>0 (temporal) prediction demons on a non-stationary random walk.

This is Step 3 of the Alberta Plan: moving from supervised prediction
to General Value Function (GVF) predictions with the Horde architecture.

Usage:
    python "examples/The Alberta Plan/Step3/horde_random_walk.py" --output-dir output/
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from alberta_framework import (
    DemonType,
    EMANormalizer,
    GVFSpec,
    HordeLearner,
    ObGDBounding,
    Timer,
    create_horde_spec,
    run_horde_learning_loop,
)
from alberta_framework.streams.synthetic import RandomWalkStream


def main(output_dir: Path, num_steps: int = 5000, seed: int = 42) -> None:
    """Run Horde learning on random walk stream."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dim = 10
    key = jr.key(seed)
    k_stream, k_learner = jr.split(key)

    # Define GVF demons
    demons = [
        # Single-step prediction demons (gamma=0, like rlsecd)
        GVFSpec(
            name="instantaneous",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        ),
        # Short-horizon temporal demon
        GVFSpec(
            name="short_horizon",
            demon_type=DemonType.PREDICTION,
            gamma=0.5,
            lamda=0.0,
            cumulant_index=0,
        ),
        # Long-horizon temporal demon
        GVFSpec(
            name="long_horizon",
            demon_type=DemonType.PREDICTION,
            gamma=0.9,
            lamda=0.0,
            cumulant_index=0,
        ),
    ]
    horde_spec = create_horde_spec(demons)

    # Create Horde learner
    horde = HordeLearner(
        horde_spec=horde_spec,
        hidden_sizes=(64, 32),
        step_size=0.1,
        sparsity=0.9,
        normalizer=EMANormalizer(decay=0.99),
        bounder=ObGDBounding(kappa=2.0),
    )

    state = horde.init(feature_dim, k_learner)

    # Generate stream data
    stream = RandomWalkStream(feature_dim=feature_dim, drift_rate=0.001)
    stream_state = stream.init(k_stream)

    observations = []
    cumulants_list = []
    for i in range(num_steps):
        timestep, stream_state = stream.step(stream_state, jnp.array(i))
        observations.append(timestep.observation)
        # All demons predict the same cumulant (target) for this demo
        cumulants_list.append(
            jnp.full(len(demons), timestep.target.squeeze())
        )

    obs_array = jnp.stack(observations)
    cum_array = jnp.stack(cumulants_list)
    # next_obs = obs shifted by 1 (standard TD setup)
    next_obs_array = jnp.concatenate([obs_array[1:], obs_array[:1]], axis=0)

    # Run learning
    with Timer("Horde learning loop"):
        result = run_horde_learning_loop(
            horde, state, obs_array, cum_array, next_obs_array
        )

    # Report results
    for i, demon in enumerate(demons):
        final_se = float(result.per_demon_metrics[-1, i, 0])
        mean_se_last100 = float(
            jnp.nanmean(result.per_demon_metrics[-100:, i, 0])
        )
        print(
            f"  {demon.name} (gamma={demon.gamma}): "
            f"final_SE={final_se:.4f}, mean_SE_last100={mean_se_last100:.4f}"
        )

    # Save config
    import json

    config = horde.to_config()
    config_path = output_dir / "horde_random_walk_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Horde on Random Walk stream"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step3"),
    )
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.output_dir, args.num_steps, args.seed)
