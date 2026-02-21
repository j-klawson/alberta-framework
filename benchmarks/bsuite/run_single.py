#!/usr/bin/env python3
"""Run a single bsuite experiment for one agent.

Usage:
    # Continuing mode (default -- Alberta Plan aligned)
    python benchmarks/bsuite/run_single.py \
        --agent autostep --bsuite_id catch/0 --save_path output/bsuite

    # Standard episodic mode (for bsuite score compatibility)
    python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --mode standard

    # With representation logging
    python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --log-representation

    # Scythe placeholder (no-op for now)
    python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --use-scythe
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import bsuite
from bsuite import sweep
from bsuite.baselines import experiment

from alberta_framework import Timer
from benchmarks.bsuite.agents import adam_dqn, autostep_dqn, lms_dqn
from benchmarks.bsuite.configs import CONFIGS
from benchmarks.bsuite.wrappers import ContinuingWrapper

logger = logging.getLogger(__name__)

AGENT_FACTORIES = {
    "autostep": autostep_dqn.default_agent,
    "lms": lms_dqn.default_agent,
    "adam": adam_dqn.default_agent,
}


def make_agent(
    agent_type: str,
    obs_spec: Any,
    action_spec: Any,
    config_name: str | None = None,
    seed: int = 0,
    log_representation: bool = False,
    log_interval: int = 100,
) -> Any:
    """Create an agent from a type name and optional config override.

    Parameters
    ----------
    agent_type : str
        One of 'autostep', 'lms', 'adam'.
    obs_spec : dm_env.specs.Array
        Observation spec from the environment.
    action_spec : dm_env.specs.DiscreteArray
        Action spec from the environment.
    config_name : str, optional
        Config key from CONFIGS. If None, uses the agent_type as the key.
    seed : int
        Random seed.
    log_representation : bool
        Whether to log representation snapshots.
    log_interval : int
        Steps between representation snapshots.

    Returns
    -------
    Agent
        The configured agent.
    """
    config_key = config_name or agent_type
    if config_key in CONFIGS:
        config = CONFIGS[config_key]
        kwargs = dict(config.kwargs)
    else:
        kwargs = {}

    # Extract the factory based on agent_type
    actual_type = CONFIGS[config_key].agent_type if config_key in CONFIGS else agent_type
    if actual_type not in AGENT_FACTORIES:
        msg = f"Unknown agent type: {actual_type}. Choose from: {list(AGENT_FACTORIES.keys())}"
        raise ValueError(msg)

    factory = AGENT_FACTORIES[actual_type]

    # Add common kwargs
    kwargs["seed"] = seed
    if actual_type != "adam":
        kwargs["log_representation"] = log_representation
        kwargs["log_interval"] = log_interval

    return factory(obs_spec, action_spec, **kwargs)


def run_continuing(
    agent: Any,
    env: Any,
    num_steps: int,
) -> None:
    """Run agent-environment loop in continuing mode.

    Unlike bsuite's experiment.run() which is episode-based, this runs
    for a fixed number of steps with no episode boundaries.

    Parameters
    ----------
    agent : base.Agent
        The agent.
    env : ContinuingWrapper
        The continuing-wrapped environment.
    num_steps : int
        Total number of steps to run.
    """
    timestep = env.reset()
    for _ in range(num_steps):
        action = agent.select_action(timestep)
        new_timestep = env.step(action)
        agent.update(timestep, action, new_timestep)
        timestep = new_timestep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single bsuite experiment")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent config name (e.g., 'autostep', 'lms', 'adam', 'autostep_bottleneck')",
    )
    parser.add_argument(
        "--bsuite_id",
        type=str,
        required=True,
        help="bsuite experiment id (e.g., 'catch/0', 'catch_scale/3')",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/bsuite",
        help="Directory to save results (default: output/bsuite)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="continuing",
        choices=["continuing", "standard"],
        help="Environment mode (default: continuing)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps for continuing mode (default: from bsuite episode count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--log-representation",
        action="store_true",
        help="Log representation utility snapshots",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Steps between representation snapshots (default: 100)",
    )
    parser.add_argument(
        "--use-scythe",
        action="store_true",
        help="Scythe unit-replacement placeholder (currently no-op)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose terminal logging",
    )

    args = parser.parse_args()

    if args.use_scythe:
        warnings.warn(
            "Scythe integration is a placeholder and currently a no-op. "
            "It will be activated when unit-replacement is added to the framework.",
            stacklevel=2,
        )

    # Create save directory for this agent
    agent_save_path = str(Path(args.save_path) / args.agent)

    with Timer(f"bsuite {args.bsuite_id} ({args.agent}, {args.mode})"):
        if args.mode == "standard":
            # Standard episodic mode: use bsuite's built-in logging and runner
            env = bsuite.load_and_record(
                bsuite_id=args.bsuite_id,
                save_path=agent_save_path,
                logging_mode="csv",
                overwrite=args.overwrite,
            )
            agent = make_agent(
                agent_type=args.agent,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=args.agent,
                seed=args.seed,
                log_representation=args.log_representation,
                log_interval=args.log_interval,
            )
            num_episodes = sweep.EPISODES[args.bsuite_id]
            experiment.run(agent, env, num_episodes, verbose=args.verbose)
        else:
            # Continuing mode: wrap environment with ContinuingWrapper
            raw_env = bsuite.load_and_record(
                bsuite_id=args.bsuite_id,
                save_path=agent_save_path,
                logging_mode="csv",
                overwrite=args.overwrite,
            )
            env = ContinuingWrapper(raw_env, mode="continuing")
            agent = make_agent(
                agent_type=args.agent,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=args.agent,
                seed=args.seed,
                log_representation=args.log_representation,
                log_interval=args.log_interval,
            )
            # Default num_steps: estimate from bsuite episode count
            # Rough heuristic: num_episodes * 1000 steps per episode
            num_steps = args.num_steps
            if num_steps is None:
                num_episodes = sweep.EPISODES[args.bsuite_id]
                num_steps = num_episodes * 1000
                logger.info(
                    "Defaulting to %d steps (from %d episodes * 1000)",
                    num_steps,
                    num_episodes,
                )
            run_continuing(agent, env, num_steps)

    # Save representation log if enabled
    if args.log_representation and hasattr(agent, "save_representation_log"):
        rep_path = Path(agent_save_path) / f"representation_{args.bsuite_id.replace('/', '_')}.json"
        agent.save_representation_log(rep_path)
        print(f"Representation log saved to {rep_path}")


if __name__ == "__main__":
    main()
