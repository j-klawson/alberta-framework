#!/usr/bin/env python3
"""Run a sweep of bsuite experiments across all agents.

Usage:
    # Run selected experiments with all standard agents
    python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --experiments catch catch_scale

    # Include bottleneck variants
    python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --bottleneck

    # Continual multi-task sequence (same agent, multiple tasks in order)
    python benchmarks/bsuite/run_sweep.py --save_path output/bsuite \
        --continual-sequence catch/0 cartpole/0 bandit/0

    # Run all primary experiments
    python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --all-primary
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import bsuite
from bsuite import sweep
from bsuite.baselines import experiment

from alberta_framework import Timer
from benchmarks.bsuite.configs import (
    ALL_EXPERIMENTS,
    BOTTLENECK_AGENTS,
    PRIMARY_EXPERIMENTS,
    SECONDARY_EXPERIMENTS,
    STANDARD_AGENTS,
)
from benchmarks.bsuite.run_single import make_agent, run_continuing
from benchmarks.bsuite.wrappers import ContinuingWrapper

logger = logging.getLogger(__name__)


def get_bsuite_ids_for_experiment(experiment_name: str) -> list[str]:
    """Get all bsuite_ids for a given experiment name.

    Parameters
    ----------
    experiment_name : str
        Experiment name (e.g., 'catch', 'catch_scale').

    Returns
    -------
    list of str
        All bsuite_ids matching this experiment.
    """
    return [
        bsuite_id
        for bsuite_id in sweep.SWEEP
        if bsuite_id.split("/")[0] == experiment_name
    ]


def run_agent_on_id(
    agent_name: str,
    bsuite_id: str,
    save_path: str,
    mode: str = "continuing",
    num_steps: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Run a single agent on a single bsuite_id.

    Parameters
    ----------
    agent_name : str
        Agent config name.
    bsuite_id : str
        bsuite experiment id.
    save_path : str
        Base directory for results.
    mode : str
        'continuing' or 'standard'.
    num_steps : int, optional
        Steps for continuing mode.
    seed : int
        Random seed.
    overwrite : bool
        Overwrite existing results.
    verbose : bool
        Verbose logging.
    """
    agent_save_path = str(Path(save_path) / agent_name)

    try:
        if mode == "standard":
            env = bsuite.load_and_record(
                bsuite_id=bsuite_id,
                save_path=agent_save_path,
                logging_mode="csv",
                overwrite=overwrite,
            )
            agent = make_agent(
                agent_type=agent_name,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=agent_name,
                seed=seed,
            )
            num_episodes = sweep.EPISODES[bsuite_id]
            experiment.run(agent, env, num_episodes, verbose=verbose)
        else:
            raw_env = bsuite.load_and_record(
                bsuite_id=bsuite_id,
                save_path=agent_save_path,
                logging_mode="csv",
                overwrite=overwrite,
            )
            env = ContinuingWrapper(raw_env, mode="continuing")
            agent = make_agent(
                agent_type=agent_name,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=agent_name,
                seed=seed,
            )
            steps = num_steps
            if steps is None:
                num_episodes = sweep.EPISODES[bsuite_id]
                steps = num_episodes * 1000
            run_continuing(agent, env, steps)
    except Exception:
        logger.exception("Failed: %s on %s", agent_name, bsuite_id)
        raise


def run_continual_sequence(
    agent_name: str,
    bsuite_ids: list[str],
    save_path: str,
    steps_per_task: int = 10000,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    """Run a single persistent agent across a sequence of environments.

    The same agent instance (with persistent state) is run across
    multiple environments in sequence. This tests whether the trunk
    retains useful representations across task switches.

    Parameters
    ----------
    agent_name : str
        Agent config name.
    bsuite_ids : list of str
        Sequence of bsuite_ids to run in order.
    save_path : str
        Base directory for results.
    steps_per_task : int
        Steps to run on each task. Default: 10000.
    seed : int
        Random seed.
    overwrite : bool
        Overwrite existing results.
    """
    agent_save_path = str(Path(save_path) / f"{agent_name}_continual")
    agent = None

    for task_idx, bsuite_id in enumerate(bsuite_ids):
        print(f"  Task {task_idx + 1}/{len(bsuite_ids)}: {bsuite_id}")

        raw_env = bsuite.load_and_record(
            bsuite_id=bsuite_id,
            save_path=agent_save_path,
            logging_mode="csv",
            overwrite=overwrite,
        )
        env = ContinuingWrapper(raw_env, mode="continuing")

        if agent is None:
            # First task: create the agent
            agent = make_agent(
                agent_type=agent_name,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=agent_name,
                seed=seed,
            )
        # Agent persists across tasks -- same state continues

        run_continuing(agent, env, steps_per_task)
        print(f"    Completed {steps_per_task} steps on {bsuite_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bsuite sweep across agents")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/bsuite",
        help="Base directory for results (default: output/bsuite)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=None,
        help="Experiment names to run (e.g., catch catch_scale)",
    )
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Run all primary experiments (scale + noise variants)",
    )
    parser.add_argument(
        "--all-secondary",
        action="store_true",
        help="Run all secondary experiments (basic + credit assignment)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (primary + secondary)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        type=str,
        default=None,
        help="Agent names to run (default: all standard agents)",
    )
    parser.add_argument(
        "--bottleneck",
        action="store_true",
        help="Include bottleneck variants",
    )
    parser.add_argument(
        "--continual-sequence",
        nargs="+",
        type=str,
        default=None,
        help="Run continual multi-task sequence with these bsuite_ids",
    )
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=10000,
        help="Steps per task in continual sequence (default: 10000)",
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
        help="Override number of steps for continuing mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose terminal logging"
    )
    parser.add_argument(
        "--use-scythe",
        action="store_true",
        help="Scythe unit-replacement placeholder (currently no-op)",
    )

    args = parser.parse_args()

    if args.use_scythe:
        warnings.warn(
            "Scythe integration is a placeholder and currently a no-op.",
            stacklevel=2,
        )

    # Handle continual sequence mode
    if args.continual_sequence:
        agent_names = args.agents or STANDARD_AGENTS
        with Timer("Continual sequence sweep"):
            for agent_name in agent_names:
                print(f"\n=== Agent: {agent_name} ===")
                run_continual_sequence(
                    agent_name=agent_name,
                    bsuite_ids=args.continual_sequence,
                    save_path=args.save_path,
                    steps_per_task=args.steps_per_task,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
        return

    # Determine experiments
    if args.all:
        experiment_names = ALL_EXPERIMENTS
    elif args.all_primary:
        experiment_names = PRIMARY_EXPERIMENTS
    elif args.all_secondary:
        experiment_names = SECONDARY_EXPERIMENTS
    elif args.experiments:
        experiment_names = args.experiments
    else:
        experiment_names = ["catch"]  # Default to just catch

    # Determine agents
    agent_names = args.agents or STANDARD_AGENTS
    if args.bottleneck:
        agent_names = agent_names + BOTTLENECK_AGENTS

    # Collect all bsuite_ids
    all_bsuite_ids: list[str] = []
    for exp_name in experiment_names:
        ids = get_bsuite_ids_for_experiment(exp_name)
        if not ids:
            logger.warning("No bsuite_ids found for experiment: %s", exp_name)
        all_bsuite_ids.extend(ids)

    total_runs = len(agent_names) * len(all_bsuite_ids)
    print(
        f"Running {total_runs} total experiments "
        f"({len(agent_names)} agents x {len(all_bsuite_ids)} bsuite_ids)"
    )

    with Timer("bsuite sweep"):
        run_count = 0
        for agent_name in agent_names:
            for bsuite_id in all_bsuite_ids:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] {agent_name} on {bsuite_id}")
                try:
                    run_agent_on_id(
                        agent_name=agent_name,
                        bsuite_id=bsuite_id,
                        save_path=args.save_path,
                        mode=args.mode,
                        num_steps=args.num_steps,
                        seed=args.seed,
                        overwrite=args.overwrite,
                        verbose=args.verbose,
                    )
                except Exception:
                    logger.exception(
                        "Failed: %s on %s, continuing...", agent_name, bsuite_id
                    )

    print(f"\nResults saved to {args.save_path}")


if __name__ == "__main__":
    main()
