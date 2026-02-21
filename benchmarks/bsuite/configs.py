"""Hyperparameter configurations for bsuite benchmarks.

Defines standard and bottleneck variants for each agent type.
Bottleneck variants force the agent to manage limited capacity,
testing whether Autostep's per-weight adaptation allocates capacity
more efficiently than global optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a bsuite benchmark agent.

    Attributes
    ----------
    agent_type : str
        Agent factory module name (e.g., 'autostep', 'lms', 'adam').
    label : str
        Human-readable label for plots and tables.
    kwargs : dict
        Keyword arguments passed to the agent factory.
    """

    agent_type: str
    label: str
    kwargs: dict[str, Any] = field(default_factory=dict)


# Architecture variants
STANDARD = {"hidden_sizes": (64, 64)}
BOTTLENECK_SMALL = {"hidden_sizes": (16, 16)}
BOTTLENECK_TINY = {"hidden_sizes": (32,)}


# Standard configurations
CONFIGS: dict[str, AgentConfig] = {
    "autostep": AgentConfig(
        agent_type="autostep",
        label="Autostep+ObGD",
        kwargs={
            **STANDARD,
            "initial_step_size": 0.01,
            "meta_step_size": 0.01,
            "tau": 10000.0,
            "kappa": 2.0,
            "normalizer_decay": 0.99,
        },
    ),
    "autostep_bottleneck": AgentConfig(
        agent_type="autostep",
        label="Autostep+ObGD (16,16)",
        kwargs={
            **BOTTLENECK_SMALL,
            "initial_step_size": 0.01,
            "meta_step_size": 0.01,
            "tau": 10000.0,
            "kappa": 2.0,
            "normalizer_decay": 0.99,
        },
    ),
    "autostep_bottleneck_tiny": AgentConfig(
        agent_type="autostep",
        label="Autostep+ObGD (32,)",
        kwargs={
            **BOTTLENECK_TINY,
            "initial_step_size": 0.01,
            "meta_step_size": 0.01,
            "tau": 10000.0,
            "kappa": 2.0,
            "normalizer_decay": 0.99,
        },
    ),
    "lms": AgentConfig(
        agent_type="lms",
        label="LMS+ObGD",
        kwargs={
            **STANDARD,
            "step_size": 0.001,
            "kappa": 2.0,
            "normalizer_decay": 0.99,
        },
    ),
    "lms_bottleneck": AgentConfig(
        agent_type="lms",
        label="LMS+ObGD (16,16)",
        kwargs={
            **BOTTLENECK_SMALL,
            "step_size": 0.001,
            "kappa": 2.0,
            "normalizer_decay": 0.99,
        },
    ),
    "adam": AgentConfig(
        agent_type="adam",
        label="Adam",
        kwargs={
            **STANDARD,
            "learning_rate": 1e-3,
        },
    ),
    "adam_bottleneck": AgentConfig(
        agent_type="adam",
        label="Adam (16,16)",
        kwargs={
            **BOTTLENECK_SMALL,
            "learning_rate": 1e-3,
        },
    ),
}


# Experiments of interest
PRIMARY_EXPERIMENTS = [
    "catch_scale",
    "cartpole_scale",
    "bandit_scale",
    "mnist_scale",
    "catch_noise",
    "cartpole_noise",
    "mnist_noise",
]

SECONDARY_EXPERIMENTS = [
    "catch",
    "cartpole",
    "bandit",
    "mnist",
    "discounting_chain",
]

ALL_EXPERIMENTS = PRIMARY_EXPERIMENTS + SECONDARY_EXPERIMENTS

STANDARD_AGENTS = ["autostep", "lms", "adam"]
BOTTLENECK_AGENTS = ["autostep_bottleneck", "lms_bottleneck", "adam_bottleneck"]
