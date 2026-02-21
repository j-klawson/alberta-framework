"""LMS DQN agent: Q-learning with fixed step-size (no adaptation baseline).

Uses the same MultiHeadMLPLearner architecture as Autostep DQN but with
a fixed scalar step-size (LMS optimizer). This isolates the benefit of
per-weight step-size adaptation.
"""

from __future__ import annotations

import dm_env

from alberta_framework import (
    EMANormalizer,
    MultiHeadMLPLearner,
    ObGDBounding,
)
from benchmarks.bsuite.agents.base import AlbertaAgent


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (64, 64),
    step_size: float = 0.001,
    kappa: float = 2.0,
    normalizer_decay: float = 0.99,
    discount: float = 0.99,
    epsilon: float = 0.05,
    seed: int = 0,
    log_representation: bool = False,
    log_interval: int = 100,
) -> AlbertaAgent:
    """Create an LMS DQN agent with default hyperparameters.

    Parameters
    ----------
    obs_spec : dm_env.specs.Array
        Observation specification.
    action_spec : dm_env.specs.DiscreteArray
        Action specification.
    hidden_sizes : tuple of int
        Hidden layer sizes. Default: (64, 64).
    step_size : float
        Fixed learning rate. Default: 0.001.
    kappa : float
        ObGD bounding parameter. Default: 2.0.
    normalizer_decay : float
        EMA decay for online normalization. Default: 0.99.
    discount : float
        Q-learning discount factor. Default: 0.99.
    epsilon : float
        Exploration rate. Default: 0.05.
    seed : int
        Random seed. Default: 0.
    log_representation : bool
        Whether to log representation snapshots. Default: False.
    log_interval : int
        Steps between representation snapshots. Default: 100.

    Returns
    -------
    AlbertaAgent
        Configured LMS DQN agent.
    """
    learner = MultiHeadMLPLearner(
        n_heads=action_spec.num_values,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=kappa),
        normalizer=EMANormalizer(decay=normalizer_decay),
    )
    return AlbertaAgent(
        obs_spec=obs_spec,
        action_spec=action_spec,
        learner=learner,
        discount=discount,
        epsilon=epsilon,
        seed=seed,
        log_representation=log_representation,
        log_interval=log_interval,
    )
