"""AlbertaAgent: bridges bsuite's Agent ABC to MultiHeadMLPLearner.

Each action maps to a prediction head (Q-learning with multi-head MLP).
NaN target masking ensures only the taken action's head is updated per step.
The shared trunk learns features -- directly testing Step 2's representation learning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dm_env
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from bsuite.baselines import base

from alberta_framework import MultiHeadMLPLearner, MultiHeadMLPState


class AlbertaAgent(base.Agent):  # type: ignore[misc]
    """Bridges bsuite's mutable Agent interface to the framework's immutable MultiHeadMLPLearner.

    Uses MultiHeadMLPLearner with n_heads = num_actions as the Q-function.
    Each head predicts Q(s, a_i). Only the taken action's head gets a target
    on each step (NaN masking for all others).

    Parameters
    ----------
    obs_spec : dm_env.specs.Array
        Observation specification from the environment.
    action_spec : dm_env.specs.DiscreteArray
        Action specification from the environment.
    learner : MultiHeadMLPLearner
        Pre-configured multi-head MLP learner with n_heads = num_actions.
    discount : float
        Discount factor for Q-learning targets. Default: 0.99.
    epsilon : float
        Exploration rate for epsilon-greedy action selection. Default: 0.05.
    seed : int
        Random seed. Default: 0.
    log_representation : bool
        Whether to log representation utility snapshots. Default: False.
    log_interval : int
        Steps between representation snapshots. Default: 100.
    """

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        learner: MultiHeadMLPLearner,
        discount: float = 0.99,
        epsilon: float = 0.05,
        seed: int = 0,
        log_representation: bool = False,
        log_interval: int = 100,
    ) -> None:
        self._learner = learner
        self._num_actions: int = action_spec.num_values
        self._discount = discount
        self._epsilon = epsilon
        self._key = jr.key(seed)

        # Validate n_heads matches action space
        if learner.n_heads != self._num_actions:
            msg = (
                f"Learner n_heads ({learner.n_heads}) must match "
                f"num_actions ({self._num_actions})"
            )
            raise ValueError(msg)

        # Initialize learner state
        feature_dim = int(np.prod(obs_spec.shape))
        self._state: MultiHeadMLPState = learner.init(
            feature_dim=feature_dim, key=jr.key(seed)
        )

        # JIT-compiled functions
        self._jit_predict = jax.jit(learner.predict)
        self._jit_update = jax.jit(learner.update)

        # Representation utility logging
        self._log_representation = log_representation
        self._log_interval = log_interval
        self._step_count = 0
        self._representation_log: list[dict[str, Any]] = []

    @property
    def state(self) -> MultiHeadMLPState:
        """Current learner state (for testing/inspection)."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of update steps taken."""
        return self._step_count

    @property
    def representation_log(self) -> list[dict[str, Any]]:
        """Representation utility snapshots."""
        return self._representation_log

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Select action via epsilon-greedy over Q-values.

        Parameters
        ----------
        timestep : dm_env.TimeStep
            Current environment timestep.

        Returns
        -------
        int
            Selected action.
        """
        self._key, subkey = jr.split(self._key)
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        q_values = self._jit_predict(self._state, obs)

        # Epsilon-greedy with random tie-breaking
        explore = jr.uniform(subkey) < self._epsilon
        self._key, subkey = jr.split(self._key)

        if explore:
            action = int(jr.randint(subkey, (), 0, self._num_actions))
        else:
            # Break ties randomly among max Q-values
            max_q = jnp.max(q_values)
            is_max = q_values >= max_q - 1e-6
            # Uniform among maximal actions
            probs = is_max / jnp.sum(is_max)
            action = int(jr.choice(subkey, self._num_actions, p=probs))

        return action

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        """Update the agent with a transition.

        Computes a Q-learning TD target and updates only the taken action's
        head via NaN masking.

        Parameters
        ----------
        timestep : dm_env.TimeStep
            Timestep before the action.
        action : int
            Action taken.
        new_timestep : dm_env.TimeStep
            Timestep after the action.
        """
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        next_obs = jnp.asarray(new_timestep.observation, dtype=jnp.float32).flatten()

        # Use environment's discount signal
        # - ContinuingWrapper: discount=0 at pseudo-boundaries, gamma otherwise
        # - Standard episodic: discount=0 at true terminals
        discount = new_timestep.discount
        if discount is None:
            discount = self._discount

        # Q-learning target: r + discount * max_a' Q(s', a')
        next_q = self._jit_predict(self._state, next_obs)
        td_target = new_timestep.reward + float(discount) * jnp.max(next_q)

        # NaN-masked targets: only the taken action gets a target
        targets = jnp.full(self._num_actions, jnp.nan)
        targets = targets.at[action].set(td_target)

        # Update learner
        result = self._jit_update(self._state, obs, targets)
        self._state = result.state

        self._step_count += 1

        # Representation logging
        if self._log_representation and self._step_count % self._log_interval == 0:
            self._log_representation_snapshot()

    def _log_representation_snapshot(self) -> None:
        """Record a snapshot of the representation utility."""
        snapshot: dict[str, Any] = {"step": self._step_count}

        # Per-head mean step-sizes (from optimizer states)
        head_step_sizes = []
        for i in range(self._num_actions):
            w_opt, _b_opt = self._state.head_optimizer_states[i]
            if hasattr(w_opt, "step_sizes"):
                head_step_sizes.append(float(jnp.mean(w_opt.step_sizes)))
            elif hasattr(w_opt, "step_size"):
                head_step_sizes.append(float(w_opt.step_size))
            else:
                head_step_sizes.append(0.0)
        snapshot["head_step_sizes"] = head_step_sizes

        # Trunk trace magnitudes per layer
        trunk_trace_norms = []
        for i in range(0, len(self._state.trunk_traces), 2):
            w_trace = self._state.trunk_traces[i]
            trunk_trace_norms.append(float(jnp.mean(jnp.abs(w_trace))))
        snapshot["trunk_trace_norms"] = trunk_trace_norms

        # Trunk optimizer step-sizes per layer (if adaptive)
        trunk_step_sizes = []
        for i in range(0, len(self._state.trunk_optimizer_states), 2):
            opt_state = self._state.trunk_optimizer_states[i]
            if hasattr(opt_state, "step_sizes"):
                trunk_step_sizes.append(float(jnp.mean(opt_state.step_sizes)))
            elif hasattr(opt_state, "step_size"):
                trunk_step_sizes.append(float(opt_state.step_size))
            else:
                trunk_step_sizes.append(0.0)
        snapshot["trunk_step_sizes"] = trunk_step_sizes

        self._representation_log.append(snapshot)

    def save_representation_log(self, path: str | Path) -> None:
        """Save representation log to JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._representation_log, f, indent=2)

    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Get Q-values for an observation (for analysis/debugging).

        Parameters
        ----------
        observation : np.ndarray
            Raw observation from the environment.

        Returns
        -------
        np.ndarray
            Q-values for each action.
        """
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        q_values = self._jit_predict(self._state, obs)
        return np.asarray(q_values)
