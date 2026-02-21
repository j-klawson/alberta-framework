"""Adam DQN agent: standalone haiku/optax baseline.

This agent does NOT wrap the Alberta Framework. It uses the same (64, 64)
MLP architecture but with standard Adam optimization via optax. No LayerNorm,
no sparse init, no ObGD bounding. This makes performance differences directly
attributable to Autostep + ObGD + normalization.
"""

from __future__ import annotations

from typing import Any

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from bsuite.baselines import base


class AdamDQNAgent(base.Agent):  # type: ignore[misc]
    """Q-learning agent using haiku/optax Adam optimizer.

    Standard DQN architecture without Alberta Framework components.
    Serves as an external baseline to isolate the benefit of Autostep,
    ObGD bounding, and online normalization.

    Parameters
    ----------
    obs_spec : dm_env.specs.Array
        Observation specification.
    action_spec : dm_env.specs.DiscreteArray
        Action specification.
    hidden_sizes : tuple of int
        Hidden layer sizes. Default: (64, 64).
    learning_rate : float
        Adam learning rate. Default: 1e-3.
    discount : float
        Q-learning discount factor. Default: 0.99.
    epsilon : float
        Exploration rate. Default: 0.05.
    seed : int
        Random seed. Default: 0.
    """

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        hidden_sizes: tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon: float = 0.05,
        seed: int = 0,
    ) -> None:
        self._num_actions: int = action_spec.num_values
        self._discount = discount
        self._epsilon = epsilon
        self._key = jr.key(seed)

        feature_dim = int(np.prod(obs_spec.shape))

        # Build haiku network
        def network_fn(obs: jnp.ndarray) -> jnp.ndarray:
            layers: list[Any] = []
            for h in hidden_sizes:
                layers.extend([hk.Linear(h), jax.nn.leaky_relu])
            layers.append(hk.Linear(self._num_actions))
            return hk.Sequential(layers)(obs)  # type: ignore[no-any-return]

        self._network: Any = hk.without_apply_rng(hk.transform(network_fn))

        # Initialize
        dummy_obs = jnp.zeros(feature_dim, dtype=jnp.float32)
        self._key, init_key = jr.split(self._key)
        self._params = self._network.init(init_key, dummy_obs)

        # Optimizer
        self._optimizer = optax.adam(learning_rate)
        self._opt_state = self._optimizer.init(self._params)

        # JIT-compiled functions
        self._jit_apply = jax.jit(self._network.apply)
        self._jit_update = jax.jit(self._sgd_step)
        self._step_count = 0

    @property
    def step_count(self) -> int:
        """Number of update steps taken."""
        return self._step_count

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Select action via epsilon-greedy over Q-values."""
        self._key, subkey = jr.split(self._key)
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        q_values = self._jit_apply(self._params, obs)

        explore = jr.uniform(subkey) < self._epsilon
        self._key, subkey = jr.split(self._key)

        if explore:
            action = int(jr.randint(subkey, (), 0, self._num_actions))
        else:
            max_q = jnp.max(q_values)
            is_max = q_values >= max_q - 1e-6
            probs = is_max / jnp.sum(is_max)
            action = int(jr.choice(subkey, self._num_actions, p=probs))

        return action

    def _sgd_step(
        self,
        params: Any,
        opt_state: Any,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        target: jnp.ndarray,
    ) -> tuple[Any, Any, jnp.ndarray]:
        """Single SGD update step."""

        def loss_fn(p: Any) -> jnp.ndarray:
            q_values = self._network.apply(p, obs)
            q_a = q_values[action]
            return jnp.square(target - q_a)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self._optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        """Update the agent with a transition."""
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        next_obs = jnp.asarray(new_timestep.observation, dtype=jnp.float32).flatten()

        discount = new_timestep.discount
        if discount is None:
            discount = self._discount

        # Q-learning target
        next_q = self._jit_apply(self._params, next_obs)
        td_target = new_timestep.reward + float(discount) * jnp.max(next_q)

        action_arr = jnp.array(action, dtype=jnp.int32)
        target_arr = jnp.array(td_target, dtype=jnp.float32)

        self._params, self._opt_state, _ = self._jit_update(
            self._params, self._opt_state, obs, action_arr, target_arr
        )
        self._step_count += 1


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    discount: float = 0.99,
    epsilon: float = 0.05,
    seed: int = 0,
) -> AdamDQNAgent:
    """Create an Adam DQN agent with default hyperparameters.

    Parameters
    ----------
    obs_spec : dm_env.specs.Array
        Observation specification.
    action_spec : dm_env.specs.DiscreteArray
        Action specification.
    hidden_sizes : tuple of int
        Hidden layer sizes. Default: (64, 64).
    learning_rate : float
        Adam learning rate. Default: 1e-3.
    discount : float
        Q-learning discount factor. Default: 0.99.
    epsilon : float
        Exploration rate. Default: 0.05.
    seed : int
        Random seed. Default: 0.

    Returns
    -------
    AdamDQNAgent
        Configured Adam DQN agent.
    """
    return AdamDQNAgent(
        obs_spec=obs_spec,
        action_spec=action_spec,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        seed=seed,
    )
