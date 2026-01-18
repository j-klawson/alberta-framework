"""Gymnasium environment wrappers as experience streams.

This module wraps Gymnasium environments to provide temporally-uniform experience
streams compatible with the Alberta Framework's learners.

Supports multiple prediction modes:
- REWARD: Predict immediate reward from (state, action)
- NEXT_STATE: Predict next state from (state, action)
- VALUE: Predict cumulative return via TD learning
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterator

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.types import TimeStep

if TYPE_CHECKING:
    import gymnasium


class PredictionMode(Enum):
    """Mode for what the stream predicts.

    REWARD: Predict immediate reward from (state, action)
    NEXT_STATE: Predict next state from (state, action)
    VALUE: Predict cumulative return (TD learning with bootstrap)
    """

    REWARD = "reward"
    NEXT_STATE = "next_state"
    VALUE = "value"


def _flatten_space(space: "gymnasium.spaces.Space") -> int:
    """Get the flattened dimension of a Gymnasium space.

    Args:
        space: A Gymnasium space (Box, Discrete, MultiDiscrete)

    Returns:
        Integer dimension of the flattened space

    Raises:
        ValueError: If space type is not supported
    """
    import gymnasium

    if isinstance(space, gymnasium.spaces.Box):
        return int(jnp.prod(jnp.array(space.shape)))
    elif isinstance(space, gymnasium.spaces.Discrete):
        return 1
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return len(space.nvec)
    else:
        raise ValueError(
            f"Unsupported space type: {type(space).__name__}. "
            "Supported types: Box, Discrete, MultiDiscrete"
        )


def _flatten_observation(obs: Any, space: "gymnasium.spaces.Space") -> Array:
    """Flatten an observation to a 1D JAX array.

    Args:
        obs: Observation from the environment
        space: The observation space

    Returns:
        Flattened observation as a 1D JAX array
    """
    import gymnasium

    if isinstance(space, gymnasium.spaces.Box):
        return jnp.asarray(obs, dtype=jnp.float32).flatten()
    elif isinstance(space, gymnasium.spaces.Discrete):
        return jnp.array([float(obs)], dtype=jnp.float32)
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return jnp.asarray(obs, dtype=jnp.float32)
    else:
        raise ValueError(f"Unsupported space type: {type(space).__name__}")


def _flatten_action(action: Any, space: "gymnasium.spaces.Space") -> Array:
    """Flatten an action to a 1D JAX array.

    Args:
        action: Action for the environment
        space: The action space

    Returns:
        Flattened action as a 1D JAX array
    """
    import gymnasium

    if isinstance(space, gymnasium.spaces.Box):
        return jnp.asarray(action, dtype=jnp.float32).flatten()
    elif isinstance(space, gymnasium.spaces.Discrete):
        return jnp.array([float(action)], dtype=jnp.float32)
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return jnp.asarray(action, dtype=jnp.float32)
    else:
        raise ValueError(f"Unsupported space type: {type(space).__name__}")


def make_random_policy(
    env: "gymnasium.Env", seed: int = 0
) -> Callable[[Array], Any]:
    """Create a random action policy for an environment.

    Args:
        env: Gymnasium environment
        seed: Random seed

    Returns:
        A callable that takes an observation and returns a random action
    """
    import gymnasium

    rng = jr.key(seed)
    action_space = env.action_space

    def policy(obs: Array) -> Any:
        nonlocal rng
        rng, key = jr.split(rng)

        if isinstance(action_space, gymnasium.spaces.Discrete):
            return int(jr.randint(key, (), 0, action_space.n))
        elif isinstance(action_space, gymnasium.spaces.Box):
            # Sample uniformly between low and high
            low = jnp.asarray(action_space.low, dtype=jnp.float32)
            high = jnp.asarray(action_space.high, dtype=jnp.float32)
            return jr.uniform(key, action_space.shape, minval=low, maxval=high)
        elif isinstance(action_space, gymnasium.spaces.MultiDiscrete):
            nvec = action_space.nvec
            return [
                int(jr.randint(jr.fold_in(key, i), (), 0, n))
                for i, n in enumerate(nvec)
            ]
        else:
            raise ValueError(f"Unsupported action space: {type(action_space).__name__}")

    return policy


def make_epsilon_greedy_policy(
    base_policy: Callable[[Array], Any],
    env: "gymnasium.Env",
    epsilon: float = 0.1,
    seed: int = 0,
) -> Callable[[Array], Any]:
    """Wrap a policy with epsilon-greedy exploration.

    Args:
        base_policy: The greedy policy to wrap
        env: Gymnasium environment (for random action sampling)
        epsilon: Probability of taking a random action
        seed: Random seed

    Returns:
        Epsilon-greedy policy
    """
    random_policy = make_random_policy(env, seed + 1)
    rng = jr.key(seed)

    def policy(obs: Array) -> Any:
        nonlocal rng
        rng, key = jr.split(rng)

        if jr.uniform(key) < epsilon:
            return random_policy(obs)
        return base_policy(obs)

    return policy


class GymnasiumStream:
    """Experience stream from a Gymnasium environment.

    Wraps a Gymnasium environment as an infinite experience stream.
    Episodes auto-reset when terminated or truncated.

    Attributes:
        mode: Prediction mode (REWARD, NEXT_STATE, VALUE)
        gamma: Discount factor for VALUE mode
        include_action_in_features: Whether to include action in features
        episode_count: Number of completed episodes
    """

    def __init__(
        self,
        env: "gymnasium.Env",
        mode: PredictionMode = PredictionMode.REWARD,
        policy: Callable[[Array], Any] | None = None,
        gamma: float = 0.99,
        include_action_in_features: bool = True,
        seed: int = 0,
    ):
        """Initialize the Gymnasium stream.

        Args:
            env: Gymnasium environment instance
            mode: What to predict (REWARD, NEXT_STATE, VALUE)
            policy: Action selection function. If None, uses random policy
            gamma: Discount factor for VALUE mode
            include_action_in_features: If True, features = concat(obs, action).
                If False, features = obs only
            seed: Random seed for environment resets and random policy
        """
        self._env = env
        self._mode = mode
        self._gamma = gamma
        self._include_action_in_features = include_action_in_features
        self._seed = seed
        self._reset_count = 0

        # Create random policy if none provided
        if policy is None:
            self._policy = make_random_policy(env, seed)
        else:
            self._policy = policy

        # Compute dimensions
        self._obs_dim = _flatten_space(env.observation_space)
        self._action_dim = _flatten_space(env.action_space)

        if include_action_in_features:
            self._feature_dim = self._obs_dim + self._action_dim
        else:
            self._feature_dim = self._obs_dim

        # Target dimension depends on mode
        if mode == PredictionMode.NEXT_STATE:
            self._target_dim = self._obs_dim
        else:  # REWARD or VALUE
            self._target_dim = 1

        # State tracking
        self._current_obs: Array | None = None
        self._episode_count = 0
        self._step_count = 0

        # For VALUE mode: store previous prediction for bootstrap
        self._value_estimator: Callable[[Array], float] | None = None

    @property
    def feature_dim(self) -> int:
        """Return the dimension of feature vectors."""
        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """Return the dimension of target vectors."""
        return self._target_dim

    @property
    def episode_count(self) -> int:
        """Return the number of completed episodes."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Return the total number of steps taken."""
        return self._step_count

    @property
    def mode(self) -> PredictionMode:
        """Return the prediction mode."""
        return self._mode

    def set_value_estimator(self, estimator: Callable[[Array], float]) -> None:
        """Set the value estimator for proper TD learning in VALUE mode.

        Args:
            estimator: Function mapping observation features to value estimate
        """
        self._value_estimator = estimator

    def _get_reset_seed(self) -> int:
        """Get the seed for the next environment reset."""
        seed = self._seed + self._reset_count
        self._reset_count += 1
        return seed

    def _construct_features(self, obs: Array, action: Array) -> Array:
        """Construct feature vector from observation and action.

        Args:
            obs: Flattened observation
            action: Flattened action

        Returns:
            Feature vector
        """
        if self._include_action_in_features:
            return jnp.concatenate([obs, action])
        return obs

    def _construct_target(
        self,
        reward: float,
        next_obs: Array,
        terminated: bool,
    ) -> Array:
        """Construct target based on prediction mode.

        Args:
            reward: Immediate reward
            next_obs: Flattened next observation
            terminated: Whether episode terminated (not truncated)

        Returns:
            Target vector
        """
        if self._mode == PredictionMode.REWARD:
            return jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))

        elif self._mode == PredictionMode.NEXT_STATE:
            return next_obs

        elif self._mode == PredictionMode.VALUE:
            # TD target: r + gamma * V(s')
            # If terminal, bootstrap is 0
            if terminated:
                return jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))

            if self._value_estimator is not None:
                # Use provided value estimator for bootstrap
                next_value = self._value_estimator(next_obs)
            else:
                # Without estimator, use Monte Carlo-style (just reward)
                # This is a simplified bootstrap using 0
                next_value = 0.0

            target = reward + self._gamma * next_value
            return jnp.atleast_1d(jnp.array(target, dtype=jnp.float32))

        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step.

        Returns:
            TimeStep with features and target
        """
        # Reset environment if needed
        if self._current_obs is None:
            raw_obs, _ = self._env.reset(seed=self._get_reset_seed())
            self._current_obs = _flatten_observation(raw_obs, self._env.observation_space)

        # Get action from policy
        action = self._policy(self._current_obs)
        flat_action = _flatten_action(action, self._env.action_space)

        # Step environment
        raw_next_obs, reward, terminated, truncated, _ = self._env.step(action)
        next_obs = _flatten_observation(raw_next_obs, self._env.observation_space)

        # Construct features and target
        features = self._construct_features(self._current_obs, flat_action)
        target = self._construct_target(float(reward), next_obs, terminated)

        # Update state
        self._step_count += 1

        if terminated or truncated:
            self._episode_count += 1
            self._current_obs = None  # Will reset on next call
        else:
            self._current_obs = next_obs

        return TimeStep(observation=features, target=target)


class TDStream:
    """Experience stream for proper TD learning with value function bootstrap.

    This stream integrates with a learner to use its predictions for
    bootstrapping in TD targets. Unlike GymnasiumStream's VALUE mode,
    this properly uses the current value estimate V(s') from the learner.

    Usage:
        stream = TDStream(env)
        learner = LinearLearner(optimizer=IDBD())
        state = learner.init(stream.feature_dim)

        for step, timestep in enumerate(stream):
            result = learner.update(state, timestep.observation, timestep.target)
            state = result.state
            # Update stream with new value function
            stream.update_value_function(lambda x: learner.predict(state, x))
    """

    def __init__(
        self,
        env: "gymnasium.Env",
        policy: Callable[[Array], Any] | None = None,
        gamma: float = 0.99,
        include_action_in_features: bool = False,
        seed: int = 0,
    ):
        """Initialize the TD stream.

        Args:
            env: Gymnasium environment instance
            policy: Action selection function. If None, uses random policy
            gamma: Discount factor
            include_action_in_features: If True, learn Q(s,a). If False, learn V(s)
            seed: Random seed
        """
        self._env = env
        self._gamma = gamma
        self._include_action_in_features = include_action_in_features
        self._seed = seed
        self._reset_count = 0

        # Create random policy if none provided
        if policy is None:
            self._policy = make_random_policy(env, seed)
        else:
            self._policy = policy

        # Compute dimensions
        self._obs_dim = _flatten_space(env.observation_space)
        self._action_dim = _flatten_space(env.action_space)

        if include_action_in_features:
            self._feature_dim = self._obs_dim + self._action_dim
        else:
            self._feature_dim = self._obs_dim

        # State tracking
        self._current_obs: Array | None = None
        self._episode_count = 0
        self._step_count = 0

        # Value function for bootstrapping (initially returns 0)
        self._value_fn: Callable[[Array], float] = lambda x: 0.0

    @property
    def feature_dim(self) -> int:
        """Return the dimension of feature vectors."""
        return self._feature_dim

    @property
    def episode_count(self) -> int:
        """Return the number of completed episodes."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Return the total number of steps taken."""
        return self._step_count

    def update_value_function(self, value_fn: Callable[[Array], float]) -> None:
        """Update the value function used for TD bootstrapping.

        Args:
            value_fn: Function mapping features to value estimate
        """
        self._value_fn = value_fn

    def _get_reset_seed(self) -> int:
        """Get the seed for the next environment reset."""
        seed = self._seed + self._reset_count
        self._reset_count += 1
        return seed

    def _construct_features(self, obs: Array, action: Array) -> Array:
        """Construct feature vector from observation and action."""
        if self._include_action_in_features:
            return jnp.concatenate([obs, action])
        return obs

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step with TD target.

        The target is: r + gamma * V(s') where V uses the current value function.

        Returns:
            TimeStep with features and TD target
        """
        # Reset environment if needed
        if self._current_obs is None:
            raw_obs, _ = self._env.reset(seed=self._get_reset_seed())
            self._current_obs = _flatten_observation(raw_obs, self._env.observation_space)

        # Get action from policy
        action = self._policy(self._current_obs)
        flat_action = _flatten_action(action, self._env.action_space)

        # Step environment
        raw_next_obs, reward, terminated, truncated, _ = self._env.step(action)
        next_obs = _flatten_observation(raw_next_obs, self._env.observation_space)

        # Construct features
        features = self._construct_features(self._current_obs, flat_action)
        next_features = self._construct_features(next_obs, flat_action)

        # Compute TD target: r + gamma * V(s')
        if terminated:
            # Terminal state: no bootstrap
            target = jnp.atleast_1d(jnp.array(reward, dtype=jnp.float32))
        else:
            bootstrap = self._value_fn(next_features)
            target_val = reward + self._gamma * float(bootstrap)
            target = jnp.atleast_1d(jnp.array(target_val, dtype=jnp.float32))

        # Update state
        self._step_count += 1

        if terminated or truncated:
            self._episode_count += 1
            self._current_obs = None  # Will reset on next call
        else:
            self._current_obs = next_obs

        return TimeStep(observation=features, target=target)


def make_gymnasium_stream(
    env_id: str,
    mode: PredictionMode = PredictionMode.REWARD,
    policy: Callable[[Array], Any] | None = None,
    gamma: float = 0.99,
    include_action_in_features: bool = True,
    seed: int = 0,
    **env_kwargs: Any,
) -> GymnasiumStream:
    """Factory function to create a GymnasiumStream from an environment ID.

    Args:
        env_id: Gymnasium environment ID (e.g., "CartPole-v1")
        mode: What to predict (REWARD, NEXT_STATE, VALUE)
        policy: Action selection function. If None, uses random policy
        gamma: Discount factor for VALUE mode
        include_action_in_features: If True, features = concat(obs, action)
        seed: Random seed
        **env_kwargs: Additional arguments passed to gymnasium.make()

    Returns:
        GymnasiumStream wrapping the environment
    """
    import gymnasium

    env = gymnasium.make(env_id, **env_kwargs)
    return GymnasiumStream(
        env=env,
        mode=mode,
        policy=policy,
        gamma=gamma,
        include_action_in_features=include_action_in_features,
        seed=seed,
    )
