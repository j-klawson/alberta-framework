"""Synthetic non-stationary experience streams for testing continual learning.

These streams generate non-stationary supervised learning problems where
the target function changes over time, testing the learner's ability to
track and adapt.
"""

from collections.abc import Iterator

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.types import TimeStep


class RandomWalkTarget:
    """Non-stationary stream where target weights drift via random walk.

    The true target function is linear: y* = w_true @ x + noise
    where w_true evolves via random walk at each time step.

    This tests the learner's ability to continuously track a moving target.

    Attributes:
        feature_dim: Dimension of observation vectors
        drift_rate: Standard deviation of weight drift per step
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        drift_rate: float = 0.001,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
        seed: int = 0,
    ):
        """Initialize the random walk target stream.

        Args:
            feature_dim: Dimension of the feature/observation vectors
            drift_rate: Std dev of weight changes per step (controls non-stationarity)
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
            seed: Random seed for reproducibility
        """
        self._feature_dim = feature_dim
        self._drift_rate = drift_rate
        self._noise_std = noise_std
        self._feature_std = feature_std
        self._key = jr.key(seed)
        self._true_weights: Array | None = None

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def true_weights(self) -> Array | None:
        """Return current true weights (for debugging/visualization)."""
        return self._true_weights

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step.

        Returns:
            TimeStep with observation and target
        """
        # Split keys for different random operations
        self._key, key_init, key_drift, key_x, key_noise = jr.split(self._key, 5)

        # Initialize weights if first step
        if self._true_weights is None:
            self._true_weights = jr.normal(key_init, (self._feature_dim,), dtype=jnp.float32)
        else:
            # Random walk update to weights
            drift = jr.normal(key_drift, (self._feature_dim,), dtype=jnp.float32)
            self._true_weights = self._true_weights + self._drift_rate * drift

        # Generate observation
        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target with noise
        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        y_star = jnp.dot(self._true_weights, x) + noise

        return TimeStep(observation=x, target=jnp.atleast_1d(y_star))


class AbruptChangeTarget:
    """Non-stationary stream with sudden target weight changes.

    Target weights remain constant for a period, then abruptly change
    to new random values. Tests the learner's ability to detect and
    rapidly adapt to distribution shifts.

    Attributes:
        feature_dim: Dimension of observation vectors
        change_interval: Number of steps between weight changes
        noise_std: Standard deviation of observation noise
    """

    def __init__(
        self,
        feature_dim: int,
        change_interval: int = 1000,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
        seed: int = 0,
    ):
        """Initialize the abrupt change stream.

        Args:
            feature_dim: Dimension of feature vectors
            change_interval: Steps between abrupt weight changes
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
            seed: Random seed
        """
        self._feature_dim = feature_dim
        self._change_interval = change_interval
        self._noise_std = noise_std
        self._feature_std = feature_std
        self._key = jr.key(seed)
        self._true_weights: Array | None = None
        self._step_count = 0

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def true_weights(self) -> Array | None:
        """Return current true weights."""
        return self._true_weights

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step."""
        self._key, key_weights, key_x, key_noise = jr.split(self._key, 4)

        # Initialize or change weights at intervals
        if self._true_weights is None or self._step_count % self._change_interval == 0:
            self._true_weights = jr.normal(key_weights, (self._feature_dim,), dtype=jnp.float32)

        self._step_count += 1

        # Generate observation
        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target
        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        y_star = jnp.dot(self._true_weights, x) + noise

        return TimeStep(observation=x, target=jnp.atleast_1d(y_star))


class SuttonExperiment1Stream:
    """Non-stationary stream replicating Experiment 1 from Sutton 1992.

    This stream implements the exact task from Sutton's IDBD paper:
    - 20 real-valued inputs drawn from N(0, 1)
    - Only first 5 inputs are relevant (weights are ±1)
    - Last 15 inputs are irrelevant (weights are 0)
    - Every change_interval steps, one of the 5 relevant signs is flipped

    Reference: Sutton, R.S. (1992). "Adapting Bias by Gradient Descent:
    An Incremental Version of Delta-Bar-Delta"

    Attributes:
        num_relevant: Number of relevant inputs (default 5)
        num_irrelevant: Number of irrelevant inputs (default 15)
        change_interval: Steps between sign changes (default 20)
    """

    def __init__(
        self,
        num_relevant: int = 5,
        num_irrelevant: int = 15,
        change_interval: int = 20,
        seed: int = 0,
    ):
        """Initialize the Sutton Experiment 1 stream.

        Args:
            num_relevant: Number of relevant inputs with ±1 weights
            num_irrelevant: Number of irrelevant inputs with 0 weights
            change_interval: Number of steps between sign flips
            seed: Random seed for reproducibility
        """
        self._num_relevant = num_relevant
        self._num_irrelevant = num_irrelevant
        self._change_interval = change_interval
        self._key = jr.key(seed)
        self._step_count = 0

        # Initialize signs for relevant inputs (all +1 initially)
        self._signs: Array | None = None

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._num_relevant + self._num_irrelevant

    @property
    def true_weights(self) -> Array | None:
        """Return current true weights (signs for relevant, zeros for irrelevant)."""
        if self._signs is None:
            return None
        zeros = jnp.zeros(self._num_irrelevant, dtype=jnp.float32)
        return jnp.concatenate([self._signs, zeros])

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step.

        At each step:
        1. If at a change interval, flip one random sign
        2. Generate random inputs from N(0, 1)
        3. Compute target as sum of relevant inputs weighted by signs

        Returns:
            TimeStep with observation and target
        """
        self._key, key_x, key_which = jr.split(self._key, 3)

        # Initialize signs if first step
        if self._signs is None:
            self._signs = jnp.ones(self._num_relevant, dtype=jnp.float32)

        # Flip one random sign at change intervals (but not at step 0)
        if self._step_count > 0 and self._step_count % self._change_interval == 0:
            # Select which sign to flip (0 to num_relevant-1)
            idx_to_flip = jr.randint(key_which, (), 0, self._num_relevant)
            # Flip the sign: multiply by -1 at the selected index
            flip_mask = jnp.ones(self._num_relevant, dtype=jnp.float32)
            flip_mask = flip_mask.at[idx_to_flip].set(-1.0)
            self._signs = self._signs * flip_mask

        self._step_count += 1

        # Generate observation from N(0, 1)
        x = jr.normal(key_x, (self.feature_dim,), dtype=jnp.float32)

        # Compute target: sum of first num_relevant inputs weighted by signs
        # y* = s1*x1 + s2*x2 + ... + s_k*x_k (no noise)
        y_star = jnp.dot(self._signs, x[: self._num_relevant])

        return TimeStep(observation=x, target=jnp.atleast_1d(y_star))


class CyclicTarget:
    """Non-stationary stream that cycles between known weight configurations.

    Weights cycle through a fixed set of configurations. Tests whether
    the learner can re-adapt quickly to previously seen targets.

    Attributes:
        feature_dim: Dimension of observation vectors
        cycle_length: Number of steps per configuration before switching
        num_configurations: Number of weight configurations to cycle through
    """

    def __init__(
        self,
        feature_dim: int,
        cycle_length: int = 500,
        num_configurations: int = 4,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
        seed: int = 0,
    ):
        """Initialize the cyclic target stream.

        Args:
            feature_dim: Dimension of feature vectors
            cycle_length: Steps spent in each configuration
            num_configurations: Number of configurations to cycle through
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
            seed: Random seed
        """
        self._feature_dim = feature_dim
        self._cycle_length = cycle_length
        self._num_configurations = num_configurations
        self._noise_std = noise_std
        self._feature_std = feature_std
        self._key = jr.key(seed)
        self._step_count = 0

        # Pre-generate all weight configurations
        key_configs, self._key = jr.split(self._key)
        self._configurations = jr.normal(
            key_configs, (num_configurations, feature_dim), dtype=jnp.float32
        )

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def true_weights(self) -> Array:
        """Return current true weights."""
        config_idx = (self._step_count // self._cycle_length) % self._num_configurations
        return self._configurations[config_idx]

    @property
    def current_configuration_index(self) -> int:
        """Return the index of the current weight configuration."""
        return (self._step_count // self._cycle_length) % self._num_configurations

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TimeStep:
        """Generate the next time step."""
        self._key, key_x, key_noise = jr.split(self._key, 3)

        # Get current configuration
        config_idx = (self._step_count // self._cycle_length) % self._num_configurations
        true_weights = self._configurations[config_idx]

        self._step_count += 1

        # Generate observation
        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target
        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        y_star = jnp.dot(true_weights, x) + noise

        return TimeStep(observation=x, target=jnp.atleast_1d(y_star))
