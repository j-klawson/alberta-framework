"""Synthetic non-stationary experience streams for testing continual learning.

These streams generate non-stationary supervised learning problems where
the target function changes over time, testing the learner's ability to
track and adapt.

All streams use JAX-compatible pure functions that work with jax.lax.scan.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.types import TimeStep
from alberta_framework.streams.base import ScanStream


class RandomWalkState(NamedTuple):
    """State for RandomWalkStream.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current true target weights
    """

    key: Array
    true_weights: Array


class RandomWalkStream:
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
    ):
        """Initialize the random walk target stream.

        Args:
            feature_dim: Dimension of the feature/observation vectors
            drift_rate: Std dev of weight changes per step (controls non-stationarity)
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        self._feature_dim = feature_dim
        self._drift_rate = drift_rate
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    def init(self, key: Array) -> RandomWalkState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random weights
        """
        key, subkey = jr.split(key)
        weights = jr.normal(subkey, (self._feature_dim,), dtype=jnp.float32)
        return RandomWalkState(key=key, true_weights=weights)

    def step(self, state: RandomWalkState, idx: Array) -> tuple[TimeStep, RandomWalkState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, k_drift, k_x, k_noise = jr.split(state.key, 4)

        # Drift weights
        drift = jr.normal(k_drift, state.true_weights.shape, dtype=jnp.float32)
        new_weights = state.true_weights + self._drift_rate * drift

        # Generate observation and target
        x = self._feature_std * jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)
        noise = self._noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, x) + noise

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = RandomWalkState(key=key, true_weights=new_weights)

        return timestep, new_state


class AbruptChangeState(NamedTuple):
    """State for AbruptChangeStream.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current true target weights
        step_count: Number of steps taken
    """

    key: Array
    true_weights: Array
    step_count: Array


class AbruptChangeStream:
    """Non-stationary stream with sudden target weight changes.

    Target weights remain constant for a period, then abruptly change
    to new random values. Tests the learner's ability to detect and
    rapidly adapt to distribution shifts.

    Attributes:
        feature_dim: Dimension of observation vectors
        change_interval: Number of steps between weight changes
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        change_interval: int = 1000,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the abrupt change stream.

        Args:
            feature_dim: Dimension of feature vectors
            change_interval: Steps between abrupt weight changes
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        self._feature_dim = feature_dim
        self._change_interval = change_interval
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    def init(self, key: Array) -> AbruptChangeState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state
        """
        key, subkey = jr.split(key)
        weights = jr.normal(subkey, (self._feature_dim,), dtype=jnp.float32)
        return AbruptChangeState(
            key=key,
            true_weights=weights,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: AbruptChangeState, idx: Array) -> tuple[TimeStep, AbruptChangeState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, key_weights, key_x, key_noise = jr.split(state.key, 4)

        # Determine if we should change weights
        should_change = state.step_count % self._change_interval == 0

        # Generate new weights (always generated but only used if should_change)
        new_random_weights = jr.normal(key_weights, (self._feature_dim,), dtype=jnp.float32)

        # Use jnp.where to conditionally update weights (JIT-compatible)
        new_weights = jnp.where(should_change, new_random_weights, state.true_weights)

        # Generate observation
        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target
        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, x) + noise

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = AbruptChangeState(
            key=key,
            true_weights=new_weights,
            step_count=state.step_count + 1,
        )

        return timestep, new_state


class SuttonExperiment1State(NamedTuple):
    """State for SuttonExperiment1Stream.

    Attributes:
        key: JAX random key for generating randomness
        signs: Signs (+1/-1) for the relevant inputs
        step_count: Number of steps taken
    """

    key: Array
    signs: Array
    step_count: Array


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
    ):
        """Initialize the Sutton Experiment 1 stream.

        Args:
            num_relevant: Number of relevant inputs with ±1 weights
            num_irrelevant: Number of irrelevant inputs with 0 weights
            change_interval: Number of steps between sign flips
        """
        self._num_relevant = num_relevant
        self._num_irrelevant = num_irrelevant
        self._change_interval = change_interval

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._num_relevant + self._num_irrelevant

    def init(self, key: Array) -> SuttonExperiment1State:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with all +1 signs
        """
        signs = jnp.ones(self._num_relevant, dtype=jnp.float32)
        return SuttonExperiment1State(
            key=key,
            signs=signs,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self, state: SuttonExperiment1State, idx: Array
    ) -> tuple[TimeStep, SuttonExperiment1State]:
        """Generate one time step.

        At each step:
        1. If at a change interval (and not step 0), flip one random sign
        2. Generate random inputs from N(0, 1)
        3. Compute target as sum of relevant inputs weighted by signs

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, key_x, key_which = jr.split(state.key, 3)

        # Determine if we should flip a sign (not at step 0)
        should_flip = (state.step_count > 0) & (state.step_count % self._change_interval == 0)

        # Select which sign to flip
        idx_to_flip = jr.randint(key_which, (), 0, self._num_relevant)

        # Create flip mask
        flip_mask = jnp.where(
            jnp.arange(self._num_relevant) == idx_to_flip,
            jnp.array(-1.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )

        # Apply flip mask conditionally
        new_signs = jnp.where(should_flip, state.signs * flip_mask, state.signs)

        # Generate observation from N(0, 1)
        x = jr.normal(key_x, (self.feature_dim,), dtype=jnp.float32)

        # Compute target: sum of first num_relevant inputs weighted by signs
        target = jnp.dot(new_signs, x[: self._num_relevant])

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = SuttonExperiment1State(
            key=key,
            signs=new_signs,
            step_count=state.step_count + 1,
        )

        return timestep, new_state


class CyclicState(NamedTuple):
    """State for CyclicStream.

    Attributes:
        key: JAX random key for generating randomness
        configurations: Pre-generated weight configurations
        step_count: Number of steps taken
    """

    key: Array
    configurations: Array
    step_count: Array


class CyclicStream:
    """Non-stationary stream that cycles between known weight configurations.

    Weights cycle through a fixed set of configurations. Tests whether
    the learner can re-adapt quickly to previously seen targets.

    Attributes:
        feature_dim: Dimension of observation vectors
        cycle_length: Number of steps per configuration before switching
        num_configurations: Number of weight configurations to cycle through
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        cycle_length: int = 500,
        num_configurations: int = 4,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the cyclic target stream.

        Args:
            feature_dim: Dimension of feature vectors
            cycle_length: Steps spent in each configuration
            num_configurations: Number of configurations to cycle through
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        self._feature_dim = feature_dim
        self._cycle_length = cycle_length
        self._num_configurations = num_configurations
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    def init(self, key: Array) -> CyclicState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with pre-generated configurations
        """
        key, key_configs = jr.split(key)
        configurations = jr.normal(
            key_configs,
            (self._num_configurations, self._feature_dim),
            dtype=jnp.float32,
        )
        return CyclicState(
            key=key,
            configurations=configurations,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: CyclicState, idx: Array) -> tuple[TimeStep, CyclicState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, key_x, key_noise = jr.split(state.key, 3)

        # Get current configuration index
        config_idx = (state.step_count // self._cycle_length) % self._num_configurations
        true_weights = state.configurations[config_idx]

        # Generate observation
        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target
        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        target = jnp.dot(true_weights, x) + noise

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = CyclicState(
            key=key,
            configurations=state.configurations,
            step_count=state.step_count + 1,
        )

        return timestep, new_state


class PeriodicChangeState(NamedTuple):
    """State for PeriodicChangeStream.

    Attributes:
        key: JAX random key for generating randomness
        base_weights: Base target weights (center of oscillation)
        phases: Per-weight phase offsets
        step_count: Number of steps taken
    """

    key: Array
    base_weights: Array
    phases: Array
    step_count: Array


class PeriodicChangeStream:
    """Non-stationary stream where target weights oscillate sinusoidally.

    Target weights follow: w(t) = base + amplitude * sin(2π * t / period + phase)
    where each weight has a random phase offset for diversity.

    This tests the learner's ability to track predictable periodic changes,
    which is qualitatively different from random drift or abrupt changes.

    Attributes:
        feature_dim: Dimension of observation vectors
        period: Number of steps for one complete oscillation
        amplitude: Magnitude of weight oscillation
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        period: int = 1000,
        amplitude: float = 1.0,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the periodic change stream.

        Args:
            feature_dim: Dimension of feature vectors
            period: Steps for one complete oscillation cycle
            amplitude: Magnitude of weight oscillations around base
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        self._feature_dim = feature_dim
        self._period = period
        self._amplitude = amplitude
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    def init(self, key: Array) -> PeriodicChangeState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random base weights and phases
        """
        key, key_weights, key_phases = jr.split(key, 3)
        base_weights = jr.normal(key_weights, (self._feature_dim,), dtype=jnp.float32)
        # Random phases in [0, 2π) for each weight
        phases = jr.uniform(key_phases, (self._feature_dim,), minval=0.0, maxval=2.0 * jnp.pi)
        return PeriodicChangeState(
            key=key,
            base_weights=base_weights,
            phases=phases,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self, state: PeriodicChangeState, idx: Array
    ) -> tuple[TimeStep, PeriodicChangeState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        del idx  # unused
        key, key_x, key_noise = jr.split(state.key, 3)

        # Compute oscillating weights: w(t) = base + amplitude * sin(2π * t / period + phase)
        t = state.step_count.astype(jnp.float32)
        oscillation = self._amplitude * jnp.sin(
            2.0 * jnp.pi * t / self._period + state.phases
        )
        true_weights = state.base_weights + oscillation

        # Generate observation
        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target
        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        target = jnp.dot(true_weights, x) + noise

        timestep = TimeStep(observation=x, target=jnp.atleast_1d(target))
        new_state = PeriodicChangeState(
            key=key,
            base_weights=state.base_weights,
            phases=state.phases,
            step_count=state.step_count + 1,
        )

        return timestep, new_state


class ScaledStreamState(NamedTuple):
    """State for ScaledStreamWrapper.

    Attributes:
        inner_state: State of the wrapped stream
    """

    inner_state: tuple[Any, ...]  # Generic state from wrapped stream


class ScaledStreamWrapper:
    """Wrapper that applies per-feature scaling to any stream's observations.

    This wrapper multiplies each feature of the observation by a corresponding
    scale factor. Useful for testing how learners handle features at different
    scales, which is important for understanding normalization benefits.

    Example:
        >>> stream = ScaledStreamWrapper(
        ...     AbruptChangeStream(feature_dim=10, change_interval=1000),
        ...     feature_scales=jnp.array([0.001, 0.01, 0.1, 1.0, 10.0,
        ...                               100.0, 1000.0, 0.001, 0.01, 0.1])
        ... )

    Attributes:
        inner_stream: The wrapped stream instance
        feature_scales: Per-feature scale factors (must match feature_dim)
    """

    def __init__(self, inner_stream: ScanStream[Any], feature_scales: Array):
        """Initialize the scaled stream wrapper.

        Args:
            inner_stream: Stream to wrap (must implement ScanStream protocol)
            feature_scales: Array of scale factors, one per feature. Must have
                shape (feature_dim,) matching the inner stream's feature_dim.

        Raises:
            ValueError: If feature_scales length doesn't match inner stream's feature_dim
        """
        self._inner_stream: ScanStream[Any] = inner_stream
        self._feature_scales = jnp.asarray(feature_scales, dtype=jnp.float32)

        if self._feature_scales.shape[0] != inner_stream.feature_dim:
            raise ValueError(
                f"feature_scales length ({self._feature_scales.shape[0]}) "
                f"must match inner stream's feature_dim ({inner_stream.feature_dim})"
            )

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return int(self._inner_stream.feature_dim)

    @property
    def inner_stream(self) -> ScanStream[Any]:
        """Return the wrapped stream."""
        return self._inner_stream

    @property
    def feature_scales(self) -> Array:
        """Return the per-feature scale factors."""
        return self._feature_scales

    def init(self, key: Array) -> ScaledStreamState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state wrapping the inner stream's state
        """
        inner_state = self._inner_stream.init(key)
        return ScaledStreamState(inner_state=inner_state)

    def step(self, state: ScaledStreamState, idx: Array) -> tuple[TimeStep, ScaledStreamState]:
        """Generate one time step with scaled observations.

        Args:
            state: Current stream state
            idx: Current step index

        Returns:
            Tuple of (timestep with scaled observation, new_state)
        """
        timestep, new_inner_state = self._inner_stream.step(state.inner_state, idx)

        # Scale the observation
        scaled_observation = timestep.observation * self._feature_scales

        scaled_timestep = TimeStep(
            observation=scaled_observation,
            target=timestep.target,
        )

        new_state = ScaledStreamState(inner_state=new_inner_state)
        return scaled_timestep, new_state


def make_scale_range(
    feature_dim: int,
    min_scale: float = 0.001,
    max_scale: float = 1000.0,
    log_spaced: bool = True,
) -> Array:
    """Create a per-feature scale array spanning a range.

    Utility function to generate scale factors for ScaledStreamWrapper.

    Args:
        feature_dim: Number of features
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        log_spaced: If True, scales are logarithmically spaced (default).
            If False, scales are linearly spaced.

    Returns:
        Array of shape (feature_dim,) with scale factors

    Example:
        >>> scales = make_scale_range(10, min_scale=0.01, max_scale=100.0)
        >>> stream = ScaledStreamWrapper(RandomWalkStream(10), scales)
    """
    if log_spaced:
        return jnp.logspace(
            jnp.log10(min_scale),
            jnp.log10(max_scale),
            feature_dim,
            dtype=jnp.float32,
        )
    else:
        return jnp.linspace(min_scale, max_scale, feature_dim, dtype=jnp.float32)


# Backward-compatible aliases
RandomWalkTarget = RandomWalkStream
AbruptChangeTarget = AbruptChangeStream
CyclicTarget = CyclicStream
PeriodicChangeTarget = PeriodicChangeStream
