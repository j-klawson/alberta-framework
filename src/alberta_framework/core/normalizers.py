"""Online feature normalization for continual learning.

Implements online (streaming) normalization that updates estimates of mean
and variance at every time step, following the principle of temporal uniformity.

Reference: Welford's online algorithm for numerical stability.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class NormalizerState(NamedTuple):
    """State for online feature normalization.

    Uses Welford's online algorithm for numerically stable estimation
    of running mean and variance.

    Attributes:
        mean: Running mean estimate per feature
        var: Running variance estimate per feature
        sample_count: Number of samples seen
        decay: Exponential decay factor for estimates (1.0 = no decay, pure online)
    """

    mean: Array  # Shape: (feature_dim,)
    var: Array  # Shape: (feature_dim,)
    sample_count: Array  # Scalar
    decay: Array  # Scalar


class OnlineNormalizer:
    """Online feature normalizer for continual learning.

    Normalizes features using running estimates of mean and standard deviation:
        x_normalized = (x - mean) / (std + epsilon)

    The normalizer updates its estimates at every time step, following
    temporal uniformity. Uses exponential moving average for non-stationary
    environments.

    Attributes:
        epsilon: Small constant for numerical stability
        decay: Exponential decay for running estimates (0.99 = slower adaptation)
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        decay: float = 0.99,
    ):
        """Initialize the online normalizer.

        Args:
            epsilon: Small constant added to std for numerical stability
            decay: Exponential decay factor for running estimates.
                   Lower values adapt faster to changes.
                   1.0 means pure online average (no decay).
        """
        self._epsilon = epsilon
        self._decay = decay

    def init(self, feature_dim: int) -> NormalizerState:
        """Initialize normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state with zero mean and unit variance
        """
        return NormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            sample_count=jnp.array(0.0, dtype=jnp.float32),
            decay=jnp.array(self._decay, dtype=jnp.float32),
        )

    def normalize(
        self,
        state: NormalizerState,
        observation: Array,
    ) -> tuple[Array, NormalizerState]:
        """Normalize observation and update running statistics.

        This method both normalizes the current observation AND updates
        the running statistics, maintaining temporal uniformity.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        # Update count
        new_count = state.sample_count + 1.0

        # Compute effective decay (ramp up from 0 to target decay)
        # This prevents instability in early steps
        effective_decay = jnp.minimum(
            state.decay,
            1.0 - 1.0 / (new_count + 1.0)
        )

        # Update mean using exponential moving average
        delta = observation - state.mean
        new_mean = state.mean + (1.0 - effective_decay) * delta

        # Update variance using exponential moving average of squared deviations
        # This is a simplified Welford's algorithm adapted for EMA
        delta2 = observation - new_mean
        new_var = effective_decay * state.var + (1.0 - effective_decay) * delta * delta2

        # Ensure variance is positive
        new_var = jnp.maximum(new_var, self._epsilon)

        # Normalize using updated statistics
        std = jnp.sqrt(new_var)
        normalized = (observation - new_mean) / (std + self._epsilon)

        new_state = NormalizerState(
            mean=new_mean,
            var=new_var,
            sample_count=new_count,
            decay=state.decay,
        )

        return normalized, new_state

    def normalize_only(
        self,
        state: NormalizerState,
        observation: Array,
    ) -> Array:
        """Normalize observation without updating statistics.

        Useful for inference or when you want to normalize multiple
        observations with the same statistics.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Normalized observation
        """
        std = jnp.sqrt(state.var)
        return (observation - state.mean) / (std + self._epsilon)

    def update_only(
        self,
        state: NormalizerState,
        observation: Array,
    ) -> NormalizerState:
        """Update statistics without returning normalized observation.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Updated normalizer state
        """
        _, new_state = self.normalize(state, observation)
        return new_state


def create_normalizer_state(
    feature_dim: int,
    decay: float = 0.99,
) -> NormalizerState:
    """Create initial normalizer state.

    Convenience function for creating normalizer state without
    instantiating the OnlineNormalizer class.

    Args:
        feature_dim: Dimension of feature vectors
        decay: Exponential decay factor

    Returns:
        Initial normalizer state
    """
    return NormalizerState(
        mean=jnp.zeros(feature_dim, dtype=jnp.float32),
        var=jnp.ones(feature_dim, dtype=jnp.float32),
        sample_count=jnp.array(0.0, dtype=jnp.float32),
        decay=jnp.array(decay, dtype=jnp.float32),
    )
