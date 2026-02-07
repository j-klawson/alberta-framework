"""Online feature normalization for continual learning.

Implements online (streaming) normalization that updates estimates of mean
and variance at every time step, following the principle of temporal uniformity.

Two normalizer variants are provided:

- ``EMANormalizer``: Exponential moving average estimates of mean and variance.
  Suitable for non-stationary distributions where recent observations should
  be weighted more heavily.

- ``WelfordNormalizer``: Welford's online algorithm for numerically stable
  estimation of cumulative sample mean and variance with Bessel's correction.
  Suitable for stationary distributions.
"""

from abc import ABC, abstractmethod

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


@chex.dataclass(frozen=True)
class EMANormalizerState:
    """State for EMA-based online feature normalization.

    Uses exponential moving average to estimate running mean and variance,
    suitable for non-stationary distributions.

    Attributes:
        mean: Running mean estimate per feature
        var: Running variance estimate per feature
        sample_count: Number of samples seen
        decay: Exponential decay factor for estimates (1.0 = no decay, pure online)
    """

    mean: Float[Array, " feature_dim"]
    var: Float[Array, " feature_dim"]
    sample_count: Float[Array, ""]
    decay: Float[Array, ""]


@chex.dataclass(frozen=True)
class WelfordNormalizerState:
    """State for Welford's online normalization algorithm.

    Uses Welford's algorithm for numerically stable estimation of cumulative
    sample mean and variance with Bessel's correction.

    Attributes:
        mean: Running mean estimate per feature
        var: Running variance estimate per feature (Bessel-corrected)
        sample_count: Number of samples seen
        p: Sum of squared deviations from the current mean (M2 accumulator)
    """

    mean: Float[Array, " feature_dim"]
    var: Float[Array, " feature_dim"]
    sample_count: Float[Array, ""]
    p: Float[Array, " feature_dim"]


AnyNormalizerState = EMANormalizerState | WelfordNormalizerState


class Normalizer[StateT: (EMANormalizerState, WelfordNormalizerState)](ABC):
    """Abstract base class for online feature normalizers.

    Normalizes features using running estimates of mean and standard deviation:
    ``x_normalized = (x - mean) / (std + epsilon)``

    The normalizer updates its estimates at every time step, following
    temporal uniformity.

    Subclasses must implement ``init`` and ``normalize``. The ``normalize_only``
    and ``update_only`` methods have default implementations.

    Attributes:
        epsilon: Small constant for numerical stability
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initialize the normalizer.

        Args:
            epsilon: Small constant added to std for numerical stability
        """
        self._epsilon = epsilon

    @abstractmethod
    def init(self, feature_dim: int) -> StateT:
        """Initialize normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state with zero mean and unit variance
        """
        ...

    @abstractmethod
    def normalize(
        self,
        state: StateT,
        observation: Array,
    ) -> tuple[Array, StateT]:
        """Normalize observation and update running statistics.

        This method both normalizes the current observation AND updates
        the running statistics, maintaining temporal uniformity.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        ...

    def normalize_only(
        self,
        state: StateT,
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
        state: StateT,
        observation: Array,
    ) -> StateT:
        """Update statistics without returning normalized observation.

        Args:
            state: Current normalizer state
            observation: Raw feature vector

        Returns:
            Updated normalizer state
        """
        _, new_state = self.normalize(state, observation)
        return new_state


class EMANormalizer(Normalizer[EMANormalizerState]):
    """Online feature normalizer using exponential moving average.

    Estimates mean and variance via EMA, suitable for non-stationary
    environments where recent observations should be weighted more heavily.

    The effective decay ramps up from 0 to the target decay over early steps
    to prevent instability.

    Attributes:
        epsilon: Small constant for numerical stability
        decay: Exponential decay for running estimates (0.99 = slower adaptation)
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        decay: float = 0.99,
    ):
        """Initialize the EMA normalizer.

        Args:
            epsilon: Small constant added to std for numerical stability
            decay: Exponential decay factor for running estimates.
                   Lower values adapt faster to changes.
                   1.0 means pure online average (no decay).
        """
        super().__init__(epsilon=epsilon)
        self._decay = decay

    def init(self, feature_dim: int) -> EMANormalizerState:
        """Initialize EMA normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state with zero mean and unit variance
        """
        return EMANormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            sample_count=jnp.array(0.0, dtype=jnp.float32),
            decay=jnp.array(self._decay, dtype=jnp.float32),
        )

    def normalize(
        self,
        state: EMANormalizerState,
        observation: Array,
    ) -> tuple[Array, EMANormalizerState]:
        """Normalize observation and update EMA running statistics.

        Args:
            state: Current EMA normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        # Update count
        new_count = state.sample_count + 1.0

        # Compute effective decay (ramp up from 0 to target decay)
        # This prevents instability in early steps
        effective_decay = jnp.minimum(state.decay, 1.0 - 1.0 / (new_count + 1.0))

        # Update mean using exponential moving average
        delta = observation - state.mean
        new_mean = state.mean + (1.0 - effective_decay) * delta

        # Update variance using exponential moving average of squared deviations
        delta2 = observation - new_mean
        new_var = effective_decay * state.var + (1.0 - effective_decay) * delta * delta2

        # Ensure variance is positive
        new_var = jnp.maximum(new_var, self._epsilon)

        # Normalize using updated statistics
        std = jnp.sqrt(new_var)
        normalized = (observation - new_mean) / (std + self._epsilon)

        new_state = EMANormalizerState(
            mean=new_mean,
            var=new_var,
            sample_count=new_count,
            decay=state.decay,
        )

        return normalized, new_state


class WelfordNormalizer(Normalizer[WelfordNormalizerState]):
    """Online feature normalizer using Welford's algorithm.

    Computes cumulative sample mean and variance with Bessel's correction,
    suitable for stationary distributions. Numerically stable for large
    sample counts.

    Reference: Welford 1962, "Note on a Method for Calculating Corrected
    Sums of Squares and Products"

    Attributes:
        epsilon: Small constant for numerical stability
    """

    def init(self, feature_dim: int) -> WelfordNormalizerState:
        """Initialize Welford normalizer state.

        Args:
            feature_dim: Dimension of feature vectors

        Returns:
            Initial normalizer state with zero mean and unit variance
        """
        return WelfordNormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            var=jnp.ones(feature_dim, dtype=jnp.float32),
            sample_count=jnp.array(0.0, dtype=jnp.float32),
            p=jnp.zeros(feature_dim, dtype=jnp.float32),
        )

    def normalize(
        self,
        state: WelfordNormalizerState,
        observation: Array,
    ) -> tuple[Array, WelfordNormalizerState]:
        """Normalize observation and update Welford running statistics.

        Uses Welford's online algorithm:
        1. Increment count
        2. Update mean incrementally
        3. Update sum of squared deviations (p / M2)
        4. Compute variance with Bessel's correction when count >= 2

        Args:
            state: Current Welford normalizer state
            observation: Raw feature vector

        Returns:
            Tuple of (normalized_observation, new_state)
        """
        new_count = state.sample_count + 1.0

        # Welford's incremental mean update
        delta = observation - state.mean
        new_mean = state.mean + delta / new_count

        # Update sum of squared deviations: p += (x - old_mean) * (x - new_mean)
        delta2 = observation - new_mean
        new_p = state.p + delta * delta2

        # Bessel-corrected variance; use unit variance when count < 2
        new_var = jnp.where(
            new_count >= 2.0,
            new_p / (new_count - 1.0),
            jnp.ones_like(new_p),
        )

        # Normalize using updated statistics
        std = jnp.sqrt(new_var)
        normalized = (observation - new_mean) / (std + self._epsilon)

        new_state = WelfordNormalizerState(
            mean=new_mean,
            var=new_var,
            sample_count=new_count,
            p=new_p,
        )

        return normalized, new_state
