"""Learning units for continual learning.

Implements learners that combine function approximation with optimizers
for temporally-uniform learning. Uses JAX's scan for efficient JIT-compiled
training loops.
"""

from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from alberta_framework.core.normalizers import NormalizerState, OnlineNormalizer
from alberta_framework.core.optimizers import LMS, Optimizer
from alberta_framework.core.types import (
    AutostepState,
    IDBDState,
    LearnerState,
    LMSState,
    Observation,
    Prediction,
    Target,
)
from alberta_framework.streams.base import ScanStream

# Type alias for any optimizer type
AnyOptimizer = Optimizer[LMSState] | Optimizer[IDBDState] | Optimizer[AutostepState]

# Type variable for stream state
StreamStateT = TypeVar("StreamStateT")


class UpdateResult(NamedTuple):
    """Result of a learner update step.

    Attributes:
        state: Updated learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics [squared_error, error, ...]
    """

    state: LearnerState
    prediction: Prediction
    error: Array
    metrics: Array


class LinearLearner:
    """Linear function approximator with pluggable optimizer.

    Computes predictions as: y = w @ x + b

    The learner maintains weights and bias, delegating the adaptation
    of learning rates to the optimizer (e.g., LMS or IDBD).

    This follows the Alberta Plan philosophy of temporal uniformity:
    every component updates at every time step.

    Attributes:
        optimizer: The optimizer to use for weight updates
    """

    def __init__(self, optimizer: AnyOptimizer | None = None):
        """Initialize the linear learner.

        Args:
            optimizer: Optimizer for weight updates. Defaults to LMS(0.01)
        """
        self._optimizer: AnyOptimizer = optimizer or LMS(step_size=0.01)

    def init(self, feature_dim: int) -> LearnerState:
        """Initialize learner state.

        Args:
            feature_dim: Dimension of the input feature vector

        Returns:
            Initial learner state with zero weights and bias
        """
        optimizer_state = self._optimizer.init(feature_dim)

        return LearnerState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            optimizer_state=optimizer_state,
        )

    def predict(self, state: LearnerState, observation: Observation) -> Prediction:
        """Compute prediction for an observation.

        Args:
            state: Current learner state
            observation: Input feature vector

        Returns:
            Scalar prediction y = w @ x + b
        """
        return jnp.atleast_1d(jnp.dot(state.weights, observation) + state.bias)

    def update(
        self,
        state: LearnerState,
        observation: Observation,
        target: Target,
    ) -> UpdateResult:
        """Update learner given observation and target.

        Performs one step of the learning algorithm:
        1. Compute prediction
        2. Compute error
        3. Get weight updates from optimizer
        4. Apply updates to weights and bias

        Args:
            state: Current learner state
            observation: Input feature vector
            target: Desired output

        Returns:
            UpdateResult with new state, prediction, error, and metrics
        """
        # Make prediction
        prediction = self.predict(state, observation)

        # Compute error (target - prediction)
        error = jnp.squeeze(target) - jnp.squeeze(prediction)

        # Get update from optimizer
        # Note: type ignore needed because we can't statically prove optimizer_state
        # matches the optimizer's expected state type (though they will at runtime)
        opt_update = self._optimizer.update(
            state.optimizer_state,  # type: ignore[arg-type]
            error,
            observation,
        )

        # Apply updates
        new_weights = state.weights + opt_update.weight_delta
        new_bias = state.bias + opt_update.bias_delta

        new_state = LearnerState(
            weights=new_weights,
            bias=new_bias,
            optimizer_state=opt_update.new_state,
        )

        # Pack metrics as array for scan compatibility
        # Format: [squared_error, error, mean_step_size (if adaptive)]
        squared_error = error**2
        mean_step_size = opt_update.metrics.get("mean_step_size", 0.0)
        metrics = jnp.array([squared_error, error, mean_step_size], dtype=jnp.float32)

        return UpdateResult(
            state=new_state,
            prediction=prediction,
            error=jnp.atleast_1d(error),
            metrics=metrics,
        )


def run_learning_loop(
    learner: LinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: LearnerState | None = None,
) -> tuple[LearnerState, Array]:
    """Run the learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.
    It returns metrics as a fixed-size array rather than a list of dicts.

    Args:
        learner: The learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)

    Returns:
        Tuple of (final_state, metrics_array) where metrics_array has shape
        (num_steps, 3) with columns [squared_error, error, mean_step_size]
    """
    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    def step_fn(carry, idx):
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        return (result.state, new_s_state), result.metrics

    (final_learner, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )

    return final_learner, metrics


class NormalizedLearnerState(NamedTuple):
    """State for a learner with online feature normalization.

    Attributes:
        learner_state: Underlying learner state (weights, bias, optimizer)
        normalizer_state: Online normalizer state (mean, var estimates)
    """

    learner_state: LearnerState
    normalizer_state: NormalizerState


class NormalizedUpdateResult(NamedTuple):
    """Result of a normalized learner update step.

    Attributes:
        state: Updated normalized learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics [squared_error, error, mean_step_size, normalizer_mean_var]
    """

    state: NormalizedLearnerState
    prediction: Prediction
    error: Array
    metrics: Array


class NormalizedLinearLearner:
    """Linear learner with online feature normalization.

    Wraps a LinearLearner with online feature normalization, following
    the Alberta Plan's approach to handling varying feature scales.

    Normalization is applied to features before prediction and learning:
        x_normalized = (x - mean) / (std + epsilon)

    The normalizer statistics update at every time step, maintaining
    temporal uniformity.

    Attributes:
        learner: Underlying linear learner
        normalizer: Online feature normalizer
    """

    def __init__(
        self,
        optimizer: AnyOptimizer | None = None,
        normalizer: OnlineNormalizer | None = None,
    ):
        """Initialize the normalized linear learner.

        Args:
            optimizer: Optimizer for weight updates. Defaults to LMS(0.01)
            normalizer: Feature normalizer. Defaults to OnlineNormalizer()
        """
        self._learner = LinearLearner(optimizer=optimizer or LMS(step_size=0.01))
        self._normalizer = normalizer or OnlineNormalizer()

    def init(self, feature_dim: int) -> NormalizedLearnerState:
        """Initialize normalized learner state.

        Args:
            feature_dim: Dimension of the input feature vector

        Returns:
            Initial state with zero weights and unit variance estimates
        """
        return NormalizedLearnerState(
            learner_state=self._learner.init(feature_dim),
            normalizer_state=self._normalizer.init(feature_dim),
        )

    def predict(
        self,
        state: NormalizedLearnerState,
        observation: Observation,
    ) -> Prediction:
        """Compute prediction for an observation.

        Normalizes the observation using current statistics before prediction.

        Args:
            state: Current normalized learner state
            observation: Raw (unnormalized) input feature vector

        Returns:
            Scalar prediction y = w @ normalize(x) + b
        """
        normalized_obs = self._normalizer.normalize_only(
            state.normalizer_state, observation
        )
        return self._learner.predict(state.learner_state, normalized_obs)

    def update(
        self,
        state: NormalizedLearnerState,
        observation: Observation,
        target: Target,
    ) -> NormalizedUpdateResult:
        """Update learner given observation and target.

        Performs one step of the learning algorithm:
        1. Normalize observation (and update normalizer statistics)
        2. Compute prediction using normalized features
        3. Compute error
        4. Get weight updates from optimizer
        5. Apply updates

        Args:
            state: Current normalized learner state
            observation: Raw (unnormalized) input feature vector
            target: Desired output

        Returns:
            NormalizedUpdateResult with new state, prediction, error, and metrics
        """
        # Normalize observation and update normalizer state
        normalized_obs, new_normalizer_state = self._normalizer.normalize(
            state.normalizer_state, observation
        )

        # Delegate to underlying learner
        result = self._learner.update(
            state.learner_state,
            normalized_obs,
            target,
        )

        # Build combined state
        new_state = NormalizedLearnerState(
            learner_state=result.state,
            normalizer_state=new_normalizer_state,
        )

        # Add normalizer metrics to the metrics array
        normalizer_mean_var = jnp.mean(new_normalizer_state.var)
        metrics = jnp.concatenate([result.metrics, jnp.array([normalizer_mean_var])])

        return NormalizedUpdateResult(
            state=new_state,
            prediction=result.prediction,
            error=result.error,
            metrics=metrics,
        )


def run_normalized_learning_loop(
    learner: NormalizedLinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: NormalizedLearnerState | None = None,
) -> tuple[NormalizedLearnerState, Array]:
    """Run the learning loop with normalization using jax.lax.scan.

    Args:
        learner: The normalized learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)

    Returns:
        Tuple of (final_state, metrics_array) where metrics_array has shape
        (num_steps, 4) with columns [squared_error, error, mean_step_size, normalizer_mean_var]
    """
    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    def step_fn(carry, idx):
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        return (result.state, new_s_state), result.metrics

    (final_learner, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )

    return final_learner, metrics


def metrics_to_dicts(metrics: Array, normalized: bool = False) -> list[dict[str, float]]:
    """Convert metrics array to list of dicts for backward compatibility.

    Args:
        metrics: Array of shape (num_steps, 3) or (num_steps, 4)
        normalized: If True, expects 4 columns including normalizer_mean_var

    Returns:
        List of metric dictionaries
    """
    result = []
    for row in metrics:
        d = {
            "squared_error": float(row[0]),
            "error": float(row[1]),
            "mean_step_size": float(row[2]),
        }
        if normalized and len(row) > 3:
            d["normalizer_mean_var"] = float(row[3])
        result.append(d)
    return result
