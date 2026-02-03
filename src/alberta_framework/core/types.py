"""Type definitions for the Alberta Framework.

This module defines the core data types used throughout the framework,
using chex dataclasses for JAX compatibility and jaxtyping for shape annotations.
"""

from typing import TYPE_CHECKING

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

if TYPE_CHECKING:
    from alberta_framework.core.learners import NormalizedLearnerState

# Type aliases for clarity
Observation = Array  # x_t: feature vector
Target = Array  # y*_t: desired output
Prediction = Array  # y_t: model output
Reward = float  # r_t: scalar reward


@chex.dataclass(frozen=True)
class TimeStep:
    """Single experience from an experience stream.

    Attributes:
        observation: Feature vector x_t
        target: Desired output y*_t (for supervised learning)
    """

    observation: Float[Array, " feature_dim"]
    target: Float[Array, " 1"]


@chex.dataclass(frozen=True)
class LMSState:
    """State for the LMS (Least Mean Square) optimizer.

    LMS uses a fixed step-size, so state only tracks the step-size parameter.

    Attributes:
        step_size: Fixed learning rate alpha
    """

    step_size: Float[Array, ""]


@chex.dataclass(frozen=True)
class IDBDState:
    """State for the IDBD (Incremental Delta-Bar-Delta) optimizer.

    IDBD maintains per-weight adaptive step-sizes that are meta-learned
    based on the correlation of successive gradients.

    Reference: Sutton 1992, "Adapting Bias by Gradient Descent"

    Attributes:
        log_step_sizes: Log of per-weight step-sizes (log alpha_i)
        traces: Per-weight traces h_i for gradient correlation
        meta_step_size: Meta learning rate beta for adapting step-sizes
        bias_step_size: Step-size for the bias term
        bias_trace: Trace for the bias term
    """

    log_step_sizes: Float[Array, " feature_dim"]  # log(alpha_i) for numerical stability
    traces: Float[Array, " feature_dim"]  # h_i: trace of weight-feature products
    meta_step_size: Float[Array, ""]  # beta: step-size for the step-sizes
    bias_step_size: Float[Array, ""]  # Step-size for bias
    bias_trace: Float[Array, ""]  # Trace for bias


@chex.dataclass(frozen=True)
class AutostepState:
    """State for the Autostep optimizer.

    Autostep is a tuning-free step-size adaptation algorithm that normalizes
    gradients to prevent large updates and adapts step-sizes based on
    gradient correlation.

    Reference: Mahmood et al. 2012, "Tuning-free step-size adaptation"

    Attributes:
        step_sizes: Per-weight step-sizes (alpha_i)
        traces: Per-weight traces for gradient correlation (h_i)
        normalizers: Running max absolute gradient per weight (v_i)
        meta_step_size: Meta learning rate mu for adapting step-sizes
        normalizer_decay: Decay factor for the normalizer (tau)
        bias_step_size: Step-size for the bias term
        bias_trace: Trace for the bias term
        bias_normalizer: Normalizer for the bias gradient
    """

    step_sizes: Float[Array, " feature_dim"]  # alpha_i
    traces: Float[Array, " feature_dim"]  # h_i
    normalizers: Float[Array, " feature_dim"]  # v_i: running max of |gradient|
    meta_step_size: Float[Array, ""]  # mu
    normalizer_decay: Float[Array, ""]  # tau
    bias_step_size: Float[Array, ""]
    bias_trace: Float[Array, ""]
    bias_normalizer: Float[Array, ""]


@chex.dataclass(frozen=True)
class LearnerState:
    """State for a linear learner.

    Attributes:
        weights: Weight vector for linear prediction
        bias: Bias term
        optimizer_state: State maintained by the optimizer
    """

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    optimizer_state: LMSState | IDBDState | AutostepState


@chex.dataclass(frozen=True)
class StepSizeTrackingConfig:
    """Configuration for recording per-weight step-sizes during training.

    Attributes:
        interval: Record step-sizes every N steps
        include_bias: Whether to also record the bias step-size
    """

    interval: int
    include_bias: bool = True


@chex.dataclass(frozen=True)
class StepSizeHistory:
    """History of per-weight step-sizes recorded during training.

    Attributes:
        step_sizes: Per-weight step-sizes at each recording, shape (num_recordings, num_weights)
        bias_step_sizes: Bias step-sizes at each recording, shape (num_recordings,) or None
        recording_indices: Step indices where recordings were made, shape (num_recordings,)
        normalizers: Autostep's per-weight normalizers (v_i) at each recording,
            shape (num_recordings, num_weights) or None. Only populated for Autostep optimizer.
    """

    step_sizes: Float[Array, "num_recordings feature_dim"]
    bias_step_sizes: Float[Array, " num_recordings"] | None
    recording_indices: Int[Array, " num_recordings"]
    normalizers: Float[Array, "num_recordings feature_dim"] | None = None


@chex.dataclass(frozen=True)
class NormalizerTrackingConfig:
    """Configuration for recording per-feature normalizer state during training.

    Attributes:
        interval: Record normalizer state every N steps
    """

    interval: int


@chex.dataclass(frozen=True)
class NormalizerHistory:
    """History of per-feature normalizer state recorded during training.

    Used for analyzing how the OnlineNormalizer adapts to distribution shifts
    (reactive lag diagnostic).

    Attributes:
        means: Per-feature mean estimates at each recording, shape (num_recordings, feature_dim)
        variances: Per-feature variance estimates at each recording,
            shape (num_recordings, feature_dim)
        recording_indices: Step indices where recordings were made, shape (num_recordings,)
    """

    means: Float[Array, "num_recordings feature_dim"]
    variances: Float[Array, "num_recordings feature_dim"]
    recording_indices: Int[Array, " num_recordings"]


@chex.dataclass(frozen=True)
class BatchedLearningResult:
    """Result from batched learning loop across multiple seeds.

    Used with `run_learning_loop_batched` for vmap-based GPU parallelization.

    Attributes:
        states: Batched learner states - each array has shape (num_seeds, ...)
        metrics: Metrics array with shape (num_seeds, num_steps, 3)
            where columns are [squared_error, error, mean_step_size]
        step_size_history: Optional step-size history with batched shapes,
            or None if tracking was disabled
    """

    states: LearnerState  # Batched: each array has shape (num_seeds, ...)
    metrics: Float[Array, "num_seeds num_steps 3"]
    step_size_history: StepSizeHistory | None


@chex.dataclass(frozen=True)
class BatchedNormalizedResult:
    """Result from batched normalized learning loop across multiple seeds.

    Used with `run_normalized_learning_loop_batched` for vmap-based GPU parallelization.

    Attributes:
        states: Batched normalized learner states - each array has shape (num_seeds, ...)
        metrics: Metrics array with shape (num_seeds, num_steps, 4)
            where columns are [squared_error, error, mean_step_size, normalizer_mean_var]
        step_size_history: Optional step-size history with batched shapes,
            or None if tracking was disabled
        normalizer_history: Optional normalizer history with batched shapes,
            or None if tracking was disabled
    """

    states: "NormalizedLearnerState"  # Batched: each array has shape (num_seeds, ...)
    metrics: Float[Array, "num_seeds num_steps 4"]
    step_size_history: StepSizeHistory | None
    normalizer_history: NormalizerHistory | None


def create_lms_state(step_size: float = 0.01) -> LMSState:
    """Create initial LMS optimizer state.

    Args:
        step_size: Fixed learning rate

    Returns:
        Initial LMS state
    """
    return LMSState(step_size=jnp.array(step_size, dtype=jnp.float32))


def create_idbd_state(
    feature_dim: int,
    initial_step_size: float = 0.01,
    meta_step_size: float = 0.01,
) -> IDBDState:
    """Create initial IDBD optimizer state.

    Args:
        feature_dim: Dimension of the feature vector
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate for adapting step-sizes

    Returns:
        Initial IDBD state
    """
    return IDBDState(
        log_step_sizes=jnp.full(feature_dim, jnp.log(initial_step_size), dtype=jnp.float32),
        traces=jnp.zeros(feature_dim, dtype=jnp.float32),
        meta_step_size=jnp.array(meta_step_size, dtype=jnp.float32),
        bias_step_size=jnp.array(initial_step_size, dtype=jnp.float32),
        bias_trace=jnp.array(0.0, dtype=jnp.float32),
    )


def create_autostep_state(
    feature_dim: int,
    initial_step_size: float = 0.01,
    meta_step_size: float = 0.01,
    normalizer_decay: float = 0.99,
) -> AutostepState:
    """Create initial Autostep optimizer state.

    Args:
        feature_dim: Dimension of the feature vector
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate for adapting step-sizes
        normalizer_decay: Decay factor for gradient normalizers

    Returns:
        Initial Autostep state
    """
    return AutostepState(
        step_sizes=jnp.full(feature_dim, initial_step_size, dtype=jnp.float32),
        traces=jnp.zeros(feature_dim, dtype=jnp.float32),
        normalizers=jnp.ones(feature_dim, dtype=jnp.float32),
        meta_step_size=jnp.array(meta_step_size, dtype=jnp.float32),
        normalizer_decay=jnp.array(normalizer_decay, dtype=jnp.float32),
        bias_step_size=jnp.array(initial_step_size, dtype=jnp.float32),
        bias_trace=jnp.array(0.0, dtype=jnp.float32),
        bias_normalizer=jnp.array(1.0, dtype=jnp.float32),
    )
