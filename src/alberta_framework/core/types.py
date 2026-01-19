"""Type definitions for the Alberta Framework.

This module defines the core data types used throughout the framework,
following JAX conventions with immutable NamedTuples for state management.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# Type aliases for clarity
Observation = Array  # x_t: feature vector
Target = Array  # y*_t: desired output
Prediction = Array  # y_t: model output
Reward = float  # r_t: scalar reward


class TimeStep(NamedTuple):
    """Single experience from an experience stream.

    Attributes:
        observation: Feature vector x_t
        target: Desired output y*_t (for supervised learning)
    """

    observation: Observation
    target: Target


class LearnerState(NamedTuple):
    """State for a linear learner.

    Attributes:
        weights: Weight vector for linear prediction
        bias: Bias term
        optimizer_state: State maintained by the optimizer
    """

    weights: Array
    bias: Array
    optimizer_state: "LMSState | IDBDState | AutostepState"


class LMSState(NamedTuple):
    """State for the LMS (Least Mean Square) optimizer.

    LMS uses a fixed step-size, so state only tracks the step-size parameter.

    Attributes:
        step_size: Fixed learning rate alpha
    """

    step_size: Array


class IDBDState(NamedTuple):
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

    log_step_sizes: Array  # log(alpha_i) for numerical stability
    traces: Array  # h_i: trace of weight-feature products
    meta_step_size: Array  # beta: step-size for the step-sizes
    bias_step_size: Array  # Step-size for bias
    bias_trace: Array  # Trace for bias


class AutostepState(NamedTuple):
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

    step_sizes: Array  # alpha_i
    traces: Array  # h_i
    normalizers: Array  # v_i: running max of |gradient|
    meta_step_size: Array  # mu
    normalizer_decay: Array  # tau
    bias_step_size: Array
    bias_trace: Array
    bias_normalizer: Array


class StepSizeTrackingConfig(NamedTuple):
    """Configuration for recording per-weight step-sizes during training.

    Attributes:
        interval: Record step-sizes every N steps
        include_bias: Whether to also record the bias step-size
    """

    interval: int
    include_bias: bool = True


class StepSizeHistory(NamedTuple):
    """History of per-weight step-sizes recorded during training.

    Attributes:
        step_sizes: Per-weight step-sizes at each recording, shape (num_recordings, num_weights)
        bias_step_sizes: Bias step-sizes at each recording, shape (num_recordings,) or None
        recording_indices: Step indices where recordings were made, shape (num_recordings,)
    """

    step_sizes: Array  # (num_recordings, num_weights)
    bias_step_sizes: Array | None  # (num_recordings,) or None
    recording_indices: Array  # (num_recordings,)


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
