"""Optimizers for continual learning.

Implements LMS (fixed step-size baseline), IDBD (meta-learned step-sizes),
and Autostep (tuning-free step-size adaptation) for Step 1 of the Alberta Plan.

References:
- Sutton 1992, "Adapting Bias by Gradient Descent: An Incremental
  Version of Delta-Bar-Delta"
- Mahmood et al. 2012, "Tuning-free step-size adaptation"
"""

from abc import ABC, abstractmethod

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from alberta_framework.core.types import AutostepState, IDBDState, LMSState


@chex.dataclass(frozen=True)
class OptimizerUpdate:
    """Result of an optimizer update step.

    Attributes:
        weight_delta: Change to apply to weights
        bias_delta: Change to apply to bias
        new_state: Updated optimizer state
        metrics: Dictionary of metrics for logging (values are JAX arrays for scan compatibility)
    """

    weight_delta: Float[Array, " feature_dim"]
    bias_delta: Float[Array, ""]
    new_state: LMSState | IDBDState | AutostepState
    metrics: dict[str, Array]


class Optimizer[StateT: (LMSState, IDBDState, AutostepState)](ABC):
    """Base class for optimizers."""

    @abstractmethod
    def init(self, feature_dim: int) -> StateT:
        """Initialize optimizer state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            Initial optimizer state
        """
        ...

    @abstractmethod
    def update(
        self,
        state: StateT,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute weight updates given prediction error.

        Args:
            state: Current optimizer state
            error: Prediction error (target - prediction)
            observation: Current observation/feature vector

        Returns:
            OptimizerUpdate with deltas and new state
        """
        ...


class LMS(Optimizer[LMSState]):
    """Least Mean Square optimizer with fixed step-size.

    The simplest gradient-based optimizer: `w_{t+1} = w_t + alpha * delta * x_t`

    This serves as a baseline. The challenge is that the optimal step-size
    depends on the problem and changes as the task becomes non-stationary.

    Attributes:
        step_size: Fixed learning rate alpha
    """

    def __init__(self, step_size: float = 0.01):
        """Initialize LMS optimizer.

        Args:
            step_size: Fixed learning rate
        """
        self._step_size = step_size

    def init(self, feature_dim: int) -> LMSState:
        """Initialize LMS state.

        Args:
            feature_dim: Dimension of weight vector (unused for LMS)

        Returns:
            LMS state containing the step-size
        """
        return LMSState(step_size=jnp.array(self._step_size, dtype=jnp.float32))

    def update(
        self,
        state: LMSState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute LMS weight update.

        Update rule: `delta_w = alpha * error * x`

        Args:
            state: Current LMS state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight and bias deltas
        """
        alpha = state.step_size
        error_scalar = jnp.squeeze(error)

        # Weight update: alpha * error * x
        weight_delta = alpha * error_scalar * observation

        # Bias update: alpha * error
        bias_delta = alpha * error_scalar

        return OptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=state,  # LMS state doesn't change
            metrics={"step_size": alpha},
        )


class IDBD(Optimizer[IDBDState]):
    """Incremental Delta-Bar-Delta optimizer.

    IDBD maintains per-weight adaptive step-sizes that are meta-learned
    based on gradient correlation. When successive gradients agree in sign,
    the step-size for that weight increases. When they disagree, it decreases.

    This implements Sutton's 1992 algorithm for adapting step-sizes online
    without requiring manual tuning.

    Reference: Sutton, R.S. (1992). "Adapting Bias by Gradient Descent:
    An Incremental Version of Delta-Bar-Delta"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate beta for adapting step-sizes
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
    ):
        """Initialize IDBD optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate beta for adapting step-sizes
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size

    def init(self, feature_dim: int) -> IDBDState:
        """Initialize IDBD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            IDBD state with per-weight step-sizes and traces
        """
        return IDBDState(
            log_step_sizes=jnp.full(
                feature_dim, jnp.log(self._initial_step_size), dtype=jnp.float32
            ),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            bias_step_size=jnp.array(self._initial_step_size, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: IDBDState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute IDBD weight update with adaptive step-sizes.

        The IDBD algorithm:

        1. Compute step-sizes: `alpha_i = exp(log_alpha_i)`
        2. Update weights: `w_i += alpha_i * error * x_i`
        3. Update log step-sizes: `log_alpha_i += beta * error * x_i * h_i`
        4. Update traces: `h_i = h_i * max(0, 1 - alpha_i * x_i^2) + alpha_i * error * x_i`

        The trace h_i tracks the correlation between current and past gradients.
        When gradients consistently point the same direction, h_i grows,
        leading to larger step-sizes.

        Args:
            state: Current IDBD state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        beta = state.meta_step_size

        # Current step-sizes (exponentiate log values)
        alphas = jnp.exp(state.log_step_sizes)

        # Weight updates: alpha_i * error * x_i
        weight_delta = alphas * error_scalar * observation

        # Meta-update: adapt step-sizes based on gradient correlation
        # log_alpha_i += beta * error * x_i * h_i
        gradient_correlation = error_scalar * observation * state.traces
        new_log_step_sizes = state.log_step_sizes + beta * gradient_correlation

        # Clip log step-sizes to prevent numerical issues
        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)

        # Update traces: h_i = h_i * decay + alpha_i * error * x_i
        # decay = max(0, 1 - alpha_i * x_i^2)
        decay = jnp.maximum(0.0, 1.0 - alphas * observation**2)
        new_traces = state.traces * decay + alphas * error_scalar * observation

        # Bias updates (similar logic but scalar)
        bias_alpha = state.bias_step_size
        bias_delta = bias_alpha * error_scalar

        # Update bias step-size
        bias_gradient_correlation = error_scalar * state.bias_trace
        new_bias_step_size = bias_alpha * jnp.exp(beta * bias_gradient_correlation)
        new_bias_step_size = jnp.clip(new_bias_step_size, 1e-6, 1.0)

        # Update bias trace
        bias_decay = jnp.maximum(0.0, 1.0 - bias_alpha)
        new_bias_trace = state.bias_trace * bias_decay + bias_alpha * error_scalar

        new_state = IDBDState(
            log_step_sizes=new_log_step_sizes,
            traces=new_traces,
            meta_step_size=beta,
            bias_step_size=new_bias_step_size,
            bias_trace=new_bias_trace,
        )

        return OptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=new_state,
            metrics={
                "mean_step_size": jnp.mean(alphas),
                "min_step_size": jnp.min(alphas),
                "max_step_size": jnp.max(alphas),
            },
        )


class Autostep(Optimizer[AutostepState]):
    """Autostep optimizer with tuning-free step-size adaptation.

    Autostep normalizes gradients to prevent large updates and adapts
    per-weight step-sizes based on gradient correlation. The key innovation
    is automatic normalization that makes the algorithm robust to different
    feature scales.

    The algorithm maintains:
    - Per-weight step-sizes that adapt based on gradient correlation
    - Running max of absolute gradients for normalization
    - Traces for detecting consistent gradient directions

    Reference: Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012).
    "Tuning-free step-size adaptation"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate mu for adapting step-sizes
        normalizer_decay: Decay factor tau for gradient normalizers
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        normalizer_decay: float = 0.99,
    ):
        """Initialize Autostep optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate for adapting step-sizes
            normalizer_decay: Decay factor for gradient normalizers (higher = slower decay)
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._normalizer_decay = normalizer_decay

    def init(self, feature_dim: int) -> AutostepState:
        """Initialize Autostep state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            Autostep state with per-weight step-sizes, traces, and normalizers
        """
        return AutostepState(
            step_sizes=jnp.full(feature_dim, self._initial_step_size, dtype=jnp.float32),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            normalizers=jnp.ones(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            normalizer_decay=jnp.array(self._normalizer_decay, dtype=jnp.float32),
            bias_step_size=jnp.array(self._initial_step_size, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_normalizer=jnp.array(1.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: AutostepState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute Autostep weight update with normalized gradients.

        The Autostep algorithm:

        1. Compute gradient: `g_i = error * x_i`
        2. Normalize gradient: `g_i' = g_i / max(|g_i|, v_i)`
        3. Update weights: `w_i += alpha_i * g_i'`
        4. Update step-sizes: `alpha_i *= exp(mu * g_i' * h_i)`
        5. Update traces: `h_i = h_i * (1 - alpha_i) + alpha_i * g_i'`
        6. Update normalizers: `v_i = max(|g_i|, v_i * tau)`

        Args:
            state: Current Autostep state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        mu = state.meta_step_size
        tau = state.normalizer_decay

        # Compute raw gradient
        gradient = error_scalar * observation

        # Normalize gradient using running max
        abs_gradient = jnp.abs(gradient)
        normalizer = jnp.maximum(abs_gradient, state.normalizers)
        normalized_gradient = gradient / (normalizer + 1e-8)

        # Compute weight delta using normalized gradient
        weight_delta = state.step_sizes * normalized_gradient

        # Update step-sizes based on gradient correlation
        gradient_correlation = normalized_gradient * state.traces
        new_step_sizes = state.step_sizes * jnp.exp(mu * gradient_correlation)

        # Clip step-sizes to prevent instability
        new_step_sizes = jnp.clip(new_step_sizes, 1e-8, 1.0)

        # Update traces with decay based on step-size
        trace_decay = 1.0 - state.step_sizes
        new_traces = state.traces * trace_decay + state.step_sizes * normalized_gradient

        # Update normalizers with decay
        new_normalizers = jnp.maximum(abs_gradient, state.normalizers * tau)

        # Bias updates (similar logic)
        bias_gradient = error_scalar
        abs_bias_gradient = jnp.abs(bias_gradient)
        bias_normalizer = jnp.maximum(abs_bias_gradient, state.bias_normalizer)
        normalized_bias_gradient = bias_gradient / (bias_normalizer + 1e-8)

        bias_delta = state.bias_step_size * normalized_bias_gradient

        bias_correlation = normalized_bias_gradient * state.bias_trace
        new_bias_step_size = state.bias_step_size * jnp.exp(mu * bias_correlation)
        new_bias_step_size = jnp.clip(new_bias_step_size, 1e-8, 1.0)

        bias_trace_decay = 1.0 - state.bias_step_size
        new_bias_trace = (
            state.bias_trace * bias_trace_decay + state.bias_step_size * normalized_bias_gradient
        )

        new_bias_normalizer = jnp.maximum(abs_bias_gradient, state.bias_normalizer * tau)

        new_state = AutostepState(
            step_sizes=new_step_sizes,
            traces=new_traces,
            normalizers=new_normalizers,
            meta_step_size=mu,
            normalizer_decay=tau,
            bias_step_size=new_bias_step_size,
            bias_trace=new_bias_trace,
            bias_normalizer=new_bias_normalizer,
        )

        return OptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=new_state,
            metrics={
                "mean_step_size": jnp.mean(state.step_sizes),
                "min_step_size": jnp.min(state.step_sizes),
                "max_step_size": jnp.max(state.step_sizes),
                "mean_normalizer": jnp.mean(state.normalizers),
            },
        )
