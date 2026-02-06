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

from alberta_framework.core.types import (
    AutostepState,
    AutoTDIDBDState,
    IDBDState,
    LMSState,
    ObGDState,
    TDIDBDState,
)


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
    new_state: LMSState | IDBDState | AutostepState | ObGDState
    metrics: dict[str, Array]


class Optimizer[StateT: (LMSState, IDBDState, AutostepState, ObGDState)](ABC):
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


class ObGD(Optimizer[ObGDState]):
    """Observation-bounded Gradient Descent optimizer.

    ObGD prevents overshooting by dynamically bounding the effective step-size
    based on the magnitude of the prediction error and eligibility traces.
    When the combined update magnitude would be too large, the step-size is
    scaled down to prevent the prediction from overshooting the target.

    This is the deep-network generalization of Autostep's overshooting
    prevention, designed for streaming reinforcement learning.

    For supervised learning (gamma=0, lamda=0), traces equal the current
    observation each step, making ObGD equivalent to LMS with dynamic
    step-size bounding.

    The ObGD algorithm:

    1. Update traces: `z = gamma * lamda * z + observation`
    2. Compute bound: `M = alpha * kappa * max(|error|, 1) * (||z_w||_1 + |z_b|)`
    3. Effective step: `alpha_eff = min(alpha, alpha / M)` (i.e. `alpha / max(M, 1)`)
    4. Weight delta: `delta_w = alpha_eff * error * z_w`
    5. Bias delta: `delta_b = alpha_eff * error * z_b`

    Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning
    Finally Works"

    Attributes:
        step_size: Base learning rate alpha
        kappa: Bounding sensitivity parameter (higher = more conservative)
        gamma: Discount factor for trace decay (0 for supervised learning)
        lamda: Eligibility trace decay parameter (0 for supervised learning)
    """

    def __init__(
        self,
        step_size: float = 1.0,
        kappa: float = 2.0,
        gamma: float = 0.0,
        lamda: float = 0.0,
    ):
        """Initialize ObGD optimizer.

        Args:
            step_size: Base learning rate (default: 1.0)
            kappa: Bounding sensitivity parameter (default: 2.0)
            gamma: Discount factor for trace decay (default: 0.0 for supervised)
            lamda: Eligibility trace decay parameter (default: 0.0 for supervised)
        """
        self._step_size = step_size
        self._kappa = kappa
        self._gamma = gamma
        self._lamda = lamda

    def init(self, feature_dim: int) -> ObGDState:
        """Initialize ObGD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            ObGD state with eligibility traces
        """
        return ObGDState(
            step_size=jnp.array(self._step_size, dtype=jnp.float32),
            kappa=jnp.array(self._kappa, dtype=jnp.float32),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
            gamma=jnp.array(self._gamma, dtype=jnp.float32),
            lamda=jnp.array(self._lamda, dtype=jnp.float32),
        )

    def update(
        self,
        state: ObGDState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute ObGD weight update with overshooting prevention.

        The bounding mechanism scales down the step-size when the combined
        effect of error magnitude, trace norm, and step-size would cause
        the prediction to overshoot the target.

        Args:
            state: Current ObGD state
            error: Prediction error (target - prediction)
            observation: Current observation/feature vector

        Returns:
            OptimizerUpdate with bounded weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        alpha = state.step_size
        kappa = state.kappa

        # Update eligibility traces: z = gamma * lamda * z + observation
        new_traces = state.gamma * state.lamda * state.traces + observation
        new_bias_trace = state.gamma * state.lamda * state.bias_trace + 1.0

        # Compute z_sum (L1 norm of all traces)
        z_sum = jnp.sum(jnp.abs(new_traces)) + jnp.abs(new_bias_trace)

        # Compute bounding factor: M = alpha * kappa * max(|error|, 1) * z_sum
        delta_bar = jnp.maximum(jnp.abs(error_scalar), 1.0)
        dot_product = delta_bar * z_sum * alpha * kappa

        # Effective step-size: alpha / max(M, 1)
        alpha_eff = alpha / jnp.maximum(dot_product, 1.0)

        # Weight and bias deltas
        weight_delta = alpha_eff * error_scalar * new_traces
        bias_delta = alpha_eff * error_scalar * new_bias_trace

        new_state = ObGDState(
            step_size=alpha,
            kappa=kappa,
            traces=new_traces,
            bias_trace=new_bias_trace,
            gamma=state.gamma,
            lamda=state.lamda,
        )

        return OptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=new_state,
            metrics={
                "step_size": alpha,
                "effective_step_size": alpha_eff,
                "bounding_factor": dot_product,
            },
        )


# =============================================================================
# TD Optimizers (for Step 3+ of Alberta Plan)
# =============================================================================


@chex.dataclass(frozen=True)
class TDOptimizerUpdate:
    """Result of a TD optimizer update step.

    Attributes:
        weight_delta: Change to apply to weights
        bias_delta: Change to apply to bias
        new_state: Updated optimizer state
        metrics: Dictionary of metrics for logging
    """

    weight_delta: Float[Array, " feature_dim"]
    bias_delta: Float[Array, ""]
    new_state: TDIDBDState | AutoTDIDBDState
    metrics: dict[str, Array]


class TDOptimizer[StateT: (TDIDBDState, AutoTDIDBDState)](ABC):
    """Base class for TD optimizers.

    TD optimizers handle temporal-difference learning with eligibility traces.
    They take TD error and both current and next observations as input.
    """

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
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> TDOptimizerUpdate:
        """Compute weight updates given TD error.

        Args:
            state: Current optimizer state
            td_error: TD error δ = R + γV(s') - V(s)
            observation: Current observation φ(s)
            next_observation: Next observation φ(s')
            gamma: Discount factor γ (0 at terminal)

        Returns:
            TDOptimizerUpdate with deltas and new state
        """
        ...


class TDIDBD(TDOptimizer[TDIDBDState]):
    """TD-IDBD optimizer for temporal-difference learning.

    Extends IDBD to TD learning with eligibility traces. Maintains per-weight
    adaptive step-sizes that are meta-learned based on gradient correlation
    in the TD setting.

    Two variants are supported:
    - Semi-gradient (default): Uses only φ(s) in meta-update, more stable
    - Ordinary gradient: Uses both φ(s) and φ(s'), more accurate but sensitive

    Reference: Kearney et al. 2019, "Learning Feature Relevance Through Step Size
    Adaptation in Temporal-Difference Learning"

    The semi-gradient TD-IDBD algorithm (Algorithm 3 in paper):
        1. Compute TD error: `δ = R + γ*w^T*φ(s') - w^T*φ(s)`
        2. Update meta-weights: `β_i += θ*δ*φ_i(s)*h_i`
        3. Compute step-sizes: `α_i = exp(β_i)`
        4. Update eligibility traces: `z_i = γ*λ*z_i + φ_i(s)`
        5. Update weights: `w_i += α_i*δ*z_i`
        6. Update h traces: `h_i = h_i*[1 - α_i*φ_i(s)*z_i]^+ + α_i*δ*z_i`

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate theta
        trace_decay: Eligibility trace decay lambda
        use_semi_gradient: If True, use semi-gradient variant (default)
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        trace_decay: float = 0.0,
        use_semi_gradient: bool = True,
    ):
        """Initialize TD-IDBD optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate theta for adapting step-sizes
            trace_decay: Eligibility trace decay lambda (0 = TD(0))
            use_semi_gradient: If True, use semi-gradient variant (recommended)
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._trace_decay = trace_decay
        self._use_semi_gradient = use_semi_gradient

    def init(self, feature_dim: int) -> TDIDBDState:
        """Initialize TD-IDBD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            TD-IDBD state with per-weight step-sizes, traces, and h traces
        """
        return TDIDBDState(
            log_step_sizes=jnp.full(
                feature_dim, jnp.log(self._initial_step_size), dtype=jnp.float32
            ),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            h_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            trace_decay=jnp.array(self._trace_decay, dtype=jnp.float32),
            bias_log_step_size=jnp.array(jnp.log(self._initial_step_size), dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_h_trace=jnp.array(0.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: TDIDBDState,
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> TDOptimizerUpdate:
        """Compute TD-IDBD weight update with adaptive step-sizes.

        Implements Algorithm 3 (semi-gradient) or Algorithm 4 (ordinary gradient)
        from Kearney et al. 2019.

        Args:
            state: Current TD-IDBD state
            td_error: TD error δ = R + γV(s') - V(s)
            observation: Current observation φ(s)
            next_observation: Next observation φ(s')
            gamma: Discount factor γ (0 at terminal)

        Returns:
            TDOptimizerUpdate with weight deltas and updated state
        """
        delta = jnp.squeeze(td_error)
        theta = state.meta_step_size
        lam = state.trace_decay
        gamma_scalar = jnp.squeeze(gamma)

        if self._use_semi_gradient:
            # Semi-gradient TD-IDBD (Algorithm 3)
            # β_i += θ*δ*φ_i(s)*h_i
            gradient_correlation = delta * observation * state.h_traces
            new_log_step_sizes = state.log_step_sizes + theta * gradient_correlation
        else:
            # Ordinary gradient TD-IDBD (Algorithm 4)
            # β_i -= θ*δ*[γ*φ_i(s') - φ_i(s)]*h_i
            # Note: negative sign because gradient direction is reversed
            feature_diff = gamma_scalar * next_observation - observation
            gradient_correlation = delta * feature_diff * state.h_traces
            new_log_step_sizes = state.log_step_sizes - theta * gradient_correlation

        # Clip log step-sizes to prevent numerical issues
        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)

        # Get updated step-sizes for weight update
        new_alphas = jnp.exp(new_log_step_sizes)

        # Update eligibility traces: z_i = γ*λ*z_i + φ_i(s)
        new_eligibility_traces = gamma_scalar * lam * state.eligibility_traces + observation

        # Compute weight delta: α_i*δ*z_i
        weight_delta = new_alphas * delta * new_eligibility_traces

        if self._use_semi_gradient:
            # Semi-gradient h update (Algorithm 3, line 9)
            # h_i = h_i*[1 - α_i*φ_i(s)*z_i]^+ + α_i*δ*z_i
            h_decay = jnp.maximum(0.0, 1.0 - new_alphas * observation * new_eligibility_traces)
            new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces
        else:
            # Ordinary gradient h update (Algorithm 4, line 9)
            # h_i = h_i*[1 + α_i*z_i*(γ*φ_i(s') - φ_i(s))]^+ + α_i*δ*z_i
            feature_diff = gamma_scalar * next_observation - observation
            h_decay = jnp.maximum(0.0, 1.0 + new_alphas * new_eligibility_traces * feature_diff)
            new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces

        # Bias updates (similar logic but scalar)
        if self._use_semi_gradient:
            # Semi-gradient bias meta-update
            bias_gradient_correlation = delta * state.bias_h_trace
            new_bias_log_step_size = state.bias_log_step_size + theta * bias_gradient_correlation
        else:
            # Ordinary gradient bias meta-update
            # For bias, φ(s) = 1, so feature_diff = γ - 1
            bias_feature_diff = gamma_scalar - 1.0
            bias_gradient_correlation = delta * bias_feature_diff * state.bias_h_trace
            new_bias_log_step_size = state.bias_log_step_size - theta * bias_gradient_correlation

        new_bias_log_step_size = jnp.clip(new_bias_log_step_size, -10.0, 2.0)
        new_bias_alpha = jnp.exp(new_bias_log_step_size)

        # Update bias eligibility trace
        new_bias_eligibility_trace = gamma_scalar * lam * state.bias_eligibility_trace + 1.0

        # Bias weight delta
        bias_delta = new_bias_alpha * delta * new_bias_eligibility_trace

        if self._use_semi_gradient:
            # Semi-gradient bias h update
            bias_h_decay = jnp.maximum(0.0, 1.0 - new_bias_alpha * new_bias_eligibility_trace)
            new_bias_h_trace = (
                state.bias_h_trace * bias_h_decay
                + new_bias_alpha * delta * new_bias_eligibility_trace
            )
        else:
            # Ordinary gradient bias h update
            bias_feature_diff = gamma_scalar - 1.0
            bias_h_decay = jnp.maximum(
                0.0, 1.0 + new_bias_alpha * new_bias_eligibility_trace * bias_feature_diff
            )
            new_bias_h_trace = (
                state.bias_h_trace * bias_h_decay
                + new_bias_alpha * delta * new_bias_eligibility_trace
            )

        new_state = TDIDBDState(
            log_step_sizes=new_log_step_sizes,
            eligibility_traces=new_eligibility_traces,
            h_traces=new_h_traces,
            meta_step_size=theta,
            trace_decay=lam,
            bias_log_step_size=new_bias_log_step_size,
            bias_eligibility_trace=new_bias_eligibility_trace,
            bias_h_trace=new_bias_h_trace,
        )

        return TDOptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=new_state,
            metrics={
                "mean_step_size": jnp.mean(new_alphas),
                "min_step_size": jnp.min(new_alphas),
                "max_step_size": jnp.max(new_alphas),
                "mean_eligibility_trace": jnp.mean(jnp.abs(new_eligibility_traces)),
            },
        )


class AutoTDIDBD(TDOptimizer[AutoTDIDBDState]):
    """AutoStep-style normalized TD-IDBD optimizer.

    Adds AutoStep-style normalization to TDIDBD for improved stability and
    reduced sensitivity to the meta step-size theta. Includes:
    1. Normalization of the meta-weight update by a running trace of recent updates
    2. Effective step-size normalization to prevent overshooting

    Reference: Kearney et al. 2019, Algorithm 6 "AutoStep Style Normalized TIDBD(λ)"

    The AutoTDIDBD algorithm:
        1. Compute TD error: `δ = R + γ*w^T*φ(s') - w^T*φ(s)`
        2. Update normalizers: `η_i = max(|δ*[γφ_i(s')-φ_i(s)]*h_i|,
           η_i - (1/τ)*α_i*[γφ_i(s')-φ_i(s)]*z_i*(|δ*φ_i(s)*h_i| - η_i))`
        3. Normalized meta-update: `β_i -= θ*(1/η_i)*δ*[γφ_i(s')-φ_i(s)]*h_i`
        4. Effective step-size normalization: `M = max(-exp(β)*[γφ(s')-φ(s)]^T*z, 1)`
           then `β_i -= log(M)`
        5. Update weights and traces as in TIDBD

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate theta
        trace_decay: Eligibility trace decay lambda
        normalizer_decay: Decay parameter tau for normalizers
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        trace_decay: float = 0.0,
        normalizer_decay: float = 10000.0,
    ):
        """Initialize AutoTDIDBD optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate theta for adapting step-sizes
            trace_decay: Eligibility trace decay lambda (0 = TD(0))
            normalizer_decay: Decay parameter tau for normalizers (default: 10000)
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._trace_decay = trace_decay
        self._normalizer_decay = normalizer_decay

    def init(self, feature_dim: int) -> AutoTDIDBDState:
        """Initialize AutoTDIDBD state.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            AutoTDIDBD state with per-weight step-sizes, traces, h traces, and normalizers
        """
        return AutoTDIDBDState(
            log_step_sizes=jnp.full(
                feature_dim, jnp.log(self._initial_step_size), dtype=jnp.float32
            ),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            h_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            normalizers=jnp.ones(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            trace_decay=jnp.array(self._trace_decay, dtype=jnp.float32),
            normalizer_decay=jnp.array(self._normalizer_decay, dtype=jnp.float32),
            bias_log_step_size=jnp.array(jnp.log(self._initial_step_size), dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_h_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_normalizer=jnp.array(1.0, dtype=jnp.float32),
        )

    def update(
        self,
        state: AutoTDIDBDState,
        td_error: Array,
        observation: Array,
        next_observation: Array,
        gamma: Array,
    ) -> TDOptimizerUpdate:
        """Compute AutoTDIDBD weight update with normalized adaptive step-sizes.

        Implements Algorithm 6 from Kearney et al. 2019.

        Args:
            state: Current AutoTDIDBD state
            td_error: TD error δ = R + γV(s') - V(s)
            observation: Current observation φ(s)
            next_observation: Next observation φ(s')
            gamma: Discount factor γ (0 at terminal)

        Returns:
            TDOptimizerUpdate with weight deltas and updated state
        """
        delta = jnp.squeeze(td_error)
        theta = state.meta_step_size
        lam = state.trace_decay
        tau = state.normalizer_decay
        gamma_scalar = jnp.squeeze(gamma)

        # Feature difference: γ*φ(s') - φ(s)
        feature_diff = gamma_scalar * next_observation - observation

        # Current step-sizes
        alphas = jnp.exp(state.log_step_sizes)

        # Update normalizers (Algorithm 6, lines 5-7)
        # η_i = max(|δ*[γφ_i(s')-φ_i(s)]*h_i|,
        #           η_i - (1/τ)*α_i*[γφ_i(s')-φ_i(s)]*z_i*(|δ*φ_i(s)*h_i| - η_i))
        abs_weight_update = jnp.abs(delta * feature_diff * state.h_traces)
        normalizer_decay_term = (
            (1.0 / tau)
            * alphas
            * feature_diff
            * state.eligibility_traces
            * (jnp.abs(delta * observation * state.h_traces) - state.normalizers)
        )
        new_normalizers = jnp.maximum(abs_weight_update, state.normalizers - normalizer_decay_term)
        # Ensure normalizers don't go to zero
        new_normalizers = jnp.maximum(new_normalizers, 1e-8)

        # Normalized meta-update (Algorithm 6, line 9)
        # β_i -= θ*(1/η_i)*δ*[γφ_i(s')-φ_i(s)]*h_i
        normalized_gradient = delta * feature_diff * state.h_traces / new_normalizers
        new_log_step_sizes = state.log_step_sizes - theta * normalized_gradient

        # Effective step-size normalization (Algorithm 6, lines 10-11)
        # M = max(-exp(β_i)*[γφ_i(s')-φ_i(s)]^T*z_i, 1)
        # β_i -= log(M)
        effective_step_size = -jnp.sum(
            jnp.exp(new_log_step_sizes) * feature_diff * state.eligibility_traces
        )
        normalization_factor = jnp.maximum(effective_step_size, 1.0)
        new_log_step_sizes = new_log_step_sizes - jnp.log(normalization_factor)

        # Clip log step-sizes to prevent numerical issues
        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)

        # Get updated step-sizes
        new_alphas = jnp.exp(new_log_step_sizes)

        # Update eligibility traces: z_i = γ*λ*z_i + φ_i(s)
        new_eligibility_traces = gamma_scalar * lam * state.eligibility_traces + observation

        # Compute weight delta: α_i*δ*z_i
        weight_delta = new_alphas * delta * new_eligibility_traces

        # Update h traces (ordinary gradient variant, Algorithm 6 line 15)
        # h_i = h_i*[1 + α_i*[γφ_i(s')-φ_i(s)]*z_i]^+ + α_i*δ*z_i
        h_decay = jnp.maximum(0.0, 1.0 + new_alphas * feature_diff * new_eligibility_traces)
        new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces

        # Bias updates
        bias_alpha = jnp.exp(state.bias_log_step_size)
        bias_feature_diff = gamma_scalar - 1.0  # For bias, φ(s) = 1

        # Bias normalizer update
        abs_bias_weight_update = jnp.abs(delta * bias_feature_diff * state.bias_h_trace)
        bias_normalizer_decay_term = (
            (1.0 / tau)
            * bias_alpha
            * bias_feature_diff
            * state.bias_eligibility_trace
            * (jnp.abs(delta * state.bias_h_trace) - state.bias_normalizer)
        )
        new_bias_normalizer = jnp.maximum(
            abs_bias_weight_update, state.bias_normalizer - bias_normalizer_decay_term
        )
        new_bias_normalizer = jnp.maximum(new_bias_normalizer, 1e-8)

        # Normalized bias meta-update
        normalized_bias_gradient = (
            delta * bias_feature_diff * state.bias_h_trace / new_bias_normalizer
        )
        new_bias_log_step_size = state.bias_log_step_size - theta * normalized_bias_gradient

        # Effective step-size normalization for bias
        bias_effective_step_size = (
            -jnp.exp(new_bias_log_step_size) * bias_feature_diff * state.bias_eligibility_trace
        )
        bias_norm_factor = jnp.maximum(bias_effective_step_size, 1.0)
        new_bias_log_step_size = new_bias_log_step_size - jnp.log(bias_norm_factor)

        new_bias_log_step_size = jnp.clip(new_bias_log_step_size, -10.0, 2.0)
        new_bias_alpha = jnp.exp(new_bias_log_step_size)

        # Update bias eligibility trace
        new_bias_eligibility_trace = gamma_scalar * lam * state.bias_eligibility_trace + 1.0

        # Bias weight delta
        bias_delta = new_bias_alpha * delta * new_bias_eligibility_trace

        # Bias h trace update
        bias_h_decay = jnp.maximum(
            0.0, 1.0 + new_bias_alpha * bias_feature_diff * new_bias_eligibility_trace
        )
        new_bias_h_trace = (
            state.bias_h_trace * bias_h_decay + new_bias_alpha * delta * new_bias_eligibility_trace
        )

        new_state = AutoTDIDBDState(
            log_step_sizes=new_log_step_sizes,
            eligibility_traces=new_eligibility_traces,
            h_traces=new_h_traces,
            normalizers=new_normalizers,
            meta_step_size=theta,
            trace_decay=lam,
            normalizer_decay=tau,
            bias_log_step_size=new_bias_log_step_size,
            bias_eligibility_trace=new_bias_eligibility_trace,
            bias_h_trace=new_bias_h_trace,
            bias_normalizer=new_bias_normalizer,
        )

        return TDOptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=new_state,
            metrics={
                "mean_step_size": jnp.mean(new_alphas),
                "min_step_size": jnp.min(new_alphas),
                "max_step_size": jnp.max(new_alphas),
                "mean_eligibility_trace": jnp.mean(jnp.abs(new_eligibility_traces)),
                "mean_normalizer": jnp.mean(new_normalizers),
            },
        )
