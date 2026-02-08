"""Optimizers for continual learning.

Implements LMS (fixed step-size baseline), IDBD (meta-learned step-sizes),
Autostep (tuning-free step-size adaptation), and ObGD (observation-bounded)
for the Alberta Plan.

Also provides the ``Bounder`` ABC for decoupled update bounding (e.g. ObGDBounding).

References:
- Sutton 1992, "Adapting Bias by Gradient Descent: An Incremental
  Version of Delta-Bar-Delta"
- Mahmood et al. 2012, "Tuning-free step-size adaptation"
- Elsayed et al. 2024, "Streaming Deep Reinforcement Learning Finally Works"
"""

from abc import ABC, abstractmethod
from typing import Any

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from alberta_framework.core.types import (
    AutostepParamState,
    AutostepState,
    AutoTDIDBDState,
    IDBDState,
    LMSState,
    ObGDState,
    TDIDBDState,
)

# =============================================================================
# Bounder ABC
# =============================================================================


class Bounder(ABC):
    """Base class for update bounding strategies.

    A bounder takes the proposed per-parameter step arrays from an optimizer
    and optionally scales them down to prevent overshooting.
    """

    @abstractmethod
    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Bound proposed update steps.

        Args:
            steps: Per-parameter step arrays from the optimizer
            error: Prediction error scalar
            params: Current parameter values (needed by some bounders like AGC)

        Returns:
            ``(bounded_steps, metric)`` where metric is a scalar for reporting
            (e.g., scale factor for ObGD, mean clip ratio for AGC)
        """
        ...


class ObGDBounding(Bounder):
    """ObGD-style global update bounding (Elsayed et al. 2024).

    Computes a global bounding factor from the L1 norm of all proposed
    steps and the error magnitude, then uniformly scales all steps down
    if the combined update would be too large.

    For LMS with a single scalar step-size ``alpha``:
    ``total_step = alpha * z_sum``, giving
    ``M = alpha * kappa * max(|error|, 1) * z_sum`` -- identical to
    the original Elsayed et al. 2024 formula.

    Attributes:
        kappa: Bounding sensitivity parameter (higher = more conservative)
    """

    def __init__(self, kappa: float = 2.0):
        self._kappa = kappa

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Bound proposed steps using ObGD formula.

        Args:
            steps: Per-parameter step arrays
            error: Prediction error scalar
            params: Current parameter values (unused by ObGD)

        Returns:
            ``(bounded_steps, scale)`` where scale is the bounding factor
        """
        del params  # ObGD bounds based on step/error magnitude only
        error_scalar = jnp.squeeze(error)
        total_step = jnp.array(0.0)
        for s in steps:
            total_step = total_step + jnp.sum(jnp.abs(s))
        delta_bar = jnp.maximum(jnp.abs(error_scalar), 1.0)
        bound_magnitude = self._kappa * delta_bar * total_step
        scale = 1.0 / jnp.maximum(bound_magnitude, 1.0)
        bounded = tuple(scale * s for s in steps)
        return bounded, scale


def _unitwise_norm(x: Array) -> Array:
    """Compute unit-wise L2 norm.

    For 2D+ arrays (e.g. weight matrices ``(fan_in, fan_out)``):
    L2 norm over all axes except the last, with keepdims for broadcasting.
    For 1D arrays (biases): absolute value per element.
    For scalars: absolute value.
    """
    if x.ndim >= 2:
        return jnp.sqrt(jnp.sum(x**2, axis=tuple(range(x.ndim - 1)), keepdims=True))
    return jnp.abs(x)


class AGCBounding(Bounder):
    """Adaptive Gradient Clipping (Brock et al. 2021).

    Clips per-output-unit based on the ratio of gradient norm to weight norm.
    Units where ``||grad|| / max(||weight||, eps) > clip_factor`` get scaled
    down to respect the constraint.

    Unlike ObGDBounding which applies a single global scale factor, AGC
    applies fine-grained, per-unit clipping that adapts to each layer's
    weight magnitude.

    The metric returned is the fraction of units that were clipped (0.0 = no
    clipping, 1.0 = all units clipped).

    Reference: Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021).
    "High-Performance Large-Scale Image Recognition Without Normalization"
    (arXiv: 2102.06171)

    Attributes:
        clip_factor: Maximum allowed gradient-to-weight ratio (lambda). Default 0.01.
        eps: Floor for weight norm to avoid division by zero. Default 1e-3.
    """

    def __init__(self, clip_factor: float = 0.01, eps: float = 1e-3):
        self._clip_factor = clip_factor
        self._eps = eps

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        """Bound proposed steps using per-unit adaptive gradient clipping.

        For each parameter/step pair, computes unit-wise norms and clips
        units where ``|error| * ||step|| > clip_factor * max(||param||, eps)``.

        Args:
            steps: Per-parameter step arrays from the optimizer
            error: Prediction error scalar
            params: Current parameter values (used for weight norms)

        Returns:
            ``(clipped_steps, frac_clipped)`` where frac_clipped is the
            fraction of units that were clipped
        """
        error_abs = jnp.abs(jnp.squeeze(error))
        clipped = []
        total_units = 0
        clipped_units = jnp.array(0.0)

        for step, param in zip(steps, params):
            p_norm = _unitwise_norm(param)
            s_norm = _unitwise_norm(step)
            g_norm = error_abs * s_norm
            max_norm = jnp.maximum(p_norm, self._eps) * self._clip_factor
            scale = max_norm / jnp.maximum(g_norm, 1e-6)
            needs_clip = g_norm > max_norm
            clipped_step = jnp.where(needs_clip, step * scale, step)
            clipped.append(clipped_step)

            total_units += needs_clip.size
            clipped_units = clipped_units + jnp.sum(needs_clip.astype(jnp.float32))

        frac_clipped = clipped_units / jnp.maximum(total_units, 1)
        return tuple(clipped), frac_clipped


# =============================================================================
# Supervised Learning Optimizers
# =============================================================================


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


class Optimizer[StateT: (LMSState, IDBDState, AutostepState, ObGDState, AutostepParamState)](ABC):
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

    def init_for_shape(self, shape: tuple[int, ...]) -> Any:
        """Initialize optimizer state for parameters of arbitrary shape.

        Used by MLP learners where parameters are matrices/vectors of
        varying shapes. Not all optimizers support this.

        The return type varies by subclass (e.g. ``LMSState`` for LMS,
        ``AutostepParamState`` for Autostep) so the base signature uses
        ``Any``.

        Args:
            shape: Shape of the parameter array

        Returns:
            Initial optimizer state with arrays matching the given shape

        Raises:
            NotImplementedError: If the optimizer does not support this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support init_for_shape. "
            "Only LMS and Autostep currently implement this."
        )

    def update_from_gradient(
        self, state: Any, gradient: Array, error: Array | None = None
    ) -> tuple[Array, Any]:
        """Compute step delta from pre-computed gradient.

        The returned delta does NOT include the error -- the caller is
        responsible for multiplying ``error * delta`` before applying.

        The state type varies by subclass (e.g. ``LMSState`` for LMS,
        ``AutostepParamState`` for Autostep) so the base signature uses
        ``Any``.

        Args:
            state: Current optimizer state
            gradient: Pre-computed gradient (e.g. eligibility trace)
            error: Optional prediction error scalar. Optimizers with
                meta-learning (e.g. Autostep) use this for meta-gradient
                computation. LMS ignores it.

        Returns:
            ``(step, new_state)`` where step has the same shape as gradient

        Raises:
            NotImplementedError: If the optimizer does not support this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support update_from_gradient. "
            "Only LMS and Autostep currently implement this."
        )


class LMS(Optimizer[LMSState]):
    """Least Mean Square optimizer with fixed step-size.

    The simplest gradient-based optimizer: ``w_{t+1} = w_t + alpha * delta * x_t``

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

    def init_for_shape(self, shape: tuple[int, ...]) -> LMSState:
        """Initialize LMS state for arbitrary-shape parameters.

        LMS state is shape-independent (single scalar), so this returns
        the same state regardless of shape.
        """
        return LMSState(step_size=jnp.array(self._step_size, dtype=jnp.float32))

    def update_from_gradient(
        self, state: LMSState, gradient: Array, error: Array | None = None
    ) -> tuple[Array, LMSState]:
        """Compute step from gradient: ``step = alpha * gradient``.

        Args:
            state: Current LMS state
            gradient: Pre-computed gradient (any shape)
            error: Unused by LMS (accepted for interface compatibility)

        Returns:
            ``(step, state)`` -- state is unchanged for LMS
        """
        del error  # LMS doesn't meta-learn
        return state.step_size * gradient, state

    def update(
        self,
        state: LMSState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute LMS weight update.

        Update rule: ``delta_w = alpha * error * x``

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

        Following Sutton 1992, Figure 2, the operation ordering is:

        1. Meta-update: ``log_alpha_i += beta * error * x_i * h_i`` (using OLD traces)
        2. Compute NEW step-sizes: ``alpha_i = exp(log_alpha_i)``
        3. Update weights: ``w_i += alpha_i * error * x_i`` (using NEW alpha)
        4. Update traces: ``h_i = h_i * max(0, 1 - alpha_i * x_i^2) + alpha_i * error * x_i``
           (using NEW alpha)

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

        # 1. Meta-update: adapt step-sizes using OLD traces
        gradient_correlation = error_scalar * observation * state.traces
        new_log_step_sizes = state.log_step_sizes + beta * gradient_correlation

        # Clip log step-sizes to prevent numerical issues
        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)

        # 2. Compute NEW step-sizes
        new_alphas = jnp.exp(new_log_step_sizes)

        # 3. Weight updates using NEW alpha: alpha_i * error * x_i
        weight_delta = new_alphas * error_scalar * observation

        # 4. Update traces using NEW alpha: h_i = h_i * decay + alpha_i * error * x_i
        # decay = max(0, 1 - alpha_i * x_i^2)
        decay = jnp.maximum(0.0, 1.0 - new_alphas * observation**2)
        new_traces = state.traces * decay + new_alphas * error_scalar * observation

        # Bias updates (same ordering: meta-update first, then new alpha)
        bias_gradient_correlation = error_scalar * state.bias_trace
        new_bias_step_size = state.bias_step_size * jnp.exp(beta * bias_gradient_correlation)
        new_bias_step_size = jnp.clip(new_bias_step_size, 1e-6, 1.0)

        bias_delta = new_bias_step_size * error_scalar

        bias_decay = jnp.maximum(0.0, 1.0 - new_bias_step_size)
        new_bias_trace = state.bias_trace * bias_decay + new_bias_step_size * error_scalar

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
                "mean_step_size": jnp.mean(new_alphas),
                "min_step_size": jnp.min(new_alphas),
                "max_step_size": jnp.max(new_alphas),
            },
        )


class Autostep(Optimizer[AutostepState]):
    """Autostep optimizer with tuning-free step-size adaptation.

    Implements the exact algorithm from Mahmood et al. 2012, Table 1.

    The algorithm maintains per-weight step-sizes that adapt based on
    meta-gradient correlation. The key innovations are:
    - Self-regulated normalizers (v_i) that track meta-gradient magnitude
      ``|delta * x_i * h_i|`` for stable meta-updates
    - Overshoot prevention via effective step-size normalization
      ``M = max(sum(alpha_i * x_i^2), 1)``

    Per-sample update (Table 1):

    1. ``v_i = max(|delta*x_i*h_i|, v_i + (1/tau)*alpha_i*x_i^2*(|delta*x_i*h_i| - v_i))``
    2. ``alpha_i *= exp(mu * delta*x_i*h_i / v_i)`` where ``v_i > 0``
    3. ``M = max(sum(alpha_i * x_i^2), 1)``; ``alpha_i /= M``
    4. ``w_i += alpha_i * delta * x_i`` (weight update with NEW alpha)
    5. ``h_i = h_i * (1 - alpha_i * x_i^2) + alpha_i * delta * x_i`` (trace update)

    Reference: Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012).
    "Tuning-free step-size adaptation"

    Attributes:
        initial_step_size: Initial per-weight step-size
        meta_step_size: Meta learning rate mu for adapting step-sizes
        tau: Time constant for normalizer adaptation (default: 10000)
    """

    def __init__(
        self,
        initial_step_size: float = 0.01,
        meta_step_size: float = 0.01,
        tau: float = 10000.0,
    ):
        """Initialize Autostep optimizer.

        Args:
            initial_step_size: Initial value for per-weight step-sizes
            meta_step_size: Meta learning rate for adapting step-sizes
            tau: Time constant for normalizer adaptation (default: 10000).
                Higher values mean slower normalizer decay.
        """
        self._initial_step_size = initial_step_size
        self._meta_step_size = meta_step_size
        self._tau = tau

    def init(self, feature_dim: int) -> AutostepState:
        """Initialize Autostep state.

        Normalizers (v_i) and traces (h_i) are initialized to 0 per the paper.

        Args:
            feature_dim: Dimension of weight vector

        Returns:
            Autostep state with per-weight step-sizes, traces, and normalizers
        """
        return AutostepState(
            step_sizes=jnp.full(feature_dim, self._initial_step_size, dtype=jnp.float32),
            traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            normalizers=jnp.zeros(feature_dim, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            tau=jnp.array(self._tau, dtype=jnp.float32),
            bias_step_size=jnp.array(self._initial_step_size, dtype=jnp.float32),
            bias_trace=jnp.array(0.0, dtype=jnp.float32),
            bias_normalizer=jnp.array(0.0, dtype=jnp.float32),
        )

    def init_for_shape(self, shape: tuple[int, ...]) -> AutostepParamState:
        """Initialize Autostep state for arbitrary-shape parameters.

        Args:
            shape: Shape of the parameter array

        Returns:
            AutostepParamState with arrays matching the given shape
        """
        return AutostepParamState(
            step_sizes=jnp.full(shape, self._initial_step_size, dtype=jnp.float32),
            traces=jnp.zeros(shape, dtype=jnp.float32),
            normalizers=jnp.zeros(shape, dtype=jnp.float32),
            meta_step_size=jnp.array(self._meta_step_size, dtype=jnp.float32),
            tau=jnp.array(self._tau, dtype=jnp.float32),
        )

    def update_from_gradient(
        self,
        state: AutostepParamState,
        gradient: Array,
        error: Array | None = None,
    ) -> tuple[Array, AutostepParamState]:
        """Compute Autostep update from pre-computed gradient (MLP path).

        Implements the Table 1 algorithm generalized for arbitrary-shape
        parameters, where ``gradient`` plays the role of the eligibility
        trace ``z`` (prediction gradient).

        When ``error`` is provided, the full paper algorithm is used:
        meta-gradient is ``error * z * h``. When ``error`` is None,
        falls back to error-free approximation (``z * h``).

        The returned step does NOT include the error -- the caller applies
        ``param += error * step`` after optional bounding.

        Args:
            state: Current Autostep param state
            gradient: Pre-computed gradient / eligibility trace (same shape as state arrays)
            error: Optional prediction error scalar. When provided, enables
                the full paper algorithm with error-scaled meta-gradients.

        Returns:
            ``(step, new_state)`` where step has the same shape as gradient
        """
        mu = state.meta_step_size
        tau = state.tau

        z = gradient  # eligibility trace
        z_sq = z**2

        # Compute meta-gradient: δ*z*h (or z*h if error is None)
        if error is not None:
            error_scalar = jnp.squeeze(error)
            meta_gradient = error_scalar * z * state.traces
        else:
            meta_gradient = z * state.traces

        abs_meta_gradient = jnp.abs(meta_gradient)

        # Eq. 4: v_i = max(|meta_grad|, v_i + (1/τ)*α_i*z_i²*(|meta_grad| - v_i))
        v_update = state.normalizers + (1.0 / tau) * state.step_sizes * z_sq * (
            abs_meta_gradient - state.normalizers
        )
        new_normalizers = jnp.maximum(abs_meta_gradient, v_update)

        # Eq. 5: α_i *= exp(μ * meta_grad / v_i) where v_i > 0
        safe_v = jnp.maximum(new_normalizers, 1e-38)
        new_step_sizes = jnp.where(
            new_normalizers > 0,
            state.step_sizes * jnp.exp(mu * meta_gradient / safe_v),
            state.step_sizes,
        )

        # Eq. 6-7: M = max(Σ α_i*z_i², 1); α_i /= M
        effective_step = jnp.sum(new_step_sizes * z_sq)
        m_factor = jnp.maximum(effective_step, 1.0)
        new_step_sizes = new_step_sizes / m_factor

        # Clip step-sizes for numerical safety
        new_step_sizes = jnp.clip(new_step_sizes, 1e-8, 1.0)

        # Compute step: α_i * z_i (error applied externally)
        step = new_step_sizes * z

        # Trace update: h_i = h_i*(1 - α_i*z_i²) + α_i*δ*z_i
        trace_decay = 1.0 - new_step_sizes * z_sq
        if error is not None:
            new_traces = state.traces * trace_decay + new_step_sizes * error_scalar * z
        else:
            new_traces = state.traces * trace_decay + new_step_sizes * z

        new_state = AutostepParamState(
            step_sizes=new_step_sizes,
            traces=new_traces,
            normalizers=new_normalizers,
            meta_step_size=mu,
            tau=tau,
        )

        return step, new_state

    def update(
        self,
        state: AutostepState,
        error: Array,
        observation: Array,
    ) -> OptimizerUpdate:
        """Compute Autostep weight update following Mahmood et al. 2012, Table 1.

        The algorithm per sample:

        1. Eq. 4: ``v_i = max(|δ*x_i*h_i|, v_i + (1/τ)*α_i*x_i²*(|δ*x_i*h_i| - v_i))``
        2. Eq. 5: ``α_i *= exp(μ * δ*x_i*h_i / v_i)`` where ``v_i > 0``
        3. Eq. 6-7: ``M = max(Σ α_i*x_i² + α_bias, 1)``; ``α_i /= M``, ``α_bias /= M``
        4. Weight update: ``w_i += α_i * δ * x_i`` (with NEW alpha)
        5. Trace update: ``h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i``

        Args:
            state: Current Autostep state
            error: Prediction error (scalar)
            observation: Feature vector

        Returns:
            OptimizerUpdate with weight deltas and updated state
        """
        error_scalar = jnp.squeeze(error)
        mu = state.meta_step_size
        tau = state.tau

        x = observation
        x_sq = x**2

        # --- Weights ---
        # Meta-gradient: δ*x_i*h_i
        meta_gradient = error_scalar * x * state.traces
        abs_meta_gradient = jnp.abs(meta_gradient)

        # Eq. 4: v_i update (self-regulated EMA)
        v_update = state.normalizers + (1.0 / tau) * state.step_sizes * x_sq * (
            abs_meta_gradient - state.normalizers
        )
        new_normalizers = jnp.maximum(abs_meta_gradient, v_update)

        # Eq. 5: α_i *= exp(μ * meta_grad / v_i) where v_i > 0
        safe_v = jnp.maximum(new_normalizers, 1e-38)
        new_step_sizes = jnp.where(
            new_normalizers > 0,
            state.step_sizes * jnp.exp(mu * meta_gradient / safe_v),
            state.step_sizes,
        )

        # --- Bias ---
        # Meta-gradient for bias (implicit x=1): δ*h_bias
        bias_meta_gradient = error_scalar * state.bias_trace
        abs_bias_meta_gradient = jnp.abs(bias_meta_gradient)

        # Eq. 4 for bias
        bias_v_update = state.bias_normalizer + (1.0 / tau) * state.bias_step_size * (
            abs_bias_meta_gradient - state.bias_normalizer
        )
        new_bias_normalizer = jnp.maximum(abs_bias_meta_gradient, bias_v_update)

        # Eq. 5 for bias
        safe_bias_v = jnp.maximum(new_bias_normalizer, 1e-38)
        new_bias_step_size = jnp.where(
            new_bias_normalizer > 0,
            state.bias_step_size * jnp.exp(mu * bias_meta_gradient / safe_bias_v),
            state.bias_step_size,
        )

        # Eq. 6-7: Overshoot prevention (joint over weights + bias)
        # M = max(Σ α_i*x_i² + α_bias*1², 1)
        effective_step = jnp.sum(new_step_sizes * x_sq) + new_bias_step_size
        m_factor = jnp.maximum(effective_step, 1.0)
        new_step_sizes = new_step_sizes / m_factor
        new_bias_step_size = new_bias_step_size / m_factor

        # Clip step-sizes for numerical safety
        new_step_sizes = jnp.clip(new_step_sizes, 1e-8, 1.0)
        new_bias_step_size = jnp.clip(new_bias_step_size, 1e-8, 1.0)

        # Weight update with NEW alpha: α_i * δ * x_i
        weight_delta = new_step_sizes * error_scalar * x

        # Bias update: α_bias * δ
        bias_delta = new_bias_step_size * error_scalar

        # Trace update: h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i
        trace_decay = 1.0 - new_step_sizes * x_sq
        new_traces = state.traces * trace_decay + new_step_sizes * error_scalar * x

        # Bias trace: h_bias = h_bias*(1 - α_bias) + α_bias*δ
        bias_trace_decay = 1.0 - new_bias_step_size
        new_bias_trace = state.bias_trace * bias_trace_decay + new_bias_step_size * error_scalar

        new_state = AutostepState(
            step_sizes=new_step_sizes,
            traces=new_traces,
            normalizers=new_normalizers,
            meta_step_size=mu,
            tau=tau,
            bias_step_size=new_bias_step_size,
            bias_trace=new_bias_trace,
            bias_normalizer=new_bias_normalizer,
        )

        return OptimizerUpdate(
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            new_state=new_state,
            metrics={
                "mean_step_size": jnp.mean(new_step_sizes),
                "min_step_size": jnp.min(new_step_sizes),
                "max_step_size": jnp.max(new_step_sizes),
                "mean_normalizer": jnp.mean(new_normalizers),
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

    1. Update traces: ``z = gamma * lamda * z + observation``
    2. Compute bound: ``M = alpha * kappa * max(|error|, 1) * (||z_w||_1 + |z_b|)``
    3. Effective step: ``alpha_eff = min(alpha, alpha / M)`` (i.e. ``alpha / max(M, 1)``)
    4. Weight delta: ``delta_w = alpha_eff * error * z_w``
    5. Bias delta: ``delta_b = alpha_eff * error * z_b``

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
            td_error: TD error delta = R + gamma*V(s') - V(s)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal)

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
    - Semi-gradient (default): Uses only phi(s) in meta-update, more stable
    - Ordinary gradient: Uses both phi(s) and phi(s'), more accurate but sensitive

    Reference: Kearney et al. 2019, "Learning Feature Relevance Through Step Size
    Adaptation in Temporal-Difference Learning"

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
            td_error: TD error delta = R + gamma*V(s') - V(s)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal)

        Returns:
            TDOptimizerUpdate with weight deltas and updated state
        """
        delta = jnp.squeeze(td_error)
        theta = state.meta_step_size
        lam = state.trace_decay
        gamma_scalar = jnp.squeeze(gamma)

        if self._use_semi_gradient:
            gradient_correlation = delta * observation * state.h_traces
            new_log_step_sizes = state.log_step_sizes + theta * gradient_correlation
        else:
            feature_diff = gamma_scalar * next_observation - observation
            gradient_correlation = delta * feature_diff * state.h_traces
            new_log_step_sizes = state.log_step_sizes - theta * gradient_correlation

        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)
        new_alphas = jnp.exp(new_log_step_sizes)

        new_eligibility_traces = gamma_scalar * lam * state.eligibility_traces + observation
        weight_delta = new_alphas * delta * new_eligibility_traces

        if self._use_semi_gradient:
            h_decay = jnp.maximum(0.0, 1.0 - new_alphas * observation * new_eligibility_traces)
            new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces
        else:
            feature_diff = gamma_scalar * next_observation - observation
            h_decay = jnp.maximum(0.0, 1.0 + new_alphas * new_eligibility_traces * feature_diff)
            new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces

        # Bias updates
        if self._use_semi_gradient:
            bias_gradient_correlation = delta * state.bias_h_trace
            new_bias_log_step_size = state.bias_log_step_size + theta * bias_gradient_correlation
        else:
            bias_feature_diff = gamma_scalar - 1.0
            bias_gradient_correlation = delta * bias_feature_diff * state.bias_h_trace
            new_bias_log_step_size = state.bias_log_step_size - theta * bias_gradient_correlation

        new_bias_log_step_size = jnp.clip(new_bias_log_step_size, -10.0, 2.0)
        new_bias_alpha = jnp.exp(new_bias_log_step_size)

        new_bias_eligibility_trace = gamma_scalar * lam * state.bias_eligibility_trace + 1.0
        bias_delta = new_bias_alpha * delta * new_bias_eligibility_trace

        if self._use_semi_gradient:
            bias_h_decay = jnp.maximum(0.0, 1.0 - new_bias_alpha * new_bias_eligibility_trace)
            new_bias_h_trace = (
                state.bias_h_trace * bias_h_decay
                + new_bias_alpha * delta * new_bias_eligibility_trace
            )
        else:
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
    reduced sensitivity to the meta step-size theta.

    Reference: Kearney et al. 2019, Algorithm 6 "AutoStep Style Normalized TIDBD(lambda)"

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
            td_error: TD error delta = R + gamma*V(s') - V(s)
            observation: Current observation phi(s)
            next_observation: Next observation phi(s')
            gamma: Discount factor gamma (0 at terminal)

        Returns:
            TDOptimizerUpdate with weight deltas and updated state
        """
        delta = jnp.squeeze(td_error)
        theta = state.meta_step_size
        lam = state.trace_decay
        tau = state.normalizer_decay
        gamma_scalar = jnp.squeeze(gamma)

        feature_diff = gamma_scalar * next_observation - observation
        alphas = jnp.exp(state.log_step_sizes)

        # Update normalizers
        abs_weight_update = jnp.abs(delta * feature_diff * state.h_traces)
        normalizer_decay_term = (
            (1.0 / tau)
            * alphas
            * feature_diff
            * state.eligibility_traces
            * (jnp.abs(delta * observation * state.h_traces) - state.normalizers)
        )
        new_normalizers = jnp.maximum(abs_weight_update, state.normalizers - normalizer_decay_term)
        new_normalizers = jnp.maximum(new_normalizers, 1e-8)

        # Normalized meta-update
        normalized_gradient = delta * feature_diff * state.h_traces / new_normalizers
        new_log_step_sizes = state.log_step_sizes - theta * normalized_gradient

        # Effective step-size normalization
        effective_step_size = -jnp.sum(
            jnp.exp(new_log_step_sizes) * feature_diff * state.eligibility_traces
        )
        normalization_factor = jnp.maximum(effective_step_size, 1.0)
        new_log_step_sizes = new_log_step_sizes - jnp.log(normalization_factor)

        new_log_step_sizes = jnp.clip(new_log_step_sizes, -10.0, 2.0)
        new_alphas = jnp.exp(new_log_step_sizes)

        new_eligibility_traces = gamma_scalar * lam * state.eligibility_traces + observation
        weight_delta = new_alphas * delta * new_eligibility_traces

        # Update h traces
        h_decay = jnp.maximum(0.0, 1.0 + new_alphas * feature_diff * new_eligibility_traces)
        new_h_traces = state.h_traces * h_decay + new_alphas * delta * new_eligibility_traces

        # Bias updates
        bias_alpha = jnp.exp(state.bias_log_step_size)
        bias_feature_diff = gamma_scalar - 1.0

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

        normalized_bias_gradient = (
            delta * bias_feature_diff * state.bias_h_trace / new_bias_normalizer
        )
        new_bias_log_step_size = state.bias_log_step_size - theta * normalized_bias_gradient

        bias_effective_step_size = (
            -jnp.exp(new_bias_log_step_size) * bias_feature_diff * state.bias_eligibility_trace
        )
        bias_norm_factor = jnp.maximum(bias_effective_step_size, 1.0)
        new_bias_log_step_size = new_bias_log_step_size - jnp.log(bias_norm_factor)

        new_bias_log_step_size = jnp.clip(new_bias_log_step_size, -10.0, 2.0)
        new_bias_alpha = jnp.exp(new_bias_log_step_size)

        new_bias_eligibility_trace = gamma_scalar * lam * state.bias_eligibility_trace + 1.0
        bias_delta = new_bias_alpha * delta * new_bias_eligibility_trace

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
