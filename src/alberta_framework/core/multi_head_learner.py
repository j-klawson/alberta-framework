"""Multi-head MLP learner for multi-task continual learning.

Implements a shared-trunk, multi-head MLP architecture where hidden layers
are shared across prediction heads. Each head can be independently active
or inactive at each time step (NaN targets = inactive).

Architecture: ``Input -> [Dense(H) -> LayerNorm -> LeakyReLU] x N -> {Head_i: Dense(1)} x n_heads``

When ``use_layer_norm=False``:
``Input -> [Dense(H) -> LeakyReLU] x N -> {Head_i: Dense(1)} x n_heads``

The update uses VJP with accumulated cotangents to perform a single backward
pass through the trunk regardless of the number of heads.

Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning Finally Works"
"""

import math

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.normalizers import (
    AnyNormalizerState,
    EMANormalizerState,
    Normalizer,
    WelfordNormalizerState,
)
from alberta_framework.core.optimizers import (
    LMS,
    Bounder,
    Optimizer,
)
from alberta_framework.core.types import (
    AutostepParamState,
    AutostepState,
    IDBDState,
    LMSState,
    MLPParams,
    ObGDState,
)


def _extract_mean_step_size(opt_state: LMSState | AutostepParamState) -> Array:
    """Extract mean step-size from an optimizer state.

    Works at JAX trace time since it dispatches on Python-level attributes.
    """
    if hasattr(opt_state, "step_sizes"):
        # AutostepParamState
        return jnp.mean(opt_state.step_sizes)
    if hasattr(opt_state, "step_size"):
        # LMSState
        return opt_state.step_size
    return jnp.array(0.0, dtype=jnp.float32)


# =============================================================================
# Types
# =============================================================================


@chex.dataclass(frozen=True)
class MultiHeadMLPState:
    """State for a multi-head MLP learner.

    The trunk (shared hidden layers) and heads (per-task output layers)
    maintain separate parameters, optimizer states, and eligibility traces.

    Trunk optimizer states and traces use an interleaved layout
    ``(w0, b0, w1, b1, ...)`` matching the ``MLPLearner`` convention.
    Head optimizer states and traces use a nested layout
    ``((w_opt, b_opt), ...)`` indexed by head.

    Attributes:
        trunk_params: Shared hidden layer parameters
        head_params: Per-head output layer parameters.
            ``weights[i]`` / ``biases[i]`` = head *i*.
        trunk_optimizer_states: Interleaved ``(w0, b0, w1, b1, ...)``
            optimizer states for trunk layers
        head_optimizer_states: Per-head ``((w_opt, b_opt), ...)``
        trunk_traces: Interleaved ``(w0, b0, w1, b1, ...)``
            eligibility traces for trunk layers
        head_traces: Per-head ``((w_trace, b_trace), ...)``
        normalizer_state: Optional online feature normalizer state
        step_count: Scalar step counter
    """

    trunk_params: MLPParams
    head_params: MLPParams
    trunk_optimizer_states: tuple[LMSState | AutostepState | AutostepParamState, ...]
    head_optimizer_states: tuple  # tuple of (w_opt, b_opt) tuples
    trunk_traces: tuple[Array, ...]
    head_traces: tuple  # tuple of (w_trace, b_trace) tuples
    normalizer_state: AnyNormalizerState | None = None
    step_count: Array = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class MultiHeadMLPUpdateResult:
    """Result of a multi-head MLP learner update step.

    Attributes:
        state: Updated multi-head MLP learner state
        predictions: Predictions from all heads, shape ``(n_heads,)``
        errors: Prediction errors, shape ``(n_heads,)``. NaN for inactive heads.
        per_head_metrics: Per-head metrics, shape ``(n_heads, 3)``.
            Columns: ``[squared_error, raw_error, mean_step_size]``.
            NaN for inactive heads.
        trunk_bounding_metric: Scalar trunk bounding metric
    """

    state: MultiHeadMLPState
    predictions: Float[Array, " n_heads"]
    errors: Float[Array, " n_heads"]
    per_head_metrics: Float[Array, "n_heads 3"]
    trunk_bounding_metric: Float[Array, ""]


@chex.dataclass(frozen=True)
class MultiHeadLearningResult:
    """Result from multi-head learning loop.

    Attributes:
        state: Final multi-head MLP learner state
        per_head_metrics: Per-head metrics over time,
            shape ``(num_steps, n_heads, 3)``
    """

    state: MultiHeadMLPState
    per_head_metrics: Float[Array, "num_steps n_heads 3"]


@chex.dataclass(frozen=True)
class BatchedMultiHeadResult:
    """Result from batched multi-head learning loop.

    Attributes:
        states: Batched multi-head MLP learner states
        per_head_metrics: Per-head metrics,
            shape ``(n_seeds, num_steps, n_heads, 3)``
    """

    states: MultiHeadMLPState
    per_head_metrics: Float[Array, "n_seeds num_steps n_heads 3"]


# =============================================================================
# Type alias (mirrors learners.py)
# =============================================================================

AnyOptimizer = (
    Optimizer[LMSState]
    | Optimizer[IDBDState]
    | Optimizer[AutostepState]
    | Optimizer[ObGDState]
    | Optimizer[AutostepParamState]
)


# =============================================================================
# MultiHeadMLPLearner
# =============================================================================


class MultiHeadMLPLearner:
    """Multi-head MLP with shared trunk and independent prediction heads.

    Architecture:
    ``Input -> [Dense(H) -> LayerNorm -> LeakyReLU] x N -> {Head_i: Dense(1)} x n_heads``

    All hidden layers are shared (the *trunk*). Each head is an independent
    linear projection from the last hidden representation to a scalar.

    The ``update`` method uses VJP with accumulated cotangents so that
    only one backward pass through the trunk is needed regardless of the
    number of active heads.

    Attributes:
        n_heads: Number of prediction heads
        hidden_sizes: Tuple of hidden layer sizes
        optimizer: Optimizer for per-weight step-size adaptation
        bounder: Optional update bounder (e.g. ObGDBounding)
        normalizer: Optional feature normalizer
        use_layer_norm: Whether to apply parameterless layer normalization
        gamma: Discount factor for trace decay
        lamda: Eligibility trace decay parameter
        sparsity: Fraction of weights zeroed out per output neuron
        leaky_relu_slope: Negative slope for LeakyReLU activation
    """

    def __init__(
        self,
        n_heads: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        gamma: float = 0.0,
        lamda: float = 0.0,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
    ):
        """Initialize the multi-head MLP learner.

        Args:
            n_heads: Number of prediction heads
            hidden_sizes: Tuple of hidden layer sizes (default: two layers of 128)
            optimizer: Optimizer for weight updates. Defaults to LMS(step_size).
                Must support ``init_for_shape`` and ``update_from_gradient``.
            step_size: Base learning rate (used only when optimizer is None)
            bounder: Optional update bounder (e.g. ObGDBounding)
            gamma: Discount factor for trace decay (default: 0.0 for supervised)
            lamda: Eligibility trace decay parameter (default: 0.0)
            normalizer: Optional feature normalizer
            sparsity: Fraction of weights zeroed out per neuron (default: 0.9)
            leaky_relu_slope: Negative slope for LeakyReLU (default: 0.01)
            use_layer_norm: Whether to apply parameterless layer normalization
                (default: True)
        """
        self._n_heads = n_heads
        self._hidden_sizes = hidden_sizes
        self._optimizer: AnyOptimizer = optimizer or LMS(step_size=step_size)
        self._bounder = bounder
        self._gamma = gamma
        self._lamda = lamda
        self._normalizer = normalizer
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm

    @property
    def n_heads(self) -> int:
        """Number of prediction heads."""
        return self._n_heads

    @property
    def normalizer(
        self,
    ) -> Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None:
        """The feature normalizer, or None if normalization is disabled."""
        return self._normalizer

    def init(self, feature_dim: int, key: Array) -> MultiHeadMLPState:
        """Initialize multi-head MLP learner state with sparse weights.

        Args:
            feature_dim: Dimension of the input feature vector
            key: JAX random key for weight initialization

        Returns:
            Initial state with sparse trunk weights, zero biases, and
            per-head output layers
        """
        # Trunk: [feature_dim, *hidden_sizes] — all hidden layers
        trunk_layer_sizes = [feature_dim, *self._hidden_sizes]

        trunk_weights: list[Array] = []
        trunk_biases: list[Array] = []
        trunk_traces: list[Array] = []
        trunk_opt_states: list[LMSState | AutostepParamState] = []

        for i in range(len(trunk_layer_sizes) - 1):
            fan_out = trunk_layer_sizes[i + 1]
            fan_in = trunk_layer_sizes[i]
            key, subkey = jax.random.split(key)
            w = sparse_init(subkey, (fan_out, fan_in), sparsity=self._sparsity)
            b = jnp.zeros(fan_out, dtype=jnp.float32)
            trunk_weights.append(w)
            trunk_biases.append(b)
            # Interleaved traces and optimizer states: w0, b0, w1, b1, ...
            trunk_traces.append(jnp.zeros_like(w))
            trunk_traces.append(jnp.zeros_like(b))
            trunk_opt_states.append(self._optimizer.init_for_shape(w.shape))
            trunk_opt_states.append(self._optimizer.init_for_shape(b.shape))

        trunk_params = MLPParams(
            weights=tuple(trunk_weights),
            biases=tuple(trunk_biases),
        )

        # Heads: n_heads output layers, each (1, H_last)
        h_last = self._hidden_sizes[-1]
        head_weights: list[Array] = []
        head_biases: list[Array] = []
        head_traces_list: list[tuple[Array, Array]] = []
        head_opt_states_list: list[tuple] = []

        for _ in range(self._n_heads):
            key, subkey = jax.random.split(key)
            w = sparse_init(subkey, (1, h_last), sparsity=self._sparsity)
            b = jnp.zeros(1, dtype=jnp.float32)
            head_weights.append(w)
            head_biases.append(b)
            head_traces_list.append((jnp.zeros_like(w), jnp.zeros_like(b)))
            head_opt_states_list.append((
                self._optimizer.init_for_shape(w.shape),
                self._optimizer.init_for_shape(b.shape),
            ))

        head_params = MLPParams(
            weights=tuple(head_weights),
            biases=tuple(head_biases),
        )

        normalizer_state = None
        if self._normalizer is not None:
            normalizer_state = self._normalizer.init(feature_dim)

        return MultiHeadMLPState(
            trunk_params=trunk_params,
            head_params=head_params,
            trunk_optimizer_states=tuple(trunk_opt_states),
            head_optimizer_states=tuple(head_opt_states_list),
            trunk_traces=tuple(trunk_traces),
            head_traces=tuple(head_traces_list),
            normalizer_state=normalizer_state,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @staticmethod
    def _trunk_forward(
        weights: tuple[Array, ...],
        biases: tuple[Array, ...],
        observation: Array,
        leaky_relu_slope: float,
        use_layer_norm: bool = True,
    ) -> Array:
        """Pure forward pass through trunk (hidden layers only).

        Args:
            weights: Tuple of weight matrices for hidden layers
            biases: Tuple of bias vectors for hidden layers
            observation: Input feature vector
            leaky_relu_slope: Negative slope for LeakyReLU
            use_layer_norm: Whether to apply parameterless layer normalization

        Returns:
            Hidden representation of shape ``(H_last,)``
        """
        x = observation
        for i in range(len(weights)):
            x = weights[i] @ x + biases[i]
            if use_layer_norm:
                mean = jnp.mean(x)
                var = jnp.var(x)
                x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = jnp.where(x >= 0, x, leaky_relu_slope * x)
        return x

    @staticmethod
    def _head_forward(head_w: Array, head_b: Array, hidden: Array) -> Array:
        """Forward a single head: ``squeeze(head_w @ hidden + head_b)``.

        Args:
            head_w: Head weight matrix, shape ``(1, H_last)``
            head_b: Head bias vector, shape ``(1,)``
            hidden: Trunk hidden representation, shape ``(H_last,)``

        Returns:
            Scalar prediction
        """
        return jnp.squeeze(head_w @ hidden + head_b)

    def predict(self, state: MultiHeadMLPState, observation: Array) -> Array:
        """Compute predictions from all heads.

        Args:
            state: Current multi-head MLP learner state
            observation: Input feature vector

        Returns:
            Array of shape ``(n_heads,)`` with one prediction per head
        """
        obs = observation
        if self._normalizer is not None and state.normalizer_state is not None:
            obs = self._normalizer.normalize_only(state.normalizer_state, observation)

        hidden = self._trunk_forward(
            state.trunk_params.weights,
            state.trunk_params.biases,
            obs,
            self._leaky_relu_slope,
            self._use_layer_norm,
        )

        predictions = []
        for i in range(self._n_heads):
            pred = self._head_forward(
                state.head_params.weights[i],
                state.head_params.biases[i],
                hidden,
            )
            predictions.append(pred)

        return jnp.array(predictions)

    def update(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        targets: Array,
    ) -> MultiHeadMLPUpdateResult:
        """Update multi-head MLP given observation and per-head targets.

        Uses VJP with accumulated cotangents for a single backward pass
        through the trunk. Error from each active head is folded into the
        trunk gradient before trace accumulation.

        Args:
            state: Current state
            observation: Input feature vector
            targets: Per-head targets, shape ``(n_heads,)``.
                NaN = inactive head.

        Returns:
            MultiHeadMLPUpdateResult with updated state, predictions,
            errors, and per-head metrics
        """
        n_heads = self._n_heads
        gamma_lamda = jnp.array(self._gamma * self._lamda, dtype=jnp.float32)

        # 1. Handle NaN targets
        active_mask = ~jnp.isnan(targets)  # (n_heads,)
        safe_targets = jnp.where(active_mask, targets, 0.0)

        # 2. Normalize observation if needed
        obs = observation
        new_normalizer_state = state.normalizer_state
        if self._normalizer is not None and state.normalizer_state is not None:
            obs, new_normalizer_state = self._normalizer.normalize(
                state.normalizer_state, observation
            )

        # 3. Forward trunk via VJP
        slope = self._leaky_relu_slope
        ln = self._use_layer_norm

        def trunk_fn(
            weights: tuple[Array, ...], biases: tuple[Array, ...]
        ) -> Array:
            return self._trunk_forward(weights, biases, obs, slope, ln)

        hidden, trunk_vjp_fn = jax.vjp(
            trunk_fn,
            state.trunk_params.weights,
            state.trunk_params.biases,
        )

        # 4. Per-head forward + compute errors + accumulate cotangent
        h_last = hidden.shape[0]
        cotangent = jnp.zeros(h_last, dtype=jnp.float32)
        predictions_list: list[Array] = []
        errors_list: list[Array] = []

        for i in range(n_heads):
            pred_i = self._head_forward(
                state.head_params.weights[i],
                state.head_params.biases[i],
                hidden,
            )
            error_i = safe_targets[i] - pred_i
            masked_error_i = jnp.where(active_mask[i], error_i, 0.0)

            predictions_list.append(pred_i)
            errors_list.append(jnp.where(active_mask[i], error_i, jnp.nan))

            # Accumulate cotangent: error_i * d(pred_i)/d(hidden)
            # d(pred_i)/d(hidden) = head_w_i squeezed to (H_last,)
            cotangent = cotangent + masked_error_i * jnp.squeeze(
                state.head_params.weights[i]
            )

        predictions_arr = jnp.array(predictions_list)
        errors_arr = jnp.array(errors_list)

        # 5. One backward pass through trunk
        trunk_weight_grads, trunk_bias_grads = trunk_vjp_fn(cotangent)
        # These grads are already error-weighted

        # 6. Update trunk traces and optimizer
        n_trunk_layers = len(state.trunk_params.weights)
        new_trunk_traces: list[Array] = []
        trunk_steps: list[Array] = []
        new_trunk_opt_states: list[LMSState | AutostepParamState] = []

        for i in range(n_trunk_layers):
            # Weight trace (index 2*i)
            new_wt = gamma_lamda * state.trunk_traces[2 * i] + trunk_weight_grads[i]
            new_trunk_traces.append(new_wt)
            w_step, new_w_opt = self._optimizer.update_from_gradient(
                state.trunk_optimizer_states[2 * i], new_wt, error=None
            )
            trunk_steps.append(w_step)
            new_trunk_opt_states.append(new_w_opt)

            # Bias trace (index 2*i + 1)
            new_bt = gamma_lamda * state.trunk_traces[2 * i + 1] + trunk_bias_grads[i]
            new_trunk_traces.append(new_bt)
            b_step, new_b_opt = self._optimizer.update_from_gradient(
                state.trunk_optimizer_states[2 * i + 1], new_bt, error=None
            )
            trunk_steps.append(b_step)
            new_trunk_opt_states.append(new_b_opt)

        # Trunk bounding (pseudo_error=1.0 since error is in gradient)
        trunk_bounding_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None:
            trunk_params_flat: list[Array] = []
            for i in range(n_trunk_layers):
                trunk_params_flat.append(state.trunk_params.weights[i])
                trunk_params_flat.append(state.trunk_params.biases[i])
            bounded_trunk_steps, trunk_bounding_metric = self._bounder.bound(
                tuple(trunk_steps), jnp.array(1.0), tuple(trunk_params_flat)
            )
            trunk_steps = list(bounded_trunk_steps)

        # Apply trunk updates (no error multiply -- error already in gradient)
        new_trunk_weights: list[Array] = []
        new_trunk_biases: list[Array] = []
        for i in range(n_trunk_layers):
            new_trunk_weights.append(
                state.trunk_params.weights[i] + trunk_steps[2 * i]
            )
            new_trunk_biases.append(
                state.trunk_params.biases[i] + trunk_steps[2 * i + 1]
            )

        new_trunk_params = MLPParams(
            weights=tuple(new_trunk_weights),
            biases=tuple(new_trunk_biases),
        )

        # 7. Per-head updates
        new_head_weights: list[Array] = []
        new_head_biases: list[Array] = []
        new_head_traces_list: list[tuple[Array, Array]] = []
        new_head_opt_states_list: list[tuple] = []
        per_head_metrics_list: list[Array] = []

        for i in range(n_heads):
            head_w = state.head_params.weights[i]
            head_b = state.head_params.biases[i]
            old_w_trace, old_b_trace = state.head_traces[i]
            old_w_opt, old_b_opt = state.head_optimizer_states[i]

            # Head prediction gradient: d(pred_i)/d(head_w) = hidden
            w_grad = hidden.reshape(1, -1)  # (1, H_last)
            b_grad = jnp.ones(1, dtype=jnp.float32)

            # Update traces
            new_w_trace = gamma_lamda * old_w_trace + w_grad
            new_b_trace = gamma_lamda * old_b_trace + b_grad

            # Error for this head (masked to 0 for inactive)
            error_i = jnp.where(
                active_mask[i], safe_targets[i] - predictions_list[i], 0.0
            )

            # Optimizer step (with error for meta-learning)
            w_step, new_w_opt = self._optimizer.update_from_gradient(
                old_w_opt, new_w_trace, error=error_i
            )
            b_step, new_b_opt = self._optimizer.update_from_gradient(
                old_b_opt, new_b_trace, error=error_i
            )

            # Head bounding
            if self._bounder is not None:
                bounded_head_steps, _ = self._bounder.bound(
                    (w_step, b_step), error_i, (head_w, head_b)
                )
                w_step, b_step = bounded_head_steps

            # Apply: param += error_i * step
            new_w = head_w + error_i * w_step
            new_b = head_b + error_i * b_step

            # Mask: for inactive heads, keep old state
            new_w = jnp.where(active_mask[i], new_w, head_w)
            new_b = jnp.where(active_mask[i], new_b, head_b)
            new_w_trace = jnp.where(active_mask[i], new_w_trace, old_w_trace)
            new_b_trace = jnp.where(active_mask[i], new_b_trace, old_b_trace)

            # Mask optimizer states back to old for inactive heads
            new_w_opt = jax.tree.map(
                lambda new, old: jnp.where(active_mask[i], new, old),
                new_w_opt,
                old_w_opt,
            )
            new_b_opt = jax.tree.map(
                lambda new, old: jnp.where(active_mask[i], new, old),
                new_b_opt,
                old_b_opt,
            )

            new_head_weights.append(new_w)
            new_head_biases.append(new_b)
            new_head_traces_list.append((new_w_trace, new_b_trace))
            new_head_opt_states_list.append((new_w_opt, new_b_opt))

            # Per-head metrics
            se_i = jnp.where(active_mask[i], error_i**2, jnp.nan)
            raw_error_i = jnp.where(active_mask[i], error_i, jnp.nan)
            mean_ss_i = _extract_mean_step_size(new_w_opt)
            mean_ss_i = jnp.where(active_mask[i], mean_ss_i, jnp.nan)
            per_head_metrics_list.append(
                jnp.array([se_i, raw_error_i, mean_ss_i])
            )

        new_head_params = MLPParams(
            weights=tuple(new_head_weights),
            biases=tuple(new_head_biases),
        )

        new_state = MultiHeadMLPState(
            trunk_params=new_trunk_params,
            head_params=new_head_params,
            trunk_optimizer_states=tuple(new_trunk_opt_states),
            head_optimizer_states=tuple(new_head_opt_states_list),
            trunk_traces=tuple(new_trunk_traces),
            head_traces=tuple(new_head_traces_list),
            normalizer_state=new_normalizer_state,
            step_count=state.step_count + 1,
        )

        per_head_metrics = jnp.stack(per_head_metrics_list)  # (n_heads, 3)

        return MultiHeadMLPUpdateResult(
            state=new_state,
            predictions=predictions_arr,
            errors=errors_arr,
            per_head_metrics=per_head_metrics,
            trunk_bounding_metric=trunk_bounding_metric,
        )


def multi_head_metrics_to_dicts(
    result: MultiHeadMLPUpdateResult,
) -> list[dict[str, float] | None]:
    """Convert per-head metrics array to list of dicts for online use.

    Active heads get a dict with keys ``'squared_error'``, ``'error'``,
    ``'mean_step_size'``. Inactive heads get ``None``.

    Args:
        result: Update result from ``MultiHeadMLPLearner.update``

    Returns:
        List of ``n_heads`` entries, one per head
    """
    output: list[dict[str, float] | None] = []
    for i in range(result.per_head_metrics.shape[0]):
        se = float(result.per_head_metrics[i, 0])
        if math.isnan(se):
            output.append(None)
        else:
            output.append(
                {
                    "squared_error": se,
                    "error": float(result.per_head_metrics[i, 1]),
                    "mean_step_size": float(result.per_head_metrics[i, 2]),
                }
            )
    return output


# =============================================================================
# Learning Loops
# =============================================================================


def run_multi_head_learning_loop(
    learner: MultiHeadMLPLearner,
    state: MultiHeadMLPState,
    observations: Array,
    targets: Array,
) -> MultiHeadLearningResult:
    """Run multi-head learning loop using ``jax.lax.scan``.

    Scans over pre-provided observation and target arrays. This is
    designed for settings where data comes from an external source
    (e.g. security event logs) rather than from a ``ScanStream``.

    Args:
        learner: Multi-head MLP learner
        state: Initial learner state
        observations: Input observations, shape ``(num_steps, feature_dim)``
        targets: Per-head targets, shape ``(num_steps, n_heads)``.
            NaN = inactive head for that step.

    Returns:
        ``MultiHeadLearningResult`` with final state and per-head metrics
        of shape ``(num_steps, n_heads, 3)``
    """

    def step_fn(
        carry: MultiHeadMLPState, inputs: tuple[Array, Array]
    ) -> tuple[MultiHeadMLPState, Array]:
        l_state = carry
        obs, tgt = inputs
        result = learner.update(l_state, obs, tgt)
        return result.state, result.per_head_metrics

    final_state, per_head_metrics = jax.lax.scan(
        step_fn, state, (observations, targets)
    )

    return MultiHeadLearningResult(
        state=final_state,
        per_head_metrics=per_head_metrics,
    )


def run_multi_head_learning_loop_batched(
    learner: MultiHeadMLPLearner,
    observations: Array,
    targets: Array,
    keys: Array,
) -> BatchedMultiHeadResult:
    """Run multi-head learning loop across seeds using ``jax.vmap``.

    Each seed produces an independently initialized state (different
    sparse weight masks). All seeds share the same observations and
    targets.

    Args:
        learner: Multi-head MLP learner
        observations: Shared observations, shape ``(num_steps, feature_dim)``
        targets: Shared targets, shape ``(num_steps, n_heads)``.
            NaN = inactive head.
        keys: JAX random keys, shape ``(n_seeds,)`` or ``(n_seeds, 2)``

    Returns:
        ``BatchedMultiHeadResult`` with batched states and per-head metrics
        of shape ``(n_seeds, num_steps, n_heads, 3)``
    """
    feature_dim = observations.shape[1]

    def single_run(key: Array) -> tuple[MultiHeadMLPState, Array]:
        init_state = learner.init(feature_dim, key)
        result = run_multi_head_learning_loop(
            learner, init_state, observations, targets
        )
        return result.state, result.per_head_metrics

    batched_states, batched_metrics = jax.vmap(single_run)(keys)

    return BatchedMultiHeadResult(
        states=batched_states,
        per_head_metrics=batched_metrics,
    )
