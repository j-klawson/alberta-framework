"""Learning units for continual learning.

Implements learners that combine function approximation with optimizers
for temporally-uniform learning. Uses JAX's scan for efficient JIT-compiled
training loops.
"""

from typing import Protocol, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.normalizers import (
    EMANormalizer,
    EMANormalizerState,
    Normalizer,
    WelfordNormalizerState,
)
from alberta_framework.core.optimizers import LMS, TDIDBD, Optimizer, TDOptimizer
from alberta_framework.core.types import (
    AutostepState,
    AutoTDIDBDState,
    BatchedLearningResult,
    BatchedMLPNormalizedResult,
    BatchedMLPResult,
    BatchedNormalizedResult,
    IDBDState,
    LearnerState,
    LMSState,
    MLPLearnerState,
    MLPObGDState,
    MLPParams,
    NormalizerHistory,
    NormalizerTrackingConfig,
    ObGDState,
    Observation,
    Prediction,
    StepSizeHistory,
    StepSizeTrackingConfig,
    Target,
    TDIDBDState,
    TDLearnerState,
    TDTimeStep,
)
from alberta_framework.streams.base import ScanStream

# Type variable for TD stream state
StateT = TypeVar("StateT")

# Type alias for any optimizer type
AnyOptimizer = (
    Optimizer[LMSState] | Optimizer[IDBDState] | Optimizer[AutostepState] | Optimizer[ObGDState]
)

# Type alias for any TD optimizer type
AnyTDOptimizer = TDOptimizer[TDIDBDState] | TDOptimizer[AutoTDIDBDState]


@chex.dataclass(frozen=True)
class UpdateResult:
    """Result of a learner update step.

    Attributes:
        state: Updated learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics [squared_error, error, ...]
    """

    state: LearnerState
    prediction: Prediction
    error: Float[Array, ""]
    metrics: Float[Array, " 3"]


@chex.dataclass(frozen=True)
class NormalizedLearnerState:
    """State for a learner with online feature normalization.

    Attributes:
        learner_state: Underlying learner state (weights, bias, optimizer)
        normalizer_state: Online normalizer state (mean, var estimates)
    """

    learner_state: LearnerState
    normalizer_state: EMANormalizerState | WelfordNormalizerState


@chex.dataclass(frozen=True)
class NormalizedUpdateResult:
    """Result of a normalized learner update step.

    Attributes:
        state: Updated normalized learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics [squared_error, error, mean_step_size, normalizer_mean_var]
    """

    state: NormalizedLearnerState
    prediction: Prediction
    error: Float[Array, ""]
    metrics: Float[Array, " 4"]


class LinearLearner:
    """Linear function approximator with pluggable optimizer.

    Computes predictions as: `y = w @ x + b`

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
            Scalar prediction `y = w @ x + b`
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
            state.optimizer_state,
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


def run_learning_loop[StreamStateT](
    learner: LinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: LearnerState | None = None,
    step_size_tracking: StepSizeTrackingConfig | None = None,
) -> tuple[LearnerState, Array] | tuple[LearnerState, Array, StepSizeHistory]:
    """Run the learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.
    It returns metrics as a fixed-size array rather than a list of dicts.

    Args:
        learner: The learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)
        step_size_tracking: Optional config for recording per-weight step-sizes.
            When provided, returns a 3-tuple including StepSizeHistory.

    Returns:
        If step_size_tracking is None:
            Tuple of (final_state, metrics_array) where metrics_array has shape
            (num_steps, 3) with columns [squared_error, error, mean_step_size]
        If step_size_tracking is provided:
            Tuple of (final_state, metrics_array, step_size_history)

    Raises:
        ValueError: If step_size_tracking.interval is less than 1 or greater than num_steps
    """
    # Validate tracking config
    if step_size_tracking is not None:
        if step_size_tracking.interval < 1:
            raise ValueError(
                f"step_size_tracking.interval must be >= 1, got {step_size_tracking.interval}"
            )
        if step_size_tracking.interval > num_steps:
            raise ValueError(
                f"step_size_tracking.interval ({step_size_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    feature_dim = stream.feature_dim

    if step_size_tracking is None:
        # Original behavior without tracking
        def step_fn(
            carry: tuple[LearnerState, StreamStateT], idx: Array
        ) -> tuple[tuple[LearnerState, StreamStateT], Array]:
            l_state, s_state = carry
            timestep, new_s_state = stream.step(s_state, idx)
            result = learner.update(l_state, timestep.observation, timestep.target)
            return (result.state, new_s_state), result.metrics

        (final_learner, _), metrics = jax.lax.scan(
            step_fn, (learner_state, stream_state), jnp.arange(num_steps)
        )

        return final_learner, metrics

    else:
        # Step-size tracking enabled
        interval = step_size_tracking.interval
        include_bias = step_size_tracking.include_bias
        num_recordings = num_steps // interval

        # Pre-allocate history arrays
        step_size_history = jnp.zeros((num_recordings, feature_dim), dtype=jnp.float32)
        bias_history = jnp.zeros(num_recordings, dtype=jnp.float32) if include_bias else None
        recording_indices = jnp.zeros(num_recordings, dtype=jnp.int32)

        # Check if we need to track Autostep normalizers
        # We detect this at trace time by checking the initial optimizer state
        track_normalizers = hasattr(learner_state.optimizer_state, "normalizers")
        normalizer_history = (
            jnp.zeros((num_recordings, feature_dim), dtype=jnp.float32)
            if track_normalizers
            else None
        )

        def step_fn_with_tracking(
            carry: tuple[LearnerState, StreamStateT, Array, Array | None, Array, Array | None],
            idx: Array,
        ) -> tuple[
            tuple[LearnerState, StreamStateT, Array, Array | None, Array, Array | None],
            Array,
        ]:
            l_state, s_state, ss_history, b_history, rec_indices, norm_history = carry

            # Perform learning step
            timestep, new_s_state = stream.step(s_state, idx)
            result = learner.update(l_state, timestep.observation, timestep.target)

            # Check if we should record at this step (idx % interval == 0)
            should_record = (idx % interval) == 0
            recording_idx = idx // interval

            # Extract current step-sizes
            # Use hasattr checks at trace time (this works because the type is fixed)
            opt_state = result.state.optimizer_state
            if hasattr(opt_state, "log_step_sizes"):
                # IDBD stores log step-sizes
                weight_ss = jnp.exp(opt_state.log_step_sizes)
                bias_ss = opt_state.bias_step_size
            elif hasattr(opt_state, "step_sizes"):
                # Autostep stores step-sizes directly
                weight_ss = opt_state.step_sizes
                bias_ss = opt_state.bias_step_size
            else:
                # LMS has a single fixed step-size
                weight_ss = jnp.full(feature_dim, opt_state.step_size)
                bias_ss = opt_state.step_size

            # Conditionally update history arrays
            new_ss_history = jax.lax.cond(
                should_record,
                lambda _: ss_history.at[recording_idx].set(weight_ss),
                lambda _: ss_history,
                None,
            )

            new_b_history = b_history
            if b_history is not None:
                new_b_history = jax.lax.cond(
                    should_record,
                    lambda _: b_history.at[recording_idx].set(bias_ss),
                    lambda _: b_history,
                    None,
                )

            new_rec_indices = jax.lax.cond(
                should_record,
                lambda _: rec_indices.at[recording_idx].set(idx),
                lambda _: rec_indices,
                None,
            )

            # Track Autostep normalizers (v_i) if applicable
            new_norm_history = norm_history
            if norm_history is not None and hasattr(opt_state, "normalizers"):
                new_norm_history = jax.lax.cond(
                    should_record,
                    lambda _: norm_history.at[recording_idx].set(opt_state.normalizers),
                    lambda _: norm_history,
                    None,
                )

            return (
                result.state,
                new_s_state,
                new_ss_history,
                new_b_history,
                new_rec_indices,
                new_norm_history,
            ), result.metrics

        initial_carry = (
            learner_state,
            stream_state,
            step_size_history,
            bias_history,
            recording_indices,
            normalizer_history,
        )

        (
            (
                final_learner,
                _,
                final_ss_history,
                final_b_history,
                final_rec_indices,
                final_norm_history,
            ),
            metrics,
        ) = jax.lax.scan(step_fn_with_tracking, initial_carry, jnp.arange(num_steps))

        history = StepSizeHistory(
            step_sizes=final_ss_history,
            bias_step_sizes=final_b_history,
            recording_indices=final_rec_indices,
            normalizers=final_norm_history,
        )

        return final_learner, metrics, history


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
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
    ):
        """Initialize the normalized linear learner.

        Args:
            optimizer: Optimizer for weight updates. Defaults to LMS(0.01)
            normalizer: Feature normalizer. Defaults to EMANormalizer()
        """
        self._learner = LinearLearner(optimizer=optimizer or LMS(step_size=0.01))
        self._normalizer = normalizer or EMANormalizer()

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
        normalized_obs = self._normalizer.normalize_only(state.normalizer_state, observation)
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


def run_normalized_learning_loop[StreamStateT](
    learner: NormalizedLinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: NormalizedLearnerState | None = None,
    step_size_tracking: StepSizeTrackingConfig | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> (
    tuple[NormalizedLearnerState, Array]
    | tuple[NormalizedLearnerState, Array, StepSizeHistory]
    | tuple[NormalizedLearnerState, Array, NormalizerHistory]
    | tuple[NormalizedLearnerState, Array, StepSizeHistory, NormalizerHistory]
):
    """Run the learning loop with normalization using jax.lax.scan.

    Args:
        learner: The normalized learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)
        step_size_tracking: Optional config for recording per-weight step-sizes.
            When provided, returns StepSizeHistory including Autostep normalizers if applicable.
        normalizer_tracking: Optional config for recording per-feature normalizer state.
            When provided, returns NormalizerHistory with means and variances over time.

    Returns:
        If no tracking:
            Tuple of (final_state, metrics_array) where metrics_array has shape
            (num_steps, 4) with columns [squared_error, error, mean_step_size, normalizer_mean_var]
        If step_size_tracking only:
            Tuple of (final_state, metrics_array, step_size_history)
        If normalizer_tracking only:
            Tuple of (final_state, metrics_array, normalizer_history)
        If both:
            Tuple of (final_state, metrics_array, step_size_history, normalizer_history)

    Raises:
        ValueError: If tracking interval is invalid
    """
    # Validate tracking configs
    if step_size_tracking is not None:
        if step_size_tracking.interval < 1:
            raise ValueError(
                f"step_size_tracking.interval must be >= 1, got {step_size_tracking.interval}"
            )
        if step_size_tracking.interval > num_steps:
            raise ValueError(
                f"step_size_tracking.interval ({step_size_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    if normalizer_tracking is not None:
        if normalizer_tracking.interval < 1:
            raise ValueError(
                f"normalizer_tracking.interval must be >= 1, got {normalizer_tracking.interval}"
            )
        if normalizer_tracking.interval > num_steps:
            raise ValueError(
                f"normalizer_tracking.interval ({normalizer_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    feature_dim = stream.feature_dim

    # No tracking - simple case
    if step_size_tracking is None and normalizer_tracking is None:

        def step_fn(
            carry: tuple[NormalizedLearnerState, StreamStateT], idx: Array
        ) -> tuple[tuple[NormalizedLearnerState, StreamStateT], Array]:
            l_state, s_state = carry
            timestep, new_s_state = stream.step(s_state, idx)
            result = learner.update(l_state, timestep.observation, timestep.target)
            return (result.state, new_s_state), result.metrics

        (final_learner, _), metrics = jax.lax.scan(
            step_fn, (learner_state, stream_state), jnp.arange(num_steps)
        )

        return final_learner, metrics

    # Tracking enabled - need to set up history arrays
    ss_interval = step_size_tracking.interval if step_size_tracking else num_steps + 1
    norm_interval = normalizer_tracking.interval if normalizer_tracking else num_steps + 1

    ss_num_recordings = num_steps // ss_interval if step_size_tracking else 0
    norm_num_recordings = num_steps // norm_interval if normalizer_tracking else 0

    # Pre-allocate step-size history arrays
    ss_history = (
        jnp.zeros((ss_num_recordings, feature_dim), dtype=jnp.float32)
        if step_size_tracking
        else None
    )
    ss_bias_history = (
        jnp.zeros(ss_num_recordings, dtype=jnp.float32)
        if step_size_tracking and step_size_tracking.include_bias
        else None
    )
    ss_rec_indices = jnp.zeros(ss_num_recordings, dtype=jnp.int32) if step_size_tracking else None

    # Check if we need to track Autostep normalizers
    track_autostep_normalizers = hasattr(learner_state.learner_state.optimizer_state, "normalizers")
    ss_normalizers = (
        jnp.zeros((ss_num_recordings, feature_dim), dtype=jnp.float32)
        if step_size_tracking and track_autostep_normalizers
        else None
    )

    # Pre-allocate normalizer state history arrays
    norm_means = (
        jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
        if normalizer_tracking
        else None
    )
    norm_vars = (
        jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
        if normalizer_tracking
        else None
    )
    norm_rec_indices = (
        jnp.zeros(norm_num_recordings, dtype=jnp.int32) if normalizer_tracking else None
    )

    def step_fn_with_tracking(
        carry: tuple[
            NormalizedLearnerState,
            StreamStateT,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
        ],
        idx: Array,
    ) -> tuple[
        tuple[
            NormalizedLearnerState,
            StreamStateT,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
            Array | None,
        ],
        Array,
    ]:
        (
            l_state,
            s_state,
            ss_hist,
            ss_bias_hist,
            ss_rec,
            ss_norm,
            n_means,
            n_vars,
            n_rec,
        ) = carry

        # Perform learning step
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)

        # Step-size tracking
        new_ss_hist = ss_hist
        new_ss_bias_hist = ss_bias_hist
        new_ss_rec = ss_rec
        new_ss_norm = ss_norm

        if ss_hist is not None:
            should_record_ss = (idx % ss_interval) == 0
            recording_idx = idx // ss_interval

            # Extract current step-sizes from the inner learner state
            opt_state = result.state.learner_state.optimizer_state
            if hasattr(opt_state, "log_step_sizes"):
                # IDBD stores log step-sizes
                weight_ss = jnp.exp(opt_state.log_step_sizes)
                bias_ss = opt_state.bias_step_size
            elif hasattr(opt_state, "step_sizes"):
                # Autostep stores step-sizes directly
                weight_ss = opt_state.step_sizes
                bias_ss = opt_state.bias_step_size
            else:
                # LMS has a single fixed step-size
                weight_ss = jnp.full(feature_dim, opt_state.step_size)
                bias_ss = opt_state.step_size

            new_ss_hist = jax.lax.cond(
                should_record_ss,
                lambda _: ss_hist.at[recording_idx].set(weight_ss),
                lambda _: ss_hist,
                None,
            )

            if ss_bias_hist is not None:
                new_ss_bias_hist = jax.lax.cond(
                    should_record_ss,
                    lambda _: ss_bias_hist.at[recording_idx].set(bias_ss),
                    lambda _: ss_bias_hist,
                    None,
                )

            if ss_rec is not None:
                new_ss_rec = jax.lax.cond(
                    should_record_ss,
                    lambda _: ss_rec.at[recording_idx].set(idx),
                    lambda _: ss_rec,
                    None,
                )

            # Track Autostep normalizers (v_i) if applicable
            if ss_norm is not None and hasattr(opt_state, "normalizers"):
                new_ss_norm = jax.lax.cond(
                    should_record_ss,
                    lambda _: ss_norm.at[recording_idx].set(opt_state.normalizers),
                    lambda _: ss_norm,
                    None,
                )

        # Normalizer state tracking
        new_n_means = n_means
        new_n_vars = n_vars
        new_n_rec = n_rec

        if n_means is not None:
            should_record_norm = (idx % norm_interval) == 0
            norm_recording_idx = idx // norm_interval

            norm_state = result.state.normalizer_state

            new_n_means = jax.lax.cond(
                should_record_norm,
                lambda _: n_means.at[norm_recording_idx].set(norm_state.mean),
                lambda _: n_means,
                None,
            )

            if n_vars is not None:
                new_n_vars = jax.lax.cond(
                    should_record_norm,
                    lambda _: n_vars.at[norm_recording_idx].set(norm_state.var),
                    lambda _: n_vars,
                    None,
                )

            if n_rec is not None:
                new_n_rec = jax.lax.cond(
                    should_record_norm,
                    lambda _: n_rec.at[norm_recording_idx].set(idx),
                    lambda _: n_rec,
                    None,
                )

        return (
            result.state,
            new_s_state,
            new_ss_hist,
            new_ss_bias_hist,
            new_ss_rec,
            new_ss_norm,
            new_n_means,
            new_n_vars,
            new_n_rec,
        ), result.metrics

    initial_carry = (
        learner_state,
        stream_state,
        ss_history,
        ss_bias_history,
        ss_rec_indices,
        ss_normalizers,
        norm_means,
        norm_vars,
        norm_rec_indices,
    )

    (
        (
            final_learner,
            _,
            final_ss_hist,
            final_ss_bias_hist,
            final_ss_rec,
            final_ss_norm,
            final_n_means,
            final_n_vars,
            final_n_rec,
        ),
        metrics,
    ) = jax.lax.scan(step_fn_with_tracking, initial_carry, jnp.arange(num_steps))

    # Build return values based on what was tracked
    ss_history_result = None
    if step_size_tracking is not None and final_ss_hist is not None:
        ss_history_result = StepSizeHistory(
            step_sizes=final_ss_hist,
            bias_step_sizes=final_ss_bias_hist,
            recording_indices=final_ss_rec,
            normalizers=final_ss_norm,
        )

    norm_history_result = None
    if normalizer_tracking is not None and final_n_means is not None:
        norm_history_result = NormalizerHistory(
            means=final_n_means,
            variances=final_n_vars,
            recording_indices=final_n_rec,
        )

    # Return appropriate tuple based on what was tracked
    if ss_history_result is not None and norm_history_result is not None:
        return final_learner, metrics, ss_history_result, norm_history_result
    elif ss_history_result is not None:
        return final_learner, metrics, ss_history_result
    elif norm_history_result is not None:
        return final_learner, metrics, norm_history_result
    else:
        return final_learner, metrics


def run_learning_loop_batched[StreamStateT](
    learner: LinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    keys: Array,
    learner_state: LearnerState | None = None,
    step_size_tracking: StepSizeTrackingConfig | None = None,
) -> BatchedLearningResult:
    """Run learning loop across multiple seeds in parallel using jax.vmap.

    This function provides GPU parallelization for multi-seed experiments,
    typically achieving 2-5x speedup over sequential execution.

    Args:
        learner: The learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run per seed
        keys: JAX random keys with shape (num_seeds,) or (num_seeds, 2)
        learner_state: Initial state (if None, will be initialized from stream).
            The same initial state is used for all seeds.
        step_size_tracking: Optional config for recording per-weight step-sizes.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)

    Returns:
        BatchedLearningResult containing:
            - states: Batched final states with shape (num_seeds, ...) for each array
            - metrics: Array of shape (num_seeds, num_steps, 3)
            - step_size_history: Batched history or None if tracking disabled

    Examples:
    ```python
    import jax.random as jr
    from alberta_framework import LinearLearner, IDBD, RandomWalkStream
    from alberta_framework import run_learning_loop_batched

    stream = RandomWalkStream(feature_dim=10)
    learner = LinearLearner(optimizer=IDBD())

    # Run 30 seeds in parallel
    keys = jr.split(jr.key(42), 30)
    result = run_learning_loop_batched(learner, stream, num_steps=10000, keys=keys)

    # result.metrics has shape (30, 10000, 3)
    mean_error = result.metrics[:, :, 0].mean(axis=0)  # Average over seeds
    ```
    """

    # Define single-seed function that returns consistent structure
    def single_seed_run(key: Array) -> tuple[LearnerState, Array, StepSizeHistory | None]:
        result = run_learning_loop(
            learner, stream, num_steps, key, learner_state, step_size_tracking
        )
        if step_size_tracking is not None:
            state, metrics, history = cast(tuple[LearnerState, Array, StepSizeHistory], result)
            return state, metrics, history
        else:
            state, metrics = cast(tuple[LearnerState, Array], result)
            # Return None for history to maintain consistent output structure
            return state, metrics, None

    # vmap over the keys dimension
    batched_states, batched_metrics, batched_history = jax.vmap(single_seed_run)(keys)

    # Reconstruct batched history if tracking was enabled
    if step_size_tracking is not None and batched_history is not None:
        batched_step_size_history = StepSizeHistory(
            step_sizes=batched_history.step_sizes,
            bias_step_sizes=batched_history.bias_step_sizes,
            recording_indices=batched_history.recording_indices,
            normalizers=batched_history.normalizers,
        )
    else:
        batched_step_size_history = None

    return BatchedLearningResult(
        states=batched_states,
        metrics=batched_metrics,
        step_size_history=batched_step_size_history,
    )


def run_normalized_learning_loop_batched[StreamStateT](
    learner: NormalizedLinearLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    keys: Array,
    learner_state: NormalizedLearnerState | None = None,
    step_size_tracking: StepSizeTrackingConfig | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> BatchedNormalizedResult:
    """Run normalized learning loop across multiple seeds in parallel using jax.vmap.

    This function provides GPU parallelization for multi-seed experiments with
    normalized learners, typically achieving 2-5x speedup over sequential execution.

    Args:
        learner: The normalized learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run per seed
        keys: JAX random keys with shape (num_seeds,) or (num_seeds, 2)
        learner_state: Initial state (if None, will be initialized from stream).
            The same initial state is used for all seeds.
        step_size_tracking: Optional config for recording per-weight step-sizes.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)
        normalizer_tracking: Optional config for recording normalizer state.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)

    Returns:
        BatchedNormalizedResult containing:
            - states: Batched final states with shape (num_seeds, ...) for each array
            - metrics: Array of shape (num_seeds, num_steps, 4)
            - step_size_history: Batched history or None if tracking disabled
            - normalizer_history: Batched history or None if tracking disabled

    Examples:
    ```python
    import jax.random as jr
    from alberta_framework import NormalizedLinearLearner, IDBD, RandomWalkStream
    from alberta_framework import run_normalized_learning_loop_batched

    stream = RandomWalkStream(feature_dim=10)
    learner = NormalizedLinearLearner(optimizer=IDBD())

    # Run 30 seeds in parallel
    keys = jr.split(jr.key(42), 30)
    result = run_normalized_learning_loop_batched(
        learner, stream, num_steps=10000, keys=keys
    )

    # result.metrics has shape (30, 10000, 4)
    mean_error = result.metrics[:, :, 0].mean(axis=0)  # Average over seeds
    ```
    """

    # Define single-seed function that returns consistent structure
    def single_seed_run(
        key: Array,
    ) -> tuple[NormalizedLearnerState, Array, StepSizeHistory | None, NormalizerHistory | None]:
        result = run_normalized_learning_loop(
            learner, stream, num_steps, key, learner_state, step_size_tracking, normalizer_tracking
        )

        # Unpack based on what tracking was enabled
        if step_size_tracking is not None and normalizer_tracking is not None:
            state, metrics, ss_history, norm_history = cast(
                tuple[NormalizedLearnerState, Array, StepSizeHistory, NormalizerHistory],
                result,
            )
            return state, metrics, ss_history, norm_history
        elif step_size_tracking is not None:
            state, metrics, ss_history = cast(
                tuple[NormalizedLearnerState, Array, StepSizeHistory], result
            )
            return state, metrics, ss_history, None
        elif normalizer_tracking is not None:
            state, metrics, norm_history = cast(
                tuple[NormalizedLearnerState, Array, NormalizerHistory], result
            )
            return state, metrics, None, norm_history
        else:
            state, metrics = cast(tuple[NormalizedLearnerState, Array], result)
            return state, metrics, None, None

    # vmap over the keys dimension
    batched_states, batched_metrics, batched_ss_history, batched_norm_history = jax.vmap(
        single_seed_run
    )(keys)

    # Reconstruct batched histories if tracking was enabled
    if step_size_tracking is not None and batched_ss_history is not None:
        batched_step_size_history = StepSizeHistory(
            step_sizes=batched_ss_history.step_sizes,
            bias_step_sizes=batched_ss_history.bias_step_sizes,
            recording_indices=batched_ss_history.recording_indices,
            normalizers=batched_ss_history.normalizers,
        )
    else:
        batched_step_size_history = None

    if normalizer_tracking is not None and batched_norm_history is not None:
        batched_normalizer_history = NormalizerHistory(
            means=batched_norm_history.means,
            variances=batched_norm_history.variances,
            recording_indices=batched_norm_history.recording_indices,
        )
    else:
        batched_normalizer_history = None

    return BatchedNormalizedResult(
        states=batched_states,
        metrics=batched_metrics,
        step_size_history=batched_step_size_history,
        normalizer_history=batched_normalizer_history,
    )


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


# =============================================================================
# MLP Learner (Step 2 of Alberta Plan)
# =============================================================================


@chex.dataclass(frozen=True)
class MLPUpdateResult:
    """Result of an MLP learner update step.

    Attributes:
        state: Updated MLP learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics [squared_error, error, effective_step_size]
    """

    state: MLPLearnerState
    prediction: Prediction
    error: Float[Array, ""]
    metrics: Float[Array, " 3"]


class MLPLearner:
    """Multi-layer perceptron with ObGD optimizer for streaming learning.

    Architecture: `Input -> [Dense(H) -> LayerNorm -> LeakyReLU] x N -> Dense(1)`

    Uses parameterless layer normalization and sparse initialization following
    Elsayed et al. 2024. The ObGD optimizer prevents overshooting by dynamically
    bounding the effective step-size.

    Internal gradient computation uses `jax.grad` on a pure forward function.
    ObGD bounding is applied globally across all parameter traces.

    Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning
    Finally Works"

    Attributes:
        hidden_sizes: Tuple of hidden layer sizes
        step_size: Base learning rate for ObGD
        kappa: Bounding sensitivity for ObGD
        gamma: Discount factor for trace decay
        lamda: Eligibility trace decay parameter
        sparsity: Fraction of weights zeroed out per output neuron
        leaky_relu_slope: Negative slope for LeakyReLU activation
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 128),
        step_size: float = 1.0,
        kappa: float = 2.0,
        gamma: float = 0.0,
        lamda: float = 0.0,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
    ):
        """Initialize MLP learner.

        Args:
            hidden_sizes: Tuple of hidden layer sizes (default: two layers of 128)
            step_size: Base learning rate for ObGD (default: 1.0)
            kappa: Bounding sensitivity for ObGD (default: 2.0)
            gamma: Discount factor for trace decay (default: 0.0 for supervised)
            lamda: Eligibility trace decay parameter (default: 0.0 for supervised)
            sparsity: Fraction of weights zeroed out per output neuron (default: 0.9)
            leaky_relu_slope: Negative slope for LeakyReLU (default: 0.01)
        """
        self._hidden_sizes = hidden_sizes
        self._step_size = step_size
        self._kappa = kappa
        self._gamma = gamma
        self._lamda = lamda
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope

    def init(self, feature_dim: int, key: Array) -> MLPLearnerState:
        """Initialize MLP learner state with sparse weights.

        Args:
            feature_dim: Dimension of the input feature vector
            key: JAX random key for weight initialization

        Returns:
            Initial MLP learner state with sparse weights and zero biases
        """
        # Build layer sizes: [feature_dim, hidden1, hidden2, ..., 1]
        layer_sizes = [feature_dim, *self._hidden_sizes, 1]

        weights_list = []
        biases_list = []
        weight_traces_list = []
        bias_traces_list = []

        for i in range(len(layer_sizes) - 1):
            fan_out = layer_sizes[i + 1]
            fan_in = layer_sizes[i]
            key, subkey = jax.random.split(key)
            w = sparse_init(subkey, (fan_out, fan_in), sparsity=self._sparsity)
            b = jnp.zeros(fan_out, dtype=jnp.float32)
            weights_list.append(w)
            biases_list.append(b)
            weight_traces_list.append(jnp.zeros_like(w))
            bias_traces_list.append(jnp.zeros_like(b))

        params = MLPParams(
            weights=tuple(weights_list),
            biases=tuple(biases_list),
        )

        optimizer_state = MLPObGDState(
            step_size=jnp.array(self._step_size, dtype=jnp.float32),
            kappa=jnp.array(self._kappa, dtype=jnp.float32),
            weight_traces=tuple(weight_traces_list),
            bias_traces=tuple(bias_traces_list),
            gamma=jnp.array(self._gamma, dtype=jnp.float32),
            lamda=jnp.array(self._lamda, dtype=jnp.float32),
        )

        return MLPLearnerState(params=params, optimizer_state=optimizer_state)

    @staticmethod
    def _forward(
        weights: tuple[Array, ...],
        biases: tuple[Array, ...],
        observation: Array,
        leaky_relu_slope: float,
    ) -> Array:
        """Pure forward pass for use with jax.grad.

        Args:
            weights: Tuple of weight matrices
            biases: Tuple of bias vectors
            observation: Input feature vector
            leaky_relu_slope: Negative slope for LeakyReLU

        Returns:
            Scalar prediction
        """
        x = observation
        num_layers = len(weights)
        for i in range(num_layers - 1):
            x = weights[i] @ x + biases[i]
            # Parameterless layer normalization
            mean = jnp.mean(x)
            var = jnp.var(x)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            # LeakyReLU
            x = jnp.where(x >= 0, x, leaky_relu_slope * x)
        # Output layer (no activation)
        x = weights[-1] @ x + biases[-1]
        return jnp.squeeze(x)

    def predict(self, state: MLPLearnerState, observation: Observation) -> Prediction:
        """Compute prediction for an observation.

        Args:
            state: Current MLP learner state
            observation: Input feature vector

        Returns:
            Scalar prediction
        """
        y = self._forward(
            state.params.weights,
            state.params.biases,
            observation,
            self._leaky_relu_slope,
        )
        return jnp.atleast_1d(y)

    def update(
        self,
        state: MLPLearnerState,
        observation: Observation,
        target: Target,
    ) -> MLPUpdateResult:
        """Update MLP given observation and target.

        Performs one step of the learning algorithm:
        1. Compute prediction and error
        2. Compute gradients via jax.grad on the forward pass
        3. Update eligibility traces
        4. Apply ObGD bounding across all parameters
        5. Apply bounded weight updates

        Args:
            state: Current MLP learner state
            observation: Input feature vector
            target: Desired output

        Returns:
            MLPUpdateResult with new state, prediction, error, and metrics
        """
        target_scalar = jnp.squeeze(target)
        opt = state.optimizer_state

        # Forward pass for prediction
        prediction = self.predict(state, observation)
        error = target_scalar - jnp.squeeze(prediction)

        # Compute gradients w.r.t. prediction (for ObGD: grad of prediction w.r.t. params)
        # We need -grad(prediction) because ObGD uses: w += alpha_eff * error * z
        # and z accumulates gradients of the prediction
        slope = self._leaky_relu_slope

        def pred_fn(weights: tuple[Array, ...], biases: tuple[Array, ...]) -> Array:
            return self._forward(weights, biases, observation, slope)

        weight_grads, bias_grads = jax.grad(pred_fn, argnums=(0, 1))(
            state.params.weights, state.params.biases
        )

        # Update eligibility traces: z = gamma * lamda * z + grad
        gamma_lamda = opt.gamma * opt.lamda
        new_weight_traces = tuple(
            gamma_lamda * zt + gt for zt, gt in zip(opt.weight_traces, weight_grads)
        )
        new_bias_traces = tuple(
            gamma_lamda * zt + gt for zt, gt in zip(opt.bias_traces, bias_grads)
        )

        # Compute global z_sum (L1 norm across all traces)
        z_sum = jnp.array(0.0)
        for zt in new_weight_traces:
            z_sum = z_sum + jnp.sum(jnp.abs(zt))
        for zt in new_bias_traces:
            z_sum = z_sum + jnp.sum(jnp.abs(zt))

        # ObGD bounding
        alpha = opt.step_size
        kappa = opt.kappa
        delta_bar = jnp.maximum(jnp.abs(error), 1.0)
        dot_product = delta_bar * z_sum * alpha * kappa
        alpha_eff = alpha / jnp.maximum(dot_product, 1.0)

        # Apply updates: w += alpha_eff * error * z
        new_weights = tuple(
            w + alpha_eff * error * zt
            for w, zt in zip(state.params.weights, new_weight_traces)
        )
        new_biases = tuple(
            b + alpha_eff * error * zt
            for b, zt in zip(state.params.biases, new_bias_traces)
        )

        new_params = MLPParams(weights=new_weights, biases=new_biases)
        new_opt_state = MLPObGDState(
            step_size=alpha,
            kappa=kappa,
            weight_traces=new_weight_traces,
            bias_traces=new_bias_traces,
            gamma=opt.gamma,
            lamda=opt.lamda,
        )
        new_state = MLPLearnerState(params=new_params, optimizer_state=new_opt_state)

        squared_error = error**2
        metrics = jnp.array([squared_error, error, alpha_eff], dtype=jnp.float32)

        return MLPUpdateResult(
            state=new_state,
            prediction=prediction,
            error=jnp.atleast_1d(error),
            metrics=metrics,
        )


def run_mlp_learning_loop[StreamStateT](
    learner: MLPLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: MLPLearnerState | None = None,
) -> tuple[MLPLearnerState, Array]:
    """Run the MLP learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.

    Args:
        learner: The MLP learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream and weight initialization
        learner_state: Initial state (if None, will be initialized from stream)

    Returns:
        Tuple of (final_state, metrics_array) where metrics_array has shape
        (num_steps, 3) with columns [squared_error, error, effective_step_size]
    """
    # Split key for initialization
    stream_key, init_key = jax.random.split(key)

    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim, init_key)
    stream_state = stream.init(stream_key)

    def step_fn(
        carry: tuple[MLPLearnerState, StreamStateT], idx: Array
    ) -> tuple[tuple[MLPLearnerState, StreamStateT], Array]:
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        return (result.state, new_s_state), result.metrics

    (final_learner, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )

    return final_learner, metrics


def run_mlp_learning_loop_batched[StreamStateT](
    learner: MLPLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    keys: Array,
    learner_state: MLPLearnerState | None = None,
) -> BatchedMLPResult:
    """Run MLP learning loop across multiple seeds in parallel using jax.vmap.

    This function provides GPU parallelization for multi-seed MLP experiments,
    typically achieving 2-5x speedup over sequential execution.

    Args:
        learner: The MLP learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run per seed
        keys: JAX random keys with shape (num_seeds,) or (num_seeds, 2)
        learner_state: Initial state (if None, will be initialized from stream).
            The same initial state is used for all seeds.

    Returns:
        BatchedMLPResult containing:
            - states: Batched final states with shape (num_seeds, ...) for each array
            - metrics: Array of shape (num_seeds, num_steps, 3)

    Examples:
    ```python
    import jax.random as jr
    from alberta_framework import MLPLearner, RandomWalkStream
    from alberta_framework import run_mlp_learning_loop_batched

    stream = RandomWalkStream(feature_dim=10)
    learner = MLPLearner(hidden_sizes=(128, 128))

    # Run 30 seeds in parallel
    keys = jr.split(jr.key(42), 30)
    result = run_mlp_learning_loop_batched(learner, stream, num_steps=10000, keys=keys)

    # result.metrics has shape (30, 10000, 3)
    mean_error = result.metrics[:, :, 0].mean(axis=0)  # Average over seeds
    ```
    """

    def single_seed_run(key: Array) -> tuple[MLPLearnerState, Array]:
        return run_mlp_learning_loop(
            learner, stream, num_steps, key, learner_state
        )

    batched_states, batched_metrics = jax.vmap(single_seed_run)(keys)

    return BatchedMLPResult(
        states=batched_states,
        metrics=batched_metrics,
    )


# =============================================================================
# Normalized MLP Learner (Step 2 of Alberta Plan)
# =============================================================================


@chex.dataclass(frozen=True)
class NormalizedMLPLearnerState:
    """State for an MLP learner with online feature normalization.

    Attributes:
        learner_state: Underlying MLP learner state (params, optimizer)
        normalizer_state: Online normalizer state (mean, var estimates)
    """

    learner_state: MLPLearnerState
    normalizer_state: EMANormalizerState | WelfordNormalizerState


@chex.dataclass(frozen=True)
class NormalizedMLPUpdateResult:
    """Result of a normalized MLP learner update step.

    Attributes:
        state: Updated normalized MLP learner state
        prediction: Prediction made before update
        error: Prediction error
        metrics: Array of metrics [squared_error, error, effective_step_size, normalizer_mean_var]
    """

    state: NormalizedMLPLearnerState
    prediction: Prediction
    error: Float[Array, ""]
    metrics: Float[Array, " 4"]


class NormalizedMLPLearner:
    """MLP learner with online feature normalization.

    Wraps an MLPLearner with online feature normalization, following
    the Alberta Plan's approach to handling varying feature scales.

    Normalization is applied to features before prediction and learning:
        x_normalized = (x - mean) / (std + epsilon)

    The normalizer statistics update at every time step, maintaining
    temporal uniformity.

    Unlike NormalizedLinearLearner which exposes optimizer params directly,
    this class accepts a pre-constructed MLPLearner to avoid duplicating
    its constructor parameters.

    Attributes:
        learner: Underlying MLP learner
        normalizer: Online feature normalizer
    """

    def __init__(
        self,
        learner: MLPLearner,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
    ):
        """Initialize the normalized MLP learner.

        Args:
            learner: Pre-constructed MLP learner
            normalizer: Feature normalizer. Defaults to EMANormalizer()
        """
        self._learner = learner
        self._normalizer = normalizer or EMANormalizer()

    def init(self, feature_dim: int, key: Array) -> NormalizedMLPLearnerState:
        """Initialize normalized MLP learner state.

        Args:
            feature_dim: Dimension of the input feature vector
            key: JAX random key for weight initialization

        Returns:
            Initial state with sparse weights and unit variance estimates
        """
        return NormalizedMLPLearnerState(
            learner_state=self._learner.init(feature_dim, key),
            normalizer_state=self._normalizer.init(feature_dim),
        )

    def predict(
        self,
        state: NormalizedMLPLearnerState,
        observation: Observation,
    ) -> Prediction:
        """Compute prediction for an observation.

        Normalizes the observation using current statistics before prediction.

        Args:
            state: Current normalized MLP learner state
            observation: Raw (unnormalized) input feature vector

        Returns:
            Scalar prediction
        """
        normalized_obs = self._normalizer.normalize_only(state.normalizer_state, observation)
        return self._learner.predict(state.learner_state, normalized_obs)

    def update(
        self,
        state: NormalizedMLPLearnerState,
        observation: Observation,
        target: Target,
    ) -> NormalizedMLPUpdateResult:
        """Update MLP given observation and target.

        Performs one step of the learning algorithm:
        1. Normalize observation (and update normalizer statistics)
        2. Delegate to underlying MLP learner
        3. Append normalizer_mean_var metric

        Args:
            state: Current normalized MLP learner state
            observation: Raw (unnormalized) input feature vector
            target: Desired output

        Returns:
            NormalizedMLPUpdateResult with new state, prediction, error, and metrics
        """
        # Normalize observation and update normalizer state
        normalized_obs, new_normalizer_state = self._normalizer.normalize(
            state.normalizer_state, observation
        )

        # Delegate to underlying MLP learner
        result = self._learner.update(
            state.learner_state,
            normalized_obs,
            target,
        )

        # Build combined state
        new_state = NormalizedMLPLearnerState(
            learner_state=result.state,
            normalizer_state=new_normalizer_state,
        )

        # Add normalizer metrics to the metrics array
        normalizer_mean_var = jnp.mean(new_normalizer_state.var)
        metrics = jnp.concatenate([result.metrics, jnp.array([normalizer_mean_var])])

        return NormalizedMLPUpdateResult(
            state=new_state,
            prediction=result.prediction,
            error=result.error,
            metrics=metrics,
        )


def run_mlp_normalized_learning_loop[StreamStateT](
    learner: NormalizedMLPLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: NormalizedMLPLearnerState | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> (
    tuple[NormalizedMLPLearnerState, Array]
    | tuple[NormalizedMLPLearnerState, Array, NormalizerHistory]
):
    """Run the normalized MLP learning loop using jax.lax.scan.

    Args:
        learner: The normalized MLP learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run
        key: JAX random key for stream and weight initialization
        learner_state: Initial state (if None, will be initialized from stream)
        normalizer_tracking: Optional config for recording per-feature normalizer state.
            When provided, returns NormalizerHistory with means and variances over time.

    Returns:
        If no tracking:
            Tuple of (final_state, metrics_array) where metrics_array has shape
            (num_steps, 4) with columns [squared_error, error, effective_step_size,
            normalizer_mean_var]
        If normalizer_tracking:
            Tuple of (final_state, metrics_array, normalizer_history)

    Raises:
        ValueError: If normalizer_tracking.interval is invalid
    """
    # Validate tracking config
    if normalizer_tracking is not None:
        if normalizer_tracking.interval < 1:
            raise ValueError(
                f"normalizer_tracking.interval must be >= 1, got {normalizer_tracking.interval}"
            )
        if normalizer_tracking.interval > num_steps:
            raise ValueError(
                f"normalizer_tracking.interval ({normalizer_tracking.interval}) "
                f"must be <= num_steps ({num_steps})"
            )

    # Split key for initialization
    stream_key, init_key = jax.random.split(key)

    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim, init_key)
    stream_state = stream.init(stream_key)

    feature_dim = stream.feature_dim

    if normalizer_tracking is None:
        # Simple case without tracking
        def step_fn(
            carry: tuple[NormalizedMLPLearnerState, StreamStateT], idx: Array
        ) -> tuple[tuple[NormalizedMLPLearnerState, StreamStateT], Array]:
            l_state, s_state = carry
            timestep, new_s_state = stream.step(s_state, idx)
            result = learner.update(l_state, timestep.observation, timestep.target)
            return (result.state, new_s_state), result.metrics

        (final_learner, _), metrics = jax.lax.scan(
            step_fn, (learner_state, stream_state), jnp.arange(num_steps)
        )

        return final_learner, metrics

    # Tracking enabled
    norm_interval = normalizer_tracking.interval
    norm_num_recordings = num_steps // norm_interval

    norm_means = jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
    norm_vars = jnp.zeros((norm_num_recordings, feature_dim), dtype=jnp.float32)
    norm_rec_indices = jnp.zeros(norm_num_recordings, dtype=jnp.int32)

    def step_fn_with_tracking(
        carry: tuple[NormalizedMLPLearnerState, StreamStateT, Array, Array, Array],
        idx: Array,
    ) -> tuple[
        tuple[NormalizedMLPLearnerState, StreamStateT, Array, Array, Array],
        Array,
    ]:
        l_state, s_state, n_means, n_vars, n_rec = carry

        # Perform learning step
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)

        # Normalizer state tracking
        should_record = (idx % norm_interval) == 0
        recording_idx = idx // norm_interval

        norm_state = result.state.normalizer_state

        new_n_means = jax.lax.cond(
            should_record,
            lambda _: n_means.at[recording_idx].set(norm_state.mean),
            lambda _: n_means,
            None,
        )

        new_n_vars = jax.lax.cond(
            should_record,
            lambda _: n_vars.at[recording_idx].set(norm_state.var),
            lambda _: n_vars,
            None,
        )

        new_n_rec = jax.lax.cond(
            should_record,
            lambda _: n_rec.at[recording_idx].set(idx),
            lambda _: n_rec,
            None,
        )

        return (
            result.state,
            new_s_state,
            new_n_means,
            new_n_vars,
            new_n_rec,
        ), result.metrics

    initial_carry = (
        learner_state,
        stream_state,
        norm_means,
        norm_vars,
        norm_rec_indices,
    )

    (
        (final_learner, _, final_n_means, final_n_vars, final_n_rec),
        metrics,
    ) = jax.lax.scan(step_fn_with_tracking, initial_carry, jnp.arange(num_steps))

    norm_history = NormalizerHistory(
        means=final_n_means,
        variances=final_n_vars,
        recording_indices=final_n_rec,
    )

    return final_learner, metrics, norm_history


def run_mlp_normalized_learning_loop_batched[StreamStateT](
    learner: NormalizedMLPLearner,
    stream: ScanStream[StreamStateT],
    num_steps: int,
    keys: Array,
    learner_state: NormalizedMLPLearnerState | None = None,
    normalizer_tracking: NormalizerTrackingConfig | None = None,
) -> BatchedMLPNormalizedResult:
    """Run normalized MLP learning loop across multiple seeds in parallel using jax.vmap.

    Args:
        learner: The normalized MLP learner to train
        stream: Experience stream providing (observation, target) pairs
        num_steps: Number of learning steps to run per seed
        keys: JAX random keys with shape (num_seeds,) or (num_seeds, 2)
        learner_state: Initial state (if None, will be initialized from stream).
            The same initial state is used for all seeds.
        normalizer_tracking: Optional config for recording normalizer state.
            When provided, history arrays have shape (num_seeds, num_recordings, ...)

    Returns:
        BatchedMLPNormalizedResult containing:
            - states: Batched final states with shape (num_seeds, ...) for each array
            - metrics: Array of shape (num_seeds, num_steps, 4)
            - normalizer_history: Batched history or None if tracking disabled
    """

    def single_seed_run(
        key: Array,
    ) -> tuple[NormalizedMLPLearnerState, Array, NormalizerHistory | None]:
        result = run_mlp_normalized_learning_loop(
            learner, stream, num_steps, key, learner_state, normalizer_tracking
        )

        if normalizer_tracking is not None:
            state, metrics, norm_history = cast(
                tuple[NormalizedMLPLearnerState, Array, NormalizerHistory], result
            )
            return state, metrics, norm_history
        else:
            state, metrics = cast(tuple[NormalizedMLPLearnerState, Array], result)
            return state, metrics, None

    batched_states, batched_metrics, batched_norm_history = jax.vmap(single_seed_run)(keys)

    if normalizer_tracking is not None and batched_norm_history is not None:
        batched_normalizer_history = NormalizerHistory(
            means=batched_norm_history.means,
            variances=batched_norm_history.variances,
            recording_indices=batched_norm_history.recording_indices,
        )
    else:
        batched_normalizer_history = None

    return BatchedMLPNormalizedResult(
        states=batched_states,
        metrics=batched_metrics,
        normalizer_history=batched_normalizer_history,
    )


# =============================================================================
# TD Learning (for Step 3+ of Alberta Plan)
# =============================================================================


@chex.dataclass(frozen=True)
class TDUpdateResult:
    """Result of a TD learner update step.

    Attributes:
        state: Updated TD learner state
        prediction: Value prediction V(s) before update
        td_error: TD error δ = R + γV(s') - V(s)
        metrics: Array of metrics [squared_td_error, td_error, mean_step_size, ...]
    """

    state: TDLearnerState
    prediction: Prediction
    td_error: Float[Array, ""]
    metrics: Float[Array, " 4"]


class TDLinearLearner:
    """Linear function approximator for TD learning.

    Computes value predictions as: `V(s) = w @ φ(s) + b`

    The learner maintains weights, bias, and eligibility traces, delegating
    the adaptation of learning rates to the TD optimizer (e.g., TDIDBD).

    This follows the Alberta Plan philosophy of temporal uniformity:
    every component updates at every time step.

    Reference: Kearney et al. 2019, "Learning Feature Relevance Through Step Size
    Adaptation in Temporal-Difference Learning"

    Attributes:
        optimizer: The TD optimizer to use for weight updates
    """

    def __init__(self, optimizer: AnyTDOptimizer | None = None):
        """Initialize the TD linear learner.

        Args:
            optimizer: TD optimizer for weight updates. Defaults to TDIDBD()
        """
        self._optimizer: AnyTDOptimizer = optimizer or TDIDBD()

    def init(self, feature_dim: int) -> TDLearnerState:
        """Initialize TD learner state.

        Args:
            feature_dim: Dimension of the input feature vector

        Returns:
            Initial TD learner state with zero weights and bias
        """
        optimizer_state = self._optimizer.init(feature_dim)

        return TDLearnerState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            optimizer_state=optimizer_state,
        )

    def predict(self, state: TDLearnerState, observation: Observation) -> Prediction:
        """Compute value prediction for an observation.

        Args:
            state: Current TD learner state
            observation: Input feature vector φ(s)

        Returns:
            Scalar value prediction `V(s) = w @ φ(s) + b`
        """
        return jnp.atleast_1d(jnp.dot(state.weights, observation) + state.bias)

    def update(
        self,
        state: TDLearnerState,
        observation: Observation,
        reward: Array,
        next_observation: Observation,
        gamma: Array,
    ) -> TDUpdateResult:
        """Update learner given a TD transition.

        Performs one step of TD learning:
        1. Compute V(s) and V(s')
        2. Compute TD error δ = R + γV(s') - V(s)
        3. Get weight updates from TD optimizer
        4. Apply updates to weights and bias

        Args:
            state: Current TD learner state
            observation: Current observation φ(s)
            reward: Reward R received
            next_observation: Next observation φ(s')
            gamma: Discount factor γ (0 at terminal states)

        Returns:
            TDUpdateResult with new state, prediction, TD error, and metrics
        """
        # Compute predictions
        prediction = self.predict(state, observation)
        next_prediction = self.predict(state, next_observation)

        # Compute TD error: δ = R + γV(s') - V(s)
        gamma_scalar = jnp.squeeze(gamma)
        td_error = (
            jnp.squeeze(reward)
            + gamma_scalar * jnp.squeeze(next_prediction)
            - jnp.squeeze(prediction)
        )

        # Get update from TD optimizer
        opt_update = self._optimizer.update(
            state.optimizer_state,
            td_error,
            observation,
            next_observation,
            gamma,
        )

        # Apply updates
        new_weights = state.weights + opt_update.weight_delta
        new_bias = state.bias + opt_update.bias_delta

        new_state = TDLearnerState(
            weights=new_weights,
            bias=new_bias,
            optimizer_state=opt_update.new_state,
        )

        # Pack metrics as array for scan compatibility
        # Format: [squared_td_error, td_error, mean_step_size, mean_eligibility_trace]
        squared_td_error = td_error**2
        mean_step_size = opt_update.metrics.get("mean_step_size", 0.0)
        mean_elig_trace = opt_update.metrics.get("mean_eligibility_trace", 0.0)
        metrics = jnp.array(
            [squared_td_error, td_error, mean_step_size, mean_elig_trace],
            dtype=jnp.float32,
        )

        return TDUpdateResult(
            state=new_state,
            prediction=prediction,
            td_error=jnp.atleast_1d(td_error),
            metrics=metrics,
        )


class TDStream(Protocol[StateT]):
    """Protocol for TD experience streams.

    TD streams produce (s, r, s', γ) tuples for temporal-difference learning.
    """

    feature_dim: int

    def init(self, key: Array) -> StateT:
        """Initialize stream state."""
        ...

    def step(self, state: StateT, idx: Array) -> tuple[TDTimeStep, StateT]:
        """Generate next TD transition."""
        ...


def run_td_learning_loop[StreamStateT](
    learner: TDLinearLearner,
    stream: TDStream[StreamStateT],
    num_steps: int,
    key: Array,
    learner_state: TDLearnerState | None = None,
) -> tuple[TDLearnerState, Array]:
    """Run the TD learning loop using jax.lax.scan.

    This is a JIT-compiled learning loop that uses scan for efficiency.
    It returns metrics as a fixed-size array rather than a list of dicts.

    Args:
        learner: The TD learner to train
        stream: TD experience stream providing (s, r, s', γ) tuples
        num_steps: Number of learning steps to run
        key: JAX random key for stream initialization
        learner_state: Initial state (if None, will be initialized from stream)

    Returns:
        Tuple of (final_state, metrics_array) where metrics_array has shape
        (num_steps, 4) with columns [squared_td_error, td_error, mean_step_size,
        mean_eligibility_trace]
    """
    # Initialize states
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    def step_fn(
        carry: tuple[TDLearnerState, StreamStateT], idx: Array
    ) -> tuple[tuple[TDLearnerState, StreamStateT], Array]:
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(
            l_state,
            timestep.observation,
            timestep.reward,
            timestep.next_observation,
            timestep.gamma,
        )
        return (result.state, new_s_state), result.metrics

    (final_learner, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )

    return final_learner, metrics
