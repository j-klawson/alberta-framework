"""Learning units for continual learning.

Implements learners that combine function approximation with optimizers
for temporally-uniform learning. Uses JAX's scan for efficient JIT-compiled
training loops.
"""

from typing import cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from alberta_framework.core.normalizers import NormalizerState, OnlineNormalizer
from alberta_framework.core.optimizers import LMS, Optimizer
from alberta_framework.core.types import (
    AutostepState,
    BatchedLearningResult,
    BatchedNormalizedResult,
    IDBDState,
    LearnerState,
    LMSState,
    NormalizerHistory,
    NormalizerTrackingConfig,
    Observation,
    Prediction,
    StepSizeHistory,
    StepSizeTrackingConfig,
    Target,
)
from alberta_framework.streams.base import ScanStream

# Type alias for any optimizer type
AnyOptimizer = Optimizer[LMSState] | Optimizer[IDBDState] | Optimizer[AutostepState]


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
    normalizer_state: NormalizerState


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
            state.optimizer_state,            error,
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
        bias_history = (
            jnp.zeros(num_recordings, dtype=jnp.float32) if include_bias else None
        )
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
            carry: tuple[
                LearnerState, StreamStateT, Array, Array | None, Array, Array | None
            ],
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
                    lambda _: norm_history.at[recording_idx].set(
                        opt_state.normalizers                    ),
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
            final_learner,
            _,
            final_ss_history,
            final_b_history,
            final_rec_indices,
            final_norm_history,
        ), metrics = jax.lax.scan(
            step_fn_with_tracking, initial_carry, jnp.arange(num_steps)
        )

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
    norm_interval = (
        normalizer_tracking.interval if normalizer_tracking else num_steps + 1
    )

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
    ss_rec_indices = (
        jnp.zeros(ss_num_recordings, dtype=jnp.int32) if step_size_tracking else None
    )

    # Check if we need to track Autostep normalizers
    track_autostep_normalizers = hasattr(
        learner_state.learner_state.optimizer_state, "normalizers"
    )
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
                    lambda _: ss_norm.at[recording_idx].set(
                        opt_state.normalizers                    ),
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
        final_learner,
        _,
        final_ss_hist,
        final_ss_bias_hist,
        final_ss_rec,
        final_ss_norm,
        final_n_means,
        final_n_vars,
        final_n_rec,
    ), metrics = jax.lax.scan(
        step_fn_with_tracking, initial_carry, jnp.arange(num_steps)
    )

    # Build return values based on what was tracked
    ss_history_result = None
    if step_size_tracking is not None and final_ss_hist is not None:
        ss_history_result = StepSizeHistory(
            step_sizes=final_ss_hist,
            bias_step_sizes=final_ss_bias_hist,
            recording_indices=final_ss_rec,            normalizers=final_ss_norm,
        )

    norm_history_result = None
    if normalizer_tracking is not None and final_n_means is not None:
        norm_history_result = NormalizerHistory(
            means=final_n_means,
            variances=final_n_vars,            recording_indices=final_n_rec,        )

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
            state, metrics, history = cast(
                tuple[LearnerState, Array, StepSizeHistory], result
            )
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
    ) -> tuple[
        NormalizedLearnerState, Array, StepSizeHistory | None, NormalizerHistory | None
    ]:
        result = run_normalized_learning_loop(
            learner, stream, num_steps, key, learner_state,
            step_size_tracking, normalizer_tracking
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
    batched_states, batched_metrics, batched_ss_history, batched_norm_history = (
        jax.vmap(single_seed_run)(keys)
    )

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
