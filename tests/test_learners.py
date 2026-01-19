"""Tests for LinearLearner."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import (
    Autostep,
    IDBD,
    LMS,
    LinearLearner,
    RandomWalkStream,
    StepSizeHistory,
    StepSizeTrackingConfig,
    metrics_to_dicts,
    run_learning_loop,
)


class TestLinearLearner:
    """Tests for the LinearLearner class."""

    def test_init_creates_zero_weights(self, feature_dim):
        """Learner should initialize with zero weights and bias."""
        learner = LinearLearner()
        state = learner.init(feature_dim)

        assert state.weights.shape == (feature_dim,)
        assert jnp.allclose(state.weights, 0.0)
        assert jnp.isclose(state.bias, 0.0)

    def test_predict_returns_correct_shape(self, feature_dim, sample_observation):
        """Prediction should return scalar (as 1D array)."""
        learner = LinearLearner()
        state = learner.init(feature_dim)

        prediction = learner.predict(state, sample_observation)

        assert prediction.shape == (1,)

    def test_predict_with_zero_weights_is_bias(self, feature_dim, sample_observation):
        """With zero weights, prediction should equal bias."""
        learner = LinearLearner()
        state = learner.init(feature_dim)

        prediction = learner.predict(state, sample_observation)

        assert jnp.isclose(prediction[0], state.bias)

    def test_update_reduces_error(self, feature_dim, sample_observation, sample_target):
        """Update should move prediction closer to target."""
        learner = LinearLearner(optimizer=LMS(step_size=0.1))
        state = learner.init(feature_dim)

        # Get initial error
        initial_pred = learner.predict(state, sample_observation)
        initial_error = abs(float(sample_target[0] - initial_pred[0]))

        # Do several updates
        for _ in range(10):
            result = learner.update(state, sample_observation, sample_target)
            state = result.state

        # Error should have decreased
        final_pred = learner.predict(state, sample_observation)
        final_error = abs(float(sample_target[0] - final_pred[0]))

        assert final_error < initial_error

    def test_update_returns_correct_metrics_array(self, feature_dim, sample_observation, sample_target):
        """Update should return metrics array with squared error."""
        learner = LinearLearner()
        state = learner.init(feature_dim)

        result = learner.update(state, sample_observation, sample_target)

        # Metrics are now an array [squared_error, error, mean_step_size]
        assert result.metrics.shape == (3,)
        assert result.metrics[0] >= 0  # squared_error

    def test_works_with_idbd_optimizer(self, feature_dim, sample_observation, sample_target):
        """Learner should work correctly with IDBD optimizer."""
        learner = LinearLearner(optimizer=IDBD())
        state = learner.init(feature_dim)

        result = learner.update(state, sample_observation, sample_target)

        assert result.state is not None
        # Metrics array: [squared_error, error, mean_step_size]
        assert result.metrics.shape == (3,)


class TestRunLearningLoop:
    """Tests for the run_learning_loop helper function."""

    def test_returns_correct_number_of_metrics(self, rng_key):
        """Should return metrics for each step."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner()

        num_steps = 100
        _, metrics = run_learning_loop(learner, stream, num_steps, rng_key)

        # Metrics is now an array of shape (num_steps, 3)
        assert metrics.shape == (num_steps, 3)

    def test_returns_valid_final_state(self, rng_key):
        """Final state should have correct structure."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner()

        state, _ = run_learning_loop(learner, stream, num_steps=50, key=rng_key)

        assert state.weights.shape == (5,)
        assert jnp.all(jnp.isfinite(state.weights))

    def test_can_resume_from_existing_state(self, rng_key):
        """Should be able to continue from a previous state."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner()

        # First run
        key1, key2 = jr.split(rng_key)
        state1, _ = run_learning_loop(learner, stream, num_steps=50, key=key1)

        # Continue from state1 with new key for stream
        state2, _ = run_learning_loop(
            learner, stream, num_steps=50, key=key2, learner_state=state1
        )

        # Weights should have changed
        assert not jnp.allclose(state1.weights, state2.weights)

    def test_error_decreases_on_stationary_target(self, rng_key):
        """On a stationary target, error should decrease over time."""
        # Use zero drift for stationary target
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.0)
        learner = LinearLearner(optimizer=LMS(step_size=0.01))

        _, metrics = run_learning_loop(learner, stream, num_steps=1000, key=rng_key)

        # Convert to dicts for easier access
        metrics_list = metrics_to_dicts(metrics)

        # Compare first 100 vs last 100 average error
        early_error = sum(m["squared_error"] for m in metrics_list[:100]) / 100
        late_error = sum(m["squared_error"] for m in metrics_list[-100:]) / 100

        assert late_error < early_error

    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same results."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner()

        state1, metrics1 = run_learning_loop(learner, stream, num_steps=50, key=rng_key)
        state2, metrics2 = run_learning_loop(learner, stream, num_steps=50, key=rng_key)

        assert jnp.allclose(state1.weights, state2.weights)
        assert jnp.allclose(metrics1, metrics2)


class TestStepSizeTracking:
    """Tests for step-size tracking in run_learning_loop."""

    def test_returns_3_tuple_when_tracking_enabled(self, rng_key):
        """Should return 3-tuple (state, metrics, history) when tracking enabled."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner(optimizer=IDBD())
        config = StepSizeTrackingConfig(interval=10)

        result = run_learning_loop(
            learner, stream, num_steps=100, key=rng_key, step_size_tracking=config
        )

        assert len(result) == 3
        state, metrics, history = result
        assert state is not None
        assert metrics is not None
        assert isinstance(history, StepSizeHistory)

    def test_returns_2_tuple_when_tracking_disabled(self, rng_key):
        """Should return 2-tuple (state, metrics) when tracking disabled (backward compat)."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner(optimizer=IDBD())

        result = run_learning_loop(learner, stream, num_steps=100, key=rng_key)

        assert len(result) == 2
        state, metrics = result
        assert state is not None
        assert metrics is not None

    def test_history_shape_based_on_interval(self, rng_key):
        """History should have correct shape based on interval."""
        feature_dim = 10
        num_steps = 1000
        interval = 100
        expected_recordings = num_steps // interval  # 10

        stream = RandomWalkStream(feature_dim=feature_dim)
        learner = LinearLearner(optimizer=IDBD())
        config = StepSizeTrackingConfig(interval=interval)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=num_steps, key=rng_key, step_size_tracking=config
        )

        assert history.step_sizes.shape == (expected_recordings, feature_dim)
        assert history.bias_step_sizes.shape == (expected_recordings,)
        assert history.recording_indices.shape == (expected_recordings,)

    def test_lms_returns_constant_step_sizes(self, rng_key):
        """LMS should return constant step-sizes throughout training."""
        step_size = 0.05
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner(optimizer=LMS(step_size=step_size))
        config = StepSizeTrackingConfig(interval=10)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=100, key=rng_key, step_size_tracking=config
        )

        # All step-sizes should be equal to the fixed step_size
        assert jnp.allclose(history.step_sizes, step_size)
        assert jnp.allclose(history.bias_step_sizes, step_size)

    def test_idbd_step_sizes_evolve(self, rng_key):
        """IDBD step-sizes should change over training."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.01)
        learner = LinearLearner(optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.1))
        config = StepSizeTrackingConfig(interval=100)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=10000, key=rng_key, step_size_tracking=config
        )

        # First and last recordings should differ
        first_mean = jnp.mean(history.step_sizes[0])
        last_mean = jnp.mean(history.step_sizes[-1])
        assert not jnp.isclose(first_mean, last_mean, rtol=0.1)

    def test_autostep_step_sizes_evolve(self, rng_key):
        """Autostep step-sizes should change over training."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.01)
        learner = LinearLearner(optimizer=Autostep(initial_step_size=0.01, meta_step_size=0.1))
        config = StepSizeTrackingConfig(interval=100)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=10000, key=rng_key, step_size_tracking=config
        )

        # First and last recordings should differ
        first_mean = jnp.mean(history.step_sizes[0])
        last_mean = jnp.mean(history.step_sizes[-1])
        assert not jnp.isclose(first_mean, last_mean, rtol=0.1)

    def test_interval_1_records_every_step(self, rng_key):
        """Interval of 1 should record at every step."""
        num_steps = 50
        feature_dim = 3
        stream = RandomWalkStream(feature_dim=feature_dim)
        learner = LinearLearner(optimizer=IDBD())
        config = StepSizeTrackingConfig(interval=1)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=num_steps, key=rng_key, step_size_tracking=config
        )

        assert history.step_sizes.shape == (num_steps, feature_dim)
        # Recording indices should be 0, 1, 2, ..., num_steps-1
        expected_indices = jnp.arange(num_steps)
        assert jnp.allclose(history.recording_indices, expected_indices)

    def test_interval_equals_num_steps_records_once(self, rng_key):
        """Interval equal to num_steps should record once at step 0."""
        num_steps = 100
        feature_dim = 5
        stream = RandomWalkStream(feature_dim=feature_dim)
        learner = LinearLearner(optimizer=IDBD())
        config = StepSizeTrackingConfig(interval=num_steps)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=num_steps, key=rng_key, step_size_tracking=config
        )

        assert history.step_sizes.shape == (1, feature_dim)
        assert history.recording_indices[0] == 0

    def test_invalid_interval_zero_raises_error(self, rng_key):
        """Interval of 0 should raise ValueError."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner()
        config = StepSizeTrackingConfig(interval=0)

        with pytest.raises(ValueError, match="interval must be >= 1"):
            run_learning_loop(
                learner, stream, num_steps=100, key=rng_key, step_size_tracking=config
            )

    def test_invalid_interval_greater_than_num_steps_raises_error(self, rng_key):
        """Interval greater than num_steps should raise ValueError."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner()
        config = StepSizeTrackingConfig(interval=200)

        with pytest.raises(ValueError, match="must be <= num_steps"):
            run_learning_loop(
                learner, stream, num_steps=100, key=rng_key, step_size_tracking=config
            )

    def test_include_bias_false_returns_none(self, rng_key):
        """When include_bias=False, bias_step_sizes should be None."""
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner(optimizer=IDBD())
        config = StepSizeTrackingConfig(interval=10, include_bias=False)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=100, key=rng_key, step_size_tracking=config
        )

        assert history.bias_step_sizes is None
        assert history.step_sizes is not None

    def test_recording_indices_correct(self, rng_key):
        """Recording indices should match expected values based on interval."""
        num_steps = 100
        interval = 25
        stream = RandomWalkStream(feature_dim=5)
        learner = LinearLearner(optimizer=IDBD())
        config = StepSizeTrackingConfig(interval=interval)

        _, _, history = run_learning_loop(
            learner, stream, num_steps=num_steps, key=rng_key, step_size_tracking=config
        )

        # Should record at steps 0, 25, 50, 75
        expected_indices = jnp.array([0, 25, 50, 75])
        assert jnp.allclose(history.recording_indices, expected_indices)
