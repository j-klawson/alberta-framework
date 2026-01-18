"""Tests for LinearLearner."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import (
    IDBD,
    LMS,
    LinearLearner,
    RandomWalkStream,
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
