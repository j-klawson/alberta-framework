"""Tests for LMS, IDBD, and Autostep optimizers."""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import Autostep, IDBD, LMS


class TestLMS:
    """Tests for the LMS optimizer."""

    def test_init_creates_correct_state(self):
        """LMS init should return state with specified step size."""
        optimizer = LMS(step_size=0.05)
        state = optimizer.init(feature_dim=10)

        assert state.step_size == pytest.approx(0.05)

    def test_update_computes_correct_delta(self, sample_observation):
        """LMS update should compute delta = alpha * error * x."""
        optimizer = LMS(step_size=0.1)
        state = optimizer.init(feature_dim=len(sample_observation))

        error = jnp.array(2.0)
        result = optimizer.update(state, error, sample_observation)

        expected_delta = 0.1 * 2.0 * sample_observation
        chex.assert_trees_all_close(result.weight_delta, expected_delta)
        assert result.bias_delta == pytest.approx(0.1 * 2.0)

    def test_state_unchanged_after_update(self):
        """LMS state should not change after update (fixed step-size)."""
        optimizer = LMS(step_size=0.01)
        state = optimizer.init(feature_dim=5)

        observation = jnp.ones(5)
        error = jnp.array(1.0)
        result = optimizer.update(state, error, observation)

        assert result.new_state.step_size == state.step_size


class TestIDBD:
    """Tests for the IDBD optimizer."""

    def test_init_creates_correct_state(self):
        """IDBD init should create per-weight step-sizes and traces."""
        optimizer = IDBD(initial_step_size=0.01, meta_step_size=0.001)
        state = optimizer.init(feature_dim=10)

        chex.assert_shape(state.log_step_sizes, (10,))
        chex.assert_shape(state.traces, (10,))
        chex.assert_trees_all_close(jnp.exp(state.log_step_sizes), jnp.full(10, 0.01))
        chex.assert_trees_all_close(state.traces, jnp.zeros(10))
        assert state.meta_step_size == pytest.approx(0.001)

    def test_update_returns_correct_shapes(self, sample_observation):
        """IDBD update should return correctly shaped deltas."""
        optimizer = IDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        error = jnp.array(1.0)
        result = optimizer.update(state, error, sample_observation)

        chex.assert_shape(result.weight_delta, sample_observation.shape)
        chex.assert_shape(result.new_state.log_step_sizes, sample_observation.shape)
        chex.assert_shape(result.new_state.traces, sample_observation.shape)

    def test_step_sizes_adapt_with_consistent_gradients(self):
        """Step-sizes should increase when gradients consistently agree."""
        optimizer = IDBD(initial_step_size=0.1, meta_step_size=0.1)
        feature_dim = 5
        state = optimizer.init(feature_dim=feature_dim)

        # Consistent positive error and positive observation
        observation = jnp.ones(feature_dim)
        error = jnp.array(1.0)

        initial_step_sizes = jnp.exp(state.log_step_sizes)

        # Run multiple updates with consistent gradients
        for _ in range(10):
            result = optimizer.update(state, error, observation)
            state = result.new_state

        final_step_sizes = jnp.exp(state.log_step_sizes)

        # Step-sizes should have increased due to consistent gradient direction
        # (traces build up positive correlation)
        assert jnp.mean(final_step_sizes) >= jnp.mean(initial_step_sizes)

    def test_metrics_contain_step_size_info(self, sample_observation):
        """IDBD update should return step-size statistics in metrics."""
        optimizer = IDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        error = jnp.array(1.0)
        result = optimizer.update(state, error, sample_observation)

        assert "mean_step_size" in result.metrics
        assert "min_step_size" in result.metrics
        assert "max_step_size" in result.metrics


class TestAutostep:
    """Tests for the Autostep optimizer."""

    def test_init_creates_correct_state(self):
        """Autostep init should create per-weight step-sizes, traces, and normalizers."""
        optimizer = Autostep(initial_step_size=0.01, meta_step_size=0.001)
        state = optimizer.init(feature_dim=10)

        chex.assert_shape(state.step_sizes, (10,))
        chex.assert_shape(state.traces, (10,))
        chex.assert_shape(state.normalizers, (10,))
        chex.assert_trees_all_close(state.step_sizes, jnp.full(10, 0.01))
        chex.assert_trees_all_close(state.traces, jnp.zeros(10))
        chex.assert_trees_all_close(state.normalizers, jnp.ones(10))
        assert state.meta_step_size == pytest.approx(0.001)

    def test_update_returns_correct_shapes(self, sample_observation):
        """Autostep update should return correctly shaped deltas."""
        optimizer = Autostep()
        state = optimizer.init(feature_dim=len(sample_observation))

        error = jnp.array(1.0)
        result = optimizer.update(state, error, sample_observation)

        chex.assert_shape(result.weight_delta, sample_observation.shape)
        chex.assert_shape(result.new_state.step_sizes, sample_observation.shape)
        chex.assert_shape(result.new_state.traces, sample_observation.shape)
        chex.assert_shape(result.new_state.normalizers, sample_observation.shape)

    def test_normalizers_adapt_to_gradient_magnitude(self):
        """Normalizers should increase when gradients are large."""
        optimizer = Autostep(initial_step_size=0.1, meta_step_size=0.1)
        feature_dim = 5
        state = optimizer.init(feature_dim=feature_dim)

        # Large observation should produce large gradients
        large_observation = jnp.ones(feature_dim) * 10.0
        error = jnp.array(1.0)

        initial_normalizers = state.normalizers

        result = optimizer.update(state, error, large_observation)

        # Normalizers should have increased to handle large gradients
        chex.assert_trees_all_equal_comparator(
            lambda x, y: jnp.all(x >= y),
            lambda x, y: f"Expected {x} >= {y}",
            result.new_state.normalizers,
            initial_normalizers,
        )

    def test_step_sizes_adapt_with_consistent_gradients(self):
        """Step-sizes should increase when gradients consistently agree."""
        optimizer = Autostep(initial_step_size=0.1, meta_step_size=0.1)
        feature_dim = 5
        state = optimizer.init(feature_dim=feature_dim)

        observation = jnp.ones(feature_dim)
        error = jnp.array(1.0)

        initial_step_sizes = state.step_sizes

        # Run multiple updates with consistent gradients
        for _ in range(10):
            result = optimizer.update(state, error, observation)
            state = result.new_state

        final_step_sizes = state.step_sizes

        # Step-sizes should have increased on average
        assert jnp.mean(final_step_sizes) >= jnp.mean(initial_step_sizes)

    def test_metrics_contain_normalizer_info(self, sample_observation):
        """Autostep update should return normalizer statistics in metrics."""
        optimizer = Autostep()
        state = optimizer.init(feature_dim=len(sample_observation))

        error = jnp.array(1.0)
        result = optimizer.update(state, error, sample_observation)

        assert "mean_step_size" in result.metrics
        assert "min_step_size" in result.metrics
        assert "max_step_size" in result.metrics
        assert "mean_normalizer" in result.metrics


class TestOptimizerComparison:
    """Integration tests comparing LMS, IDBD, and Autostep behavior."""

    def test_all_optimizers_produce_valid_updates(self, sample_observation):
        """All optimizers should produce finite, non-zero updates."""
        lms = LMS(step_size=0.01)
        idbd = IDBD(initial_step_size=0.01)
        autostep = Autostep(initial_step_size=0.01)

        lms_state = lms.init(len(sample_observation))
        idbd_state = idbd.init(len(sample_observation))
        autostep_state = autostep.init(len(sample_observation))

        error = jnp.array(1.0)

        lms_result = lms.update(lms_state, error, sample_observation)
        idbd_result = idbd.update(idbd_state, error, sample_observation)
        autostep_result = autostep.update(autostep_state, error, sample_observation)

        # All should produce finite updates
        chex.assert_tree_all_finite(lms_result.weight_delta)
        chex.assert_tree_all_finite(idbd_result.weight_delta)
        chex.assert_tree_all_finite(autostep_result.weight_delta)

        # All should produce non-zero updates for non-zero error
        assert jnp.any(lms_result.weight_delta != 0)
        assert jnp.any(idbd_result.weight_delta != 0)
        assert jnp.any(autostep_result.weight_delta != 0)
