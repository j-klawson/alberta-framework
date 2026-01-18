"""Tests for online feature normalization."""

import jax.numpy as jnp
import pytest

from alberta_framework import OnlineNormalizer, NormalizerState, create_normalizer_state


class TestOnlineNormalizer:
    """Tests for the OnlineNormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """OnlineNormalizer init should create state with zero mean, unit variance."""
        normalizer = OnlineNormalizer()
        state = normalizer.init(feature_dim)

        assert state.mean.shape == (feature_dim,)
        assert state.var.shape == (feature_dim,)
        assert jnp.allclose(state.mean, 0.0)
        assert jnp.allclose(state.var, 1.0)
        assert state.sample_count == 0.0

    def test_normalize_updates_statistics(self, sample_observation):
        """Normalizing should update mean and variance estimates."""
        normalizer = OnlineNormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, new_state = normalizer.normalize(state, sample_observation)

        # Count should increase
        assert new_state.sample_count == 1.0

        # Mean should have moved toward the observation
        assert not jnp.allclose(new_state.mean, state.mean)

    def test_normalize_returns_finite_values(self, sample_observation):
        """Normalized output should always be finite."""
        normalizer = OnlineNormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, _ = normalizer.normalize(state, sample_observation)

        assert jnp.all(jnp.isfinite(normalized))

    def test_normalize_only_does_not_update_state(self, sample_observation):
        """normalize_only should not modify the state."""
        normalizer = OnlineNormalizer()
        state = normalizer.init(len(sample_observation))

        # First update state
        _, state = normalizer.normalize(state, sample_observation)
        original_count = state.sample_count

        # normalize_only should not change count
        _ = normalizer.normalize_only(state, sample_observation)

        assert state.sample_count == original_count

    def test_update_only_does_not_return_normalized(self, sample_observation):
        """update_only should only update state, returning new state."""
        normalizer = OnlineNormalizer()
        state = normalizer.init(len(sample_observation))

        new_state = normalizer.update_only(state, sample_observation)

        assert isinstance(new_state, NormalizerState)
        assert new_state.sample_count == 1.0

    def test_repeated_updates_converge(self, sample_observation):
        """Mean and variance should converge with repeated identical inputs."""
        normalizer = OnlineNormalizer(decay=0.9)
        state = normalizer.init(len(sample_observation))

        # Repeatedly normalize the same observation
        for _ in range(100):
            _, state = normalizer.normalize(state, sample_observation)

        # Mean should be close to the observation
        # (not exact due to decay and numerical issues)
        assert jnp.allclose(state.mean, sample_observation, atol=0.5)

    def test_normalized_output_has_zero_mean_unit_var_asymptotically(self):
        """After many samples from standard normal, output should be ~N(0,1)."""
        import jax.random as jr

        normalizer = OnlineNormalizer(decay=0.99)
        feature_dim = 5
        state = normalizer.init(feature_dim)

        # Generate many samples
        key = jr.key(42)
        normalized_outputs = []

        for i in range(1000):
            key, subkey = jr.split(key)
            obs = jr.normal(subkey, (feature_dim,), dtype=jnp.float32)
            normalized, state = normalizer.normalize(state, obs)
            if i >= 100:  # Skip warmup
                normalized_outputs.append(normalized)

        # Stack and compute statistics
        all_normalized = jnp.stack(normalized_outputs)
        mean_of_normalized = jnp.mean(all_normalized, axis=0)
        var_of_normalized = jnp.var(all_normalized, axis=0)

        # Should be close to N(0,1)
        assert jnp.allclose(mean_of_normalized, 0.0, atol=0.3)
        assert jnp.allclose(var_of_normalized, 1.0, atol=0.5)


class TestCreateNormalizerState:
    """Tests for the create_normalizer_state convenience function."""

    def test_creates_valid_state(self):
        """Should create a valid NormalizerState."""
        state = create_normalizer_state(feature_dim=10, decay=0.95)

        assert isinstance(state, NormalizerState)
        assert state.mean.shape == (10,)
        assert state.decay == pytest.approx(0.95)

    def test_default_decay(self):
        """Default decay should be 0.99."""
        state = create_normalizer_state(feature_dim=5)
        assert state.decay == pytest.approx(0.99)
