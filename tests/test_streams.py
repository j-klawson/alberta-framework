"""Tests for experience streams."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import (
    AbruptChangeStream,
    CyclicStream,
    RandomWalkStream,
    SuttonExperiment1Stream,
    TimeStep,
)


class TestRandomWalkStream:
    """Tests for the RandomWalkStream class."""

    def test_init_creates_valid_state(self, rng_key):
        """Stream init should create valid state with correct shapes."""
        stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)
        state = stream.init(rng_key)

        assert state.key is not None
        assert state.true_weights.shape == (10,)

    def test_step_produces_valid_timestep(self, rng_key):
        """Step should produce valid observation and target."""
        stream = RandomWalkStream(feature_dim=10)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        assert timestep.observation.shape == (10,)
        assert timestep.target.shape == (1,)
        assert jnp.all(jnp.isfinite(timestep.observation))
        assert jnp.all(jnp.isfinite(timestep.target))

    def test_feature_dim_property(self):
        """Feature dim property should return correct dimension."""
        stream = RandomWalkStream(feature_dim=20)
        assert stream.feature_dim == 20

    def test_weights_drift_over_time(self, rng_key):
        """True weights should change from step to step."""
        stream = RandomWalkStream(feature_dim=10, drift_rate=0.1)  # High drift
        state = stream.init(rng_key)

        initial_weights = state.true_weights

        for i in range(10):
            _, state = stream.step(state, jnp.array(i))

        # Weights should have changed
        assert not jnp.allclose(initial_weights, state.true_weights)

    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same sequence."""
        stream = RandomWalkStream(feature_dim=10)

        state1 = stream.init(rng_key)
        timestep1, _ = stream.step(state1, jnp.array(0))

        state2 = stream.init(rng_key)
        timestep2, _ = stream.step(state2, jnp.array(0))

        assert jnp.allclose(timestep1.observation, timestep2.observation)
        assert jnp.allclose(timestep1.target, timestep2.target)

    def test_targets_are_non_constant(self, rng_key):
        """Targets should vary due to random features and noise."""
        stream = RandomWalkStream(feature_dim=10)
        state = stream.init(rng_key)

        targets = []
        for i in range(100):
            timestep, state = stream.step(state, jnp.array(i))
            targets.append(float(timestep.target[0]))

        # Targets should not all be the same
        assert len(set(targets)) > 1


class TestAbruptChangeStream:
    """Tests for the AbruptChangeStream class."""

    def test_weights_change_at_interval(self, rng_key):
        """Weights should change at specified interval."""
        stream = AbruptChangeStream(feature_dim=10, change_interval=10)
        state = stream.init(rng_key)

        # Step count starts at 0, changes happen when step_count % interval == 0
        # So first change at step 0 (initial), then step 10, 20, etc.
        weights_at_0 = state.true_weights.copy()

        # Run 9 steps (step_count goes 0->9)
        for i in range(9):
            _, state = stream.step(state, jnp.array(i))

        weights_at_9 = state.true_weights

        # Run one more step (step_count becomes 10)
        _, state = stream.step(state, jnp.array(9))

        weights_at_10 = state.true_weights

        # Weights should have changed at step 10
        assert not jnp.allclose(weights_at_0, weights_at_10)

    def test_generates_valid_timesteps(self, rng_key):
        """Should generate valid TimeStep instances."""
        stream = AbruptChangeStream(feature_dim=5)
        state = stream.init(rng_key)

        for i in range(50):
            timestep, state = stream.step(state, jnp.array(i))
            assert isinstance(timestep, TimeStep)
            assert jnp.all(jnp.isfinite(timestep.observation))


class TestSuttonExperiment1Stream:
    """Tests for the SuttonExperiment1Stream class."""

    def test_correct_feature_dim(self):
        """Feature dim should be num_relevant + num_irrelevant."""
        stream = SuttonExperiment1Stream(num_relevant=5, num_irrelevant=15)
        assert stream.feature_dim == 20

    def test_initial_signs_are_positive(self, rng_key):
        """All initial signs should be +1."""
        stream = SuttonExperiment1Stream()
        state = stream.init(rng_key)

        assert jnp.all(state.signs == 1.0)

    def test_sign_flips_at_interval(self, rng_key):
        """One sign should flip every change_interval steps."""
        stream = SuttonExperiment1Stream(change_interval=20)
        state = stream.init(rng_key)

        initial_signs = state.signs.copy()

        # Step past the change interval
        for i in range(21):
            _, state = stream.step(state, jnp.array(i))

        # At least one sign should have changed
        assert not jnp.allclose(initial_signs, state.signs)

        # Exactly one sign should be different
        num_changes = jnp.sum(initial_signs != state.signs)
        assert num_changes == 1

    def test_target_only_depends_on_relevant_inputs(self, rng_key):
        """Target should only depend on first num_relevant inputs."""
        stream = SuttonExperiment1Stream(num_relevant=5, num_irrelevant=15)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        # At step 0, no flip happens (step_count > 0 check), so signs remain all 1
        # Target = sum of first 5 inputs (weighted by signs which are all 1)
        expected = jnp.sum(timestep.observation[:5])

        assert jnp.isclose(timestep.target[0], expected, rtol=1e-5)


class TestCyclicStream:
    """Tests for the CyclicStream class."""

    def test_cycles_through_configurations(self, rng_key):
        """Should cycle through configurations."""
        stream = CyclicStream(
            feature_dim=10,
            cycle_length=5,
            num_configurations=4,
        )
        state = stream.init(rng_key)

        # Track which configuration index is used
        config_indices = []
        for i in range(25):  # Go through all 4 configs plus more
            config_idx = (state.step_count // 5) % 4
            config_indices.append(int(config_idx))
            _, state = stream.step(state, jnp.array(i))

        # Should see all 4 configurations
        assert 0 in config_indices
        assert 1 in config_indices
        assert 2 in config_indices
        assert 3 in config_indices

    def test_same_config_produces_consistent_weights(self, rng_key):
        """Same configuration should use same weights."""
        stream = CyclicStream(
            feature_dim=10,
            cycle_length=10,
            num_configurations=2,
            noise_std=0.0,  # No noise for easier testing
        )
        state = stream.init(rng_key)

        # Get the stored configurations
        config0_weights = state.configurations[0]

        # After one full cycle, we should be back to config 0
        for i in range(20):  # Go through both configs
            _, state = stream.step(state, jnp.array(i))

        # Config 0 weights should be unchanged (stored in configurations)
        assert jnp.allclose(config0_weights, state.configurations[0])

    def test_generates_valid_timesteps(self, rng_key):
        """Should generate valid TimeStep instances."""
        stream = CyclicStream(feature_dim=5)
        state = stream.init(rng_key)

        for i in range(50):
            timestep, state = stream.step(state, jnp.array(i))
            assert isinstance(timestep, TimeStep)
            assert jnp.all(jnp.isfinite(timestep.observation))
