"""Tests for experience streams."""

import jax.numpy as jnp
import pytest

from alberta_toolkit import (
    AbruptChangeTarget,
    CyclicTarget,
    RandomWalkTarget,
    TimeStep,
)


class TestRandomWalkTarget:
    """Tests for RandomWalkTarget stream."""

    def test_feature_dim_property(self):
        """Stream should report correct feature dimension."""
        stream = RandomWalkTarget(feature_dim=10)
        assert stream.feature_dim == 10

    def test_generates_timesteps(self):
        """Stream should generate TimeStep instances."""
        stream = RandomWalkTarget(feature_dim=5)
        timestep = next(stream)

        assert isinstance(timestep, TimeStep)
        assert timestep.observation.shape == (5,)
        assert timestep.target.shape == (1,)

    def test_observations_are_finite(self):
        """Generated observations should be finite."""
        stream = RandomWalkTarget(feature_dim=10, seed=42)

        for i, timestep in enumerate(stream):
            if i >= 100:
                break
            assert jnp.all(jnp.isfinite(timestep.observation))
            assert jnp.all(jnp.isfinite(timestep.target))

    def test_targets_are_non_constant(self):
        """Targets should vary due to random features and noise."""
        stream = RandomWalkTarget(feature_dim=10, seed=42)

        targets = []
        for i, timestep in enumerate(stream):
            if i >= 100:
                break
            targets.append(float(timestep.target[0]))

        # Targets should not all be the same
        assert len(set(targets)) > 1

    def test_true_weights_drift(self):
        """True weights should change due to random walk."""
        stream = RandomWalkTarget(feature_dim=5, drift_rate=0.1, seed=42)

        _ = next(stream)
        initial_weights = stream.true_weights.copy()

        for _ in range(100):
            _ = next(stream)

        final_weights = stream.true_weights

        # Weights should have drifted
        assert not jnp.allclose(initial_weights, final_weights)

    def test_reproducible_with_seed(self):
        """Same seed should produce same sequence."""
        stream1 = RandomWalkTarget(feature_dim=5, seed=123)
        stream2 = RandomWalkTarget(feature_dim=5, seed=123)

        for i in range(10):
            ts1 = next(stream1)
            ts2 = next(stream2)
            assert jnp.allclose(ts1.observation, ts2.observation)
            assert jnp.allclose(ts1.target, ts2.target)


class TestAbruptChangeTarget:
    """Tests for AbruptChangeTarget stream."""

    def test_weights_change_at_interval(self):
        """Weights should change abruptly at specified intervals."""
        stream = AbruptChangeTarget(
            feature_dim=5,
            change_interval=10,
            seed=42,
        )

        # Get weights at start
        _ = next(stream)
        weights_at_0 = stream.true_weights.copy()

        # Advance to step 9 (still same configuration)
        for _ in range(9):
            _ = next(stream)
        weights_at_9 = stream.true_weights.copy()

        # Step 10 should trigger change
        _ = next(stream)
        weights_at_10 = stream.true_weights.copy()

        # Steps 0-9 should have same weights
        assert jnp.allclose(weights_at_0, weights_at_9)

        # Step 10 should have different weights
        assert not jnp.allclose(weights_at_9, weights_at_10)

    def test_generates_valid_timesteps(self):
        """Should generate valid TimeStep instances."""
        stream = AbruptChangeTarget(feature_dim=5)

        for i, timestep in enumerate(stream):
            if i >= 50:
                break
            assert isinstance(timestep, TimeStep)
            assert jnp.all(jnp.isfinite(timestep.observation))


class TestCyclicTarget:
    """Tests for CyclicTarget stream."""

    def test_cycles_through_configurations(self):
        """Weights should cycle through configurations."""
        stream = CyclicTarget(
            feature_dim=5,
            cycle_length=10,
            num_configurations=3,
            seed=42,
        )

        # Collect configuration indices over multiple cycles
        indices = []
        for i, _ in enumerate(stream):
            if i >= 60:  # Two full cycles
                break
            indices.append(stream.current_configuration_index)

        # After calling next(), step_count is incremented, so index uses (step+1)
        # Should cycle: 0,0,...,0 (9 times), 1,1,...,1 (10 times), etc.
        expected = [(i + 1) // 10 % 3 for i in range(60)]
        assert indices == expected

    def test_returns_to_previous_configurations(self):
        """Weights should return to exact previous values when cycling."""
        stream = CyclicTarget(
            feature_dim=5,
            cycle_length=5,
            num_configurations=2,
            seed=42,
        )

        # Get weights for config 0
        _ = next(stream)
        config0_weights = stream.true_weights.copy()

        # Skip to config 1, then back to config 0
        for _ in range(9):  # Steps 1-9
            _ = next(stream)

        # Step 10 should be back to config 0
        _ = next(stream)
        config0_again = stream.true_weights

        assert jnp.allclose(config0_weights, config0_again)

    def test_generates_valid_timesteps(self):
        """Should generate valid TimeStep instances."""
        stream = CyclicTarget(feature_dim=5)

        for i, timestep in enumerate(stream):
            if i >= 50:
                break
            assert isinstance(timestep, TimeStep)
            assert jnp.all(jnp.isfinite(timestep.observation))


class TestStreamIterator:
    """Tests for stream iterator behavior."""

    def test_can_use_in_for_loop(self):
        """Streams should work with Python for loops."""
        stream = RandomWalkTarget(feature_dim=5)

        count = 0
        for timestep in stream:
            count += 1
            if count >= 10:
                break

        assert count == 10

    def test_iter_returns_self(self):
        """__iter__ should return self."""
        stream = RandomWalkTarget(feature_dim=5)
        assert iter(stream) is stream
