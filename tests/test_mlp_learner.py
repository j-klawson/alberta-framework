"""Tests for the MLPLearner and run_mlp_learning_loop."""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import (
    BatchedMLPNormalizedResult,
    BatchedMLPResult,
    EMANormalizer,
    MLPLearner,
    NormalizedMLPLearner,
    NormalizerTrackingConfig,
    RandomWalkStream,
    WelfordNormalizer,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
    run_mlp_normalized_learning_loop,
    run_mlp_normalized_learning_loop_batched,
)


class TestMLPLearner:
    """Tests for the MLPLearner class."""

    def test_correct_param_shapes_single_hidden(self):
        """MLP with one hidden layer should have correct param shapes."""
        learner = MLPLearner(hidden_sizes=(32,), sparsity=0.0)
        state = learner.init(feature_dim=10, key=jr.key(42))

        # Layer 0: 10 -> 32
        chex.assert_shape(state.params.weights[0], (32, 10))
        chex.assert_shape(state.params.biases[0], (32,))
        # Layer 1: 32 -> 1
        chex.assert_shape(state.params.weights[1], (1, 32))
        chex.assert_shape(state.params.biases[1], (1,))

        assert len(state.params.weights) == 2
        assert len(state.params.biases) == 2

    def test_correct_param_shapes_two_hidden(self):
        """MLP with two hidden layers should have correct param shapes."""
        learner = MLPLearner(hidden_sizes=(64, 32), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        # Layer 0: 5 -> 64
        chex.assert_shape(state.params.weights[0], (64, 5))
        chex.assert_shape(state.params.biases[0], (64,))
        # Layer 1: 64 -> 32
        chex.assert_shape(state.params.weights[1], (32, 64))
        chex.assert_shape(state.params.biases[1], (32,))
        # Layer 2: 32 -> 1
        chex.assert_shape(state.params.weights[2], (1, 32))
        chex.assert_shape(state.params.biases[2], (1,))

        assert len(state.params.weights) == 3

    def test_predict_returns_scalar(self):
        """Predict should return a 1-d array (scalar prediction)."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        prediction = learner.predict(state, observation)

        chex.assert_shape(prediction, (1,))
        chex.assert_tree_all_finite(prediction)

    def test_update_returns_correct_result(self):
        """Update should return MLPUpdateResult with correct shapes."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)

        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (3,))
        chex.assert_tree_all_finite(result.metrics)

        # State should have same structure
        assert len(result.state.params.weights) == len(state.params.weights)

    def test_update_reduces_error(self):
        """Multiple updates on a fixed target should reduce error."""
        learner = MLPLearner(hidden_sizes=(16,), step_size=0.1, kappa=2.0, sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])
        target = jnp.array([2.0])

        initial_error = abs(float(learner.predict(state, observation)[0]) - 2.0)

        # Run several updates
        for _ in range(50):
            result = learner.update(state, observation, target)
            state = result.state

        final_error = abs(float(learner.predict(state, observation)[0]) - 2.0)

        # Error should decrease
        assert final_error < initial_error

    def test_deterministic_with_same_key(self):
        """Same key should produce identical initial states."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.5)

        state1 = learner.init(feature_dim=5, key=jr.key(42))
        state2 = learner.init(feature_dim=5, key=jr.key(42))

        for w1, w2 in zip(state1.params.weights, state2.params.weights):
            chex.assert_trees_all_close(w1, w2)

    def test_sparse_init_applied(self):
        """Weights should be sparse when sparsity > 0."""
        learner = MLPLearner(hidden_sizes=(32,), sparsity=0.9)
        state = learner.init(feature_dim=10, key=jr.key(42))

        # First layer weights should be ~90% sparse
        zeros = jnp.sum(state.params.weights[0] == 0)
        total = state.params.weights[0].size
        sparsity = float(zeros) / total

        assert sparsity > 0.85  # Allow some tolerance

    def test_biases_initialized_to_zero(self):
        """All biases should be initialized to zero."""
        learner = MLPLearner(hidden_sizes=(32, 16), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        for bias in state.params.biases:
            chex.assert_trees_all_close(bias, jnp.zeros_like(bias))

    def test_traces_initialized_to_zero(self):
        """All optimizer traces should be initialized to zero."""
        learner = MLPLearner(hidden_sizes=(32,))
        state = learner.init(feature_dim=5, key=jr.key(42))

        for wt in state.optimizer_state.weight_traces:
            chex.assert_trees_all_close(wt, jnp.zeros_like(wt))
        for bt in state.optimizer_state.bias_traces:
            chex.assert_trees_all_close(bt, jnp.zeros_like(bt))


class TestRunMLPLearningLoop:
    """Tests for the run_mlp_learning_loop function."""

    def test_scan_loop_produces_correct_shapes(self):
        """Scan loop should return metrics with shape (num_steps, 3)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 3))
        chex.assert_tree_all_finite(metrics)

        # State should have correct param shapes
        chex.assert_shape(state.params.weights[0], (16, 5))
        chex.assert_shape(state.params.weights[1], (1, 16))

    def test_scan_loop_deterministic(self):
        """Same key should produce identical results."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)

        _, metrics1 = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        _, metrics2 = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )

        chex.assert_trees_all_close(metrics1, metrics2)

    def test_scan_loop_with_provided_state(self):
        """Should accept a pre-initialized state."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        initial_state = learner.init(feature_dim=5, key=jr.key(0))

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42),
            learner_state=initial_state,
        )

        chex.assert_shape(metrics, (50, 3))
        chex.assert_tree_all_finite(metrics)


class TestBatchedMLPLearningLoop:
    """Tests for the run_mlp_learning_loop_batched function."""

    def test_batched_returns_correct_shapes(self):
        """Batched loop should return metrics with shape (num_seeds, num_steps, 3)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        num_seeds = 4
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 3))
        chex.assert_tree_all_finite(result.metrics)

        # Check batched param shapes
        chex.assert_shape(result.states.params.weights[0], (num_seeds, 16, 5))
        chex.assert_shape(result.states.params.weights[1], (num_seeds, 1, 16))
        chex.assert_shape(result.states.params.biases[0], (num_seeds, 16))
        chex.assert_shape(result.states.params.biases[1], (num_seeds, 1))

    def test_batched_matches_sequential(self):
        """Batched results should match sequential results for each seed."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        num_seeds = 3
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)

        # Run batched
        batched_result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        # Run sequential
        for i in range(num_seeds):
            state_i, metrics_i = run_mlp_learning_loop(
                learner, stream, num_steps=num_steps, key=keys[i]
            )
            chex.assert_trees_all_close(
                batched_result.metrics[i], metrics_i, rtol=1e-4
            )

    def test_batched_deterministic(self):
        """Same keys should produce identical batched results."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)

        keys = jr.split(jr.key(42), 3)

        result1 = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )
        result2 = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )

        chex.assert_trees_all_close(result1.metrics, result2.metrics)

    def test_batched_different_keys_different_results(self):
        """Different seeds should produce different metrics."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)

        keys = jr.split(jr.key(42), 3)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )

        # Different seeds should give different final metrics
        assert not jnp.allclose(result.metrics[0], result.metrics[1])
        assert not jnp.allclose(result.metrics[0], result.metrics[2])


class TestNormalizedMLPLearner:
    """Tests for the NormalizedMLPLearner class."""

    def test_correct_param_shapes(self):
        """NormalizedMLPLearner should have correct param shapes."""
        mlp = MLPLearner(hidden_sizes=(32,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        state = learner.init(feature_dim=10, key=jr.key(42))

        # MLP layer shapes
        chex.assert_shape(state.learner_state.params.weights[0], (32, 10))
        chex.assert_shape(state.learner_state.params.biases[0], (32,))
        chex.assert_shape(state.learner_state.params.weights[1], (1, 32))
        chex.assert_shape(state.learner_state.params.biases[1], (1,))

        # Normalizer state
        chex.assert_shape(state.normalizer_state.mean, (10,))
        chex.assert_shape(state.normalizer_state.var, (10,))

    def test_predict_returns_scalar(self):
        """Predict should return a 1-d array."""
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        prediction = learner.predict(state, observation)

        chex.assert_shape(prediction, (1,))
        chex.assert_tree_all_finite(prediction)

    def test_update_returns_correct_shapes(self):
        """Update should return NormalizedMLPUpdateResult with 4-column metrics."""
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)

        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (4,))
        chex.assert_tree_all_finite(result.metrics)

    def test_normalizer_state_updates(self):
        """Normalizer state should change after an update."""
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)

        # Mean should have changed from zeros
        assert not jnp.allclose(
            result.state.normalizer_state.mean, state.normalizer_state.mean
        )

    def test_works_with_ema_normalizer(self):
        """Should work with EMANormalizer."""
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp, normalizer=EMANormalizer(decay=0.95))
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)
        chex.assert_shape(result.metrics, (4,))
        chex.assert_tree_all_finite(result.metrics)

    def test_works_with_welford_normalizer(self):
        """Should work with WelfordNormalizer."""
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp, normalizer=WelfordNormalizer())
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)
        chex.assert_shape(result.metrics, (4,))
        chex.assert_tree_all_finite(result.metrics)


class TestRunMLPNormalizedLearningLoop:
    """Tests for the run_mlp_normalized_learning_loop function."""

    def test_scan_loop_produces_correct_shapes(self):
        """Scan loop should return metrics with shape (num_steps, 4)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)

        state, metrics = run_mlp_normalized_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 4))
        chex.assert_tree_all_finite(metrics)

        # State should have correct param shapes
        chex.assert_shape(state.learner_state.params.weights[0], (16, 5))
        chex.assert_shape(state.learner_state.params.weights[1], (1, 16))

    def test_scan_loop_deterministic(self):
        """Same key should produce identical results."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)

        _, metrics1 = run_mlp_normalized_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        _, metrics2 = run_mlp_normalized_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )

        chex.assert_trees_all_close(metrics1, metrics2)

    def test_tracking_returns_3_tuple(self):
        """With normalizer tracking, should return 3-tuple."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        config = NormalizerTrackingConfig(interval=10)

        result = run_mlp_normalized_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42),
            normalizer_tracking=config,
        )

        assert len(result) == 3
        state, metrics, norm_history = result

        chex.assert_shape(metrics, (100, 4))
        chex.assert_shape(norm_history.means, (10, 5))
        chex.assert_shape(norm_history.variances, (10, 5))
        chex.assert_shape(norm_history.recording_indices, (10,))

    def test_invalid_interval_raises(self):
        """Invalid tracking interval should raise ValueError."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)

        with pytest.raises(ValueError, match="must be >= 1"):
            run_mlp_normalized_learning_loop(
                learner, stream, num_steps=100, key=jr.key(42),
                normalizer_tracking=NormalizerTrackingConfig(interval=0),
            )

        with pytest.raises(ValueError, match="must be <= num_steps"):
            run_mlp_normalized_learning_loop(
                learner, stream, num_steps=100, key=jr.key(42),
                normalizer_tracking=NormalizerTrackingConfig(interval=200),
            )

    def test_with_provided_state(self):
        """Should accept a pre-initialized state."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        initial_state = learner.init(feature_dim=5, key=jr.key(0))

        state, metrics = run_mlp_normalized_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42),
            learner_state=initial_state,
        )

        chex.assert_shape(metrics, (50, 4))
        chex.assert_tree_all_finite(metrics)


class TestBatchedMLPNormalizedLearningLoop:
    """Tests for the run_mlp_normalized_learning_loop_batched function."""

    def test_batched_returns_correct_shapes(self):
        """Batched loop should return metrics with shape (num_seeds, num_steps, 4)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        num_seeds = 4
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_normalized_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        assert isinstance(result, BatchedMLPNormalizedResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 4))
        chex.assert_tree_all_finite(result.metrics)
        assert result.normalizer_history is None

        # Check batched param shapes
        chex.assert_shape(
            result.states.learner_state.params.weights[0], (num_seeds, 16, 5)
        )
        chex.assert_shape(
            result.states.learner_state.params.weights[1], (num_seeds, 1, 16)
        )

    def test_batched_matches_sequential(self):
        """Batched results should match sequential results for each seed."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        num_seeds = 3
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)

        # Run batched
        batched_result = run_mlp_normalized_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        # Run sequential
        for i in range(num_seeds):
            state_i, metrics_i = run_mlp_normalized_learning_loop(
                learner, stream, num_steps=num_steps, key=keys[i]
            )
            chex.assert_trees_all_close(
                batched_result.metrics[i], metrics_i, rtol=1e-4
            )

    def test_batched_with_tracking(self):
        """Batched with tracking should return correct shapes."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        mlp = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner = NormalizedMLPLearner(mlp)
        num_seeds = 3
        num_steps = 50
        config = NormalizerTrackingConfig(interval=10)

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_normalized_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys,
            normalizer_tracking=config,
        )

        assert isinstance(result, BatchedMLPNormalizedResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 4))
        assert result.normalizer_history is not None
        chex.assert_shape(result.normalizer_history.means, (num_seeds, 5, 5))
        chex.assert_shape(result.normalizer_history.variances, (num_seeds, 5, 5))
        chex.assert_shape(result.normalizer_history.recording_indices, (num_seeds, 5))
