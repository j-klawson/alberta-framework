"""Pytest configuration and fixtures for Alberta Toolkit tests."""

import pytest
import jax.numpy as jnp
import jax.random as jr


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX random key."""
    return jr.key(42)


@pytest.fixture
def feature_dim():
    """Default feature dimension for tests."""
    return 10


@pytest.fixture
def sample_observation(feature_dim, rng_key):
    """Generate a sample observation vector."""
    return jr.normal(rng_key, (feature_dim,), dtype=jnp.float32)


@pytest.fixture
def sample_target():
    """Generate a sample target value."""
    return jnp.array([1.5], dtype=jnp.float32)
