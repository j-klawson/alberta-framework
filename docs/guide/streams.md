# Experience Streams

Streams generate the data for learning experiments. All streams implement the `ExperienceStream` protocol.

## Synthetic Streams

These streams generate non-stationary supervised learning problems.

### RandomWalkTarget

The true weights drift according to a random walk:

\[
w^*_{t+1} = w^*_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2_{\text{walk}})
\]

```python
from alberta_framework.streams import RandomWalkTarget
import jax.random as jr

stream = RandomWalkTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),
    walk_std=0.01,      # Drift speed
    noise_std=0.1,      # Observation noise
    feature_std=1.0,    # Feature scale
)
```

**Use case**: Continuous, gradual non-stationarity.

### AbruptChangeTarget

The true weights change abruptly at random intervals:

```python
from alberta_framework.streams import AbruptChangeTarget

stream = AbruptChangeTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),
    change_prob=0.001,   # Probability of change per step
    noise_std=0.1,
)
```

**Use case**: Concept drift, sudden distribution shifts.

### CyclicTarget

The true weights cycle through a fixed set of configurations:

```python
from alberta_framework.streams import CyclicTarget

stream = CyclicTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),
    num_configurations=4,   # Number of weight sets
    steps_per_config=1000,  # Steps before switching
    noise_std=0.1,
)
```

**Use case**: Periodic patterns, recurring tasks.

## Stream Protocol

All streams implement:

```python
class ExperienceStream(Protocol):
    @property
    def feature_dim(self) -> int:
        """Dimension of observation vectors."""
        ...

    def __iter__(self) -> Iterator[TimeStep]:
        """Return self as iterator."""
        ...

    def __next__(self) -> TimeStep:
        """Generate next experience."""
        ...
```

## TimeStep

Each stream yields `TimeStep` objects:

```python
from alberta_framework.core.types import TimeStep

# TimeStep fields
timestep.observation  # jax.Array of shape (feature_dim,)
timestep.target       # jax.Array of shape () or (1,)
timestep.reward       # Optional[jax.Array], for RL streams
```

## Creating Custom Streams

Implement the `ExperienceStream` protocol:

```python
from typing import Iterator
from alberta_framework.core.types import TimeStep
from alberta_framework.streams.base import ExperienceStream
import jax.numpy as jnp
import jax.random as jr

class SinusoidalTarget(ExperienceStream):
    """Target follows a sinusoidal pattern."""

    def __init__(self, feature_dim: int, key: jr.PRNGKey, period: int = 1000):
        self._feature_dim = feature_dim
        self._key = key
        self._period = period
        self._step = 0

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __iter__(self) -> Iterator[TimeStep]:
        return self

    def __next__(self) -> TimeStep:
        self._key, k1, k2 = jr.split(self._key, 3)

        # Generate observation
        observation = jr.normal(k1, (self._feature_dim,))

        # Sinusoidal target
        phase = 2 * jnp.pi * self._step / self._period
        target = jnp.sin(phase) + 0.1 * jr.normal(k2, ())

        self._step += 1

        return TimeStep(
            observation=observation,
            target=target,
            reward=None,
        )
```

## Combining with Learners

Streams integrate with the learning loop:

```python
from alberta_framework import LinearLearner, IDBD, run_learning_loop

stream = RandomWalkTarget(feature_dim=10, key=jr.PRNGKey(0))
learner = LinearLearner(optimizer=IDBD())

state, metrics = run_learning_loop(
    learner=learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(42),
)
```
