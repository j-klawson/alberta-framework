# Core Concepts

This guide explains the foundational concepts of the Alberta Framework.

## The Alberta Plan

The Alberta Plan is a roadmap for building continual learning AI systems. This framework implements **Step 1**: demonstrating that adaptive step-size methods can match or beat hand-tuned baselines on non-stationary supervised learning.

## Temporal Uniformity

The framework's core principle is **temporal uniformity**: every component updates at every time step. This means:

- No batch processing
- No epochs or passes over data
- Learning happens incrementally, one sample at a time

This reflects the reality of continual learning where data arrives as a stream.

## Key Components

### Experience Streams

Streams generate `TimeStep` objects containing:

- `observation`: Feature vector \(x_t \in \mathbb{R}^d\)
- `target`: Value to predict \(y_t \in \mathbb{R}\)
- `reward`: Optional reward signal (for RL streams)

```python
from alberta_framework.streams import RandomWalkTarget

stream = RandomWalkTarget(feature_dim=10, key=jr.PRNGKey(0))
for timestep in stream:
    print(f"x: {timestep.observation.shape}, y: {timestep.target}")
```

### Optimizers

Optimizers compute weight updates given a prediction error:

| Optimizer | Description |
|-----------|-------------|
| **LMS** | Fixed step-size baseline |
| **IDBD** | Per-weight adaptive step-sizes via gradient correlation |
| **Autostep** | Tuning-free adaptation with gradient normalization |

All optimizers follow the `Optimizer` protocol:

```python
class Optimizer(Protocol):
    def init(self, feature_dim: int) -> State: ...
    def update(self, state: State, error: Array, observation: Array) -> OptimizerUpdate: ...
```

### Learners

Learners combine a prediction model with an optimizer:

```python
from alberta_framework import LinearLearner, IDBD

learner = LinearLearner(optimizer=IDBD())
state = learner.init(feature_dim=10, key=jr.PRNGKey(0))
prediction = learner.predict(state, observation)
result = learner.update(state, error, observation)
```

The `NormalizedLinearLearner` adds online feature normalization.

## Immutable State

All state is represented as immutable `NamedTuple` objects:

- `LearnerState`: Weights and optimizer state
- `LMSState`, `IDBDState`, `AutostepState`: Optimizer-specific state
- `NormalizerState`: Running statistics for normalization

This design enables:

- JAX transformations (`jit`, `vmap`, `grad`)
- Reproducible experiments
- Easy serialization

## The Learning Loop

The `run_learning_loop` function encapsulates the training process:

```python
state, metrics = run_learning_loop(
    learner=learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(42),
)
```

Each step:
1. Gets the next `TimeStep` from the stream
2. Makes a prediction
3. Computes the error
4. Updates weights and optimizer state
5. Records metrics

## Metrics

The learning loop returns a list of metric dictionaries:

```python
metrics[-1]  # Last step metrics
# {
#     'squared_error': 0.0123,
#     'mean_step_size': 0.015,  # For adaptive optimizers
#     ...
# }
```

Use `compute_tracking_error` to aggregate over a window:

```python
from alberta_framework.utils import compute_tracking_error

final_error = compute_tracking_error(metrics, window=1000)
```
