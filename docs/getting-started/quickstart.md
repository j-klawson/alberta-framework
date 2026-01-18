# Quick Start

This guide walks through a basic example comparing LMS and IDBD on a non-stationary prediction problem.

## Basic Concepts

The framework has three core components:

1. **Streams**: Generate experience (observations and targets)
2. **Optimizers**: Compute weight updates (LMS, IDBD, Autostep)
3. **Learners**: Combine predictions with optimization

## Your First Experiment

```python
import jax.random as jr
from alberta_framework import (
    LinearLearner,
    LMS,
    IDBD,
    run_learning_loop,
)
from alberta_framework.streams import RandomWalkTarget
from alberta_framework.utils import compute_tracking_error

# Create a non-stationary stream where true weights drift over time
stream = RandomWalkTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),
    walk_std=0.01,  # How fast the target drifts
    noise_std=0.1,  # Observation noise
)

# Train with fixed step-size (LMS)
lms_learner = LinearLearner(optimizer=LMS(step_size=0.01))
lms_state, lms_metrics = run_learning_loop(
    learner=lms_learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(42),
)

# Reset stream for fair comparison
stream = RandomWalkTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),  # Same seed!
    walk_std=0.01,
    noise_std=0.1,
)

# Train with adaptive step-sizes (IDBD)
idbd_learner = LinearLearner(optimizer=IDBD(initial_step_size=0.01))
idbd_state, idbd_metrics = run_learning_loop(
    learner=idbd_learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(42),
)

# Compare tracking error
lms_error = compute_tracking_error(lms_metrics, window=1000)
idbd_error = compute_tracking_error(idbd_metrics, window=1000)

print(f"LMS final tracking error:  {lms_error:.4f}")
print(f"IDBD final tracking error: {idbd_error:.4f}")
```

## Understanding the Results

- **Tracking error** measures how well the learner follows the drifting target
- IDBD adapts its step-sizes to the problem, often outperforming fixed LMS
- The key is that IDBD doesn't require manual step-size tuning

## Next Steps

- Learn about [Optimizers](../guide/optimizers.md) in detail
- Explore [Streams](../guide/streams.md) for different non-stationary patterns
- Set up [Experiments](../guide/experiments.md) with multiple seeds
