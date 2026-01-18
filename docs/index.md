# Alberta Framework

A research-first framework for the Alberta Plan: Building the foundations of Continual AI.

## Overview

The Alberta Framework implements **Step 1 of the Alberta Plan**: demonstrating that IDBD (Incremental Delta-Bar-Delta) and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS on non-stationary supervised learning problems.

**Core Philosophy**: Temporal uniformity — every component updates at every time step.

## Key Features

- **Adaptive Optimizers**: IDBD and Autostep with per-weight meta-learned step-sizes
- **Non-stationary Streams**: Random walk, abrupt change, and cyclic target generators
- **Gymnasium Integration**: Wrap RL environments as prediction streams
- **Publication-Quality Analysis**: Multi-seed experiments, statistical tests, and visualization

## Quick Example

```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, run_learning_loop
from alberta_framework.streams import RandomWalkTarget

# Create a non-stationary prediction problem
stream = RandomWalkTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),
    walk_std=0.01,
)

# Train with adaptive step-sizes
learner = LinearLearner(optimizer=IDBD(initial_step_size=0.01))
state, metrics = run_learning_loop(
    learner=learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(42),
)

print(f"Final error: {metrics[-1]['squared_error']:.4f}")
```

## Installation

```bash
pip install alberta-framework
```

For development with all optional dependencies:

```bash
pip install alberta-framework[dev,gymnasium,analysis,docs]
```

## Design Principles

- **Immutable State**: All state uses NamedTuples for JAX compatibility
- **Functional Style**: Pure functions enable `jit`, `vmap`
- **Composition**: Learners accept optimizers as parameters
- **Temporal Uniformity**: Every component updates at every time step

## Project Status

This is an early-stage research framework (v0.1.0). The API may change as we progress through the Alberta Plan.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alberta_framework,
  title = {Alberta Framework},
  author = {Lawson, Keith},
  year = {2026},
  url = {https://github.com/j-klawson/alberta-framework}
}
```

## Questions & Contact

Open an issue on [GitHub](https://github.com/j-klawson/alberta-framework/issues).

## License

Apache-2.0
