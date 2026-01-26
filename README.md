# Alberta Framework

[![CI](https://github.com/j-klawson/alberta-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/j-klawson/alberta-framework/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/alberta-framework.svg)](https://pypi.org/project/alberta-framework/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A JAX-based research framework implementing components of [The Alberta Plan](https://arxiv.org/abs/2208.11173) in the pursuit of building the foundations of Continual AI.

> "The agents are complex only because they interact with a complex world... their initial design is as simple, general, and scalable as possible." — *Sutton et al., 2022*

## Overview

The Alberta Framework provides foundational components for continual reinforcement learning research. Built on JAX for hardware acceleration, the framework emphasizes temporal uniformity every component updates at every time step, with no special training phases or batch processing.

## Project Context

This framework is developed as part of my D.Eng. work focusing on the foundations of Continual AI. For more background and context see:

* **Research Blog**: [blog.9600baud.net](https://blog.9600baud.net)
* **Replicating Sutton '92**: [The Foundation of Step-size Adaptation](https://blog.9600baud.net/sutton92.html)
* **About the Author**: [Keith Lawson](https://blog.9600baud.net/about.html)

### Roadmap

Depending on my research trajectory I may or may not implement components required for the plan. The current focus of this framework is the Step 1 Baseline Study, investigating the interaction between adaptive optimizers and online normalization.

| Step | Focus | Status |
|------|-------|--------|
| 1 | Meta-learned step-sizes (IDBD, Autostep) | **Complete** |
| 2 | Feature generation and testing | Planned |
| 3 | GVF predictions, Horde architecture | Planned |
| 4 | Actor-critic with eligibility traces | Planned |
| 5-6 | Off-policy learning, average reward | Planned |
| 7-12 | Hierarchical, multi-agent, world models | Future |

## Installation

```bash
pip install alberta-framework

# With optional dependencies
pip install alberta-framework[gymnasium]  # RL environment support
pip install alberta-framework[dev]        # Development (pytest, ruff)
```

**Requirements:** Python >= 3.13, JAX >= 0.4, NumPy >= 2.0

## Quick Start

```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop

# Non-stationary stream where target weights drift over time
stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# Learner with IDBD meta-learned step-sizes
learner = LinearLearner(optimizer=IDBD())

# JIT-compiled training via jax.lax.scan
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
```

## Core Components

### Optimizers

- **LMS**: Fixed step-size baseline
- **IDBD**: Per-weight adaptive step-sizes via gradient correlation (Sutton, 1992)
- **Autostep**: Tuning-free adaptation with gradient normalization (Mahmood et al., 2012)

### Streams

Non-stationary experience generators implementing the `ScanStream` protocol:

- `RandomWalkStream`: Gradual target drift
- `AbruptChangeStream`: Sudden target switches
- `PeriodicChangeStream`: Sinusoidal oscillation
- `DynamicScaleShiftStream`: Time-varying feature scales

### Gymnasium Integration

```python
from alberta_framework.streams.gymnasium import collect_trajectory, learn_from_trajectory, PredictionMode
import gymnasium as gym

env = gym.make("CartPole-v1")
observations, targets = collect_trajectory(env, policy, num_steps=10000, mode=PredictionMode.REWARD)
state, metrics = learn_from_trajectory(learner, observations, targets)
```

### Publication Tools

Multi-seed experiments with statistical analysis and publication-ready outputs:

```python
from alberta_framework.utils import ExperimentConfig, run_multi_seed_experiment, pairwise_comparisons

results = run_multi_seed_experiment(configs, seeds=30, parallel=True)
significance = pairwise_comparisons(results, test="ttest", correction="bonferroni")
```

## Documentation

Full documentation available at [j-klawson.github.io/alberta-framework](https://j-klawson.github.io/alberta-framework) or build locally:

```bash
pip install alberta-framework[docs]
mkdocs serve  # http://localhost:8000
```

## Contributing

Contributions are welcome, particularly for upcoming roadmap steps. Please ensure tests pass and follow the existing code style.

```bash
pytest tests/ -v
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alberta_framework,
  title = {Alberta Framework: A JAX Implementation of Alberta Plan components},
  author = {Lawson, Keith},
  year = {2026},
  url = {https://github.com/j-klawson/alberta-framework}
}
```

### Key References

```bibtex
@article{sutton2022alberta,
  title = {The Alberta Plan for AI Research},
  author = {Sutton, Richard S. and Bowling, Michael and Pilarski, Patrick M.},
  year = {2022},
  eprint = {2208.11173},
  archivePrefix = {arXiv}
}

@inproceedings{sutton1992idbd,
  title = {Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta},
  author = {Sutton, Richard S.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {1992}
}

@inproceedings{mahmood2012autostep,
  title = {Tuning-free Step-size Adaptation},
  author = {Mahmood, A. Rupam and Sutton, Richard S. and Degris, Thomas and Pilarski, Patrick M.},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing},
  year = {2012}
}
```

## License

Apache License 2.0
