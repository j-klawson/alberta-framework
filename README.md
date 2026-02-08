# Alberta Framework

[![CI](https://github.com/j-klawson/alberta-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/j-klawson/alberta-framework/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/alberta-framework.svg)](https://pypi.org/project/alberta-framework/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

> **Warning:** This framework is under active research development. The API is unstable and subject to breaking changes between releases. It is not intended for production use.

A JAX-based research framework implementing components of [The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) in the pursuit of building the foundations of Continual AI.


> "The agents are complex only because they interact with a complex world... their initial design is as simple, general, and scalable as possible." — *Sutton et al., 2022*

## Overview

The Alberta Framework provides foundational components for continual reinforcement learning research. Built on JAX for hardware acceleration, the framework emphasizes temporal uniformity every component updates at every time step, with no special training phases or batch processing.

## Project Context

This framework is developed as part of my D.Eng. work focusing on the foundations of online, continuious Reinforcement Learning (RL). For more background and context see:

* **Research Blog**: [blog.9600baud.net](https://blog.9600baud.net)
* **Replicating Sutton '92**: [The Foundation of Step-size Adaptation](https://blog.9600baud.net/sutton92.html)
* **Effects of normalizing input data**: [Demonstrating Adaptive Step-Size Algorithm Needs External Normalization](https://blog.9600baud.net/autostep-normalization.html)
* **Notes on JAX performance**: [JAX Performance: From 63 Minutes to 2 Minutes](https://blog.9600baud.net/jax-performance.html)
* **About the Author**: [Keith Lawson](https://blog.9600baud.net/about.html)

### Roadmap

Depending on my research trajectory I may or may not implement components required for the Alberta Plan. The current focus of this framework is the Step 1 Baseline Study, investigating the interaction between adaptive optimizers and online normalization.

| Step | Focus | Status |
|------|-------|--------|
| 1 | Meta-learned step-sizes (IDBD, Autostep) | **Complete** |
| 2 | Nonlinear function approximation (MLP, ObGD) | **In Progress** |
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
from alberta_framework import (
    LinearLearner, MLPLearner, LMS, IDBD, Autostep,
    ObGDBounding, AGCBounding, EMANormalizer, WelfordNormalizer,
    RandomWalkStream, run_learning_loop, run_mlp_learning_loop,
)

stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# --- Optimizers ---

# Fixed step-size baseline
learner = LinearLearner(optimizer=LMS(step_size=0.01))

# IDBD: per-weight adaptive step-sizes via gradient correlation (Sutton, 1992)
learner = LinearLearner(optimizer=IDBD())

# Autostep: tuning-free adaptation with gradient normalization (Mahmood et al., 2012)
learner = LinearLearner(optimizer=Autostep())

# --- Adding a Normalizer ---

# EMA normalization for non-stationary feature scales
learner = LinearLearner(optimizer=IDBD(), normalizer=EMANormalizer(decay=0.99))

# Welford normalization for stationary distributions
learner = LinearLearner(optimizer=Autostep(), normalizer=WelfordNormalizer())

# --- Adding a Bounder ---

# ObGD bounding prevents overshooting (Elsayed et al., 2024)
learner = LinearLearner(optimizer=Autostep(), bounder=ObGDBounding(kappa=2.0))

# --- MLP Learner ---

# MLP with Autostep + ObGD bounding + normalization
mlp = MLPLearner(
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=ObGDBounding(kappa=2.0),
    normalizer=EMANormalizer(decay=0.99),
)

# MLP with AGC bounding — per-unit clipping scaled by weight norm (Brock et al., 2021)
mlp = MLPLearner(
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=AGCBounding(clip_factor=0.01),
)

# --- Training ---

# Linear: JIT-compiled training via jax.lax.scan
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))

# MLP: same interface
state, metrics = run_mlp_learning_loop(mlp, stream, num_steps=10000, key=jr.key(42))
```

## Core Components

### Composable Architecture

Learners accept three independent, composable concerns:
- **Optimizer** — per-weight step-size adaptation (LMS, IDBD, Autostep)
- **Bounder** — optional global update bounding (ObGDBounding)
- **Normalizer** — optional online feature normalization (EMANormalizer, WelfordNormalizer)

```python
from alberta_framework import (
    LinearLearner, MLPLearner, Autostep, ObGDBounding, AGCBounding, EMANormalizer
)

# Linear learner with Autostep + normalization
learner = LinearLearner(
    optimizer=Autostep(),
    normalizer=EMANormalizer(decay=0.99),
)

# MLP with Autostep + ObGD bounding + normalization
mlp = MLPLearner(
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=ObGDBounding(kappa=2.0),
    normalizer=EMANormalizer(decay=0.99),
)

# MLP with AGC bounding (per-unit clipping scaled by weight norm)
mlp_agc = MLPLearner(
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=AGCBounding(clip_factor=0.01),
    normalizer=EMANormalizer(decay=0.99),
)
```

### Optimizers

**Supervised Learning:**
- **LMS**: Fixed step-size baseline
- **IDBD**: Per-weight adaptive step-sizes via gradient correlation (Sutton, 1992)
- **Autostep**: Tuning-free adaptation with gradient normalization (Mahmood et al., 2012)

**TD Learning:**
- **TDIDBD**: TD learning with per-weight adaptive step-sizes and eligibility traces (Kearney et al., 2019)
- **AutoTDIDBD**: TD learning with AutoStep-style normalization for improved stability

### Bounders

- **ObGDBounding**: Dynamic update bounding to prevent overshooting (Elsayed et al., 2024). Decoupled from the optimizer so it can be composed with any optimizer.
- **AGCBounding**: Adaptive Gradient Clipping — per-unit clipping scaled by weight norm (Brock et al., 2021). Finer-grained than ObGD's global scaling.

### Normalizers

Online feature normalization for handling varying feature scales:
- **EMANormalizer**: Exponential moving average — suitable for non-stationary distributions
- **WelfordNormalizer**: Welford's algorithm with Bessel's correction — suitable for stationary distributions

### MLP Learner

Multi-layer perceptron for nonlinear function approximation (Elsayed et al., 2024):

```python
from alberta_framework import MLPLearner, ObGDBounding, RandomWalkStream, run_mlp_learning_loop
import jax.random as jr

stream = RandomWalkStream(feature_dim=10)
learner = MLPLearner(
    hidden_sizes=(128, 128),
    step_size=1.0,
    bounder=ObGDBounding(kappa=2.0),
    sparsity=0.9,
)
state, metrics = run_mlp_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
```

### Streams

Non-stationary experience generators implementing the `ScanStream` protocol:

- `RandomWalkStream`: Gradual target drift
- `AbruptChangeStream`: Sudden target switches
- `PeriodicChangeStream`: Sinusoidal oscillation
- `DynamicScaleShiftStream`: Time-varying feature scales
- `ScaleDriftStream`: Continuous feature scale drift

### TD Learning

For temporal-difference learning with value function approximation:

```python
from alberta_framework import TDLinearLearner, TDIDBD, run_td_learning_loop

learner = TDLinearLearner(optimizer=TDIDBD(trace_decay=0.9))
state, metrics = run_td_learning_loop(learner, td_stream, num_steps=10000, key=jr.key(42))
```

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

@inproceedings{kearney2019tidbd,
  title = {Learning Feature Relevance Through Step Size Adaptation in Temporal-Difference Learning},
  author = {Kearney, Alex and Veeriah, Vivek and Travnik, Jaden and Sutton, Richard S. and Pilarski, Patrick M.},
  booktitle = {International Conference on Machine Learning},
  year = {2019}
}

@article{brock2021high,
  title = {High-Performance Large-Scale Image Recognition Without Normalization},
  author = {Brock, Andrew and De, Soham and Smith, Samuel L. and Simonyan, Karen},
  journal = {arXiv preprint arXiv:2102.06171},
  year = {2021}
}

@article{elsayed2024streaming,
  title = {Streaming Deep Reinforcement Learning Finally Works},
  author = {Elsayed, Mohamed and Lan, Gautham and Lim, Shuze and Mahmood, A. Rupam},
  journal = {arXiv preprint arXiv:2410.14606},
  year = {2024}
}
```

## License

Apache License 2.0
