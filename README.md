# Alberta Framework: Step 1 Baseline for Continual RL

[![CI](https://github.com/j-klawson/alberta-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/j-klawson/alberta-framework/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)

> "The agents are complex only because they interact with a complex world... their initial design is as simple, general, and scalable as possible." — *The Alberta Plan (Sutton et al., 2022)*

## The Vision: A Massive Retreat

The **Alberta Framework** is an implementation of the 12-step research roadmap proposed by Richard Sutton, Michael Bowling, and Patrick Pilarski. We embrace a "massive retreat" from the complexities of modern batch-oriented deep learning to solve the fundamental problems of **continual learning** and **meta-learning** in their simplest, most rigorous settings.

### Why this exists?

Traditional AI systems are often "frozen" after training. In contrast, this framework is built for agents that live in the **Big World**—environments that are vastly more complex than the agent itself. Our goal is to provide the "bricks" for agents that learn, adapt, and construct their own representations from a continuous, non-stationary stream of experience.

## Core Architectural Pillars

### 1. Temporal Uniformity (The 1-Step Discipline)
Every component in the framework follows the discipline of **Temporal Uniformity**. There are no special training periods, no "epochs," and no offline batch processing. If the agent learns, it learns at every time step. If it plans, it plans at every time step.

### 2. Continual Meta-Learning
The environment is non-stationary; targets drift and optimal step-sizes change. Our core learners utilize **IDBD (Incremental Delta-Bar-Delta)** to meta-learn per-feature learning rates ($\alpha_t^i$), allowing the agent to autonomously discover which signals are salient.

### 3. Functional JAX Implementation
To support the 10-year scaling vision of the Alberta Plan, the framework is built on **JAX**.
* **Stateless Pure Functions**: Every update is a transformation: `(state, experience) -> new_state`.
* **Hardware Accelerated**: Native support for JIT compilation, auto-vectorization (`vmap`), and gradients (`grad`).
* **Scale-Invariant**: Logic defined for a single weight scales seamlessly to massive architectures.

## Current Roadmap Progress

- [x] **Step 1: Representation I** - Continual supervised learning with IDBD.
- [ ] **Step 2: Representation II** - Supervised feature finding.
- [ ] **Step 3: Prediction I** - Continual GVF learning (The Horde).
- [ ] ...
- [ ] **Step 11: Prototype-AI III** - Full OaK Architecture.

## Getting Started

### Installation
```bash
# Requires Python 3.13+
pip install alberta-framework
```

### The Alberta Plan Research Roadmap

| Step | Name | Focus | Status |
|------|------|-------|--------|
| 1 | Representation I | Meta-learned step-sizes (IDBD, Autostep) | **Current** |
| 2 | Representation II | Feature generation and testing | Planned |
| 3 | Predictions | GVF predictions, Horde architecture | Planned |
| 4 | Control | Actor-critic with eligibility traces | Planned |
| 5 | Off-policy | Importance sampling, GTD methods | Planned |
| 6 | Average Reward | Differential value functions | Planned |
| 7-12 | Advanced | Hierarchical, multi-agent, world models | Future |

### Current Status: Step 1

Step 1 demonstrates that meta-learned step-sizes can match or beat hand-tuned fixed learning rates on non-stationary supervised learning problems.

**Key algorithms implemented:**
- **LMS**: Fixed step-size baseline (requires manual tuning)
- **IDBD**: Adapts step-sizes via gradient correlation (Sutton 1992)
- **Autostep**: Tuning-free adaptation with gradient normalization (Mahmood et al. 2012)
- **Online Normalization**: Streaming feature normalization for varying scales

## Installation

```bash
# From source
git clone https://github.com/j-klawson/alberta-framework.git
cd alberta-framework
pip install -e ".[dev]"

# With Gymnasium support for RL environments
pip install -e ".[gymnasium]"
```

Requirements:
- Python >= 3.13
- JAX >= 0.4
- NumPy >= 2.0
- Gymnasium >= 0.29.0 (optional, for RL environments)

## Quick Start

```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop, metrics_to_dicts

# Create a non-stationary stream where the target drifts over time
stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# Create a learner with IDBD (meta-learned step-sizes)
learner = LinearLearner(optimizer=IDBD())

# Run the learning loop (uses jax.lax.scan for JIT-compiled training)
key = jr.key(42)
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=key)

# Convert metrics array to list of dicts for analysis
metrics_list = metrics_to_dicts(metrics)
print(f"Final squared_error: {metrics_list[-1]['squared_error']:.4f}")
```

## Key Components

### Optimizers

Three step-size strategies with increasing sophistication:

```python
from alberta_framework import LMS, IDBD, Autostep

# Fixed step-size (requires tuning)
lms = LMS(step_size=0.01)

# Meta-learned step-sizes via gradient correlation
idbd = IDBD(initial_step_size=0.01, meta_step_size=0.01)

# Tuning-free with gradient normalization
autostep = Autostep(initial_step_size=0.01, meta_step_size=0.01)
```

### Learners

Linear function approximators with pluggable optimizers:

```python
from alberta_framework import LinearLearner, NormalizedLinearLearner, IDBD

# Basic learner
learner = LinearLearner(optimizer=IDBD())

# Learner with online feature normalization
# (useful when features have different scales)
normalized_learner = NormalizedLinearLearner(optimizer=IDBD())
```

### Experience Streams

Non-stationary target generators for testing continual learning. All streams implement the `ScanStream` protocol with `init(key)` and `step(state, idx)` methods for use with `jax.lax.scan`:

```python
from alberta_framework import RandomWalkStream, AbruptChangeStream, CyclicStream

# Gradual drift (target weights perform random walk)
stream1 = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# Sudden changes (target switches every N steps)
stream2 = AbruptChangeStream(feature_dim=10, change_interval=1000)

# Cyclic changes (target cycles through configurations)
stream3 = CyclicStream(feature_dim=10, num_configurations=4, cycle_length=500)
```

### Gymnasium Integration

Use any Gymnasium RL environment with the framework via trajectory collection:

```python
import gymnasium as gym
from alberta_framework import LinearLearner, IDBD, metrics_to_dicts
from alberta_framework.streams.gymnasium import (
    collect_trajectory, learn_from_trajectory, PredictionMode, make_random_policy
)

# Create environment and policy
env = gym.make("CartPole-v1")
policy = make_random_policy(env, seed=42)

# Collect trajectory (Python loop for env interaction)
observations, targets = collect_trajectory(
    env, policy, num_steps=10000, mode=PredictionMode.REWARD
)

# Learn from trajectory (JIT-compiled scan)
learner = LinearLearner(optimizer=IDBD())
state, metrics = learn_from_trajectory(learner, observations, targets)
```

**Prediction Modes**:
- `REWARD`: Predict immediate reward from (state, action)
- `NEXT_STATE`: Predict next state from (state, action)
- `VALUE`: Predict cumulative return (TD learning)

**Features**:
- Trajectory collection separates env interaction from learning
- JIT-compiled learning via `jax.lax.scan`
- Custom policy support (random by default)
- Works with Box, Discrete, and MultiDiscrete spaces

### Online Normalization

Streaming feature normalization for handling varying scales:

```python
from alberta_framework import OnlineNormalizer

normalizer = OnlineNormalizer(decay=0.99)
state = normalizer.init(feature_dim=10)

# Normalize and update statistics in one call
normalized_obs, state = normalizer.normalize(state, observation)
```

## Running Experiments

### Step 1 Demonstration

The core experiment showing IDBD/Autostep beating hand-tuned LMS:

```bash
python examples/step1_idbd_vs_lms.py
```

### Normalization Study

Demonstrates the benefit of online feature normalization:

```bash
python examples/step1_normalization_study.py
```

### Optimizer Comparison

Comprehensive comparison across different non-stationarity types:

```bash
python examples/step1_autostep_comparison.py
```

### Gymnasium Reward Prediction

Compare IDBD vs LMS on CartPole reward prediction:

```bash
pip install gymnasium
python examples/gymnasium_reward_prediction.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Mathematical Background

### The Step-Size Problem

In non-stationary environments, the optimal step-size changes over time. A step-size that's too small fails to track changes; one that's too large overshoots. Manual tuning is fragile and doesn't adapt.

### IDBD: Learning to Learn

IDBD (Sutton 1992) maintains per-weight step-sizes α_i that adapt based on gradient correlation:

```
α_i = exp(log_α_i)                                    # step-size
w_i += α_i * error * x_i                              # weight update
log_α_i += β * error * x_i * h_i                      # meta-update
h_i = h_i * max(0, 1 - α_i * x_i²) + α_i * error * x_i  # trace
```

When successive gradients agree (same sign), h_i grows positive, increasing α_i. When they disagree, h_i shrinks or goes negative, decreasing α_i.

### Autostep: Tuning-Free Adaptation

Autostep (Mahmood et al. 2012) adds gradient normalization for robustness:

```
g_i = error * x_i                     # raw gradient
g_i' = g_i / max(|g_i|, v_i)         # normalized gradient
w_i += α_i * g_i'                     # weight update
α_i *= exp(μ * g_i' * h_i)           # adapt step-size
h_i = h_i * (1 - α_i) + α_i * g_i'   # update trace
v_i = max(|g_i|, v_i * τ)            # update normalizer
```

The normalization prevents large gradients from causing instability.

## API Reference

### Core Types

```python
TimeStep(observation, target)        # Single experience from stream
LearnerState(weights, bias, optimizer_state)  # Immutable learner state
NormalizerState(mean, var, count, decay)      # Normalizer statistics
```

### Stream Protocol

```python
# ScanStream protocol - all streams implement this interface
class ScanStream(Protocol[StateT]):
    @property
    def feature_dim(self) -> int: ...
    def init(self, key: Array) -> StateT: ...
    def step(self, state: StateT, idx: Array) -> tuple[TimeStep, StateT]: ...
```

### Utility Functions

```python
from alberta_framework import (
    run_learning_loop,          # Run learning for N steps (uses jax.lax.scan)
    run_normalized_learning_loop,  # With normalization
    metrics_to_dicts,           # Convert metrics array to list of dicts
    compute_tracking_error,     # Running mean of squared error
    compute_cumulative_error,   # Cumulative error over time
    compare_learners,           # Compare multiple learners
)
```

## Publication-Quality Analysis

The framework includes tools for running rigorous multi-seed experiments with statistical analysis and publication-ready outputs.

### Installation

```bash
pip install -e ".[analysis]"  # Adds matplotlib, scipy, joblib, tqdm
```

### Multi-Seed Experiments

Run experiments across multiple seeds with optional parallelization:

```python
from alberta_framework import IDBD, LMS, LinearLearner, RandomWalkStream
from alberta_framework.utils import (
    ExperimentConfig,
    run_multi_seed_experiment,
    pairwise_comparisons,
)

configs = [
    ExperimentConfig(
        name="LMS",
        learner_factory=lambda: LinearLearner(optimizer=LMS(0.01)),
        stream_factory=lambda: RandomWalkStream(feature_dim=10),
        num_steps=10000,
    ),
    ExperimentConfig(
        name="IDBD",
        learner_factory=lambda: LinearLearner(optimizer=IDBD()),
        stream_factory=lambda: RandomWalkStream(feature_dim=10),
        num_steps=10000,
    ),
]

# Run across 30 seeds with parallel execution
results = run_multi_seed_experiment(configs, seeds=30, parallel=True)

# Statistical comparison with multiple comparison correction
significance = pairwise_comparisons(results, test="ttest", correction="bonferroni")
```

### Publication Figures

Generate publication-quality figures:

```python
from alberta_framework.utils import (
    set_publication_style,
    plot_learning_curves,
    create_comparison_figure,
    save_figure,
)

set_publication_style(font_size=10, use_latex=False)

# Learning curves with confidence intervals
fig, ax = plot_learning_curves(results, show_ci=True, log_scale=True)
save_figure(fig, "learning_curves", formats=["pdf", "png"])

# Multi-panel comparison figure
fig = create_comparison_figure(results, significance_results=significance)
save_figure(fig, "comparison", formats=["pdf", "png"])
```

### Export Results

Export results for papers and reports:

```python
from alberta_framework.utils import (
    generate_latex_table,
    generate_markdown_table,
    export_to_csv,
    export_to_json,
    save_experiment_report,
)

# Generate LaTeX table with significance markers
latex = generate_latex_table(
    results,
    significance_results=significance,
    caption="Comparison of LMS and IDBD",
    label="tab:comparison",
)

# Save complete report (CSV, JSON, LaTeX, Markdown)
artifacts = save_experiment_report(results, "output/", "my_experiment")
```

### Statistical Functions

Available statistical tests and utilities:

```python
from alberta_framework.utils import (
    compute_statistics,           # Mean, std, SEM, CI, median, IQR
    compute_timeseries_statistics,  # Statistics over time
    ttest_comparison,             # Paired/independent t-test
    mann_whitney_comparison,      # Non-parametric test
    wilcoxon_comparison,          # Paired non-parametric test
    bonferroni_correction,        # Multiple comparison correction
    holm_correction,              # Holm-Bonferroni correction
    bootstrap_ci,                 # Bootstrap confidence intervals
    cohens_d,                     # Effect size
)
```

### Running the Example

```bash
pip install -e ".[analysis]"
python examples/publication_experiment.py
```

This generates:
- Learning curve figures (PDF + PNG)
- Multi-panel comparison figures
- LaTeX and Markdown tables
- CSV and JSON result files

## Design Principles

1. **Immutable State**: All state is immutable (NamedTuples), making it JAX-friendly and easy to reason about.

2. **Composition over Inheritance**: Learners accept optimizers as parameters, normalizers wrap learners.

3. **Functional Style**: Pure functions enable `jit` compilation, `vmap` batching, and `grad` differentiation.

4. **Scan-Based Learning**: Learning loops use `jax.lax.scan` for JIT-compiled, efficient training.

5. **Temporal Uniformity**: Every component updates at every step — no special initialization phases.

## References & Citation

If you use this framework for research, please cite:

- **The Alberta Plan**: Sutton, R.S., et al. (2022). "The Alberta Plan for AI Research."
- **IDBD**: Sutton, R.S. (1992). "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta"
- **Autostep**: Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012). "Tuning-free step-size adaptation"

## Contributing

Contributions are welcome! Areas of particular interest:

- **Step 2**: Feature generation and relevance testing
- **Step 3**: General Value Functions (GVFs) and the Horde architecture
- **Performance**: JAX optimizations for faster experimentation
- **Visualization**: Tools for understanding adaptation dynamics

Please ensure all tests pass and follow the existing code style.

## Questions & Contact

Open an issue on [GitHub](https://github.com/j-klawson/alberta-framework/issues) for questions, bugs, or discussion.

## License

Apache License 2.0
