# Alberta Framework

A research-first framework for the Alberta Plan: Building the foundations of Continual AI.

## Project Overview

This framework implements Step 1 of the Alberta Plan: demonstrating that IDBD (Incremental Delta-Bar-Delta) and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS on non-stationary supervised learning problems.

**Core Philosophy**: Temporal uniformity — every component updates at every time step.

## Quick Reference

### Package Structure
```
src/alberta_framework/
├── core/
│   ├── types.py        # TimeStep, LearnerState, LMSState, IDBDState, AutostepState, StepSizeTrackingConfig, StepSizeHistory, NormalizerTrackingConfig, NormalizerHistory, BatchedLearningResult, BatchedNormalizedResult
│   ├── optimizers.py   # LMS, IDBD, Autostep optimizers
│   ├── normalizers.py  # OnlineNormalizer, NormalizerState
│   └── learners.py     # LinearLearner, NormalizedLinearLearner, run_learning_loop, run_learning_loop_batched, run_normalized_learning_loop, run_normalized_learning_loop_batched, metrics_to_dicts
├── streams/
│   ├── base.py         # ScanStream protocol (pure function interface for jax.lax.scan)
│   ├── synthetic.py    # RandomWalkStream, AbruptChangeStream, CyclicStream, PeriodicChangeStream, ScaledStreamWrapper, DynamicScaleShiftStream, ScaleDriftStream
│   └── gymnasium.py    # collect_trajectory, learn_from_trajectory, GymnasiumStream (optional)
└── utils/
    ├── metrics.py      # compute_tracking_error, compare_learners, etc.
    ├── experiments.py  # ExperimentConfig, run_multi_seed_experiment, AggregatedResults
    ├── statistics.py   # Statistical tests, CI, effect sizes (requires scipy)
    ├── visualization.py # Publication plots (requires matplotlib)
    ├── export.py       # CSV, JSON, LaTeX, Markdown export
    └── timing.py       # Timer context manager, format_duration
```

### Key Commands
```bash
# Install in dev mode
pip install -e ".[dev]"

# Install with Gymnasium support
pip install -e ".[gymnasium]"

# Run tests
pytest tests/ -v

# Run Step 1 demonstrations (The Alberta Plan)
python "examples/The Alberta Plan/Step1/idbd_lms_autostep_comparison.py"
python "examples/The Alberta Plan/Step1/normalization_study.py"

# Run Sutton 1992 replications
python "examples/The Alberta Plan/Step1/sutton1992_experiment1.py"
python "examples/The Alberta Plan/Step1/sutton1992_experiment2.py"

# Save plots to output directory (instead of displaying interactively)
python "examples/The Alberta Plan/Step1/idbd_lms_autostep_comparison.py" --output-dir output/
python "examples/The Alberta Plan/Step1/sutton1992_experiment1.py" --output-dir output/
python "examples/The Alberta Plan/Step1/sutton1992_experiment2.py" --output-dir output/

# Run Gymnasium examples (requires gymnasium)
python examples/gymnasium_reward_prediction.py

# Run publication-quality experiment
python examples/publication_experiment.py

# Build documentation (requires docs)
pip install -e ".[docs]"
mkdocs serve          # Local preview at http://localhost:8000
mkdocs build          # Build static site to site/
```

## Development Guidelines

### Design Principles
- **Immutable State**: All state uses `@chex.dataclass(frozen=True)` for JAX PyTree compatibility
- **Type Safety**: jaxtyping annotations for shape checking (`Float[Array, " feature_dim"]`)
- **Functional Style**: Pure functions enable `jit`, `vmap`, `jax.lax.scan`
- **Scan-Based Learning**: Learning loops use `jax.lax.scan` for JIT-compiled training
- **Composition**: Learners accept optimizers as parameters
- **Temporal Uniformity**: Every component updates at every time step

### JAX Conventions
- Use `jax.numpy` (imported as `jnp`) not regular numpy for array operations
- Use `jax.random` with explicit key management: `key = jr.key(seed)`
- State is immutable - return new state objects, don't mutate
- Streams use `ScanStream` protocol with `init(key)` and `step(state, idx)` methods

### Testing
- Tests are in `tests/` directory
- Use pytest fixtures from `conftest.py`
- Use chex assertions: `chex.assert_shape()`, `chex.assert_trees_all_close()`, `chex.assert_tree_all_finite()`
- All tests should pass before committing

## Key Algorithms

### LMS (Least Mean Squares)
Fixed step-size baseline optimizer:
- `w_i += alpha * error * x_i` — weight update with fixed alpha
- Simple but requires manual tuning of step-size

### IDBD (Incremental Delta-Bar-Delta)
Reference: Sutton 1992, "Adapting Bias by Gradient Descent"

Per-weight adaptive step-sizes based on gradient correlation:
1. `alpha_i = exp(log_alpha_i)` — per-weight step-sizes
2. `w_i += alpha_i * error * x_i` — weight update
3. `log_alpha_i += beta * error * x_i * h_i` — meta-update
4. `h_i = h_i * max(0, 1 - alpha_i * x_i^2) + alpha_i * error * x_i` — trace update

### Autostep
Reference: Mahmood et al. 2012, "Tuning-free step-size adaptation"

Per-weight adaptive step-sizes with gradient normalization:
1. `g_i = error * x_i` — compute gradient
2. `g_i' = g_i / max(|g_i|, v_i)` — normalize gradient
3. `w_i += alpha_i * g_i'` — weight update with normalized gradient
4. `alpha_i *= exp(mu * g_i' * h_i)` — adapt step-size
5. `h_i = h_i * (1 - alpha_i) + alpha_i * g_i'` — update trace
6. `v_i = max(|g_i|, v_i * tau)` — update normalizer

### Online Normalization
Streaming feature normalization following the Alberta Plan:
- `x_normalized = (x - mean) / (std + epsilon)`
- Mean and variance estimated via exponential moving average
- Updates at every time step (temporal uniformity)

### Success Criterion
IDBD/Autostep should beat LMS when starting from the same step-size (demonstrates adaptation).
With optimal parameters, adaptive methods should match best grid-searched LMS.

### Step-Size Tracking for Meta-Adaptation Analysis
The `run_learning_loop` and `run_normalized_learning_loop` functions support optional per-weight step-size tracking for analyzing how adaptive optimizers evolve their step-sizes during training:

```python
from alberta_framework import LinearLearner, IDBD, Autostep, StepSizeTrackingConfig, run_learning_loop
from alberta_framework.streams import RandomWalkStream
import jax.random as jr

stream = RandomWalkStream(feature_dim=10)
learner = LinearLearner(optimizer=Autostep())
config = StepSizeTrackingConfig(interval=100)  # Record every 100 steps

state, metrics, history = run_learning_loop(
    learner, stream, num_steps=10000, key=jr.key(42), step_size_tracking=config
)

# history.step_sizes: shape (100, 10) - per-weight step-sizes at each recording
# history.bias_step_sizes: shape (100,) - bias step-size at each recording
# history.recording_indices: shape (100,) - step indices where recordings were made
# history.normalizers: shape (100, 10) - Autostep's v_i normalizers (None for IDBD/LMS)
```

Key features:
- Recording happens inside the JAX scan loop (no Python loop overhead)
- Configurable interval to control memory usage
- Optional `include_bias=False` to skip bias tracking
- Works with LMS (constant), IDBD, and Autostep optimizers
- **Autostep's normalizers (v_i)** are tracked automatically when using Autostep

### Normalizer State Tracking for Reactive Lag Analysis
The `run_normalized_learning_loop` function supports tracking the OnlineNormalizer's per-feature mean and variance estimates over time. This is essential for analyzing reactive lag — how quickly the normalizer adapts to distribution shifts:

```python
from alberta_framework import (
    NormalizedLinearLearner, IDBD,
    StepSizeTrackingConfig, NormalizerTrackingConfig,
    run_normalized_learning_loop
)
from alberta_framework.streams import RandomWalkStream
import jax.random as jr

stream = RandomWalkStream(feature_dim=10)
learner = NormalizedLinearLearner(optimizer=IDBD())
ss_config = StepSizeTrackingConfig(interval=100)
norm_config = NormalizerTrackingConfig(interval=100)

# Track both step-sizes and normalizer state
state, metrics, ss_history, norm_history = run_normalized_learning_loop(
    learner, stream, num_steps=10000, key=jr.key(42),
    step_size_tracking=ss_config, normalizer_tracking=norm_config
)

# norm_history.means: shape (100, 10) - per-feature mean estimates at each recording
# norm_history.variances: shape (100, 10) - per-feature variance estimates at each recording
# norm_history.recording_indices: shape (100,) - step indices where recordings were made
```

Return value depends on tracking options:
- No tracking: `(state, metrics)` — 2-tuple
- step_size_tracking only: `(state, metrics, ss_history)` — 3-tuple
- normalizer_tracking only: `(state, metrics, norm_history)` — 3-tuple
- Both: `(state, metrics, ss_history, norm_history)` — 4-tuple

### Batched Learning Loops (vmap-based GPU Parallelization)
The `run_learning_loop_batched` and `run_normalized_learning_loop_batched` functions use `jax.vmap` to run multiple seeds in parallel, typically achieving 2-5x speedup over sequential execution:

```python
import jax.random as jr
from alberta_framework import (
    LinearLearner, IDBD, RandomWalkStream,
    run_learning_loop_batched, StepSizeTrackingConfig
)

stream = RandomWalkStream(feature_dim=10)
learner = LinearLearner(optimizer=IDBD())

# Run 30 seeds in parallel
keys = jr.split(jr.key(42), 30)
result = run_learning_loop_batched(learner, stream, num_steps=10000, keys=keys)

# result.metrics has shape (30, 10000, 3)
# result.states.weights has shape (30, 10)
mean_error = result.metrics[:, :, 0].mean(axis=0)  # Average squared error over seeds

# With step-size tracking
config = StepSizeTrackingConfig(interval=100)
result = run_learning_loop_batched(
    learner, stream, num_steps=10000, keys=keys, step_size_tracking=config
)
# result.step_size_history.step_sizes has shape (30, 100, 10)
```

Key features:
- `jax.vmap` parallelizes over seeds, not steps — memory scales with num_seeds
- `jax.lax.scan` processes steps sequentially within each seed
- Returns `BatchedLearningResult` or `BatchedNormalizedResult` NamedTuples
- Tracking histories get batched shapes: `(num_seeds, num_recordings, ...)`
- Same initial state used for all seeds (controlled variation via different keys)

For normalized learners:
```python
from alberta_framework import (
    NormalizedLinearLearner, run_normalized_learning_loop_batched,
    NormalizerTrackingConfig
)

learner = NormalizedLinearLearner(optimizer=IDBD())
result = run_normalized_learning_loop_batched(
    learner, stream, num_steps=10000, keys=keys,
    step_size_tracking=StepSizeTrackingConfig(interval=100),
    normalizer_tracking=NormalizerTrackingConfig(interval=100)
)
# result.metrics has shape (30, 10000, 4)
# result.step_size_history and result.normalizer_history both batched
```

## Gymnasium Integration

Wrap Gymnasium RL environments as experience streams for the framework.

### Prediction Modes
- **REWARD**: Predict immediate reward from (state, action)
- **NEXT_STATE**: Predict next state from (state, action)
- **VALUE**: Predict cumulative return (TD learning)

### Key Functions
- `collect_trajectory(env, policy, num_steps, mode, ...)`: Collect trajectory using Python loop
- `learn_from_trajectory(learner, observations, targets)`: Learn from trajectory using scan
- `learn_from_trajectory_normalized(learner, observations, targets)`: With normalization
- `make_gymnasium_stream(env_id, mode, ...)`: Create stream from env ID (for Python loops)
- `make_random_policy(env, seed)`: Create random action policy
- `make_epsilon_greedy_policy(base, env, epsilon, seed)`: Wrap policy with exploration

### Example Usage (Trajectory Collection - Recommended)
```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, metrics_to_dicts
from alberta_framework.streams.gymnasium import (
    collect_trajectory, learn_from_trajectory, PredictionMode, make_random_policy
)
import gymnasium as gym

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
metrics_list = metrics_to_dicts(metrics)
```

## Publication-Quality Analysis

The `utils` module provides tools for rigorous multi-seed experiments:

### Key Classes
- `ExperimentConfig`: Define experiments with learner/stream factories
- `AggregatedResults`: Results aggregated across seeds with summary statistics
- `SignificanceResult`: Statistical test results with effect sizes

### Multi-Seed Experiments
```python
from alberta_framework.utils import ExperimentConfig, run_multi_seed_experiment

configs = [ExperimentConfig(name="IDBD", learner_factory=..., stream_factory=..., num_steps=10000)]
results = run_multi_seed_experiment(configs, seeds=30, parallel=True)
```

### Statistical Analysis
- `pairwise_comparisons()`: All pairwise tests with Bonferroni/Holm correction
- `ttest_comparison()`, `mann_whitney_comparison()`, `wilcoxon_comparison()`
- `compute_statistics()`, `bootstrap_ci()`, `cohens_d()`

### Visualization
- `set_publication_style()`: Configure for academic papers
- `plot_learning_curves()`: Learning curves with confidence intervals
- `plot_final_performance_bars()`: Bar charts with significance markers
- `create_comparison_figure()`: Multi-panel comparison figure
- `save_figure()`: Export to PDF/PNG

### Export
- `generate_latex_table()`, `generate_markdown_table()`
- `export_to_csv()`, `export_to_json()`
- `save_experiment_report()`: Save all artifacts at once

### Timing
All example scripts report total runtime at completion using the `Timer` context manager:
```python
from alberta_framework import Timer

with Timer("My experiment"):
    # ... run experiment ...
# Prints: "My experiment completed in 31.34s"
```

- `Timer(name, verbose=True)`: Context manager for timing code blocks
- `format_duration(seconds)`: Format seconds as human-readable (e.g., "2m 30.50s")

## Documentation

Documentation is built with MkDocs and mkdocstrings (auto-generated API docs from docstrings).

### Local Preview
```bash
pip install -e ".[docs]"
mkdocs serve          # Preview at http://localhost:8000
mkdocs build          # Build static site to site/
```

### Structure
```
docs/
├── index.md                 # Home page
├── getting-started/         # Installation, quickstart
├── guide/                   # Concepts, optimizers, streams, experiments
├── contributing.md
└── gen_ref_pages.py         # Auto-generates API reference
mkdocs.yml                   # MkDocs configuration
```

### API Reference
The API Reference section is auto-generated from docstrings in the source code. Every public class, method, and function is documented. The `gen_ref_pages.py` script scans all `.py` files and creates reference pages using mkdocstrings.

### Docstring Style
Use NumPy-style docstrings for all public functions and classes. See `core/optimizers.py` for examples.

**Code examples**: Use fenced markdown code blocks (triple backticks with `python`) inside an `Examples` section, not doctest `>>>` format. This ensures proper syntax highlighting in mkdocstrings. See `streams/base.py` or `utils/timing.py` for examples.

**Math formulas**: Wrap inline math expressions in backticks for monospace rendering, e.g., `` `y = w @ x + b` `` or `` `alpha_i = exp(log_alpha_i)` ``. See `core/optimizers.py` for examples.

## Streams for Factorial Studies

The framework supports factorial experiment designs with multiple non-stationarity types and scale ranges:

### Non-stationarity Types
- **Drift**: `RandomWalkStream` - continuous random walk of target weights
- **Abrupt**: `AbruptChangeStream` - sudden weight changes at fixed intervals
- **Periodic**: `PeriodicChangeStream` - sinusoidal weight oscillation

### Scale Ranges with ScaledStreamWrapper
Wrap any stream to apply per-feature scaling (tests normalization benefits):
```python
from alberta_framework import ScaledStreamWrapper, AbruptChangeStream, make_scale_range
import jax.numpy as jnp

# Create scale ranges for factorial study
small = make_scale_range(10, min_scale=0.1, max_scale=10.0)    # 10^2 range
medium = make_scale_range(10, min_scale=0.01, max_scale=100.0)  # 10^4 range
large = make_scale_range(10, min_scale=0.001, max_scale=1000.0) # 10^6 range

# Wrap any stream with scaling
stream = ScaledStreamWrapper(
    AbruptChangeStream(feature_dim=10, change_interval=1000),
    feature_scales=large
)
```

### Dynamic Scale Streams (for testing normalization benefits)
Two streams with time-varying feature scales, designed to test whether external normalization (OnlineNormalizer) provides benefits beyond Autostep's internal v_i normalization:

- **DynamicScaleShiftStream**: Feature scales abruptly change at intervals
  - Both target weights AND feature scales change at (possibly different) intervals
  - Scales are log-uniform distributed within [min_scale, max_scale]
  - Target is computed from unscaled features for consistent difficulty

- **ScaleDriftStream**: Feature scales drift via bounded random walk on log-scale
  - Weights drift in linear space, scales drift in log-space
  - Log-scales are clipped to [min_log_scale, max_log_scale]
  - Tests continuous scale tracking

```python
from alberta_framework import DynamicScaleShiftStream, ScaleDriftStream

# Abrupt scale shifts every 5000 steps
stream = DynamicScaleShiftStream(
    feature_dim=20,
    scale_change_interval=5000,
    weight_change_interval=2000,
    min_scale=0.01,
    max_scale=100.0,
)

# Continuous scale drift
stream = ScaleDriftStream(
    feature_dim=20,
    weight_drift_rate=0.001,
    scale_drift_rate=0.02,
    min_log_scale=-4.0,  # exp(-4) ~ 0.018
    max_log_scale=4.0,   # exp(4) ~ 54.6
)
```

### External Normalization Study
Run the external normalization study to compare IDBD/Autostep with and without OnlineNormalizer:
```bash
python "examples/The Alberta Plan/Step1/external_normalization_study.py" --seeds 30 --output-dir output/
```

## Future Work (Out of Scope for v0.1.0)
- Step 2: Feature generation/testing
- Step 3: GVF predictions, Horde architecture
- Step 4: Actor-critic control
- Steps 5-6: Average reward formulation

## Version Management and CI/CD

After publishing to PyPI, version numbers must be considered with each commit:

- **Patch** (0.1.1): Bug fixes, documentation updates
- **Minor** (0.2.0): New features, new algorithms, API additions
- **Major** (1.0.0): Breaking changes, API redesign

Before committing changes, ask: "Does this change require a version bump?"

### GitHub Actions Workflows

- **ci.yml**: Runs tests, linting, and doc builds on push/PR to main
- **docs.yml**: Deploys documentation to GitHub Pages on push to main
- **publish.yml**: Publishes to PyPI on version tags (e.g., `v0.2.0`)

### Publishing a New Version
```bash
# 1. Update version in pyproject.toml
# 2. Commit and push changes
# 3. Create and push a version tag
git tag v0.2.0
git push --tags
# GitHub Actions handles TestPyPI -> PyPI publishing automatically
```

### PyPI Trusted Publishing Setup
The publish workflow uses OpenID Connect (no API tokens). Configure on PyPI:
1. PyPI project → Settings → Publishing → Add GitHub publisher
2. Repository: `j-klawson/alberta-framework`, Workflow: `publish.yml`, Environment: `pypi`
3. Repeat on TestPyPI with environment: `testpypi`

## Changelog

### v0.3.0 (2026-02-03)
- **FEATURE**: Migrated all state types from NamedTuple to `@chex.dataclass(frozen=True)` for DeepMind-style JAX compatibility
- **FEATURE**: Added jaxtyping shape annotations for compile-time type safety (`Float[Array, " feature_dim"]`, `PRNGKeyArray`, etc.)
- **FEATURE**: Updated test suite to use chex assertions (`chex.assert_shape`, `chex.assert_tree_all_finite`, `chex.assert_trees_all_close`)
- **DEPS**: Added `chex>=0.1.86` and `jaxtyping>=0.2.28` as required dependencies
- **DEPS**: Added `beartype>=0.18.0` as optional dev dependency for runtime type checking

### v0.2.2 (2026-02-02)
- Fixed mypy type errors in `run_learning_loop_batched` and `run_normalized_learning_loop_batched` functions
- Added `typing.cast` to properly handle conditional return type unpacking in batched learning loops
