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
│   ├── types.py        # TimeStep, LearnerState, LMSState, IDBDState, AutostepState
│   ├── optimizers.py   # LMS, IDBD, Autostep optimizers
│   ├── normalizers.py  # OnlineNormalizer, NormalizerState
│   └── learners.py     # LinearLearner, NormalizedLinearLearner, run_learning_loop
├── streams/
│   ├── base.py         # ExperienceStream protocol
│   ├── synthetic.py    # RandomWalkTarget, AbruptChangeTarget, CyclicTarget, SuttonExperiment1Stream
│   └── gymnasium.py    # GymnasiumStream, TDStream, PredictionMode (optional)
└── utils/
    ├── metrics.py      # compute_tracking_error, compare_learners, etc.
    ├── experiments.py  # ExperimentConfig, run_multi_seed_experiment, AggregatedResults
    ├── statistics.py   # Statistical tests, CI, effect sizes (requires scipy)
    ├── visualization.py # Publication plots (requires matplotlib)
    └── export.py       # CSV, JSON, LaTeX, Markdown export
```

### Key Commands
```bash
# Install in dev mode
pip install -e ".[dev]"

# Install with Gymnasium support
pip install -e ".[gymnasium]"

# Install with analysis tools (matplotlib, scipy, joblib, tqdm)
pip install -e ".[analysis]"

# Run tests
pytest tests/ -v

# Run Step 1 demonstrations
python examples/step1_idbd_vs_lms.py
python examples/step1_normalization_study.py
python examples/step1_autostep_comparison.py

# Run Sutton 1992 Experiment 1 replication
python examples/sutton1992_experiment1.py

# Run Gymnasium examples (requires gymnasium)
python examples/gymnasium_reward_prediction.py

# Run publication-quality experiment (requires analysis)
python examples/publication_experiment.py

# Build documentation (requires docs)
pip install -e ".[docs]"
mkdocs serve          # Local preview at http://localhost:8000
mkdocs build          # Build static site to site/
```

## Development Guidelines

### Design Principles
- **Immutable State**: All state uses NamedTuples for JAX compatibility
- **Functional Style**: Pure functions enable `jit`, `vmap`
- **Composition**: Learners accept optimizers as parameters
- **Temporal Uniformity**: Every component updates at every time step

### JAX Conventions
- Use `jax.numpy` (imported as `jnp`) not regular numpy for array operations
- Use `jax.random` with explicit key management
- State is immutable - return new state objects, don't mutate

### Testing
- Tests are in `tests/` directory
- Use pytest fixtures from `conftest.py`
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

## Gymnasium Integration

Wrap Gymnasium RL environments as experience streams for the framework.

### Prediction Modes
- **REWARD**: Predict immediate reward from (state, action)
- **NEXT_STATE**: Predict next state from (state, action)
- **VALUE**: Predict cumulative return (TD learning)

### Key Classes
- `GymnasiumStream`: Main wrapper, auto-resets on episode boundaries
- `TDStream`: For proper TD learning with value function bootstrap
- `PredictionMode`: Enum for prediction mode selection

### Factory Functions
- `make_gymnasium_stream(env_id, mode, ...)`: Create stream from env ID
- `make_random_policy(env, seed)`: Create random action policy
- `make_epsilon_greedy_policy(base, env, epsilon, seed)`: Wrap policy with exploration

### Example Usage
```python
from alberta_framework import LinearLearner, IDBD, run_learning_loop
from alberta_framework.streams.gymnasium import make_gymnasium_stream, PredictionMode

# Create stream from CartPole
stream = make_gymnasium_stream(
    "CartPole-v1",
    mode=PredictionMode.REWARD,
    include_action_in_features=True,
)

# Use with existing learners
learner = LinearLearner(optimizer=IDBD())
state, metrics = run_learning_loop(learner, stream, num_steps=10000)
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

### Statistical Analysis (requires scipy)
- `pairwise_comparisons()`: All pairwise tests with Bonferroni/Holm correction
- `ttest_comparison()`, `mann_whitney_comparison()`, `wilcoxon_comparison()`
- `compute_statistics()`, `bootstrap_ci()`, `cohens_d()`

### Visualization (requires matplotlib)
- `set_publication_style()`: Configure for academic papers
- `plot_learning_curves()`: Learning curves with confidence intervals
- `plot_final_performance_bars()`: Bar charts with significance markers
- `create_comparison_figure()`: Multi-panel comparison figure
- `save_figure()`: Export to PDF/PNG

### Export
- `generate_latex_table()`, `generate_markdown_table()`
- `export_to_csv()`, `export_to_json()`
- `save_experiment_report()`: Save all artifacts at once

## Documentation

Documentation is built with MkDocs and mkdocstrings (auto-generated API docs from docstrings).

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

### Docstring Style
Use NumPy-style docstrings for all public functions and classes. See `core/optimizers.py` for examples.

## Future Work (Out of Scope for v0.1.0)
- Step 2: Feature generation/testing
- Step 3: GVF predictions, Horde architecture
- Step 4: Actor-critic control
- Steps 5-6: Average reward formulation
