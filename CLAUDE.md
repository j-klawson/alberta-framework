# Alberta Framework

A research-first framework for the Alberta Plan: Building the foundations of Continual AI.

## Project Overview

This framework implements the Alberta Plan for AI Research, progressing through increasingly complex continual learning settings. Step 1 demonstrates that IDBD and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS. Step 2 extends to nonlinear function approximation with MLP and ObGD.

**Core Philosophy**: Temporal uniformity — every component updates at every time step.

## Project Architecture
- This is a multi-repo research ecosystem: alberta-framework (core RL library), chronos-sec (security experiments), and related experiment repos
- When analyzing cross-repo dependencies or planning features, always read the sibling project's current API before assuming interfaces
- After framework API changes, test downstream repos for breaking changes (e.g., result.metrics type changes)

## Python Environment
- Always activate the project's Python virtual environment before running pip install or any Python commands: `source .venv/bin/activate` (or the appropriate venv path)
- When installing packages, always quote version specifiers to avoid shell expansion: `pip install 'package>=1.0,<2.0'`
- For GPU/JAX projects, always use the [gpu] extra when installing: `pip install -e '.[gpu]'`

## Quick Reference

### Package Structure
```
src/alberta_framework/
├── core/
│   ├── types.py        # TimeStep, LearnerState, optimizer states, MLP types, TD types
│   ├── optimizers.py   # LMS, IDBD, Autostep, ObGD, TDIDBD, AutoTDIDBD optimizers; Bounder ABC, ObGDBounding, AGCBounding
│   ├── normalizers.py  # Normalizer ABC, EMANormalizer, WelfordNormalizer
│   ├── initializers.py # sparse_init (LeCun + sparsity)
│   ├── learners.py     # LinearLearner, MLPLearner, TDLinearLearner, learning loops
│   └── multi_head_learner.py  # MultiHeadMLPLearner, multi-head learning loops
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

benchmarks/
└── bsuite/
    ├── agents/
    │   ├── base.py           # AlbertaAgent: bridges bsuite.Agent <-> MultiHeadMLPLearner
    │   ├── autostep_dqn.py   # Q-learning with Autostep + ObGD (framework agent)
    │   ├── lms_dqn.py        # Q-learning with LMS fixed step-size (framework baseline)
    │   └── adam_dqn.py       # Q-learning with haiku/optax Adam (external baseline)
    ├── wrappers.py           # ContinuingWrapper: episodic → continuing conversion
    ├── configs.py            # Hyperparameter configs (standard + bottleneck variants)
    ├── run_single.py         # Run one bsuite_id for one agent
    ├── run_sweep.py          # Run sweep of bsuite_ids for all agents
    └── analysis.py           # Load results, compare agents, generate plots
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

# Run Step 2 demonstrations (MLP + ObGD)
python "examples/The Alberta Plan/Step2/linear_vs_mlp_comparison.py"
python "examples/The Alberta Plan/Step2/linear_vs_mlp_comparison.py" --output-dir output/

# Save plots to output directory (instead of displaying interactively)
python "examples/The Alberta Plan/Step1/idbd_lms_autostep_comparison.py" --output-dir output/
python "examples/The Alberta Plan/Step1/sutton1992_experiment1.py" --output-dir output/
python "examples/The Alberta Plan/Step1/sutton1992_experiment2.py" --output-dir output/

# Run Gymnasium examples (requires gymnasium)
python examples/gymnasium_reward_prediction.py

# Run publication-quality experiment
python examples/publication_experiment.py

# Run bsuite benchmarks (requires bsuite extra + local bsuite clone)
pip install -e '.[bsuite]'
# bsuite itself: PYTHONPATH=/path/to/bsuite:$PYTHONPATH (can't pip install on Python 3.13)
python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --save_path output/bsuite
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --experiments catch catch_scale
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --continual-sequence catch/0 cartpole/0
python benchmarks/bsuite/analysis.py --save_path output/bsuite --summary

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
- **Composition**: Learners accept independent optimizer, bounder, and normalizer ABCs
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

## Git Workflow
- When asked for a commit message, generate it in conventional commit format and include the scope (e.g., `fix(learners):`, `docs(plan):`, `feat(experiments):`)
- Always check current version in pyproject.toml or setup.cfg before suggesting version bumps — never assume the current version

## Testing & Linting
- After making any code edits, always run the full test suite (`pytest`) and linter (`ruff check .` and `mypy`) before presenting results
- If linting fixes break tests, fix the tests in the same pass — do not present partial fixes
- For type errors (mypy), resolve all errors in a single pass rather than introducing new type confusion across fix iterations

## Documentation Updates
- When updating CLAUDE.md with session progress, always include: date, what was accomplished, current state of the project, and next steps
- When updating study/research documents, ask the user for the exact file paths before starting — do not assume locations
- When asked to write 'in the user's voice', default to formal academic prose that is clear and direct — NOT informal or blog-style

## Key Algorithms

### LMS (Least Mean Squares)
Fixed step-size baseline optimizer:
- `w_i += alpha * error * x_i` — weight update with fixed alpha
- Simple but requires manual tuning of step-size

### IDBD (Incremental Delta-Bar-Delta)
Reference: Sutton 1992, "Adapting Bias by Gradient Descent", Figure 2

Per-weight adaptive step-sizes based on gradient correlation. Operation ordering follows the paper (meta-update first, new alpha for weight/trace):
1. `log_alpha_i += beta * error * x_i * h_i` — meta-update (using OLD traces)
2. `alpha_i = exp(log_alpha_i)` — compute NEW step-sizes
3. `w_i += alpha_i * error * x_i` — weight update (using NEW alpha)
4. `h_i = h_i * max(0, 1 - alpha_i * x_i^2) + alpha_i * error * x_i` — trace update (using NEW alpha)

### Autostep
Reference: Mahmood et al. 2012, "Tuning-free step-size adaptation", Table 1

Per-weight adaptive step-sizes with self-regulated normalizers and overshoot prevention:
1. Eq. 4: `v_i = max(|δ*x_i*h_i|, v_i + (1/τ)*α_i*x_i²*(|δ*x_i*h_i| - v_i))` — normalizer tracks meta-gradient magnitude
2. Eq. 5: `α_i *= exp(μ * δ*x_i*h_i / v_i)` where `v_i > 0` — meta-update with normalized meta-gradient
3. Eq. 6-7: `M = max(Σ α_i*x_i², 1)`; `α_i /= M` — overshoot prevention
4. `w_i += α_i * δ * x_i` — weight update (with NEW alpha, after M-normalization)
5. `h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i` — trace update (decay includes x²)

Key differences from a naive implementation:
- `v_i` normalizes the *meta-gradient* `|δ*x*h|`, not the primary gradient `|δ*x|`
- `v_i` uses self-regulated EMA (Eq. 4), not simple `max(|grad|, v*τ)`
- Normalization only applies to the meta-update, NOT to weight/trace updates
- `τ` is a time constant (default 10000), not a multiplicative decay factor
- `v_i` and `h_i` are initialized to 0 (first step: no meta-update, no normalizer update)

### Online Normalization
Streaming feature normalization following the Alberta Plan:
- `x_normalized = (x - mean) / (std + epsilon)`
- Updates at every time step (temporal uniformity)
- **Normalizer ABC** with two subclasses (follows the `Optimizer[StateT]` pattern):
  - `EMANormalizer`: Exponential moving average of mean/variance — suitable for non-stationary distributions
  - `WelfordNormalizer`: Welford's algorithm with Bessel's correction — suitable for stationary distributions
- State types: `EMANormalizerState` (mean, var, sample_count, decay), `WelfordNormalizerState` (mean, var, sample_count, p)
- Both `LinearLearner` and `MLPLearner` accept an optional `normalizer` parameter

### ObGD Bounding (Observation-bounded Gradient Descent)
Reference: Elsayed et al. 2024, "Streaming Deep Reinforcement Learning Finally Works"

Dynamic update bounding to prevent overshooting, implemented as a `Bounder` ABC (`ObGDBounding`):
1. Optimizer produces per-parameter steps from traces: `step = optimizer.update_from_gradient(state, z)`
2. `M = kappa * max(|error|, 1) * sum(|step_i|)` — compute bound
3. `scale = 1 / max(M, 1)` — bounding scale factor
4. `w += scale * error * step` — bounded weight update

**Implementation differences from Elsayed et al. 2024:**
1. **Decoupled from optimizer**: In the paper, ObGD is a complete optimizer with a fixed scalar step-size `alpha`. In our implementation, ObGD bounding is a separate `Bounder` ABC (`ObGDBounding`) applied on top of any optimizer's output. For LMS with `ObGDBounding(kappa=2.0)`: `step = alpha * z`, so `total_step = alpha * z_sum`, giving `M = kappa * max(|error|, 1) * alpha * z_sum` — identical to original.
2. **Optimizer processes trace, not error*trace**: The optimizer receives the eligibility trace `z` (or prediction gradient for supervised), and the error is multiplied after bounding.
3. **Traces managed by learner**: In the paper, traces are part of the ObGD algorithm. In our implementation, eligibility traces (`gamma`, `lamda`) are managed by the learner, making them available regardless of which optimizer is used.
4. **Generalizes to per-weight step-sizes**: For Autostep, per-weight step-sizes fold naturally into `total_step = sum(|alpha_i * f(z_i)|)`. No re-tuning of kappa needed.

### AGC Bounding (Adaptive Gradient Clipping)
Reference: Brock et al. 2021, "High-Performance Large-Scale Image Recognition Without Normalization" (arXiv: 2102.06171)

Per-unit gradient clipping scaled by weight norm, implemented as a `Bounder` ABC (`AGCBounding`):
1. Compute unit-wise parameter norm: `p_norm = unitwise_norm(param)`
2. Compute unit-wise step norm: `s_norm = unitwise_norm(step)`
3. Compute effective gradient norm: `g_norm = |error| * s_norm`
4. Compute max allowed norm: `max_norm = max(p_norm, eps) * clip_factor`
5. Clip conditionally: `clipped = where(g_norm > max_norm, step * max_norm / g_norm, step)`

Unit-wise norms by shape:
- **1D** (biases): absolute value per element
- **2D** (weight matrices `(fan_in, fan_out)`): L2 norm over axis 0, keepdims=True → shape `(1, fan_out)`

Unlike ObGDBounding which applies a single global scale factor, AGC applies fine-grained, per-unit clipping.

```python
from alberta_framework import MLPLearner, AGCBounding

learner = MLPLearner(
    hidden_sizes=(128, 128),
    step_size=1.0,
    bounder=AGCBounding(clip_factor=0.01, eps=1e-3),
)
```

### Sparse Initialization
Reference: Elsayed et al. 2024

LeCun-scale initialization with per-neuron sparsity:
- `w ~ U[-sqrt(1/fan_in), sqrt(1/fan_in)]` — LeCun uniform
- Zero out 90% of input connections per output neuron
- Creates sparser gradient flows for improved streaming stability

### MLP Learner
Reference: Elsayed et al. 2024

Architecture: `Input -> [Dense(H) -> LayerNorm -> LeakyReLU] x N -> Dense(1)`

When `use_layer_norm=False`: `Input -> [Dense(H) -> LeakyReLU] x N -> Dense(1)`
- Parameterless layer normalization (no learned scale/shift), toggleable via `use_layer_norm`
- Sparse initialization (90% default)
- Composable: accepts any `Optimizer`, optional `Bounder`, optional `Normalizer`
- Gradient computation via `jax.grad` on pure forward function
- Eligibility traces managed by learner (`gamma`, `lamda` parameters)

```python
from alberta_framework import (
    MLPLearner, ObGDBounding, AGCBounding, EMANormalizer, Autostep,
    RandomWalkStream, run_mlp_learning_loop, NormalizerTrackingConfig
)
import jax.random as jr

stream = RandomWalkStream(feature_dim=10)

# LMS + ObGD bounding (equivalent to original Elsayed et al. 2024)
learner = MLPLearner(hidden_sizes=(128, 128), step_size=1.0, bounder=ObGDBounding(kappa=2.0))
state, metrics = run_mlp_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))

# Autostep + ObGD bounding + normalization (composable)
learner = MLPLearner(
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=ObGDBounding(kappa=2.0),
    normalizer=EMANormalizer(decay=0.99),
)

# With normalizer tracking
config = NormalizerTrackingConfig(interval=100)
state, metrics, norm_history = run_mlp_learning_loop(
    learner, stream, num_steps=10000, key=jr.key(42), normalizer_tracking=config
)
# norm_history.means: shape (100, 10), norm_history.variances: shape (100, 10)

# LayerNorm ablation study (disable internal layer normalization)
learner = MLPLearner(hidden_sizes=(128, 128), step_size=1.0, use_layer_norm=False,
                     bounder=ObGDBounding(kappa=2.0))
```

### Multi-Head MLP Learner
Shared-trunk, multi-head MLP for multi-task continual learning. Designed for chronos-sec Phase 2 where a single network predicts multiple targets simultaneously.

Architecture: `Input -> [Dense(H) -> LayerNorm -> LeakyReLU] x N -> {Head_i: Dense(1)} x n_heads`

- Shared hidden trunk with independent per-head output layers
- VJP with accumulated cotangents: one backward pass through trunk regardless of n_heads
- NaN targets mark inactive heads (params/traces/optimizer states preserved)
- Same composability as MLPLearner (Optimizer, Bounder, Normalizer)
- Learning loops take pre-provided observations/targets arrays (no ScanStream)

```python
from alberta_framework import (
    MultiHeadMLPLearner, ObGDBounding, EMANormalizer, Autostep,
    run_multi_head_learning_loop, run_multi_head_learning_loop_batched,
    multi_head_metrics_to_dicts,
)
import jax.numpy as jnp
import jax.random as jr

# 4-head learner (e.g., event type, bot detection, attack stage, session value)
learner = MultiHeadMLPLearner(
    n_heads=4,
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=ObGDBounding(kappa=2.0),
    normalizer=EMANormalizer(decay=0.99),
)
state = learner.init(feature_dim=20, key=jr.key(42))

# Single update with partial active heads (head 2 inactive)
obs = jnp.ones(20)
targets = jnp.array([1.0, 0.5, jnp.nan, 3.0])  # NaN = inactive
result = learner.update(state, obs, targets)
# result.per_head_metrics: (4, 3) — [se, error, mean_step_size], NaN for inactive

# Convert to dicts for online use
dicts = multi_head_metrics_to_dicts(result)  # [dict, dict, None, dict]

# Scan loop over pre-provided data
observations = jr.normal(jr.key(1), (1000, 20))
targets = jr.normal(jr.key(2), (1000, 4))
loop_result = run_multi_head_learning_loop(learner, state, observations, targets)
# loop_result.per_head_metrics: (1000, 4, 3)

# Batched (vmap over init keys)
keys = jr.split(jr.key(42), 30)
batched = run_multi_head_learning_loop_batched(learner, observations, targets, keys)
# batched.per_head_metrics: (30, 1000, 4, 3)
```

### Success Criterion
IDBD/Autostep should beat LMS when starting from the same step-size (demonstrates adaptation).
With optimal parameters, adaptive methods should match best grid-searched LMS.

### Step-Size Tracking for Meta-Adaptation Analysis
The `run_learning_loop` function supports optional per-weight step-size tracking for analyzing how adaptive optimizers evolve their step-sizes during training:

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
The `run_learning_loop` function supports tracking the normalizer's per-feature mean and variance estimates over time when the learner has a normalizer. This is essential for analyzing reactive lag — how quickly the normalizer adapts to distribution shifts:

```python
from alberta_framework import (
    LinearLearner, IDBD, EMANormalizer,
    StepSizeTrackingConfig, NormalizerTrackingConfig,
    run_learning_loop
)
from alberta_framework.streams import RandomWalkStream
import jax.random as jr

stream = RandomWalkStream(feature_dim=10)
learner = LinearLearner(optimizer=IDBD(), normalizer=EMANormalizer())
ss_config = StepSizeTrackingConfig(interval=100)
norm_config = NormalizerTrackingConfig(interval=100)

# Track both step-sizes and normalizer state
state, metrics, ss_history, norm_history = run_learning_loop(
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
The `run_learning_loop_batched` and `run_mlp_learning_loop_batched` functions use `jax.vmap` to run multiple seeds in parallel, typically achieving 2-5x speedup over sequential execution:

```python
import jax.random as jr
from alberta_framework import (
    LinearLearner, IDBD, EMANormalizer, RandomWalkStream,
    run_learning_loop_batched, StepSizeTrackingConfig, NormalizerTrackingConfig
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
- Returns `BatchedLearningResult` or `BatchedMLPResult`
- Tracking histories get batched shapes: `(num_seeds, num_recordings, ...)`
- Same initial state used for all seeds (controlled variation via different keys)

For normalized learners (same function, learner carries normalizer):
```python
learner = LinearLearner(optimizer=IDBD(), normalizer=EMANormalizer())
result = run_learning_loop_batched(
    learner, stream, num_steps=10000, keys=keys,
    step_size_tracking=StepSizeTrackingConfig(interval=100),
    normalizer_tracking=NormalizerTrackingConfig(interval=100)
)
# result.metrics has shape (30, 10000, 4)
# result.step_size_history and result.normalizer_history both batched
```

For MLP learners:
```python
from alberta_framework import MLPLearner, ObGDBounding, run_mlp_learning_loop_batched

learner = MLPLearner(hidden_sizes=(128, 128), step_size=1.0, bounder=ObGDBounding(kappa=2.0))
keys = jr.split(jr.key(42), 30)
result = run_mlp_learning_loop_batched(learner, stream, num_steps=10000, keys=keys)
# result.metrics has shape (30, 10000, 3)
# result.states.params.weights[0] has shape (30, 128, feature_dim)
```

## Gymnasium Integration

Wrap Gymnasium RL environments as experience streams for the framework.

### Prediction Modes
- **REWARD**: Predict immediate reward from (state, action)
- **NEXT_STATE**: Predict next state from (state, action)
- **VALUE**: Predict cumulative return (TD learning)

### Key Functions
- `collect_trajectory(env, policy, num_steps, mode, ...)`: Collect trajectory using Python loop
- `learn_from_trajectory(learner, observations, targets)`: Learn from trajectory using scan (handles normalization automatically if learner has normalizer)
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
Two streams with time-varying feature scales, designed to test whether external normalization (EMANormalizer) provides benefits beyond Autostep's internal v_i normalization:

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
Run the external normalization study to compare IDBD/Autostep with and without EMANormalizer:
```bash
python "examples/The Alberta Plan/Step1/external_normalization_study.py" --seeds 30 --output-dir output/
```

## bsuite Benchmarks

Bridges the framework to [bsuite](https://github.com/google-deepmind/bsuite) for standardized RL diagnostics. Code lives in `benchmarks/` (not `src/`) because the agent layer is a *consumer* of the framework, not a core primitive.

### Design: Q-Learning as Multi-Head Prediction
Uses `MultiHeadMLPLearner` with `n_heads = num_actions` as the Q-function:
- Each head predicts Q(s, a_i)
- NaN target masking: only the taken action's head gets updated per step
- The shared trunk learns features — directly testing Step 2's representation learning

### Continuing Mode (Alberta Plan-aligned)
`ContinuingWrapper` converts episodic bsuite envs to continuing streams:
- Agent never sees FIRST or LAST timesteps — runs as one continuous stream
- `discount=0` at episode boundaries signals pseudo-termination (no bootstrap)
- Agent state persists across environment resets

### Agents
| Agent | Config key | Description |
|-------|-----------|-------------|
| Autostep DQN | `autostep` | Autostep + ObGD + EMA normalization (framework's best) |
| LMS DQN | `lms` | Fixed step-size + ObGD + EMA normalization (no-adaptation baseline) |
| Adam DQN | `adam` | Standalone haiku/optax Adam (external baseline, no framework) |

Each has `_bottleneck` variants with smaller networks `(16, 16)`.

### Representation Utility Logging
`--log-representation` flag records per-interval snapshots of:
- Per-weight step-sizes from Autostep optimizer states
- Trunk trace magnitudes per layer
- Per-head step-size means

### Dependencies
bsuite itself can't be pip-installed on Python 3.13 (`imp` module removed). Use `PYTHONPATH=../bsuite:$PYTHONPATH` instead. The `[bsuite]` extra provides dm-env, optax, dm-haiku, plotnine.

## Future Work
- Step 2 (continued): Feature generation/testing, nonlinear feature discovery
- Step 3: GVF predictions, Horde architecture
- Step 4: Actor-critic control with ObGD; add AdaptiveObGD (Appendix B of Elsayed et al. 2024) with RMSProp-style second-moment normalization
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

### v0.8.1 (2026-02-21)
- **FEATURE**: bsuite benchmark integration — bridges framework to bsuite for standardized RL diagnostics
  - `ContinuingWrapper`: converts episodic envs to continuing streams (Alberta Plan Step 6)
  - `AlbertaAgent`: bridges bsuite `Agent` ABC to `MultiHeadMLPLearner` with Q-learning
  - Three agent factories: Autostep+ObGD, LMS+ObGD, Adam (haiku/optax external baseline)
  - Hyperparameter configs with standard `(64, 64)` and bottleneck `(16, 16)` variants
  - `run_single.py` / `run_sweep.py` CLIs with `--continual-sequence` and `--use-scythe` flags
  - Analysis module: result loading, comparison plots, representation analysis, summary tables
  - Representation utility logging: per-weight step-sizes, trunk trace magnitudes, per-head metrics
  - 22 tests covering wrapper, agents, factories, representation logging, and integration
- **DEPS**: Added `[bsuite]` optional dependency group (dm-env, optax, dm-haiku, plotnine)

### v0.8.0 (2026-02-16)
- **FEATURE**: `MultiHeadMLPLearner` — shared-trunk MLP with multiple prediction heads for multi-task continual learning
  - VJP-based gradient computation with accumulated cotangents (single backward pass through trunk)
  - NaN target masking for selective head activation (inactive heads skip gradient updates)
  - Composable: accepts any `Optimizer`, optional `Bounder`, optional `Normalizer`
  - Eligibility traces managed per-head and per-trunk-layer
- **FEATURE**: `MultiHeadMLPState`, `MultiHeadMLPUpdateResult`, `MultiHeadLearningResult`, `BatchedMultiHeadResult` types
- **FEATURE**: `run_multi_head_learning_loop()` — `jax.lax.scan` over observation/target arrays with NaN masking
- **FEATURE**: `run_multi_head_learning_loop_batched()` — `jax.vmap` over initialization keys for multi-seed parallelization
- **FEATURE**: `multi_head_metrics_to_dicts()` — convert array metrics to per-head dicts for online use

### v0.7.3 (2026-02-09)
- **FEATURE**: `MLPLearner(use_layer_norm=False)` — toggle parameterless LayerNorm for ablation studies (default `True`, backwards-compatible)

### v0.7.2 (2026-02-08)
- **FIX**: IDBD operation ordering now matches Sutton 1992 Figure 2: meta-update first, then NEW alpha for weight and trace updates
- **BREAKING**: Autostep rewritten to match Mahmood et al. 2012 Table 1 exactly:
  - `v_i` now tracks meta-gradient magnitude `|δ*x*h|` (was primary gradient `|δ*x|`)
  - `v_i` uses self-regulated EMA (Eq. 4), not `max(|grad|, v*τ)`
  - Overshoot prevention via `M = max(Σ α_i*x_i², 1)` (Eq. 6-7)
  - Trace decay includes `x²`: `h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i`
  - Normalizers and traces initialized to 0 (was 1 and 0)
  - Normalization only applies to meta-update, not to weight/trace updates
- **BREAKING**: `Autostep(normalizer_decay=...)` renamed to `Autostep(tau=...)`, default changed from 0.99 to 10000.0
- **BREAKING**: `AutostepState.normalizer_decay` renamed to `AutostepState.tau`
- **BREAKING**: `AutostepParamState.normalizer_decay` renamed to `AutostepParamState.tau`
- **FEATURE**: `Autostep.update_from_gradient()` now accepts optional `error` parameter for full paper algorithm in MLP path
- **FEATURE**: `Optimizer.update_from_gradient()` base signature accepts optional `error` parameter
- **DOCS**: Updated CLAUDE.md algorithm descriptions for IDBD and Autostep

### v0.7.1 (2026-02-07)
- **FEATURE**: `AGCBounding` — Adaptive Gradient Clipping (Brock et al. 2021) as a `Bounder` ABC, per-unit clipping scaled by weight norm
- **FEATURE**: `_unitwise_norm()` helper for unit-wise L2 norm computation (1D: abs, 2D+: norm over fan-in axes)
- **DOCS**: Updated README with AGCBounding docs, example usage, and Brock et al. 2021 citation
- **DOCS**: Updated CLAUDE.md with AGC algorithm description and changelog

### v0.7.0 (2026-02-07)
- **BREAKING**: Removed `NormalizedLinearLearner`, `NormalizedMLPLearner` — use `LinearLearner(normalizer=...)` and `MLPLearner(normalizer=...)` instead
- **BREAKING**: Removed `run_normalized_learning_loop`, `run_normalized_learning_loop_batched`, `run_mlp_normalized_learning_loop`, `run_mlp_normalized_learning_loop_batched` — unified into `run_learning_loop` and `run_mlp_learning_loop` (detect normalization from learner)
- **BREAKING**: Removed `NormalizedLearnerState`, `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedNormalizedResult`, `BatchedMLPNormalizedResult`, `MLPObGDState` types
- **BREAKING**: `MLPLearner` no longer accepts `kappa` parameter — use `bounder=ObGDBounding(kappa=2.0)` instead
- **FEATURE**: `Bounder` ABC and `ObGDBounding` for decoupled update bounding (composable with any optimizer)
- **FEATURE**: `AutostepParamState` for per-parameter Autostep optimization (arbitrary array shapes)
- **FEATURE**: `Optimizer.init_for_shape()` and `Optimizer.update_from_gradient()` for shape-agnostic optimization (LMS, Autostep)
- **FEATURE**: `MLPLearner` now accepts composable `optimizer`, `bounder`, and `normalizer` parameters
- **FEATURE**: `LinearLearner` now accepts optional `bounder` and `normalizer` parameters
- **FEATURE**: Unified learning loops: 4 functions instead of 8 (linear + MLP, each with single + batched)
- **FIX**: mypy override errors — base class `init_for_shape`/`update_from_gradient` use `Any` since return type varies by subclass
- **DOCS**: Updated README with composable architecture, MLP Learner, Bounders, Normalizers, stability disclaimer, and Elsayed et al. 2024 reference
- **DOCS**: Updated CLAUDE.md with v0.7.0 API changes throughout

### v0.6.1 (2026-02-07)
- Version bump only

### v0.6.0 (2026-02-07)
- **BREAKING**: Replaced `OnlineNormalizer`, `NormalizerState`, `create_normalizer_state` with `Normalizer` ABC hierarchy
- **FEATURE**: `Normalizer` ABC with generic `StateT` constraint, following the `Optimizer[StateT]` pattern
- **FEATURE**: `EMANormalizer` — exponential moving average normalization (renamed from `OnlineNormalizer`, corrected docstrings)
- **FEATURE**: `WelfordNormalizer` — true Welford's algorithm with Bessel's correction for stationary distributions
- **FEATURE**: `EMANormalizerState`, `WelfordNormalizerState`, `AnyNormalizerState` types
- **FEATURE**: `NormalizedLinearLearner` now accepts any `Normalizer` subclass
- **FEATURE**: `NormalizedMLPLearner` — wraps `MLPLearner` with online normalization (EMA or Welford)
- **FEATURE**: `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedMLPNormalizedResult` types
- **FEATURE**: `run_mlp_normalized_learning_loop()` with optional `NormalizerTrackingConfig`
- **FEATURE**: `run_mlp_normalized_learning_loop_batched()` for vmap-based multi-seed normalized MLP training

### v0.5.3 (2026-02-06)
- **FEATURE**: `run_mlp_learning_loop_batched()` for vmap-based multi-seed MLP training with `BatchedMLPResult` return type

### v0.5.2 (2026-02-06)
- **FIX**: Resolved mypy type error in `MLPLearner` z_sum computation — replaced `sum()` over JAX arrays with explicit `jnp.array(0.0)` accumulator
- **DOCS**: Added Project Architecture, Python Environment, Git Workflow, Testing & Linting, and Documentation Updates sections to CLAUDE.md

### v0.5.0 (2026-02-06)
- **FEATURE**: ObGD (Observation-bounded Gradient Descent) optimizer with dynamic step-size bounding (Elsayed et al. 2024)
- **FEATURE**: `MLPLearner` with parameterless LayerNorm, LeakyReLU, and sparse initialization
- **FEATURE**: `sparse_init()` for LeCun-scale initialization with per-neuron sparsity
- **FEATURE**: `run_mlp_learning_loop()` for JIT-compiled MLP training via `jax.lax.scan`
- **FEATURE**: MLP types: `MLPParams`, `MLPObGDState`, `MLPLearnerState`, `MLPUpdateResult`
- **FEATURE**: ObGD types: `ObGDState`, `create_obgd_state()`
- **FEATURE**: Step 2 example: `linear_vs_mlp_comparison.py`

### v0.4.0 (2026-02-04)
- **FEATURE**: Implemented TD-IDBD optimizer for temporal-difference learning with per-weight adaptive step-sizes and eligibility traces (Kearney et al., 2019)
- **FEATURE**: Implemented AutoTDIDBD optimizer with AutoStep-style normalization for improved stability
- **FEATURE**: Added `TDLinearLearner` class for linear value function approximation in TD learning
- **FEATURE**: Added `run_td_learning_loop()` for JIT-compiled TD learning via `jax.lax.scan`
- **FEATURE**: Added TD state types: `TDIDBDState`, `AutoTDIDBDState`, `TDLearnerState`, `TDTimeStep`
- **FEATURE**: Added `TDStream` protocol for TD experience streams
- **DOCS**: Updated README with TD learning documentation and Kearney et al. 2019 reference

### v0.3.2 (2026-02-03)
- **FIX**: Relaxed test tolerance in batched vs sequential comparison tests (`rtol=1e-5`) to account for floating-point differences between vmap and sequential execution paths
- **FIX**: Added `ignore = ["F722"]` to ruff config for jaxtyping shape annotation syntax that ruff doesn't understand
- **FIX**: Removed unused `PRNGKeyArray` import from `core/types.py`

### v0.3.0 (2026-02-03)
- **FEATURE**: Migrated all state types from NamedTuple to `@chex.dataclass(frozen=True)` for DeepMind-style JAX compatibility
- **FEATURE**: Added jaxtyping shape annotations for compile-time type safety (`Float[Array, " feature_dim"]`, `PRNGKeyArray`, etc.)
- **FEATURE**: Updated test suite to use chex assertions (`chex.assert_shape`, `chex.assert_tree_all_finite`, `chex.assert_trees_all_close`)
- **DEPS**: Added `chex>=0.1.86` and `jaxtyping>=0.2.28` as required dependencies
- **DEPS**: Added `beartype>=0.18.0` as optional dev dependency for runtime type checking

### v0.2.2 (2026-02-02)
- Fixed mypy type errors in `run_learning_loop_batched` and `run_normalized_learning_loop_batched` functions
- Added `typing.cast` to properly handle conditional return type unpacking in batched learning loops
