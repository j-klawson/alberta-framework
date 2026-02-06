# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-02-06

### Added

- **ObGD Optimizer**: Observation-bounded Gradient Descent for overshooting prevention (Elsayed et al. 2024). Dynamically bounds effective step-size based on error magnitude and trace norms. Works as a linear optimizer (`ObGD`) and within the MLP learner.
- **MLPLearner**: Multi-layer perceptron with ObGD optimizer for nonlinear function approximation in the streaming setting. Architecture: `Input -> [Dense -> LayerNorm -> LeakyReLU] x N -> Dense(1)`. Configurable depth via `hidden_sizes` tuple.
- **Sparse Initialization**: `sparse_init()` function implementing LeCun-scale initialization with per-neuron sparsity (default 90%), following Elsayed et al. 2024.
- **`run_mlp_learning_loop()`**: JIT-compiled MLP training via `jax.lax.scan`, same pattern as existing linear learning loops.
- **MLP Types**: `MLPParams`, `MLPObGDState`, `MLPLearnerState`, `MLPUpdateResult` chex dataclasses.
- **ObGD Types**: `ObGDState` chex dataclass with `create_obgd_state()` factory.
- **Step 2 Example**: `linear_vs_mlp_comparison.py` comparing LinearLearner+Autostep vs MLPLearner+ObGD on RandomWalk, AbruptChange, and DynamicScaleShift streams.

### Notes

- ObGD defaults to `gamma=0, lamda=0` for supervised learning (traces = current observation). Nonzero values enable eligibility traces for future RL use (Steps 3-4).
- MLP implementation is self-contained (no Flax/Haiku dependency). Uses `jax.grad` for backpropagation and parameterless layer normalization.
- The `Optimizer` generic constraint now includes `ObGDState`, so `ObGD` can be used with `LinearLearner` as well.

## [0.1.0] - 2026-01-19

### Added

- **Core Optimizers**: LMS (baseline), IDBD (Sutton 1992), and Autostep (Mahmood et al. 2012) with per-weight adaptive step-sizes
- **Linear Learners**: `LinearLearner` and `NormalizedLinearLearner` with pluggable optimizers
- **Scan-based Learning Loops**: JIT-compiled training with `jax.lax.scan` for efficiency
- **Online Normalization**: Streaming feature normalization with exponential moving averages
- **Experience Streams**: `RandomWalkStream`, `AbruptChangeStream`, `CyclicStream`, `SuttonExperiment1Stream`
- **Gymnasium Integration**: Trajectory collection and learning from Gymnasium RL environments
- **Step-Size Tracking**: Optional per-weight step-size history recording for meta-adaptation analysis
- **Multi-Seed Experiments**: `run_multi_seed_experiment` with optional parallelization via joblib
- **Statistical Analysis**: Pairwise comparisons, confidence intervals, effect sizes (requires scipy)
- **Publication Visualization**: Learning curves, bar charts, heatmaps with matplotlib
- **Export Utilities**: CSV, JSON, LaTeX, and Markdown table generation
- **Documentation**: MkDocs-based documentation with auto-generated API reference

### Notes

- Requires Python 3.13+
- Implements Step 1 of the Alberta Plan: demonstrating that IDBD/Autostep can match or beat hand-tuned LMS
- All state uses immutable NamedTuples for JAX compatibility
- Follows temporal uniformity principle: every component updates at every time step
