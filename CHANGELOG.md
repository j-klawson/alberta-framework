# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.0] - 2026-03-15

### Added

- **SARSA agent (Step 4a)** — on-policy control via Horde architecture
  - `SARSAAgent`: wraps `HordeLearner` with epsilon-greedy action selection and SARSA target computation
  - `SARSAConfig`: configuration for n_actions, gamma, epsilon schedule
  - `SARSAState`, `SARSAUpdateResult`: immutable state and result types
  - `run_sarsa_episode`: Python loop for episodic Gymnasium environments
  - `run_sarsa_continuing`: continuing mode with pseudo-boundary handling (daemon-style)
  - `run_sarsa_from_arrays`: JIT-compiled `jax.lax.scan` for pre-collected data (security-gym)
  - Gumbel trick tie-breaking for uniform action selection among equal Q-values
  - Linear epsilon decay schedule (configurable start, end, decay steps)
  - Optional prediction demons coexist with control demons in the same Horde
  - Config serialization via `to_config()` / `from_config()` roundtrip
  - 30 new tests covering init, action selection, update logic, epsilon decay, bounding, serialization, and scan loop
  - Example: `examples/The Alberta Plan/Step4/sarsa_cartpole.py`
  - Documentation: `docs/guide/sarsa-control.md`

- **Trunk trace guard** — validation preventing `gamma * lamda > 0` on `MultiHeadMLPLearner` with hidden layers
  - VJP backward pass folds error into trunk cotangent before trace accumulation; only correct when traces reset each step
  - Linear baseline (`hidden_sizes=()`) allows any gamma/lamda
  - `HordeLearner` enforces trunk gamma=0 by design (per-head trace decay only)
  - Expanded docstrings on `MultiHeadMLPLearner` and `HordeLearner` explaining the constraint

## [0.10.0] - 2026-02-27

### Added

- **Hybrid optimizer (`head_optimizer`)** — separate optimizer for trunk vs head layers on `MLPLearner` and `MultiHeadMLPLearner`
  - `MLPLearner(head_optimizer=...)`: output layer uses `head_optimizer`, hidden layers use `optimizer`
  - `MultiHeadMLPLearner(head_optimizer=...)`: all prediction heads use `head_optimizer`, trunk uses `optimizer`
  - Enables stable LMS+ObGD for non-convex hidden layers with adaptive Autostep for the linear output head
  - Backwards compatible: `head_optimizer=None` (default) keeps all layers on the same optimizer
  - 10 new tests (6 MLPLearner, 4 MultiHeadMLPLearner)

## [0.9.0] - 2026-02-22

### Added

- **Agent lifecycle tracking** — `step_count`, `birth_timestamp`, `uptime_s` on all learner states
  - `LearnerState`, `MLPLearnerState`, `TDLearnerState`: new fields with backward-compatible defaults
  - `MultiHeadMLPState`: added `birth_timestamp` and `uptime_s` (already had `step_count`)
  - `step_count` incremented inside `update()` (JAX-traced, safe in `jax.lax.scan`)
  - `birth_timestamp` set at `init()`, immutable across updates
  - `uptime_s` accumulated after each `jax.lax.scan` completes in all learning loops
  - All learning loop functions stamp uptime: `run_learning_loop` (simple + tracking), `run_learning_loop_batched`, `run_mlp_learning_loop` (simple + tracking), `run_mlp_learning_loop_batched`, `run_td_learning_loop`, `run_multi_head_learning_loop`, `run_multi_head_learning_loop_batched`, `learn_from_trajectory`
- `agent_age_s(state)` — wall-clock seconds since agent birth
- `agent_uptime_s(state)` — cumulative active seconds inside learning loops
- 28 new lifecycle tracking tests across all learner types

## [0.8.1] - 2026-02-21

### Added

- **bsuite benchmark integration** — bridges framework to bsuite for standardized RL diagnostics
  - `ContinuingWrapper`: converts episodic envs to continuing streams (Alberta Plan Step 6)
  - `AlbertaAgent`: bridges bsuite `Agent` ABC to `MultiHeadMLPLearner` with Q-learning
  - Three agent factories: Autostep+ObGD, LMS+ObGD, Adam (haiku/optax external baseline)
  - Hyperparameter configs with standard `(64, 64)` and bottleneck `(16, 16)` variants
  - `run_single.py` / `run_sweep.py` CLIs with `--continual-sequence` and `--use-scythe` flags
  - Analysis module: result loading, comparison plots, representation analysis, summary tables
  - Representation utility logging: per-weight step-sizes, trunk trace magnitudes, per-head metrics
  - 22 tests covering wrapper, agents, factories, representation logging, and integration

### Dependencies

- Added `[bsuite]` optional dependency group (dm-env, optax, dm-haiku, plotnine)

## [0.8.0] - 2026-02-16

### Added

- **`MultiHeadMLPLearner`** — shared-trunk MLP with multiple prediction heads for multi-task continual learning
  - VJP-based gradient computation with accumulated cotangents (single backward pass through trunk)
  - NaN target masking for selective head activation (inactive heads skip gradient updates)
  - Composable: accepts any `Optimizer`, optional `Bounder`, optional `Normalizer`
  - Eligibility traces managed per-head and per-trunk-layer
- `MultiHeadMLPState`, `MultiHeadMLPUpdateResult`, `MultiHeadLearningResult`, `BatchedMultiHeadResult` types
- `run_multi_head_learning_loop()` — `jax.lax.scan` over observation/target arrays with NaN masking
- `run_multi_head_learning_loop_batched()` — `jax.vmap` over initialization keys for multi-seed parallelization
- `multi_head_metrics_to_dicts()` — convert array metrics to per-head dicts for online use

## [0.7.3] - 2026-02-09

### Added

- `MLPLearner(use_layer_norm=False)` — toggle parameterless LayerNorm for ablation studies (default `True`, backwards-compatible)

## [0.7.2] - 2026-02-08

### Fixed

- IDBD operation ordering now matches Sutton 1992 Figure 2: meta-update first, then NEW alpha for weight and trace updates

### Changed (Breaking)

- Autostep rewritten to match Mahmood et al. 2012 Table 1 exactly:
  - `v_i` now tracks meta-gradient magnitude `|δ*x*h|` (was primary gradient `|δ*x|`)
  - `v_i` uses self-regulated EMA (Eq. 4), not `max(|grad|, v*τ)`
  - Overshoot prevention via `M = max(Σ α_i*x_i², 1)` (Eq. 6-7)
  - Trace decay includes `x²`: `h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i`
  - Normalizers and traces initialized to 0 (was 1 and 0)
  - Normalization only applies to meta-update, not to weight/trace updates
- `Autostep(normalizer_decay=...)` renamed to `Autostep(tau=...)`, default changed from 0.99 to 10000.0
- `AutostepState.normalizer_decay` renamed to `AutostepState.tau`
- `AutostepParamState.normalizer_decay` renamed to `AutostepParamState.tau`

### Added

- `Autostep.update_from_gradient()` now accepts optional `error` parameter for full paper algorithm in MLP path
- `Optimizer.update_from_gradient()` base signature accepts optional `error` parameter

## [0.7.1] - 2026-02-07

### Added

- **`AGCBounding`** — Adaptive Gradient Clipping (Brock et al. 2021) as a `Bounder` ABC, per-unit clipping scaled by weight norm
- `_unitwise_norm()` helper for unit-wise L2 norm computation (1D: abs, 2D+: norm over fan-in axes)

## [0.7.0] - 2026-02-07

### Changed (Breaking)

- Removed `NormalizedLinearLearner`, `NormalizedMLPLearner` — use `LinearLearner(normalizer=...)` and `MLPLearner(normalizer=...)` instead
- Removed `run_normalized_learning_loop`, `run_normalized_learning_loop_batched`, `run_mlp_normalized_learning_loop`, `run_mlp_normalized_learning_loop_batched` — unified into `run_learning_loop` and `run_mlp_learning_loop` (detect normalization from learner)
- Removed `NormalizedLearnerState`, `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedNormalizedResult`, `BatchedMLPNormalizedResult`, `MLPObGDState` types
- `MLPLearner` no longer accepts `kappa` parameter — use `bounder=ObGDBounding(kappa=2.0)` instead

### Added

- `Bounder` ABC and `ObGDBounding` for decoupled update bounding (composable with any optimizer)
- `AutostepParamState` for per-parameter Autostep optimization (arbitrary array shapes)
- `Optimizer.init_for_shape()` and `Optimizer.update_from_gradient()` for shape-agnostic optimization (LMS, Autostep)
- `MLPLearner` now accepts composable `optimizer`, `bounder`, and `normalizer` parameters
- `LinearLearner` now accepts optional `bounder` and `normalizer` parameters
- Unified learning loops: 4 functions instead of 8 (linear + MLP, each with single + batched)

### Fixed

- mypy override errors — base class `init_for_shape`/`update_from_gradient` use `Any` since return type varies by subclass

## [0.6.1] - 2026-02-07

- Version bump only

## [0.6.0] - 2026-02-07

### Changed (Breaking)

- Replaced `OnlineNormalizer`, `NormalizerState`, `create_normalizer_state` with `Normalizer` ABC hierarchy

### Added

- `Normalizer` ABC with generic `StateT` constraint, following the `Optimizer[StateT]` pattern
- `EMANormalizer` — exponential moving average normalization (renamed from `OnlineNormalizer`, corrected docstrings)
- `WelfordNormalizer` — true Welford's algorithm with Bessel's correction for stationary distributions
- `EMANormalizerState`, `WelfordNormalizerState`, `AnyNormalizerState` types
- `NormalizedLinearLearner` now accepts any `Normalizer` subclass
- `NormalizedMLPLearner` — wraps `MLPLearner` with online normalization (EMA or Welford)
- `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedMLPNormalizedResult` types
- `run_mlp_normalized_learning_loop()` with optional `NormalizerTrackingConfig`
- `run_mlp_normalized_learning_loop_batched()` for vmap-based multi-seed normalized MLP training

## [0.5.3] - 2026-02-06

### Added

- `run_mlp_learning_loop_batched()` for vmap-based multi-seed MLP training with `BatchedMLPResult` return type

## [0.5.2] - 2026-02-06

### Fixed

- Resolved mypy type error in `MLPLearner` z_sum computation — replaced `sum()` over JAX arrays with explicit `jnp.array(0.0)` accumulator

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

## [0.4.0] - 2026-02-04

### Added

- TD-IDBD optimizer for temporal-difference learning with per-weight adaptive step-sizes and eligibility traces (Kearney et al., 2019)
- AutoTDIDBD optimizer with AutoStep-style normalization for improved stability
- `TDLinearLearner` class for linear value function approximation in TD learning
- `run_td_learning_loop()` for JIT-compiled TD learning via `jax.lax.scan`
- TD state types: `TDIDBDState`, `AutoTDIDBDState`, `TDLearnerState`, `TDTimeStep`
- `TDStream` protocol for TD experience streams

## [0.3.2] - 2026-02-03

### Fixed

- Relaxed test tolerance in batched vs sequential comparison tests (`rtol=1e-5`) to account for floating-point differences between vmap and sequential execution paths
- Added `ignore = ["F722"]` to ruff config for jaxtyping shape annotation syntax that ruff doesn't understand
- Removed unused `PRNGKeyArray` import from `core/types.py`

## [0.3.0] - 2026-02-03

### Added

- Migrated all state types from NamedTuple to `@chex.dataclass(frozen=True)` for DeepMind-style JAX compatibility
- jaxtyping shape annotations for compile-time type safety (`Float[Array, " feature_dim"]`, `PRNGKeyArray`, etc.)
- Updated test suite to use chex assertions (`chex.assert_shape`, `chex.assert_tree_all_finite`, `chex.assert_trees_all_close`)

### Dependencies

- Added `chex>=0.1.86` and `jaxtyping>=0.2.28` as required dependencies
- Added `beartype>=0.18.0` as optional dev dependency for runtime type checking

## [0.2.2] - 2026-02-02

### Fixed

- mypy type errors in `run_learning_loop_batched` and `run_normalized_learning_loop_batched` functions
- Added `typing.cast` to properly handle conditional return type unpacking in batched learning loops

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
