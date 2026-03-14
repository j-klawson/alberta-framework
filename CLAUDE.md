# Alberta Framework

A research-first framework for the Alberta Plan: Building the foundations of Continual AI.

## Project Overview

Implements the Alberta Plan for AI Research, progressing through increasingly complex continual learning settings. Step 1 (complete): IDBD/Autostep beat hand-tuned LMS. Step 2 (substantial): nonlinear function approximation with MLP, ObGD, and IDBD-MLP. Step 3 (next): GVF prediction and Horde architecture.

**Core Philosophy**: Temporal uniformity — every component updates at every time step.

## Project Architecture
- Multi-repo ecosystem: alberta-framework (core), chronos-sec (security experiments), rlsecd (daemon), security-gym (environments)
- Read sibling project APIs before assuming interfaces
- After API changes, test downstream repos for breaking changes

## Python Environment
- Activate venv before running Python: `source .venv/bin/activate`
- Quote version specifiers: `pip install 'package>=1.0,<2.0'`
- GPU/JAX: `pip install -e '.[gpu]'`

## Package Structure
```
src/alberta_framework/
├── core/
│   ├── types.py            # TimeStep, LearnerState, optimizer states, MLP types, TD types, lifecycle utilities
│   ├── optimizers.py       # LMS, IDBD, Autostep, ObGD, TDIDBD, AutoTDIDBD; Bounder ABC, ObGDBounding, AGCBounding
│   ├── normalizers.py      # Normalizer ABC, EMANormalizer, WelfordNormalizer
│   ├── initializers.py     # sparse_init (LeCun + sparsity)
│   ├── learners.py         # LinearLearner, MLPLearner, TDLinearLearner, learning loops
│   ├── multi_head_learner.py  # MultiHeadMLPLearner, multi-head learning loops
│   └── diagnostics.py      # FeatureRelevance, compute_feature_relevance, compute_feature_sensitivity
├── streams/
│   ├── base.py             # ScanStream protocol
│   ├── synthetic.py        # RandomWalkStream, AbruptChangeStream, CyclicStream, PeriodicChangeStream, ScaledStreamWrapper, DynamicScaleShiftStream, ScaleDriftStream
│   └── gymnasium.py        # collect_trajectory, learn_from_trajectory, GymnasiumStream
└── utils/
    ├── metrics.py           # compute_tracking_error, compare_learners
    ├── experiments.py       # ExperimentConfig, run_multi_seed_experiment, AggregatedResults
    ├── statistics.py        # Statistical tests, CI, effect sizes
    ├── visualization.py     # Publication plots
    ├── export.py            # CSV, JSON, LaTeX, Markdown export
    └── timing.py            # Timer context manager, format_duration

benchmarks/bsuite/          # bsuite RL diagnostics (consumer of framework, not core)
    agents/                  # AlbertaAgent, autostep_dqn, lms_dqn, adam_dqn
    wrappers.py              # ContinuingWrapper: episodic -> continuing
    configs.py, run_single.py, run_sweep.py, analysis.py
```

## Key Commands
```bash
pip install -e ".[dev]"                    # Dev install
pytest tests/ -v                           # Run tests
ruff check .                               # Lint
mypy                                       # Type check

# Examples
python "examples/The Alberta Plan/Step1/idbd_lms_autostep_comparison.py" --output-dir output/
python "examples/The Alberta Plan/Step2/linear_vs_mlp_comparison.py" --output-dir output/

# bsuite (requires PYTHONPATH=../bsuite:$PYTHONPATH on Python 3.13)
pip install -e '.[bsuite]'
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --experiments catch

# Docs
pip install -e ".[docs]" && mkdocs serve
```

## Design Principles
- **Immutable State**: `@chex.dataclass(frozen=True)` for JAX PyTree compatibility
- **Type Safety**: jaxtyping annotations (`Float[Array, " feature_dim"]`)
- **Functional Style**: Pure functions for `jit`, `vmap`, `jax.lax.scan`
- **Scan-Based Learning**: `jax.lax.scan` for JIT-compiled training loops
- **Composition**: Learners accept independent Optimizer, Bounder, and Normalizer ABCs
- **Temporal Uniformity**: Every component updates at every time step

## JAX Conventions
- `jax.numpy` (as `jnp`) not numpy; `jax.random` with explicit key management (`jr.key(seed)`)
- State is immutable — return new state objects, don't mutate
- Streams use `ScanStream` protocol: `init(key)` and `step(state, idx)`

## Testing
- Tests in `tests/`, fixtures in `conftest.py`
- Use chex assertions: `chex.assert_shape()`, `chex.assert_trees_all_close()`, `chex.assert_tree_all_finite()`
- All tests must pass before committing

## Development Rules
- After code edits, always run `pytest`, `ruff check .`, and `mypy` before presenting results
- If linting fixes break tests, fix tests in the same pass
- For mypy errors, resolve all in a single pass

## Git Workflow
- Conventional commit format with scope: `fix(learners):`, `docs(plan):`, `feat(experiments):`
- Check current version in pyproject.toml before suggesting bumps

## Documentation
- MkDocs + mkdocstrings (auto-generated API docs from docstrings)
- NumPy-style docstrings for all public functions/classes
- Code examples: fenced markdown blocks (not doctest `>>>`)
- Math: backtick-wrapped inline (`alpha_i = exp(log_alpha_i)`)
- When asked to write 'in the user's voice': formal academic prose, clear and direct

## Key Algorithms

### LMS (Least Mean Squares)
Fixed step-size baseline: `w_i += alpha * error * x_i`. Simple but requires manual tuning.

### IDBD (Incremental Delta-Bar-Delta) — Sutton 1992
Per-weight adaptive step-sizes via gradient correlation. Operation order: meta-update first (using OLD traces), then NEW alpha for weight and trace updates. See `core/optimizers.py` for full pseudocode.

**MLP Support (IDBD-MLP) — Meyer**: IDBD supports MLPs via `init_for_shape()` and `update_from_gradient()` using `IDBDParamState`. The core insight (Meyer): replace `x^2` in the h-decay term with `(dy/dw)^2` (squared prediction gradients), generalizing IDBD to arbitrary architectures. Two `h_decay_mode` options: `"prediction_grads"` (default) and `"loss_grads"` (Fisher approx). The MLP path intentionally differs from linear IDBD: meta-update uses `z * h` (no error), h-trace accumulates loss gradient direction (`-error * z`). Reference: [Meyer](https://github.com/ejmejm/phd_research/blob/main/phd/jax_core/optimizers/idbd.py)

### Autostep — Mahmood et al. 2012
Per-weight adaptive step-sizes with self-regulated normalizers (`v_i` tracks meta-gradient `|delta*x*h|`) and overshoot prevention (`M = max(sum(alpha_i*x_i^2), 1)`). Key: `tau` is a time constant (default 10000), `v_i`/`h_i` init to 0. See `core/optimizers.py` for full pseudocode.

### ObGD Bounding — Elsayed et al. 2024
Dynamic update bounding via `Bounder` ABC. Decoupled from optimizer (unlike paper). `M = kappa * max(|error|, 1) * sum(|step_i|)`, `scale = 1/max(M, 1)`. Generalizes to per-weight step-sizes (Autostep).

### AGC Bounding — Brock et al. 2021
Per-unit gradient clipping scaled by weight norm. Fine-grained unlike ObGD's global scale.

### Online Normalization
`Normalizer` ABC: `EMANormalizer` (non-stationary), `WelfordNormalizer` (stationary). Both learners accept optional `normalizer` parameter.

### MLP Learner — Elsayed et al. 2024
`Input -> [Dense -> LayerNorm -> LeakyReLU] x N -> Dense(1)`. Sparse init (90%), composable optimizer/bounder/normalizer, optional `head_optimizer` for trunk/head split. Toggleable `use_layer_norm`.

### MultiHeadMLPLearner
Shared trunk, N independent heads. VJP with accumulated cotangents. NaN targets mask inactive heads. Same composability as MLPLearner. Supports `hidden_sizes=()` for linear baseline. Serves as the foundation for the Horde architecture (Step 3) — each head is a GVF demon.

### GVF / Horde (Step 3 — Next)
General Value Functions (Sutton et al. 2011) represent knowledge as value functions with four question functions: policy π, pseudo-termination γ, pseudo-reward r (cumulant), and pseudo-terminal-reward z. A **prediction demon** has a fixed π; a **control demon** has π = greedy(q̂). The Horde is many demons learning in parallel — `MultiHeadMLPLearner` is the proto-Horde. rlsecd's 5 heads are implicit prediction demons (γ=0, π=behavior). Step 3 formalizes this with `GVFSpec` types and adds TD(λ) eligibility traces for MLP. Step 4 (SARSA) then adds a control demon to the Horde.

### Key Features (brief)
- **Single-step API**: `predict()`/`update()` with unbatched 1D obs for daemon use, JIT-compiled automatically (see `docs/guide/daemon-usage.md`)
- **Checkpoint utils**: `save_checkpoint`/`load_checkpoint`/`load_checkpoint_metadata`/`checkpoint_exists` (Orbax)
- **Config serialization**: `to_config()`/`from_config()` on all components; dispatchers: `optimizer_from_config()`, etc.
- **Feature diagnostics**: `compute_feature_relevance` (zero-cost), `compute_feature_sensitivity` (Jacobian)
- **Lifecycle tracking**: `step_count`, `birth_timestamp`, `uptime_s` on all learner states; `agent_age_s()`, `agent_uptime_s()`
- **Step-size tracking**: `StepSizeTrackingConfig` inside scan loops; Autostep `v_i` normalizers tracked automatically
- **Normalizer tracking**: `NormalizerTrackingConfig` for reactive lag analysis
- **Batched loops**: `run_*_batched()` via `jax.vmap` over seeds (2-5x speedup)
- **Gymnasium**: `collect_trajectory` + `learn_from_trajectory`; PredictionMode (REWARD, NEXT_STATE, VALUE)
- **Publication utils**: `run_multi_seed_experiment`, `pairwise_comparisons`, `plot_learning_curves`, `generate_latex_table`
- **Streams**: RandomWalk, AbruptChange, Cyclic, Periodic, ScaledStreamWrapper, DynamicScaleShift, ScaleDrift
- **bsuite**: Q-learning via MultiHeadMLPLearner (n_heads=num_actions), ContinuingWrapper, 3 agents (autostep/lms/adam)
- **Timer**: `with Timer("name"):` context manager for runtime reporting

## Version Management

Versioning: patch (bug fixes), minor (new features), major (breaking changes). Check `pyproject.toml` before bumping.

### CI/CD
- **ci.yml**: tests + linting on push/PR
- **docs.yml**: GitHub Pages deployment
- **publish.yml**: PyPI on version tags (`v0.X.Y`), uses OIDC trusted publishing

### Publishing
```bash
# Update version in pyproject.toml, commit, then:
git tag v0.X.Y && git push --tags
```

## Project Status

See **ROADMAP.md** for the Alberta Plan 12-step progression and **TODO.md** for immediate work items. See **CHANGELOG.md** for full version history.
