# TODO

Immediate next steps and near-term work items for the Alberta Framework.

## Step 2 — Remaining Work

- [ ] Neuron utility tracking (per-hidden-unit EMA of gradient magnitude)
- [ ] Feature generation and testing ("generate and test" mechanisms)
- [ ] Nonlinear feature discovery for streaming problems
- [ ] Comparison studies: MLPLearner across diverse non-stationarity types (drift, abrupt, periodic)
- [ ] AdaptiveObGD (Appendix B of Elsayed et al. 2024) — RMSProp-style second-moment normalization
- [ ] More bsuite sweep experiments and analysis (beyond catch/cartpole)

## Step 3 — GVF Predictions (Next Major Step)

- [ ] GVF specification type (cumulant, discount, policy)
- [ ] Stream TD(lambda) with MLP function approximation
- [ ] Horde architecture: many GVFs learning in parallel via MultiHeadMLPLearner
- [ ] Integration with ObGD bounding for stable GVF learning

## Step 4a — SARSA (On-Policy TD Control)

First control algorithm. Enables rlsecd to take actions in security-gym's action space.

### Core Types
- [ ] `SARSATimeStep` in `core/types.py` — extends `TDTimeStep` with `action: int` and `next_action: int`
- [ ] `SARSAUpdate` result type — includes `td_error`, `action_taken`, `q_values` for diagnostics

### SARSAAgent
- [ ] `SARSAAgent` class in `core/sarsa.py` (or `core/control.py`)
  - Wraps `MultiHeadMLPLearner` (n_heads = n_actions)
  - `select_action(state, obs, key) -> action` — ε-greedy over Q-values from all heads
  - `sarsa_update(state, obs, action, reward, next_obs, next_action, gamma) -> (state, metrics)` — SARSA target computation + head update via NaN masking
  - Configurable: `epsilon` (exploration rate), `epsilon_decay` (optional schedule), `epsilon_min`
  - All existing composable components: Optimizer + Bounder + Normalizer
- [ ] SARSA(λ) variant with eligibility traces per action-head

### Learning Loops
- [ ] `run_sarsa_episode(agent, env, state, key) -> (state, metrics)` — single episode gymnasium loop
- [ ] `run_sarsa_continuing(agent, env, state, key, num_steps) -> (state, metrics)` — continuing (non-episodic) loop for streaming environments
- [ ] Scan-compatible step function for JIT compilation

### Testing
- [ ] Unit tests: SARSA update rule correctness (known MDP with hand-computed Q-values)
- [ ] On-policy vs off-policy: verify SARSA learns different Q-values than Q-learning under ε-greedy
- [ ] Trace test: SARSA(λ=0) matches one-step SARSA
- [ ] bsuite catch/cartpole comparison: SARSA agent alongside existing DQN agents
- [ ] Integration: `SARSAAgent` with Autostep + ObGD + EMA (the winning rlsecd combo)

### Downstream Integration (rlsecd)
- [ ] Validate SARSAAgent works with 12-dim obs × 6-action security-gym space
- [ ] Benchmark throughput: SARSA predict+update must sustain >1000 evt/s on CPU (rlsecd requirement)

## rlsecd Integration

- [x] AF-1: Checkpoint utilities — `save_checkpoint`/`load_checkpoint` + `to_config()`/`from_config()` (rlsecd needs to consume)
- [x] AF-3: Document single-step learner API for daemon use (`docs/guide/daemon-usage.md`)
- [x] AF-4: JIT-compile `predict()`/`update()` on MLPLearner and MultiHeadMLPLearner (upstream)
- [x] AF-2: Get permission from Edan Meyer to publish IDBD-MLP
- [x] AF-2: Merge IDBD-MLP into main (Meyer adaptation with IDBDParamState, 18 tests)
- [ ] AF-2: IDBD-MLP 100k-event replay test in rlsecd
- [ ] AF-2: IDBD-MLP full 1.6M log stability test
- [ ] Simplify rlsecd SecurityAgent to use framework checkpoint utilities
- [ ] Simplify rlsecd SecurityAgent to use framework config serialization
- [ ] Integrate `compute_feature_relevance` into rlsecd periodic reporting (60s interval)

## Infrastructure

- [ ] Update CHANGELOG.md with each release (moved from CLAUDE.md)
- [ ] Keep bsuite running on Python 3.13 via PYTHONPATH workaround
