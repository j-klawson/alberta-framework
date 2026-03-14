# TODO

Immediate next steps and near-term work items for the Alberta Framework.

## Step 2 — Remaining Work

- [ ] Neuron utility tracking (per-hidden-unit EMA of gradient magnitude)
- [ ] Feature generation and testing ("generate and test" mechanisms)
- [ ] Nonlinear feature discovery for streaming problems
- [ ] Comparison studies: MLPLearner across diverse non-stationarity types (drift, abrupt, periodic)
- [ ] AdaptiveObGD (Appendix B of Elsayed et al. 2024) — RMSProp-style second-moment normalization
- [ ] More bsuite sweep experiments and analysis (beyond catch/cartpole)

## Step 3 — GVF Prediction & Horde (Active — Next Major Step)

Formalize rlsecd's multi-head predictions as GVF demons (Sutton et al. 2011, "Horde"), extend to temporal predictions with eligibility traces, and build the prediction infrastructure that Step 4 control will use. Per the Alberta Plan (Sutton et al. 2022, p.8): the critic in Step 4 "would presumably be that resulting from Steps 1-3."

### Phase 1: GVF Types & Demon Specification
- [ ] `GVFSpec` dataclass in `core/types.py` — four question functions: `cumulant_fn`, `gamma_fn`, `policy`, `terminal_reward_fn` (Horde §3)
- [ ] `DemonType` enum: `PREDICTION` (fixed target policy) vs `CONTROL` (π = greedy(q̂))
- [ ] `HordeSpec` — collection of `GVFSpec` entries, one per head
- [ ] Validate types by expressing rlsecd's 5 heads as `GVFSpec` instances (all γ=0, π=behavior)
- [ ] Config serialization: `GVFSpec.to_config()` / `from_config()` consistent with existing patterns

### Phase 2: TD(λ) Eligibility Traces for MLP
- [ ] Per-parameter eligibility trace arrays on `MultiHeadMLPLearner` (matching weight shapes in each layer)
- [ ] TD(λ) update rule: `e_t = γ_t * λ * e_{t-1} + ∇_θ q̂(s,a)` integrated with Optimizer/Bounder composition
- [ ] Trace decay λ configurable per demon (Horde §4: λ is an "answer function")
- [ ] Accumulating vs replacing traces option
- [ ] Integration with ObGD bounding — traces must respect bounding scale factor
- [ ] Test: TD(λ=0) reduces to existing single-step MLP update
- [ ] Test: linear MLP (`hidden_sizes=()`) with traces matches `TDLinearLearner` results

### Phase 3: Horde Learning Loop
- [ ] `HordeLearner` class (or extend `MultiHeadMLPLearner`) accepting `HordeSpec`
- [ ] Per-demon TD target computation from question functions: `δ_i = r_i + γ_i * q̂_i(s') - q̂_i(s)`
- [ ] Mixed-γ heads: some demons γ=0 (single-step), others γ>0 (temporal predictions)
- [ ] Scan-based Horde learning loop for JIT compilation
- [ ] Prediction testbed: security-gym streams with γ>0 demons (e.g., "will session become malicious in next N events?")
- [ ] Prediction testbed: random walk streams from Step 1 with TD(λ) GVF predictions
- [ ] Test: Horde with all γ=0 demons matches current `MultiHeadMLPLearner` behavior exactly

### Phase 4: Off-Policy Prediction (Stretch)
- [ ] Importance sampling ratios π(s,a)/b(s,a) per demon
- [ ] GQ(λ) or GTD(λ) for stable off-policy learning with function approximation (Maei & Sutton 2010)
- [ ] Off-policy prediction demon test: learn about a policy different from behavior
- [ ] Test on security-gym: "what would happen if we blocked this IP?" (prediction about untaken action)

## Step 4a — SARSA (On-Policy TD Control)

First control demon in the Horde. Builds on Step 3 GVF infrastructure — SARSA is a `GVFSpec` where `policy=greedy(q̂)` and `cumulant=reward`. Enables rlsecd to transition from passive prediction to active defense.

### Core Types
- [ ] `SARSAAgent` wrapping `HordeLearner` — prediction demons + one control demon
- [ ] Control demon spec: `GVFSpec(cumulant=reward, gamma=0.99, policy=greedy, type=CONTROL)`
- [ ] ε-greedy behavior policy for action selection (configurable ε, optional decay)
- [ ] Per-action heads via existing NaN-masking mechanism

### Learning Loops
- [ ] `run_sarsa_episode(agent, env, state, key)` — single episode gymnasium loop
- [ ] `run_sarsa_continuing(agent, env, state, key, num_steps)` — continuing loop for streaming environments
- [ ] Scan-compatible step function for JIT compilation

### Testing
- [ ] Unit tests: SARSA update rule correctness (known MDP with hand-computed Q-values)
- [ ] On-policy vs off-policy: verify SARSA learns different Q-values than Q-learning under ε-greedy
- [ ] Trace test: SARSA(λ=0) matches one-step SARSA
- [ ] bsuite catch/cartpole comparison: SARSA agent alongside existing DQN agents
- [ ] Integration: SARSA with Autostep + ObGD + EMA (the winning rlsecd combo)

### Downstream Integration (rlsecd)
- [ ] rlsecd `--gym-control` mode: existing 5 prediction demons + SARSA control demon
- [ ] Maps 6 security-gym actions (pass/alert/throttle/block/unblock/isolate) to action heads
- [ ] Validate throughput: predict+update must sustain >1000 evt/s on CPU
- [ ] Generate (state, action, reward, outcome) experience for autoresearch LLM oracle pipeline

## rlsecd Integration

- [x] AF-1: Checkpoint utilities — `save_checkpoint`/`load_checkpoint` + `to_config()`/`from_config()` (rlsecd needs to consume)
- [x] AF-1: Orbax checkpointing migration — replaced npz+json with `orbax-checkpoint`, added `load_checkpoint_metadata`/`checkpoint_exists`
- [x] AF-3: Document single-step learner API for daemon use (`docs/guide/daemon-usage.md`)
- [x] AF-4: JIT-compile `predict()`/`update()` on MLPLearner and MultiHeadMLPLearner (upstream)
- [x] AF-2: Get permission from Edan Meyer to publish IDBD-MLP
- [x] AF-2: Merge IDBD-MLP into main (Meyer adaptation with IDBDParamState, 18 tests)
- [ ] AF-2: IDBD-MLP 100k-event replay test in rlsecd
- [ ] AF-2: IDBD-MLP full 1.6M log stability test
- [ ] Simplify rlsecd SecurityAgent to use Orbax checkpoint utilities (format v2)
- [ ] Simplify rlsecd SecurityAgent to use framework config serialization
- [ ] Integrate `compute_feature_relevance` into rlsecd periodic reporting (60s interval)

## Infrastructure

- [ ] Update CHANGELOG.md with each release (moved from CLAUDE.md)
- [ ] Keep bsuite running on Python 3.13 via PYTHONPATH workaround
