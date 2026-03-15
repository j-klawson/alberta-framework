# TODO

Immediate next steps and near-term work items for the Alberta Framework.

## Step 2 — Remaining Work

- [ ] Neuron utility tracking (per-hidden-unit EMA of gradient magnitude)
- [ ] Feature generation and testing ("generate and test" mechanisms)
- [ ] Nonlinear feature discovery for streaming problems
- [ ] Comparison studies: MLPLearner across diverse non-stationarity types (drift, abrupt, periodic)
- [ ] AdaptiveObGD (Appendix B of Elsayed et al. 2024) — RMSProp-style second-moment normalization
- [ ] More bsuite sweep experiments and analysis (beyond catch/cartpole)

## Step 3 — GVF Prediction & Horde

Formalize rlsecd's multi-head predictions as GVF demons (Sutton et al. 2011, "Horde"), extend to temporal predictions with eligibility traces, and build the prediction infrastructure that Step 4 control will use. Per the Alberta Plan (Sutton et al. 2022, p.8): the critic in Step 4 "would presumably be that resulting from Steps 1-3."

### Phase 1: GVF Types & Demon Specification — Complete (v0.15.0)
- [x] `GVFSpec` dataclass in `core/types.py` — question functions: cumulant_index, gamma, lamda, terminal_reward
- [x] `DemonType` enum: `PREDICTION` (fixed target policy) vs `CONTROL` (π = greedy(q̂))
- [x] `HordeSpec` — collection of `GVFSpec` entries with pre-computed gamma/lambda arrays
- [x] Validate types by expressing rlsecd's 5 heads as `GVFSpec` instances (all γ=0, π=behavior)
- [x] Config serialization: `GVFSpec.to_config()` / `from_config()` consistent with existing patterns
- [x] `HordeLearner` wrapping `MultiHeadMLPLearner` with per-demon gamma/lambda and TD targets
- [x] Per-head trace decay via `per_head_gamma_lamda` on `MultiHeadMLPLearner`
- [x] Scan-based Horde learning loop + batched variant
- [x] Test: Horde with all γ=0 demons matches `MultiHeadMLPLearner` behavior exactly
- [x] Trunk trace guard: `MultiHeadMLPLearner` validates trunk `gamma * lamda = 0` when hidden layers present

### Phase 2: TD(λ) Eligibility Traces for MLP
- [ ] Per-parameter eligibility trace arrays on `MultiHeadMLPLearner` (matching weight shapes in each layer)
- [ ] TD(λ) update rule: `e_t = γ_t * λ * e_{t-1} + ∇_θ q̂(s,a)` integrated with Optimizer/Bounder composition
- [ ] Trace decay λ configurable per demon (Horde §4: λ is an "answer function")
- [ ] Accumulating vs replacing traces option
- [ ] Integration with ObGD bounding — traces must respect bounding scale factor
- [ ] Test: TD(λ=0) reduces to existing single-step MLP update
- [ ] Test: linear MLP (`hidden_sizes=()`) with traces matches `TDLinearLearner` results

### Phase 3: Off-Policy Prediction (Stretch)
- [ ] Importance sampling ratios π(s,a)/b(s,a) per demon
- [ ] GQ(λ) or GTD(λ) for stable off-policy learning with function approximation (Maei & Sutton 2010)
- [ ] Off-policy prediction demon test: learn about a policy different from behavior
- [ ] Test on security-gym: "what would happen if we blocked this IP?" (prediction about untaken action)

## Step 4a — SARSA (On-Policy TD Control) — Complete (v0.16.0)

- [x] `SARSAAgent` wrapping `HordeLearner` — control demons + optional prediction demons
- [x] Control demons: gamma=0 internally, real discount in `SARSAConfig.gamma`, SARSA target computed externally
- [x] ε-greedy action selection with Gumbel trick tie-breaking, linear epsilon decay
- [x] NaN-masking: only taken action's head updated per step
- [x] `run_sarsa_episode` — single episode gymnasium loop
- [x] `run_sarsa_continuing` — continuing loop for streaming environments (daemon-style)
- [x] `run_sarsa_from_arrays` — JIT-compiled scan for pre-collected data (security-gym pattern)
- [x] Config serialization roundtrip via `to_config()` / `from_config()`
- [x] 30 tests: init, action selection, update logic, epsilon decay, bounding, serialization, scan
- [x] Example: `sarsa_cartpole.py` (episodic CartPole)
- [x] Example: `sarsa_trading.py` (continuing FOREX/STOCKS via gym-anytrading)
- [x] Documentation: `docs/guide/sarsa-control.md`

### Remaining Step 4a Work
- [ ] bsuite catch/cartpole comparison: SARSA agent alongside existing DQN agents
- [ ] Multi-seed statistical comparison of SARSA vs Q-learning (AlbertaAgent)

### Downstream Integration (rlsecd)
- [ ] rlsecd `--gym-control` mode: existing 5 prediction demons + SARSA control demon
- [ ] Maps 6 security-gym actions (pass/alert/throttle/block/unblock/isolate) to action heads
- [ ] Validate throughput: predict+update must sustain >1000 evt/s on CPU
- [ ] Generate (state, action, reward, outcome) experience for autoresearch LLM oracle pipeline

## Step 4b — Actor-Critic (Planned)

- [ ] Stream AC(lambda): Actor-critic with eligibility traces
- [ ] Policy gradient with ObGD-style overshooting prevention
- [ ] Continuous and discrete action spaces

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
