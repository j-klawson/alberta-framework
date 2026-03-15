# Alberta Framework Roadmap

Building the foundations of Continual AI, one step at a time.

The Alberta Framework follows the 12-step "retreat and return" strategy from the Alberta Plan for AI Research (Sutton et al., 2022). Each step builds on the previous, starting from the simplest possible setting and incrementally adding complexity.

## Step 1: Meta-Learned Step-Sizes — Complete (v0.1.0–v0.4.0)

**Goal**: Demonstrate that IDBD and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS on non-stationary supervised learning problems.

**Delivered**:
- LMS, IDBD (Sutton 1992), Autostep (Mahmood et al. 2012) optimizers
- Linear learners with pluggable optimizers
- Online feature normalization (EMA and Welford)
- JIT-compiled scan-based learning loops with `jax.lax.scan`
- Batched multi-seed experiments via `jax.vmap`
- Step-size and normalizer tracking for meta-adaptation analysis
- TD-IDBD and AutoTDIDBD for temporal-difference learning (Kearney et al. 2019)
- Publication-quality experiment infrastructure (statistics, visualization, export)
- Factorial studies with multiple non-stationarity types and scale ranges

## Step 2: Nonlinear Function Approximation — In Progress (v0.5.0–v0.9.0)

**Goal**: Extend from linear to nonlinear function approximation while maintaining streaming, single-step updates. Demonstrate that ObGD's overshooting prevention enables stable MLP learning in the continual setting.

**Delivered**:
- ObGD bounding (Elsayed et al. 2024) with dynamic step-size bounding, decoupled as `Bounder` ABC
- AGC bounding (Brock et al. 2021) for per-unit adaptive gradient clipping
- `MLPLearner` with parameterless LayerNorm, LeakyReLU, sparse initialization
- Composable architecture: any Optimizer + optional Bounder + optional Normalizer
- `MultiHeadMLPLearner` for multi-task continual learning (shared trunk, NaN masking)
- bsuite benchmark integration with Q-learning agents (Autostep DQN, LMS DQN, Adam DQN)
- `ContinuingWrapper` for episodic-to-continuing conversion (Alberta Plan Step 6 preview)
- Agent lifecycle tracking (`step_count`, `birth_timestamp`, `uptime_s`)
- Representation utility logging for bsuite experiments
- Hybrid optimizer (`head_optimizer`) for trunk/head optimizer split on MLPLearner and MultiHeadMLPLearner
- Checkpoint utilities (`save_checkpoint`/`load_checkpoint`) for persisting learner state
- Learner config serialization (`to_config`/`from_config`) for all learners, optimizers, bounders, and normalizers
- Built-in JIT compilation on `MLPLearner` and `MultiHeadMLPLearner` `predict()`/`update()` methods
- Daemon usage guide (`docs/guide/daemon-usage.md`) covering single-step API, JIT warmup, checkpoints, config serialization, and feature diagnostics
- `MultiHeadMLPLearner` linear baseline support (`hidden_sizes=()`)
- Feature relevance diagnostics (`compute_feature_relevance`, `compute_feature_sensitivity`, `relevance_to_dict`) for periodic daemon reporting
- IDBD-MLP optimizer (Meyer): per-parameter adaptive step-sizes for MLPs via `IDBDParamState`, with `h_decay_mode` (`prediction_grads`/`loss_grads`)
- Orbax checkpointing: replaced hand-rolled npz+json with `orbax-checkpoint` for versioned pytree serialization; added `load_checkpoint_metadata` and `checkpoint_exists` utilities

**Planned**:
- Neuron utility tracking (per-hidden-unit EMA of gradient magnitude)
- Feature generation and testing ("generate and test" mechanisms)
- Nonlinear feature discovery for streaming problems
- Comparison studies across diverse non-stationarity types
- AdaptiveObGD (Appendix B of Elsayed et al. 2024) with RMSProp-style second-moment normalization

## Step 3: GVF Prediction & Horde — Phase 1 Complete (v0.15.0)

**Goal**: Move from supervised prediction to General Value Function (GVF) predictions using the Horde architecture. Formalize rlsecd's existing multi-head predictions as GVF demons, extend to temporal predictions (γ > 0) with eligibility traces, and build the foundation that Step 4 control will use.

This follows the Alberta Plan's Step 3 ("Prediction I: Continual GVF prediction learning" — Sutton et al. 2022, p.8): *"Repeat the above two steps for sequential, real-time settings where the data is not i.i.d., but rather is from a process with state and the task is generalized value function (GVF) prediction."* The reference architecture is Horde (Sutton et al. 2011), where knowledge is represented as a large number of approximate value functions learned in parallel, each with its own policy, pseudo-reward, pseudo-termination, and pseudo-terminal-reward.

**Key insight**: rlsecd is already a Horde — its 5 prediction heads are implicit GVF demons with γ=0 (single-step prediction) and π=behavior (passive observation). Step 3 formalizes this, then extends to temporal predictions (γ > 0) and off-policy learning, which are prerequisites for control demons in Step 4.

### Phase 1: GVF Types & Demon Specification

A GVF demon is defined by four "question functions" (Sutton et al. 2011, §3):
- **π** (policy) — what behavior is this knowledge about?
- **γ** (pseudo-termination) — when does the prediction horizon end?
- **r** (pseudo-reward / cumulant) — what signal are we predicting?
- **z** (pseudo-terminal-reward) — what value at termination?

A demon with a fixed target policy π is a **prediction demon** (knowledge). A demon whose target policy is greedy w.r.t. its own GVF (π = greedy(q̂)) is a **control demon** (goals). Conventional value functions and SARSA Q-functions are special cases.

**Deliverables**:
- `GVFSpec` dataclass: `(cumulant_fn, gamma_fn, policy, terminal_reward_fn)` — the four question functions
- `DemonType` enum: `PREDICTION` vs `CONTROL`
- `HordeSpec`: collection of `GVFSpec` entries, one per head in `MultiHeadMLPLearner`
- Formalize rlsecd's 5 heads as `GVFSpec` instances (all γ=0, π=behavior — validates the types against an existing system)

### Phase 2: TD(λ) for MLP — Eligibility Traces

Eligibility traces are essential for temporal GVF predictions (γ > 0) and for efficient credit assignment in control. We have traces for linear TD (TD-IDBD/AutoTDIDBD) but not for MLP.

**Deliverables**:
- Eligibility traces on `MultiHeadMLPLearner`: per-parameter trace arrays matching weight shapes
- TD(λ) update rule integrated with existing Optimizer/Bounder composition
- Trace decay λ configurable per demon (part of GVF answer functions, per Horde §4)
- Accumulating vs replacing traces option
- Integration with ObGD bounding for stable TD learning with MLP

### Phase 3: Horde Learning Loop

**Deliverables**:
- `HordeLearner` or extend `MultiHeadMLPLearner` to accept `HordeSpec`
- Per-demon TD targets computed from each demon's question functions
- Per-demon γ handling (some heads γ=0 single-step, others γ>0 temporal)
- Scan-based Horde learning loop for JIT compilation
- Prediction testbed: predict-next-observation on security-gym streams, random walk streams

### Phase 4: Off-Policy Prediction (Stretch)

The Horde paper uses GQ(λ) (Maei & Sutton 2010) for off-policy learning — each demon can learn about a target policy π different from the behavior policy b. This requires importance sampling ratios π(s,a)/b(s,a).

**Deliverables**:
- Importance sampling ratio computation per demon
- GQ(λ) or GTD(λ) integration for stable off-policy learning with function approximation
- Off-policy prediction demons: e.g., "what would session risk be if we blocked this IP?"

### Downstream: rlsecd as a Formal Horde

rlsecd's current 5 heads become proper GVF demons:

| Head | Cumulant (r) | γ | π | Type |
|------|-------------|---|---|------|
| is_malicious | binary label | 0 | behavior | prediction |
| attack_type | categorical | 0 | behavior | prediction |
| severity | severity score | 0 | behavior | prediction |
| session_risk | EMA risk | 0 | behavior | prediction |
| next_event_type | next event | 0 | behavior | prediction |

Step 3 extends these to temporal predictions (γ > 0): "will this session become malicious in the next N events?" Then Step 4 adds a control demon that takes defensive actions.

### Decision Point: Discounted vs Average Reward

When implementing temporal GVF predictions (γ > 0), we need to decide whether to use discounted reward, average reward, or both. The Alberta group argues that discounted reward is "a hack" — the discount factor γ conflates two distinct roles (prediction horizon and value weighting), and average-reward formulations are more natural for continuing, non-episodic agents. The Alberta Plan explicitly calls for average-reward methods in Steps 5–6, so building on discounted reward first may create technical debt. On the other hand, discounted reward is simpler to implement initially and has more established tooling (e.g., standard TD(λ), SARSA). Decide before committing to the Phase 2/3 TD target computation.

## Step 4: Control — Step 4a Complete (v0.16.0)

**Goal**: Introduce action selection. Move from prediction-only agents to control using the GVF/Horde infrastructure from Step 3. A control demon is a GVF where π = greedy(q̂) — prediction and control are the same mechanism (Sutton et al. 2011, §4).

The Alberta Plan's Step 4 ("Control I: Continual actor-critic control" — Sutton et al. 2022, p.8) says: *"The critic would presumably be that resulting from Steps 1-3."* The GVF prediction machinery from Step 3 IS the critic.

### Step 4a: SARSA — On-Policy TD Control — Complete (v0.16.0)

**Goal**: Add the first control demon to the Horde. SARSA is on-policy (behavior policy = target policy), so no importance sampling is needed — the simplest possible control demon.

**Delivered**:
- `SARSAAgent` wrapping `HordeLearner` with epsilon-greedy action selection and SARSA target computation
- `SARSAConfig`: n_actions, gamma, epsilon schedule (linear decay)
- Control demons use gamma=0 internally; real discount in `SARSAConfig.gamma` (SARSA target computed externally)
- Gumbel trick tie-breaking for uniform action selection among equal Q-values
- NaN-masking: only taken action's head receives target per step
- Mixed Horde: optional prediction demons coexist with control demons
- Three learning loops: `run_sarsa_episode` (episodic), `run_sarsa_continuing` (daemon-style), `run_sarsa_from_arrays` (JIT-compiled scan)
- Integration with all composable components (Optimizer, Bounder, Normalizer)
- Config serialization roundtrip via `to_config()` / `from_config()`
- Trunk trace guard: `MultiHeadMLPLearner` validates trunk `gamma * lamda = 0` when hidden layers present
- 30 tests, example (`sarsa_cartpole.py`), documentation (`sarsa-control.md`)

**Downstream: rlsecd as active defender**:
- rlsecd gains `--gym-control` mode: existing prediction demons + one SARSA control demon
- Maps 6 security-gym actions (pass/alert/throttle/block/unblock/isolate) to action heads
- Prediction demons continue learning knowledge; control demon learns to act
- Generates (state, action, reward, outcome) experience for autoresearch LLM oracle pipeline

### Step 4b: Actor-Critic — Planned

**Key components**:
- Stream AC(lambda): Actor-critic with eligibility traces
- Policy gradient with ObGD-style overshooting prevention
- Continuous and discrete action spaces
- Gymnasium integration for control benchmarks

## Steps 5–6: Continuing Control — Future

**Goal**: Transition from episodic to continuing (average-reward) formulations, which are more natural for long-lived agents.

**Key components**:
- Average reward TD learning
- Differential value functions
- Continuing actor-critic methods

## Steps 7–12: Intelligence — Future

**Goal**: Integrate all components into a cohesive architecture.

- Planning with learned transition models
- Subtask and option discovery (STOMP progression)
- Hierarchical architectures
- Multi-agent coordination
- OaK: the integrated proto-AI agent

## References

- Sutton, R.S. (1992). "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta"
- Sutton, R.S., Modayil, J., Delp, M., Degris, T., Pilarski, P.M., White, A., & Precup, D. (2011). "Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction." *Proc. 10th AAMAS*, pp. 761–768.
- Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012). "Tuning-free Step-size Adaptation"
- Kearney, A., Veeriah, V., Travnik, J., Pilarski, P.M., & Sutton, R.S. (2019). "Learning Feature Relevance Through Step Size Adaptation in Temporal-Difference Learning"
- Sutton, R.S., et al. (2022). "The Alberta Plan for AI Research"
- Elsayed, M., Lan, Q., Lyle, C., & Mahmood, A.R. (2024). "Streaming Deep Reinforcement Learning Finally Works"
- Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021). "High-Performance Large-Scale Image Recognition Without Normalization"
- Maei, H.R. & Sutton, R.S. (2010). "GQ(λ): A general gradient algorithm for temporal-difference prediction learning with eligibility traces." *Proc. 3rd Conf. on AGI*.
- Meyer, E. (2025). "IDBD for MLPs" — https://github.com/ejmejm/phd_research/blob/main/phd/jax_core/optimizers/idbd.py
