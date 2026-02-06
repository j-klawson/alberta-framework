# Alberta Framework Roadmap

Building the foundations of Continual AI, one step at a time.

The Alberta Framework follows the 12-step "retreat and return" strategy from the Alberta Plan for AI Research (Sutton et al., 2022). Each step builds on the previous, starting from the simplest possible setting and incrementally adding complexity.

## Step 1: Meta-Learned Step-Sizes — Complete (v0.1.0–v0.4.0)

**Goal**: Demonstrate that IDBD and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS on non-stationary supervised learning problems.

**Delivered**:
- LMS, IDBD (Sutton 1992), Autostep (Mahmood et al. 2012) optimizers
- Linear learners with pluggable optimizers
- Online feature normalization
- JIT-compiled scan-based learning loops with `jax.lax.scan`
- Batched multi-seed experiments via `jax.vmap`
- Step-size and normalizer tracking for meta-adaptation analysis
- TD-IDBD and AutoTDIDBD for temporal-difference learning (Kearney et al. 2019)
- Publication-quality experiment infrastructure (statistics, visualization, export)

## Step 2: Nonlinear Function Approximation — In Progress (v0.5.0)

**Goal**: Extend from linear to nonlinear function approximation while maintaining streaming, single-step updates. Demonstrate that ObGD's overshooting prevention enables stable MLP learning in the continual setting.

**Delivered (v0.5.0)**:
- ObGD optimizer (Elsayed et al. 2024) with dynamic step-size bounding
- MLPLearner with parameterless LayerNorm, LeakyReLU, sparse initialization
- `run_mlp_learning_loop` for JIT-compiled MLP training

**Planned**:
- Feature generation and testing ("generate and test" mechanisms)
- Nonlinear feature discovery for streaming problems
- Comparison studies across diverse non-stationarity types

## Step 3: Prediction — Planned

**Goal**: Move from supervised prediction to General Value Function (GVF) predictions. Learn to predict "anything" as a cumulant signal.

**Key components**:
- Stream TD(lambda) with linear and nonlinear function approximation
- GVF specification (cumulant, discount, policy)
- Horde architecture: many GVFs learning in parallel
- Integration with ObGD and MLP learners

## Step 4: Control — Planned

**Goal**: Introduce action selection. Move from prediction-only agents to actor-critic control.

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
- Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012). "Tuning-free Step-size Adaptation"
- Kearney, A., Veeriah, V., Travnik, J., Pilarski, P.M., & Sutton, R.S. (2019). "Learning Feature Relevance Through Step Size Adaptation in Temporal-Difference Learning"
- Sutton, R.S., et al. (2022). "The Alberta Plan for AI Research"
- Elsayed, M., Lan, Q., Lyle, C., & Mahmood, A.R. (2024). "Streaming Deep Reinforcement Learning Finally Works"
