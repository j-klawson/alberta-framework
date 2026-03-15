# SARSA Control (Step 4a)

On-policy temporal-difference control using the Horde architecture.

## Architecture

`SARSAAgent` wraps a `HordeLearner` with epsilon-greedy action selection
and SARSA target computation. Each discrete action maps to a **control
demon** (head) in the Horde.

```
Observation -> [Shared Trunk] -> {Q(s, a_0), Q(s, a_1), ..., Q(s, a_n)}
                                   Control demons (one per action)
                                 {V_pred_0, V_pred_1, ...}
                                   Optional prediction demons
```

### SARSA Target Computation

The SARSA target `r + gamma * Q(s', a')` is computed externally by the
`SARSAAgent` and passed as the cumulant to the corresponding control
demon. Control demons use `gamma=0` internally (single-step prediction
of the externally-computed target). The real discount factor lives in
`SARSAConfig.gamma`.

This design avoids modifying the Horde's TD target logic and lets
prediction demons coexist with control demons in the same Horde.

### NaN Masking

Only the taken action's head receives a target on each step. All other
Q-heads get `NaN`, preserving their parameters, traces, and optimizer
states. This is the same pattern used by bsuite's `AlbertaAgent`.

## Usage

### Basic SARSA on CartPole

```python
import jax.random as jr
from alberta_framework import (
    SARSAAgent, SARSAConfig, ObGDBounding, Autostep,
    run_sarsa_episode,
)

config = SARSAConfig(
    n_actions=2,
    gamma=0.99,
    epsilon_start=0.5,
    epsilon_end=0.01,
    epsilon_decay_steps=5000,
)

agent = SARSAAgent(
    sarsa_config=config,
    hidden_sizes=(64, 32),
    optimizer=Autostep(initial_step_size=0.01),
    bounder=ObGDBounding(kappa=2.0),
)

state = agent.init(feature_dim=4, key=jr.key(42))

import gymnasium as gym
env = gym.make("CartPole-v1")
result = run_sarsa_episode(agent, state, env, max_steps=500)
print(f"Episode reward: {result.total_reward}")
```

### Continuing Mode (Daemon-Style)

For streaming environments that never terminate (e.g., rlsecd monitoring
server logs), use `run_sarsa_continuing`:

```python
from alberta_framework import run_sarsa_continuing

result = run_sarsa_continuing(agent, state, env, num_steps=10000)
```

At episode boundaries, `gamma` is set to 0 (no bootstrapping across
resets), matching the `ContinuingWrapper` pattern from bsuite.

### Scan-Based Learning (Pre-Collected Data)

For security-gym data or other pre-collected arrays, use the JIT-compiled
scan loop:

```python
from alberta_framework import run_sarsa_from_arrays

result = run_sarsa_from_arrays(
    agent, state, observations, rewards, terminated, next_observations
)
```

### Mixed Prediction + Control Demons

Add prediction demons alongside Q-heads:

```python
from alberta_framework import GVFSpec, DemonType

pred_demons = [
    GVFSpec(
        name="is_malicious",
        demon_type=DemonType.PREDICTION,
        gamma=0.0, lamda=0.0,
        cumulant_index=0,
    ),
]

agent = SARSAAgent(
    sarsa_config=config,
    hidden_sizes=(64, 32),
    prediction_demons=pred_demons,
)
```

## Relationship to rlsecd

rlsecd's `--gym-control` mode can use `SARSAAgent` with 6 action heads
(security-gym's action space: allow, throttle, challenge, block, isolate,
alert). The existing 5 prediction heads become prediction demons in the
same Horde. The agent transitions from passive prediction to active
defense.

## Trunk Trace Constraint

`MultiHeadMLPLearner` requires trunk `gamma * lamda = 0` when hidden
layers are present. The VJP backward pass folds per-head errors into the
trunk cotangent before trace accumulation, which is only correct when
traces reset each step. `HordeLearner` enforces this by setting trunk
`gamma=0, lamda=0` and applying per-head trace decay only to head layers.
