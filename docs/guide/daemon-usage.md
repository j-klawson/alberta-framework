# Daemon / Single-Step Usage

This guide covers using alberta-framework learners in daemon-style deployments
where observations arrive one at a time (e.g. [rlsecd](https://github.com/j-klawson/rlsecd)).

## Single-Step API

Both `MLPLearner` and `MultiHeadMLPLearner` accept single unbatched 1D
observations. This is the intended usage for online, per-event processing:

```python
import jax.numpy as jnp
import jax.random as jr
from alberta_framework import MultiHeadMLPLearner, LMS, ObGDBounding, EMANormalizer

learner = MultiHeadMLPLearner(
    n_heads=5,
    hidden_sizes=(64, 64),
    optimizer=LMS(step_size=0.01),
    bounder=ObGDBounding(),
    normalizer=EMANormalizer(),
)
state = learner.init(feature_dim=12, key=jr.key(0))

# Single observation in, predictions out
observation = jnp.ones(12)
predictions = learner.predict(state, observation)  # shape (5,)

# Single-step update with NaN masking for inactive heads
targets = jnp.array([1.0, 0.5, jnp.nan, 0.3, jnp.nan])  # heads 2,4 inactive
result = learner.update(state, observation, targets)
state = result.state  # carry forward
```

## JIT Compilation

`predict()` and `update()` are JIT-compiled automatically on both
`MLPLearner` and `MultiHeadMLPLearner`. The first call triggers JAX's
tracing; subsequent calls reuse the cached compilation.

For low-latency startup (avoiding a slow first real event), run a warmup
call during initialization:

```python
# Warmup at daemon startup
dummy_obs = jnp.zeros(feature_dim)
dummy_targets = jnp.full(n_heads, jnp.nan)
learner.predict(state, dummy_obs).block_until_ready()
learner.update(state, dummy_obs, dummy_targets)
# First real event is now fast (~0.3ms vs ~20ms without JIT)
```

!!! note "Scan loops are unaffected"
    The `jax.lax.scan`-based learning loops (e.g. `run_multi_head_learning_loop`)
    already compile the outer scan. Nested JIT is a no-op in JAX, so the
    built-in JIT on `predict`/`update` adds zero overhead in scan contexts.

## Checkpoints

Save and restore learner state across daemon restarts:

```python
from alberta_framework import save_checkpoint, load_checkpoint

# Save state + daemon metadata
save_checkpoint(state, "agent.ckpt", metadata={
    "total_updates": 100_000,
    "daemon_version": "1.0",
})

# Load (template provides PyTree structure)
template = learner.init(feature_dim=12, key=jr.key(0))
loaded_state, meta = load_checkpoint(template, "agent.ckpt")
print(meta["total_updates"])  # 100000
```

## Config Serialization

Round-trip the learner configuration (architecture, optimizer, bounder,
normalizer) as a JSON-serializable dict:

```python
# Save config
config = learner.to_config()
import json
with open("learner_config.json", "w") as f:
    json.dump(config, f)

# Reconstruct learner from config
with open("learner_config.json") as f:
    config = json.load(f)
learner = MultiHeadMLPLearner.from_config(config)
state = learner.init(feature_dim=12, key=jr.key(0))
```

This pairs with checkpoints: save the config alongside the state so a
daemon can fully reconstruct itself on restart without hardcoding
architecture parameters.

## Feature Diagnostics

For periodic reporting (e.g. every 60s), extract per-feature relevance
from the learner state at zero cost:

```python
from alberta_framework import compute_feature_relevance, relevance_to_dict

relevance = compute_feature_relevance(state)
report = relevance_to_dict(
    relevance,
    feature_names=["src_ip", "dst_port", "payload_len", ...],
    head_names=["is_malicious", "attack_type", "stage", "severity", "value"],
)
# report is a JSON-serializable dict ready for logging/storage
```

For deeper analysis, compute input sensitivity via Jacobian (one forward
pass per head, ~100-500us):

```python
from alberta_framework import compute_feature_sensitivity

jacobian = compute_feature_sensitivity(learner, state, observation)
# shape: (n_heads, feature_dim) — sensitivity of each head to each input
```

## Complete Daemon Pattern

Putting it all together:

```python
import jax.numpy as jnp
import jax.random as jr
from alberta_framework import (
    MultiHeadMLPLearner, LMS, ObGDBounding, EMANormalizer,
    save_checkpoint, load_checkpoint,
    compute_feature_relevance, relevance_to_dict,
)

FEATURE_DIM = 12
N_HEADS = 5
CHECKPOINT_PATH = "agent.ckpt"

# 1. Create or restore learner
learner = MultiHeadMLPLearner(
    n_heads=N_HEADS,
    hidden_sizes=(64, 64),
    optimizer=LMS(step_size=0.01),
    bounder=ObGDBounding(),
    normalizer=EMANormalizer(),
)
state = learner.init(feature_dim=FEATURE_DIM, key=jr.key(42))

# Optional: restore from checkpoint
# template = learner.init(feature_dim=FEATURE_DIM, key=jr.key(0))
# state, meta = load_checkpoint(template, CHECKPOINT_PATH)

# 2. Warmup JIT
dummy_obs = jnp.zeros(FEATURE_DIM)
dummy_targets = jnp.full(N_HEADS, jnp.nan)
learner.predict(state, dummy_obs).block_until_ready()
learner.update(state, dummy_obs, dummy_targets)

# 3. Event loop
for event in event_source:
    obs = jnp.array(event.features, dtype=jnp.float32)

    # Predict
    predictions = learner.predict(state, obs)

    # Update (if labels available)
    if event.has_labels:
        targets = jnp.array(event.targets, dtype=jnp.float32)
        result = learner.update(state, obs, targets)
        state = result.state

    # Periodic checkpoint
    if event.step % 10_000 == 0:
        save_checkpoint(state, CHECKPOINT_PATH)

    # Periodic diagnostics
    if event.step % 1_000 == 0:
        rel = compute_feature_relevance(state)
        report = relevance_to_dict(rel)
        log_diagnostics(report)
```
