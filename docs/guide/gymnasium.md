# Gymnasium Integration

The framework can wrap Gymnasium RL environments as experience streams for prediction learning.

!!! note "Optional Dependency"
    This requires the `gymnasium` extra: `pip install alberta-framework[gymnasium]`

## Overview

Gymnasium environments become prediction problems by predicting:

- **Rewards**: Predict immediate reward from (state, action)
- **Next states**: Predict next observation from (state, action)
- **Values**: Predict cumulative return (TD learning)

## Basic Usage

```python
from alberta_framework import LinearLearner, IDBD, run_learning_loop
from alberta_framework.streams.gymnasium import (
    make_gymnasium_stream,
    PredictionMode,
)

# Create a reward prediction stream
stream = make_gymnasium_stream(
    "CartPole-v1",
    mode=PredictionMode.REWARD,
    include_action_in_features=True,
    seed=42,
)

# Train a predictor
learner = LinearLearner(optimizer=IDBD())
state, metrics = run_learning_loop(
    learner=learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(0),
)
```

## Prediction Modes

### REWARD Mode

Predict the immediate reward:

- **Features**: Current state (optionally with action)
- **Target**: Reward received

```python
stream = make_gymnasium_stream(
    "CartPole-v1",
    mode=PredictionMode.REWARD,
)
```

### NEXT_STATE Mode

Predict the next observation:

- **Features**: Current state and action
- **Target**: Next state vector

```python
stream = make_gymnasium_stream(
    "CartPole-v1",
    mode=PredictionMode.NEXT_STATE,
    include_action_in_features=True,  # Required for this mode
)
```

### VALUE Mode

Predict cumulative return (for TD learning):

- **Features**: Current state
- **Target**: Bootstrapped value estimate

```python
stream = make_gymnasium_stream(
    "CartPole-v1",
    mode=PredictionMode.VALUE,
    gamma=0.99,  # Discount factor
)
```

## TD Learning with TDStream

For proper TD learning with value function bootstrap:

```python
from alberta_framework.streams.gymnasium import TDStream
import gymnasium as gym

env = gym.make("CartPole-v1")
stream = TDStream(
    env=env,
    gamma=0.99,
    seed=42,
)

# The stream automatically computes TD targets:
# target = reward + gamma * V(next_state)
```

### Updating the Value Function

TD learning requires updating the value estimator:

```python
for step, timestep in enumerate(stream):
    # Make prediction
    prediction = learner.predict(state, timestep.observation)

    # Compute TD error
    error = timestep.target - prediction

    # Update learner
    result = learner.update(state, error, timestep.observation)
    state = result.new_state

    # Update stream's value function estimate
    stream.update_value_function(
        lambda obs: learner.predict(state, obs)
    )
```

## Custom Policies

By default, streams use a random policy. Create custom policies:

```python
from alberta_framework.streams.gymnasium import (
    make_random_policy,
    make_epsilon_greedy_policy,
)

# Random policy
random_policy = make_random_policy(env, seed=42)

# Epsilon-greedy wrapping another policy
def my_policy(obs):
    return my_action_selection(obs)

eps_policy = make_epsilon_greedy_policy(
    base_policy=my_policy,
    env=env,
    epsilon=0.1,
    seed=42,
)

# Use with stream
stream = make_gymnasium_stream(
    "CartPole-v1",
    policy=eps_policy,
)
```

## Episode Handling

Streams automatically handle episode boundaries:

- Reset environment when episode ends
- Continue generating experience seamlessly
- Track episode count via `stream.episode_count`

```python
stream = make_gymnasium_stream("CartPole-v1")

for i, timestep in enumerate(stream):
    if i >= 10000:
        break

print(f"Completed {stream.episode_count} episodes")
print(f"Total steps: {stream.step_count}")
```

## Feature Construction

The stream flattens observations and actions into feature vectors:

```python
# CartPole-v1: 4-dim state
# With action (discrete 2): 4 + 2 = 6-dim features (one-hot action)

stream = make_gymnasium_stream(
    "CartPole-v1",
    include_action_in_features=True,
)
print(f"Feature dimension: {stream.feature_dim}")  # 6
```

## Supported Environments

The framework supports:

- **Box observation spaces**: Continuous state vectors
- **Discrete action spaces**: One-hot encoded into features

Environments with complex observation spaces (Dict, Tuple) are flattened automatically.
