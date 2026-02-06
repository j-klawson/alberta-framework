"""TD learning with LMS on CartPole-v1.

Demonstrates using the Alberta Framework for value function learning
via Temporal Difference (TD) on a Gymnasium environment.
"""

import jax.numpy as jnp

from alberta_framework import LMS, LinearLearner, Timer
from alberta_framework.streams.gymnasium import TDStream, make_random_policy

try:
    import gymnasium
except ImportError:
    raise ImportError("This example requires gymnasium: pip install gymnasium")


def main():
    with Timer("Total experiment runtime"):
        # Create CartPole environment
        env = gymnasium.make("CartPole-v1")

        # Create random policy (we're just learning value function, not control)
        policy = make_random_policy(env, seed=42)

        # Create TD stream for value function learning
        # include_action_in_features=False means we learn V(s) not Q(s,a)
        stream = TDStream(
            env=env,
            policy=policy,
            gamma=0.99,
            include_action_in_features=False,
            seed=42,
        )

        # Create learner with LMS optimizer
        # Try different step sizes to see the effect
        step_size = 0.01
        learner = LinearLearner(optimizer=LMS(step_size=step_size))

        # Initialize state
        state = learner.init(stream.feature_dim)

        print("TD Learning on CartPole-v1")
        print(f"Optimizer: LMS (step_size={step_size})")
        print(f"Feature dim: {stream.feature_dim}")
        print("Discount factor (gamma): 0.99")
        print("-" * 50)

        # Run learning loop manually to update value function in stream
        num_steps = 10000
        errors = []

        for step, timestep in enumerate(stream):
            if step >= num_steps:
                break

            # Update learner
            result = learner.update(state, timestep.observation, timestep.target)
            state = result.state

            # Track squared error
            squared_error = float(jnp.squeeze(result.error) ** 2)
            errors.append(squared_error)

            # Update stream's value function for proper TD bootstrapping
            def value_fn(obs, s=state):
                return float(jnp.squeeze(learner.predict(s, obs)))
            stream.update_value_function(value_fn)

            # Print progress
            if (step + 1) % 1000 == 0:
                recent_errors = errors[-1000:]
                mean_error = sum(recent_errors) / len(recent_errors)
                print(f"Step {step + 1:5d} | Episodes: {stream.episode_count:3d} | "
                      f"Mean squared error (last 1000): {mean_error:.4f}")

        print("-" * 50)
        print(f"Final episodes completed: {stream.episode_count}")
        print(f"Final mean squared error (last 1000): {sum(errors[-1000:]) / 1000:.4f}")

        # Show learned weights
        print(f"\nLearned weights: {state.weights}")
        print(f"Learned bias: {float(state.bias):.4f}")

        # Test value predictions on a few states
        print("\nSample value predictions:")
        test_obs = jnp.array([0.0, 0.0, 0.0, 0.0])  # Centered, balanced
        v = float(jnp.squeeze(learner.predict(state, test_obs)))
        print(f"  Centered state [0,0,0,0]: V = {v:.2f}")

        test_obs = jnp.array([0.0, 0.0, 0.2, 0.0])  # Tilted right
        v = float(jnp.squeeze(learner.predict(state, test_obs)))
        print(f"  Tilted right [0,0,0.2,0]: V = {v:.2f}")

        test_obs = jnp.array([0.0, 0.0, -0.2, 0.0])  # Tilted left
        v = float(jnp.squeeze(learner.predict(state, test_obs)))
        print(f"  Tilted left [0,0,-0.2,0]: V = {v:.2f}")

        env.close()


if __name__ == "__main__":
    main()
