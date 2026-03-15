"""SARSA on CartPole-v1 with Horde architecture.

Alberta Plan Step 4a: On-policy control via SARSA with a shared-trunk
multi-head MLP. Each action maps to a control demon (head) in the Horde.

Compares Autostep vs LMS optimizers and plots:
1. Reward per episode
2. Epsilon decay schedule

Usage:
    python "examples/The Alberta Plan/Step4/sarsa_cartpole.py" --output-dir output/step4
"""

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from alberta_framework import (
    Autostep,
    ObGDBounding,
    SARSAAgent,
    SARSAConfig,
    run_sarsa_episode,
)
from alberta_framework.utils.timing import Timer


def run_experiment(
    agent: SARSAAgent,
    n_episodes: int = 200,
    max_steps: int = 500,
    seed: int = 42,
) -> list[float]:
    """Run SARSA agent on CartPole for n_episodes."""
    import jax.random as jr

    env = gym.make("CartPole-v1")
    state = agent.init(feature_dim=4, key=jr.key(seed))
    episode_rewards = []

    for ep in range(n_episodes):
        result = run_sarsa_episode(agent, state, env, max_steps=max_steps)
        state = result.state
        episode_rewards.append(result.total_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1:3d}: avg reward (last 50) = {avg:.1f}")

    env.close()
    return episode_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="SARSA on CartPole")
    parser.add_argument(
        "--output-dir", type=str, default="output/step4",
        help="Directory for output plots",
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sarsa_config = SARSAConfig(
        n_actions=2,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_steps=5000,
    )

    # Agent 1: SARSA + Autostep + ObGD
    print("Running SARSA + Autostep...")
    agent_autostep = SARSAAgent(
        sarsa_config=sarsa_config,
        hidden_sizes=(64, 32),
        optimizer=Autostep(initial_step_size=0.01),
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.9,
    )
    with Timer("Autostep"):
        rewards_autostep = run_experiment(
            agent_autostep, n_episodes=args.episodes, seed=args.seed
        )

    # Agent 2: SARSA + LMS + ObGD
    print("\nRunning SARSA + LMS...")
    agent_lms = SARSAAgent(
        sarsa_config=sarsa_config,
        hidden_sizes=(64, 32),
        step_size=0.001,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.9,
    )
    with Timer("LMS"):
        rewards_lms = run_experiment(
            agent_lms, n_episodes=args.episodes, seed=args.seed
        )

    # Plot: episode rewards
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    window = 20
    for label, rewards in [("Autostep", rewards_autostep), ("LMS", rewards_lms)]:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=label, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-ep moving avg)")
    ax.set_title("SARSA on CartPole-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sarsa_cartpole.png", dpi=150)
    print(f"\nPlot saved to {output_dir / 'sarsa_cartpole.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
