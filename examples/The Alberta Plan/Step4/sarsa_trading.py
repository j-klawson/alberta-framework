"""SARSA on non-stationary financial streams (gym-anytrading).

Alberta Plan Step 4a: On-policy control on a genuinely non-stationary,
continuing environment. Financial time series exhibit regime changes,
trend shifts, and volatility clustering — exactly the kind of
non-stationarity the Alberta Plan targets.

Uses gym-anytrading's FOREX (EUR/USD hourly) and STOCKS (GOOGL daily)
environments. Each has 2 discrete actions: Sell (0) / Buy (1).

Produces plots in the gym-anytrading README style (price chart with
red=Short / green=Long position dots), comparing:
- SARSA (Autostep+EMA+ObGD) — continual learning agent
- Random baseline — uniform random action selection

Requirements:
    pip install gym-anytrading

Usage:
    python "examples/The Alberta Plan/Step4/sarsa_trading.py" --output-dir output/step4
"""

import argparse
from pathlib import Path
from typing import Any

import gym_anytrading  # noqa: F401 — registers envs
import gymnasium as gym
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from alberta_framework import (
    Autostep,
    EMANormalizer,
    ObGDBounding,
    SARSAAgent,
    SARSAConfig,
)
from alberta_framework.utils.timing import Timer


def run_sarsa_on_env(
    agent: SARSAAgent,
    env: gym.Env[Any, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Run SARSA agent through a single episode, recording positions.

    Returns dict with total_reward, total_profit, and the env (for render_all).
    """
    state = agent.init(
        feature_dim=int(np.prod(env.observation_space.shape)),
        key=jr.key(seed),
    )

    obs, _info = env.reset(seed=seed)
    obs_flat = jnp.asarray(obs, dtype=jnp.float32).flatten()

    # Select initial action
    action, new_key = agent.select_action(state, obs_flat)
    state = state.replace(  # type: ignore[attr-defined]
        last_action=action,
        last_observation=obs_flat,
        rng_key=new_key,
    )

    total_reward = 0.0
    steps = 0

    while True:
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        next_obs_flat = jnp.asarray(next_obs, dtype=jnp.float32).flatten()
        reward_arr = jnp.array(reward, dtype=jnp.float32)
        done = terminated or truncated
        term_arr = jnp.array(done, dtype=jnp.float32)

        # Select next action
        next_action, new_key = agent.select_action(state, next_obs_flat)
        state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]

        # SARSA update
        result = agent.update(
            state, reward_arr, next_obs_flat, term_arr, next_action
        )
        state = result.state
        total_reward += float(reward)
        steps += 1

        action = next_action

        if done:
            break

    return {
        "total_reward": info["total_reward"],
        "total_profit": info["total_profit"],
        "steps": steps,
        "env": env,
    }


def run_random_on_env(
    env: gym.Env[Any, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Run random agent through a single episode."""
    env.reset(seed=seed)
    while True:
        action = env.action_space.sample()
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return {
        "total_reward": info["total_reward"],
        "total_profit": info["total_profit"],
        "env": env,
    }


def render_comparison(
    envs: dict[str, gym.Env[Any, Any]],
    infos: dict[str, dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    """Render side-by-side position plots in gym-anytrading README style."""
    n = len(envs)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, env) in zip(axes, envs.items()):
        uw = env.unwrapped
        prices = uw.prices
        pos_hist = uw._position_history

        ax.plot(prices, color="#2196F3", linewidth=1.0)

        short_ticks = []
        long_ticks = []
        for i, pos in enumerate(pos_hist):
            from gym_anytrading.envs import Positions
            if pos == Positions.Short:
                short_ticks.append(i)
            elif pos == Positions.Long:
                long_ticks.append(i)

        ax.plot(short_ticks, prices[short_ticks], "ro", markersize=4, alpha=0.7)
        ax.plot(long_ticks, prices[long_ticks], "go", markersize=4, alpha=0.7)

        info = infos[label]
        reward = float(info["total_reward"])
        profit = float(info["total_profit"])
        ax.set_title(
            f"{label}  —  Total Reward: {reward:.2f}  ~  "
            f"Total Profit: {profit:.6f}",
            fontsize=12,
        )
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def run_continuing_sarsa(
    agent: SARSAAgent,
    env_id: str,
    num_steps: int,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Run SARSA in continuing mode (agent persists across episode resets).

    Returns per-step rewards and Q-spreads for long-horizon analysis.
    """
    env = gym.make(env_id)
    state = agent.init(
        feature_dim=int(np.prod(env.observation_space.shape)),
        key=jr.key(seed),
    )

    obs, _info = env.reset()
    obs_flat = jnp.asarray(obs, dtype=jnp.float32).flatten()

    action, new_key = agent.select_action(state, obs_flat)
    state = state.replace(  # type: ignore[attr-defined]
        last_action=action,
        last_observation=obs_flat,
        rng_key=new_key,
    )

    rewards: list[float] = []
    q_spreads: list[float] = []
    episode_count = 0

    for _ in range(num_steps):
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        next_obs_flat = jnp.asarray(next_obs, dtype=jnp.float32).flatten()
        reward_arr = jnp.array(reward, dtype=jnp.float32)

        is_boundary = terminated or truncated
        term_arr = jnp.array(is_boundary, dtype=jnp.float32)

        if is_boundary:
            episode_count += 1
            next_obs_reset, _info = env.reset()
            next_obs_flat = jnp.asarray(
                next_obs_reset, dtype=jnp.float32
            ).flatten()

        next_action, new_key = agent.select_action(state, next_obs_flat)
        state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]

        result = agent.update(
            state, reward_arr, next_obs_flat, term_arr, next_action
        )
        state = result.state

        rewards.append(float(reward))
        q_spreads.append(float(result.q_values[1] - result.q_values[0]))
        action = next_action

    env.close()
    print(f"    {episode_count} episode resets over {num_steps} steps")
    return {"rewards": rewards, "q_spreads": q_spreads}


def plot_continuing_results(
    results: dict[str, dict[str, list[float]]],
    title: str,
    output_path: Path,
    window: int = 200,
) -> None:
    """Plot continuing-mode results: cumulative reward + Q-spread."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, data in results.items():
        rewards = np.array(data["rewards"])
        q_spreads = np.array(data["q_spreads"])

        cum_reward = np.cumsum(rewards)
        axes[0].plot(cum_reward, label=label, alpha=0.8)

        if len(q_spreads) > window:
            smoothed = np.convolve(
                q_spreads, np.ones(window) / window, mode="valid"
            )
            axes[1].plot(smoothed, label=label, alpha=0.8)
        else:
            axes[1].plot(q_spreads, label=label, alpha=0.8)

    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title(f"{title}: Cumulative Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(f"Q(buy) - Q(sell)\n({window}-step avg)")
    axes[1].set_title("Q-Spread (agent's buy/sell preference)")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def make_sarsa_agent(
    n_actions: int = 2,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 3000,
) -> SARSAAgent:
    """Create the standard SARSA agent for trading experiments."""
    return SARSAAgent(
        sarsa_config=SARSAConfig(
            n_actions=n_actions,
            gamma=0.99,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        ),
        hidden_sizes=(32, 16),
        optimizer=Autostep(initial_step_size=0.01),
        bounder=ObGDBounding(kappa=2.0),
        normalizer=EMANormalizer(decay=0.999),
        sparsity=0.9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SARSA on non-stationary financial streams"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/step4",
        help="Directory for output plots",
    )
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # =====================================================================
    # Part 1: README-style position plots (SARSA vs Random, single episode)
    # =====================================================================

    # --- FOREX: small window (README style) ---
    print("=== FOREX EUR/USD — README-style comparison ===")
    env_sarsa_f = gym.make("forex-v0", frame_bound=(50, 250), window_size=10)
    env_rand_f = gym.make("forex-v0", frame_bound=(50, 250), window_size=10)

    agent_f = make_sarsa_agent(epsilon_decay_steps=150)

    print("  SARSA (Autostep+EMA+ObGD):")
    with Timer("FOREX SARSA"):
        sarsa_f = run_sarsa_on_env(agent_f, env_sarsa_f, seed=seed)
    print(f"    reward={float(sarsa_f['total_reward']):.2f}, "
          f"profit={float(sarsa_f['total_profit']):.6f}")

    print("  Random baseline:")
    rand_f = run_random_on_env(env_rand_f, seed=seed)
    print(f"    reward={float(rand_f['total_reward']):.2f}, "
          f"profit={float(rand_f['total_profit']):.6f}")

    render_comparison(
        {"SARSA (Autostep+EMA+ObGD)": sarsa_f["env"], "Random": rand_f["env"]},
        {"SARSA (Autostep+EMA+ObGD)": sarsa_f, "Random": rand_f},
        "FOREX EUR/USD: SARSA vs Random",
        output_dir / "sarsa_vs_random_forex.png",
    )

    # --- STOCKS: full dataset ---
    print("\n=== STOCKS GOOGL — README-style comparison ===")
    env_sarsa_s = gym.make("stocks-v0")
    env_rand_s = gym.make("stocks-v0")

    agent_s = make_sarsa_agent(epsilon_decay_steps=1500)

    print("  SARSA (Autostep+EMA+ObGD):")
    with Timer("STOCKS SARSA"):
        sarsa_s = run_sarsa_on_env(agent_s, env_sarsa_s, seed=seed)
    print(f"    reward={float(sarsa_s['total_reward']):.2f}, "
          f"profit={float(sarsa_s['total_profit']):.6f}, "
          f"steps={sarsa_s['steps']}")

    print("  Random baseline:")
    rand_s = run_random_on_env(env_rand_s, seed=seed)
    print(f"    reward={float(rand_s['total_reward']):.2f}, "
          f"profit={float(rand_s['total_profit']):.6f}")

    render_comparison(
        {"SARSA (Autostep+EMA+ObGD)": sarsa_s["env"], "Random": rand_s["env"]},
        {"SARSA (Autostep+EMA+ObGD)": sarsa_s, "Random": rand_s},
        "GOOGL Stock: SARSA vs Random",
        output_dir / "sarsa_vs_random_stocks.png",
    )

    # --- FOREX: full dataset ---
    print("\n=== FOREX EUR/USD — Full dataset comparison ===")
    env_sarsa_ff = gym.make("forex-v0")
    env_rand_ff = gym.make("forex-v0")

    agent_ff = make_sarsa_agent(epsilon_decay_steps=4000)

    print("  SARSA (Autostep+EMA+ObGD):")
    with Timer("FOREX full SARSA"):
        sarsa_ff = run_sarsa_on_env(agent_ff, env_sarsa_ff, seed=seed)
    print(f"    reward={float(sarsa_ff['total_reward']):.2f}, "
          f"profit={float(sarsa_ff['total_profit']):.6f}, "
          f"steps={sarsa_ff['steps']}")

    print("  Random baseline:")
    rand_ff = run_random_on_env(env_rand_ff, seed=seed)
    print(f"    reward={float(rand_ff['total_reward']):.2f}, "
          f"profit={float(rand_ff['total_profit']):.6f}")

    render_comparison(
        {"SARSA (Autostep+EMA+ObGD)": sarsa_ff["env"], "Random": rand_ff["env"]},
        {"SARSA (Autostep+EMA+ObGD)": sarsa_ff, "Random": rand_ff},
        "FOREX EUR/USD (Full Year): SARSA vs Random",
        output_dir / "sarsa_vs_random_forex_full.png",
    )

    # =====================================================================
    # Part 2: Continuing mode — Autostep vs LMS long-horizon
    # =====================================================================

    print("\n=== Continuing mode: Autostep vs LMS ===")
    continuing_results = {}

    print("  FOREX Autostep:")
    agent_cont_a = make_sarsa_agent(epsilon_decay_steps=3000)
    with Timer("Continuing Autostep"):
        continuing_results["Autostep+EMA+ObGD"] = run_continuing_sarsa(
            agent_cont_a, "forex-v0", num_steps=6000, seed=seed,
        )

    print("  FOREX LMS:")
    agent_cont_l = SARSAAgent(
        sarsa_config=SARSAConfig(
            n_actions=2, gamma=0.99,
            epsilon_start=0.2, epsilon_end=0.05, epsilon_decay_steps=3000,
        ),
        hidden_sizes=(32, 16),
        step_size=0.001,
        bounder=ObGDBounding(kappa=2.0),
        normalizer=EMANormalizer(decay=0.999),
        sparsity=0.9,
    )
    with Timer("Continuing LMS"):
        continuing_results["LMS+EMA+ObGD"] = run_continuing_sarsa(
            agent_cont_l, "forex-v0", num_steps=6000, seed=seed,
        )

    plot_continuing_results(
        continuing_results,
        "SARSA on FOREX EUR/USD (Continuing)",
        output_dir / "sarsa_forex_continuing.png",
    )

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nFOREX (200 steps):  SARSA={float(sarsa_f['total_reward']):.2f}  "
          f"Random={float(rand_f['total_reward']):.2f}")
    print(f"GOOGL (full):       SARSA={float(sarsa_s['total_reward']):.2f}  "
          f"Random={float(rand_s['total_reward']):.2f}")
    print(f"FOREX (full year):  SARSA={float(sarsa_ff['total_reward']):.2f}  "
          f"Random={float(rand_ff['total_reward']):.2f}")
    print("\nContinuing FOREX (6000 steps):")
    for label, data in continuing_results.items():
        print(f"  {label}: total reward = {sum(data['rewards']):.2f}")


if __name__ == "__main__":
    main()
