"""Analysis tools for bsuite benchmark results.

Load CSV results, compare agents, generate plots, and export summaries.
Supports both bsuite's built-in scoring (standard mode) and online
performance metrics (continuing mode).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bsuite.logging import csv_load


def load_results(
    save_path: str,
    agent_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load CSV results for all agents.

    Parameters
    ----------
    save_path : str
        Base directory containing agent subdirectories.
    agent_names : list of str, optional
        Specific agent names to load. If None, loads all subdirectories.

    Returns
    -------
    dict mapping agent_name -> DataFrame
        Results for each agent.
    """
    base_path = Path(save_path)
    results: dict[str, pd.DataFrame] = {}

    if agent_names is None:
        agent_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        agent_names = [d.name for d in agent_dirs]

    for name in agent_names:
        agent_dir = base_path / name
        if not agent_dir.exists():
            continue
        try:
            df, _ = csv_load.load_bsuite(str(agent_dir))
            results[name] = df
        except Exception as e:
            print(f"Warning: Could not load results for {name}: {e}")

    return results


def load_representation_logs(
    save_path: str,
    agent_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load representation utility logs for an agent.

    Parameters
    ----------
    save_path : str
        Base directory containing agent subdirectories.
    agent_name : str
        Agent name.

    Returns
    -------
    dict mapping bsuite_id -> list of snapshots
    """
    agent_dir = Path(save_path) / agent_name
    logs: dict[str, list[dict[str, Any]]] = {}

    for json_file in agent_dir.glob("representation_*.json"):
        # Extract bsuite_id from filename: representation_catch_0.json -> catch/0
        name = json_file.stem.replace("representation_", "")
        bsuite_id = name.replace("_", "/", 1)
        with open(json_file) as f:
            logs[bsuite_id] = json.load(f)

    return logs


def compute_online_metrics(
    df: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """Compute online performance metrics from bsuite results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results DataFrame.
    window : int
        Smoothing window for running statistics. Default: 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional online metrics.
    """
    result = df.copy()

    if "total_regret" in result.columns:
        result["regret_rate"] = result.groupby("bsuite_id")["total_regret"].diff().fillna(0)
        result["running_regret_rate"] = (
            result.groupby("bsuite_id")["regret_rate"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    if "reward" in result.columns:
        result["running_reward"] = (
            result.groupby("bsuite_id")["reward"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    return result


def compare_agents_bar(
    results: dict[str, pd.DataFrame],
    metric: str = "total_regret",
    experiment_name: str | None = None,
    title: str | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Create bar chart comparing agents on a metric.

    Parameters
    ----------
    results : dict mapping agent_name -> DataFrame
        Results for each agent.
    metric : str
        Column name to compare. Default: 'total_regret'.
    experiment_name : str, optional
        Filter to this experiment. If None, uses all data.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison figure.
    """
    agent_names = []
    means = []
    stds = []

    for name, df in sorted(results.items()):
        if experiment_name:
            df = df[df["bsuite_id"].str.startswith(experiment_name)]
        if metric not in df.columns:
            continue

        # Get final value per bsuite_id
        final_vals = df.groupby("bsuite_id")[metric].last()
        agent_names.append(name)
        means.append(float(final_vals.mean()))
        stds.append(float(final_vals.std()))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agent_names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Agent Comparison: {metric}")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig


def plot_learning_curves(
    results: dict[str, pd.DataFrame],
    bsuite_id: str,
    metric: str = "total_regret",
    window: int = 100,
    title: str | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot learning curves for all agents on a single bsuite_id.

    Parameters
    ----------
    results : dict mapping agent_name -> DataFrame
        Results for each agent.
    bsuite_id : str
        The bsuite experiment id to plot.
    metric : str
        Column name to plot. Default: 'total_regret'.
    window : int
        Smoothing window. Default: 100.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The learning curves figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in sorted(results.items()):
        mask = df["bsuite_id"] == bsuite_id
        agent_df = df[mask].copy()
        if agent_df.empty or metric not in agent_df.columns:
            continue

        values = agent_df[metric].values
        if window > 1 and len(values) > window:
            smoothed = pd.Series(values).rolling(window, min_periods=1).mean().values
        else:
            smoothed = values

        ax.plot(smoothed, label=name, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{bsuite_id}: {metric}")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig


def plot_representation_evolution(
    logs: list[dict[str, Any]],
    metric: str = "trunk_step_sizes",
    title: str | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot representation utility evolution over time.

    Parameters
    ----------
    logs : list of dict
        Representation snapshots from AlbertaAgent.
    metric : str
        Key in snapshot dicts to plot. Default: 'trunk_step_sizes'.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The representation evolution figure.
    """
    steps = [snap["step"] for snap in logs]
    values = np.array([snap[metric] for snap in logs])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(values.shape[1]):
        ax.plot(steps, values[:, i], alpha=0.5, label=f"Layer {i}")

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Representation: {metric}")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def generate_summary_table(
    results: dict[str, pd.DataFrame],
    metric: str = "total_regret",
    experiments: list[str] | None = None,
    fmt: str = "markdown",
) -> str:
    """Generate a summary table comparing agents across experiments.

    Parameters
    ----------
    results : dict mapping agent_name -> DataFrame
        Results for each agent.
    metric : str
        Column to summarize. Default: 'total_regret'.
    experiments : list of str, optional
        Experiment names to include. If None, includes all.
    fmt : str
        Output format: 'markdown' or 'latex'. Default: 'markdown'.

    Returns
    -------
    str
        Formatted table.
    """
    # Collect data
    rows: list[dict[str, Any]] = []
    agent_names = sorted(results.keys())

    # Get all experiment names from data
    all_experiments: set[str] = set()
    for df in results.values():
        all_experiments.update(df["bsuite_id"].str.split("/").str[0].unique())

    if experiments:
        all_experiments = {e for e in all_experiments if e in experiments}

    for exp_name in sorted(all_experiments):
        row: dict[str, Any] = {"experiment": exp_name}
        for agent_name in agent_names:
            df = results[agent_name]
            mask = df["bsuite_id"].str.startswith(exp_name)
            exp_df = df[mask]
            if exp_df.empty or metric not in exp_df.columns:
                row[agent_name] = "N/A"
                continue
            final_vals = exp_df.groupby("bsuite_id")[metric].last()
            mean = final_vals.mean()
            std = final_vals.std()
            row[agent_name] = f"{mean:.1f} +/- {std:.1f}"
        rows.append(row)

    # Format
    if not rows:
        return "No results to display."

    summary_df = pd.DataFrame(rows)

    if fmt == "latex":
        return str(summary_df.to_latex(index=False))
    return str(summary_df.to_markdown(index=False))


def main() -> None:
    """CLI for analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bsuite results")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/bsuite",
        help="Base directory for results",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        type=str,
        default=None,
        help="Agent names to compare",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="total_regret",
        help="Metric to compare (default: total_regret)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table",
    )

    args = parser.parse_args()

    results = load_results(args.save_path, args.agents)
    if not results:
        print(f"No results found in {args.save_path}")
        return

    print(f"Loaded results for agents: {list(results.keys())}")

    if args.summary:
        table = generate_summary_table(results, metric=args.metric)
        print("\n" + table)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        compare_agents_bar(
            results,
            metric=args.metric,
            output_path=str(output_dir / "agent_comparison.png"),
        )


if __name__ == "__main__":
    main()
