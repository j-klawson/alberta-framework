#!/usr/bin/env python3
"""Publication-Quality Experiment Example.

This example demonstrates how to run multi-seed experiments with statistical
analysis and generate publication-quality figures and tables.

Usage:
    pip install -e ".[analysis]"  # Install analysis dependencies
    python examples/publication_experiment.py

This will:
1. Run IDBD, Autostep, and LMS experiments across 30 seeds
2. Compute statistical significance tests
3. Generate publication-quality figures (PDF + PNG)
4. Generate LaTeX and markdown tables
5. Save all results to output/publication_experiment/
"""

from pathlib import Path

from alberta_framework import (
    IDBD,
    LMS,
    Autostep,
    LinearLearner,
    RandomWalkStream,
    Timer,
)
from alberta_framework.utils import (
    ExperimentConfig,
    export_to_json,
    generate_latex_table,
    generate_markdown_table,
    get_final_performance,
    run_multi_seed_experiment,
    save_experiment_report,
)


def create_configs() -> list[ExperimentConfig]:
    """Create experiment configurations for LMS, IDBD, and Autostep."""

    def make_stream():
        return RandomWalkStream(feature_dim=10, drift_rate=0.001, noise_std=0.1)

    configs = [
        # LMS with various step-sizes
        ExperimentConfig(
            name="LMS_0.01",
            learner_factory=lambda: LinearLearner(optimizer=LMS(step_size=0.01)),
            stream_factory=make_stream,
            num_steps=10000,
        ),
        ExperimentConfig(
            name="LMS_0.05",
            learner_factory=lambda: LinearLearner(optimizer=LMS(step_size=0.05)),
            stream_factory=make_stream,
            num_steps=10000,
        ),
        # IDBD with default parameters
        ExperimentConfig(
            name="IDBD",
            learner_factory=lambda: LinearLearner(
                optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.05)
            ),
            stream_factory=make_stream,
            num_steps=10000,
        ),
        # Autostep
        ExperimentConfig(
            name="Autostep",
            learner_factory=lambda: LinearLearner(
                optimizer=Autostep(initial_step_size=0.01, meta_step_size=0.05)
            ),
            stream_factory=make_stream,
            num_steps=10000,
        ),
    ]
    return configs


def main() -> None:
    """Run the publication experiment."""
    with Timer("Total experiment runtime"):
        print("Publication-Quality Experiment")
        print("=" * 60)

        # Create output directory
        output_dir = Path("output/publication_experiment")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment configurations
        configs = create_configs()
        print(f"\nRunning {len(configs)} configurations across 30 seeds...")

        # Run experiments
        results = run_multi_seed_experiment(
            configs,
            seeds=30,
            parallel=True,
            show_progress=True,
        )

        print("\nExperiment complete!")

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        performance = get_final_performance(results, metric="squared_error")
        print(f"\n{'Method':<15} {'Mean Error':>15} {'Std':>15}")
        print("-" * 45)
        for name, (mean, std) in sorted(performance.items(), key=lambda x: x[1][0]):
            print(f"{name:<15} {mean:>15.6f} {std:>15.6f}")

        # Statistical analysis (requires scipy)
        significance_results = None
        try:
            from alberta_framework.utils import pairwise_comparisons

            print("\n" + "=" * 60)
            print("STATISTICAL ANALYSIS")
            print("=" * 60)

            significance_results = pairwise_comparisons(
                results,
                metric="squared_error",
                test="ttest",
                correction="bonferroni",
            )

            print("\nPairwise Comparisons (paired t-test with Bonferroni correction):")
            print(f"\n{'Comparison':<30} {'p-value':>12} {'Effect Size':>12} {'Sig.':>8}")
            print("-" * 65)
            for (name_a, name_b), result in significance_results.items():
                sig = "Yes" if result.significant else "No"
                print(
                    f"{name_a} vs {name_b:<15} {result.p_value:>12.4f} "
                    f"{result.effect_size:>12.3f} {sig:>8}"
                )
        except ImportError:
            print("\nNote: Install scipy for statistical analysis: pip install scipy")

        # Generate outputs
        print("\n" + "=" * 60)
        print("GENERATING OUTPUTS")
        print("=" * 60)

        # Save experiment report (CSV, JSON, LaTeX, markdown)
        artifacts = save_experiment_report(
            results,
            output_dir,
            "comparison",
            significance_results=significance_results,
            metric="squared_error",
        )

        print("\nGenerated files:")
        for artifact_type, path in artifacts.items():
            print(f"  - {artifact_type}: {path}")

        # Export full JSON with timeseries
        json_full_path = output_dir / "comparison_full.json"
        export_to_json(results, json_full_path, include_timeseries=True)
        print(f"  - full_json: {json_full_path}")

        # Print LaTeX table
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        latex = generate_latex_table(
            results,
            significance_results=significance_results,
            metric="squared_error",
            caption="Comparison of LMS, IDBD, and Autostep on Random Walk Target",
            label="tab:comparison",
        )
        print(latex)

        # Print markdown table
        print("\n" + "=" * 60)
        print("MARKDOWN TABLE")
        print("=" * 60)
        markdown = generate_markdown_table(
            results,
            significance_results=significance_results,
            metric="squared_error",
        )
        print(markdown)

        # Generate figures (requires matplotlib)
        try:
            from alberta_framework.utils import (
                create_comparison_figure,
                plot_learning_curves,
                save_figure,
                set_publication_style,
            )

            print("\n" + "=" * 60)
            print("GENERATING FIGURES")
            print("=" * 60)

            set_publication_style(font_size=10, use_latex=False)

            # Learning curves figure
            fig, ax = plot_learning_curves(results, metric="squared_error", show_ci=True)
            ax.set_title("Learning Curves on Random Walk Target")
            paths = save_figure(fig, output_dir / "learning_curves", formats=["pdf", "png"])
            print(f"Saved learning curves: {[str(p) for p in paths]}")

            # Multi-panel comparison figure
            fig = create_comparison_figure(
                results,
                significance_results=significance_results,
                metric="squared_error",
            )
            paths = save_figure(fig, output_dir / "comparison", formats=["pdf", "png"])
            print(f"Saved comparison figure: {[str(p) for p in paths]}")

        except ImportError:
            print("\nNote: Install matplotlib for figures: pip install matplotlib")

        print("\n" + "=" * 60)
        print(f"All outputs saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
