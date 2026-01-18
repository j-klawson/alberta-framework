# Running Experiments

This guide covers the experiment infrastructure for publication-quality research.

!!! note "Optional Dependencies"
    Full experiment support requires: `pip install alberta-framework[analysis]`

## Multi-Seed Experiments

Research requires running experiments across multiple random seeds.

### Basic Setup

```python
from alberta_framework import LinearLearner, LMS, IDBD
from alberta_framework.streams import RandomWalkTarget
from alberta_framework.utils import (
    ExperimentConfig,
    run_multi_seed_experiment,
)
import jax.random as jr

# Define experiment configurations
configs = [
    ExperimentConfig(
        name="LMS",
        learner_factory=lambda: LinearLearner(optimizer=LMS(step_size=0.01)),
        stream_factory=lambda key: RandomWalkTarget(
            feature_dim=10, key=key, walk_std=0.01
        ),
        num_steps=10000,
    ),
    ExperimentConfig(
        name="IDBD",
        learner_factory=lambda: LinearLearner(optimizer=IDBD(initial_step_size=0.01)),
        stream_factory=lambda key: RandomWalkTarget(
            feature_dim=10, key=key, walk_std=0.01
        ),
        num_steps=10000,
    ),
]

# Run across 30 seeds
results = run_multi_seed_experiment(
    configs=configs,
    seeds=30,
    parallel=True,  # Use joblib for parallelization
)
```

### Accessing Results

```python
from alberta_framework.utils import get_final_performance, get_metric_timeseries

# Get final performance for each method
for name, agg_result in results.items():
    perf = get_final_performance(agg_result, metric="squared_error", window=1000)
    print(f"{name}: {perf.mean:.4f} +/- {perf.std:.4f}")

# Get learning curves
lms_curves = get_metric_timeseries(results["LMS"], metric="squared_error")
# Shape: (num_seeds, num_steps)
```

## Statistical Analysis

### Pairwise Comparisons

```python
from alberta_framework.utils import pairwise_comparisons

# Compare all pairs of methods
comparisons = pairwise_comparisons(
    results,
    metric="squared_error",
    window=1000,
    test="welch",           # or "mann_whitney", "wilcoxon"
    correction="holm",      # Multiple comparison correction
)

for comp in comparisons:
    print(f"{comp.method_a} vs {comp.method_b}:")
    print(f"  p-value: {comp.p_value:.4f}")
    print(f"  Cohen's d: {comp.effect_size:.2f}")
    print(f"  Significant: {comp.significant}")
```

### Individual Tests

```python
from alberta_framework.utils import ttest_comparison, compute_statistics

# Two-sample t-test
result = ttest_comparison(
    results["LMS"],
    results["IDBD"],
    metric="squared_error",
)

# Summary statistics with confidence intervals
stats = compute_statistics(
    get_final_performance(results["IDBD"], "squared_error").values,
    confidence=0.95,
)
print(f"Mean: {stats.mean:.4f}")
print(f"95% CI: [{stats.ci_lower:.4f}, {stats.ci_upper:.4f}]")
```

## Visualization

### Learning Curves

```python
from alberta_framework.utils import (
    set_publication_style,
    plot_learning_curves,
    save_figure,
)

set_publication_style()

fig, ax = plot_learning_curves(
    results,
    metric="squared_error",
    window=100,           # Smoothing window
    show_individual=False, # Show mean + CI only
    ci_alpha=0.2,         # Confidence band transparency
)

save_figure(fig, "learning_curves", formats=["pdf", "png"])
```

### Performance Comparison

```python
from alberta_framework.utils import plot_final_performance_bars

fig, ax = plot_final_performance_bars(
    results,
    metric="squared_error",
    window=1000,
    show_significance=True,  # Add significance markers
)

save_figure(fig, "performance_bars")
```

### Multi-Panel Figures

```python
from alberta_framework.utils import create_comparison_figure

fig = create_comparison_figure(
    results,
    metric="squared_error",
    window=1000,
    title="IDBD vs LMS Comparison",
)

save_figure(fig, "comparison", formats=["pdf"])
```

## Export

### Tables

```python
from alberta_framework.utils import generate_latex_table, generate_markdown_table

# LaTeX table for papers
latex = generate_latex_table(
    results,
    metrics=["squared_error"],
    caption="Tracking Error Comparison",
    label="tab:results",
)

# Markdown for README
markdown = generate_markdown_table(results, metrics=["squared_error"])
```

### Data Files

```python
from alberta_framework.utils import export_to_csv, export_to_json

# CSV for external analysis
export_to_csv(results, "results.csv")

# JSON for archival
export_to_json(results, "results.json")
```

### Complete Report

```python
from alberta_framework.utils import save_experiment_report

# Save all artifacts at once
save_experiment_report(
    results,
    output_dir="experiment_output",
    formats=["pdf", "png"],
    include_tables=True,
    include_data=True,
)
```

This creates:
```
experiment_output/
├── figures/
│   ├── learning_curves.pdf
│   ├── learning_curves.png
│   ├── performance_bars.pdf
│   └── performance_bars.png
├── tables/
│   ├── results.tex
│   └── results.md
└── data/
    ├── results.csv
    └── results.json
```

## Hyperparameter Sweeps

```python
from alberta_framework.utils import extract_hyperparameter_results

# Run experiments with different step-sizes
step_sizes = [0.001, 0.01, 0.1]
all_configs = []

for alpha in step_sizes:
    all_configs.append(
        ExperimentConfig(
            name=f"LMS_alpha={alpha}",
            learner_factory=lambda a=alpha: LinearLearner(
                optimizer=LMS(step_size=a)
            ),
            stream_factory=lambda key: RandomWalkTarget(
                feature_dim=10, key=key
            ),
            num_steps=10000,
            metadata={"step_size": alpha},  # Store hyperparameters
        )
    )

results = run_multi_seed_experiment(all_configs, seeds=30)

# Extract best configuration
best = extract_hyperparameter_results(
    results,
    metric="squared_error",
    minimize=True,
)
print(f"Best step-size: {best.metadata['step_size']}")
```
