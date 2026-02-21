# bsuite Benchmarks for Alberta Framework

Bridges Alberta Framework learners to [bsuite](https://github.com/google-deepmind/bsuite) for standardized RL diagnostics.

## Design

**Q-Learning as Multi-Head Prediction**: Uses `MultiHeadMLPLearner` with `n_heads = num_actions` as the Q-function. Each head predicts Q(s, a_i). NaN target masking ensures only the taken action's head is updated per step.

**Continuing Mode** (default): Aligned with the Alberta Plan -- bsuite environments are treated as continuing streams. The `ContinuingWrapper` auto-resets at episode boundaries with `discount=0` to signal pseudo-termination without terminal states. Agent state persists across resets.

**Standard Mode**: For bsuite score compatibility. Uses episodic semantics but agent state still persists.

## Agents

| Agent | Description |
|-------|-------------|
| `autostep` | Autostep + ObGD + EMA normalization (framework's best) |
| `lms` | Fixed step-size + ObGD + EMA normalization (no-adaptation baseline) |
| `adam` | Standalone haiku/optax Adam (external baseline) |

Each agent also has `_bottleneck` variants with smaller networks `(16, 16)`.

## Installation

```bash
# From the alberta-framework root
pip install -e '.[bsuite]'

# bsuite itself (from local clone)
pip install -e ../bsuite
```

## Usage

### Single Experiment

```bash
# Continuing mode (default)
python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --save_path output/bsuite

# Standard episodic mode
python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --mode standard

# With representation logging
python benchmarks/bsuite/run_single.py --agent autostep --bsuite_id catch/0 --log-representation
```

### Sweep

```bash
# Run all standard agents on catch experiments
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --experiments catch catch_scale

# Run all primary experiments (scale + noise)
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --all-primary

# Include bottleneck variants
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --experiments catch_scale --bottleneck

# Continual multi-task sequence
python benchmarks/bsuite/run_sweep.py --save_path output/bsuite --continual-sequence catch/0 cartpole/0 bandit/0
```

### Analysis

```bash
# Print summary table
python benchmarks/bsuite/analysis.py --save_path output/bsuite --summary

# Save comparison plots
python benchmarks/bsuite/analysis.py --save_path output/bsuite --output-dir output/bsuite/plots
```

## Experiment Selection

**Primary focus** (adaptive step-sizes on scale/noise):
- `catch_scale`, `cartpole_scale`, `bandit_scale`, `mnist_scale`
- `catch_noise`, `cartpole_noise`, `mnist_noise`

**Secondary** (basic sanity + credit assignment):
- `catch`, `cartpole`, `bandit`, `mnist`, `discounting_chain`

## Tests

```bash
pytest tests/test_bsuite_agents.py -v
```
