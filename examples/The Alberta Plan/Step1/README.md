# Step 1: Meta-Learned Step-Sizes

This directory contains examples demonstrating **Step 1 of the Alberta Plan**: showing that IDBD (Incremental Delta-Bar-Delta) and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS on non-stationary supervised learning problems.

## Background

The Alberta Plan's first step is foundational: before building more complex continual learning systems, we must first demonstrate that adaptive step-size methods can automatically find good learning rates without manual tuning.

**Key insight**: In non-stationary environments, the optimal step-size:
- Changes over time as the task difficulty varies
- Differs per-weight (some features are more relevant than others)

IDBD and Autostep address this by maintaining per-weight step-sizes that adapt based on gradient correlation.

## Scripts

### Sutton 1992 Replications

These scripts replicate the foundational experiments from Sutton's 1992 IDBD paper:

| Script | Description |
|--------|-------------|
| `sutton1992_experiment1.py` | **Experiment 1: Does IDBD help?** Compares IDBD vs LMS on a 20-input regression task with 5 relevant and 15 irrelevant inputs. Sign flips every 20 steps create non-stationarity. Shows IDBD achieves less than half the error of best-tuned LMS. |
| `sutton1992_experiment2.py` | **Experiment 2: Does IDBD find optimal alpha?** Tracks how IDBD learning rates evolve over 250,000 steps (Figure 4), then verifies via grid search that IDBD converges to the optimal per-weight learning rates (Figure 5). |

### Framework Demonstrations

These scripts demonstrate the Alberta Framework's Step 1 capabilities:

| Script | Description |
|--------|-------------|
| `idbd_vs_lms.py` | Basic comparison of IDBD vs LMS optimizers with detailed analysis and visualization. Good starting point for understanding the framework. |
| `normalization_study.py` | Studies the effect of online feature normalization, comparing normalized vs unnormalized learners across different feature scales. |
| `autostep_comparison.py` | Compares IDBD, Autostep, and LMS optimizers. Autostep adds gradient normalization for robustness to feature scales. |

## Running the Examples

```bash
# Install the framework in development mode
pip install -e ".[dev]"

# Run Sutton 1992 replications
python "examples/The Alberta Plan/Step1/sutton1992_experiment1.py"
python "examples/The Alberta Plan/Step1/sutton1992_experiment2.py"

# Run framework demonstrations
python "examples/The Alberta Plan/Step1/idbd_vs_lms.py"
python "examples/The Alberta Plan/Step1/normalization_study.py"
python "examples/The Alberta Plan/Step1/autostep_comparison.py"
```

## Expected Results

### Experiment 1 (Does IDBD help?)
- LMS with best alpha (~0.04): MSE ~3.5
- IDBD over broad theta range: MSE ~1.5
- **IDBD achieves less than half the error of best-tuned LMS**

### Experiment 2 (Does IDBD find optimal alpha?)
- After 250,000 steps with theta=0.001:
  - Relevant input learning rates converge to ~0.13
  - Irrelevant input learning rates converge to <0.007
- Grid search confirms minimum error occurs at alpha ~0.13
- **IDBD automatically discovers the optimal per-weight learning rates**

## Reference

Sutton, R.S. (1992). "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta." Proceedings of the Tenth National Conference on Artificial Intelligence (AAAI-92).

Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012). "Tuning-free step-size adaptation." IEEE International Conference on Acoustics, Speech and Signal Processing.
