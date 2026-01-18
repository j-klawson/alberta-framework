# Optimizers

This guide covers the three optimizers in the Alberta Framework.

## LMS (Least Mean Squares)

The simplest optimizer with a fixed step-size.

### Algorithm

\[
w_{t+1} = w_t + \alpha \cdot \delta_t \cdot x_t
\]

Where:
- \(\alpha\) is the fixed step-size
- \(\delta_t = y_t - \hat{y}_t\) is the prediction error
- \(x_t\) is the feature vector

### Usage

```python
from alberta_framework import LMS

optimizer = LMS(step_size=0.01)
```

### When to Use

LMS serves as a baseline. Use it to:

- Establish performance benchmarks
- Compare against adaptive methods
- Validate problem setups

The main limitation is that optimal step-size depends on the problem and may change as conditions shift.

## IDBD (Incremental Delta-Bar-Delta)

IDBD maintains **per-weight adaptive step-sizes** that increase when gradients consistently agree and decrease when they conflict.

### Algorithm

1. Compute per-weight step-sizes: \(\alpha_i = \exp(\log \alpha_i)\)
2. Update weights: \(w_i \leftarrow w_i + \alpha_i \cdot \delta \cdot x_i\)
3. Update log step-sizes: \(\log \alpha_i \leftarrow \log \alpha_i + \beta \cdot \delta \cdot x_i \cdot h_i\)
4. Update traces: \(h_i \leftarrow h_i \cdot \max(0, 1 - \alpha_i \cdot x_i^2) + \alpha_i \cdot \delta \cdot x_i\)

Where \(h_i\) is a trace that tracks gradient correlation over time.

### Usage

```python
from alberta_framework import IDBD

optimizer = IDBD(
    initial_step_size=0.01,  # Starting step-size
    meta_step_size=0.01,     # How fast step-sizes adapt
)
```

### Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `initial_step_size` | Starting value for per-weight step-sizes | 0.001 - 0.1 |
| `meta_step_size` | Learning rate for step-size adaptation (\(\beta\)) | 0.001 - 0.1 |

### Reference

> Sutton, R.S. (1992). "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta"

## Autostep

Autostep combines adaptive step-sizes with **gradient normalization**, making it robust to different feature scales without manual tuning.

### Algorithm

1. Compute gradient: \(g_i = \delta \cdot x_i\)
2. Normalize: \(g'_i = g_i / \max(|g_i|, v_i)\)
3. Update weights: \(w_i \leftarrow w_i + \alpha_i \cdot g'_i\)
4. Adapt step-sizes: \(\alpha_i \leftarrow \alpha_i \cdot \exp(\mu \cdot g'_i \cdot h_i)\)
5. Update traces: \(h_i \leftarrow h_i \cdot (1 - \alpha_i) + \alpha_i \cdot g'_i\)
6. Update normalizers: \(v_i \leftarrow \max(|g_i|, v_i \cdot \tau)\)

### Usage

```python
from alberta_framework import Autostep

optimizer = Autostep(
    initial_step_size=0.01,
    meta_step_size=0.01,
    normalizer_decay=0.99,
)
```

### Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `initial_step_size` | Starting value for per-weight step-sizes | 0.001 - 0.1 |
| `meta_step_size` | Learning rate for adaptation (\(\mu\)) | 0.001 - 0.1 |
| `normalizer_decay` | Decay for gradient normalizers (\(\tau\)) | 0.9 - 0.999 |

### Reference

> Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012). "Tuning-free step-size adaptation"

## Comparison

| Feature | LMS | IDBD | Autostep |
|---------|-----|------|----------|
| Per-weight step-sizes | No | Yes | Yes |
| Gradient normalization | No | No | Yes |
| Tuning required | High | Medium | Low |
| Computational cost | Lowest | Medium | Highest |

## Choosing an Optimizer

1. **Start with IDBD** for most non-stationary problems
2. **Use Autostep** when feature scales vary significantly
3. **Use LMS** as a baseline or when you have a well-tuned step-size

## Metrics

All optimizers report metrics during training:

```python
# LMS metrics
{'step_size': 0.01}

# IDBD metrics
{'mean_step_size': 0.015, 'min_step_size': 0.001, 'max_step_size': 0.1}

# Autostep metrics
{'mean_step_size': 0.015, 'min_step_size': 0.001,
 'max_step_size': 0.1, 'mean_normalizer': 1.5}
```
