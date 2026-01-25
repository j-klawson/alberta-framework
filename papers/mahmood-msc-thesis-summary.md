### Summary of Mahmood (2010): Online Real-time Step-size Adaptation

#### Core Objective
To develop a step-size adaptation algorithm (Autostep) that inherits the tracking capabilities of IDBD but removes its sensitivity to initial parameters and numerical instability in non-stationary environments.

#### The Problem with IDBD (The Antecedent)
- **Exponential Update**: IDBD updates step-sizes via $\alpha_i = e^{\beta_i}$. In environments with large errors or unscaled features, $\beta_i$ grows linearly, causing $\alpha_i$ to grow exponentially, leading to catastrophic divergence.
- **Tuning Sensitivity**: Performance is highly dependent on a meta-learning rate ($\theta$). If $\theta$ is too high, the system diverges; if too low, it fails to track non-stationarity.

#### The Innovation: Autostep
Mahmood introduced a normalized meta-descent rule. The effective step-size for the meta-learning rate is normalized by a running average of the magnitude of the updates:
1. **Normalization Term ($v_i$)**: $v_{i, t+1} = \max(|\delta_t x_{i, t} h_{i, t}|, v_{i, t} + \frac{1}{\tau} \alpha_{i, t} x_{i, t}^2 (| \delta_t x_{i, t} h_{i, t} | - v_{i, t}))$.
2. **Update Rule**: $\beta_{i, t+1} = \beta_{i, t} + \mu \frac{\delta_t x_{i, t} h_{i, t}}{v_{i, t+1}}$.
3. **Result**: This ensures the change in the log-step-size ($\beta$) is bounded, providing "scale invariance" relative to the input features.

---

### Experimental Designs (Empirical Methodology)

#### 1. The Parameter Sensitivity Test (The "U-Curve")
- **Setup**: A stationary supervised learning task (linear regression).
- **Procedure**: Run the task hundreds of times, sweeping the meta-step-size ($\mu$ for Autostep, $\theta$ for IDBD) across several orders of magnitude.
- **Goal**: Measure Total MSE over the entire run.
- **Key Finding**: IDBD has a narrow "V" shaped sensitivity curve (works only at a specific $\theta$). Autostep has a wide, flat "U" shaped curve (works across a broad range of $\mu$).



#### 2. Non-Stationary Tracking (Bit-Stream Task)
- **Setup**: A binary input stream where the target $y$ is determined by a subset of features. The "relevant" features switch abruptly at fixed intervals.
- **Procedure**: Compare a fixed-step-size optimizer (LMS) against adaptive ones.
- **Goal**: Measure how quickly the algorithm reduces step-sizes for features that become "noise" and increases them for features that become "signal."
- **Key Finding**: Autostep successfully tracks these transitions without the "divergence spikes" common in IDBD during the moment of abrupt change.



#### 3. Irrelevant Feature Suppression
- **Setup**: 1 relevant feature and 19 irrelevant features (noise).
- **Procedure**: Start all step-sizes at a high initial value.
- **Goal**: Observe the trajectory of $\alpha_i$.
- **Key Finding**: Autostep quickly decays the step-sizes of the 19 noise features to near-zero, while maintaining a high step-size for the predictive feature, effectively performing online feature selection.



---

### Implementation Constraints for Claude Code
- **Metalearning State**: Must track weights ($w$), log-step-sizes ($\beta$), and the normalization vector ($v$).
- **Numerical Stability**: Autostep requires a small constant (e.g., $1e-8$) in the denominator of the $v$ update to prevent division by zero.
- **Batch Size**: The thesis is strictly $B=1$ (online). The logic must be implemented inside the innermost loop of the experience stream.
