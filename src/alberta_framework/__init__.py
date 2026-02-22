"""Alberta Framework: A JAX-based research framework for continual AI.

The Alberta Framework provides foundational components for continual reinforcement
learning research. Built on JAX for hardware acceleration, the framework emphasizes
temporal uniformity — every component updates at every time step, with no special
training phases or batch processing.

Roadmap
-------
| Step | Focus | Status |
|------|-------|--------|
| 1 | Meta-learned step-sizes (IDBD, Autostep) | **Complete** |
| 2 | Nonlinear function approximation (MLP, ObGD) | **In Progress** |
| 3 | GVF predictions, Horde architecture | Planned |
| 4 | Actor-critic with eligibility traces | Planned |
| 5-6 | Off-policy learning, average reward | Planned |
| 7-12 | Hierarchical, multi-agent, world models | Future |

Examples
--------
```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop

# Non-stationary stream where target weights drift over time
stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# Learner with IDBD meta-learned step-sizes
learner = LinearLearner(optimizer=IDBD())

# JIT-compiled training via jax.lax.scan
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
```

References
----------
- The Alberta Plan for AI Research (Sutton et al., 2022): https://arxiv.org/abs/2208.11173
- Adapting Bias by Gradient Descent (Sutton, 1992)
- Tuning-free Step-size Adaptation (Mahmood et al., 2012)
- Streaming Deep Reinforcement Learning Finally Works (Elsayed et al., 2024)
"""

__version__ = "0.8.0"

# Core types
# Learners
# Initializers
from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.learners import (
    LinearLearner,
    MLPLearner,
    MLPUpdateResult,
    TDLinearLearner,
    TDUpdateResult,
    UpdateResult,
    metrics_to_dicts,
    run_learning_loop,
    run_learning_loop_batched,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
    run_td_learning_loop,
)

# Multi-head learner
from alberta_framework.core.multi_head_learner import (
    BatchedMultiHeadResult,
    MultiHeadLearningResult,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    MultiHeadMLPUpdateResult,
    multi_head_metrics_to_dicts,
    run_multi_head_learning_loop,
    run_multi_head_learning_loop_batched,
)

# Normalizers
from alberta_framework.core.normalizers import (
    AnyNormalizerState,
    EMANormalizer,
    EMANormalizerState,
    Normalizer,
    WelfordNormalizer,
    WelfordNormalizerState,
)

# Optimizers
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    TDIDBD,
    AGCBounding,
    Autostep,
    AutoTDIDBD,
    Bounder,
    ObGD,
    ObGDBounding,
    Optimizer,
    TDOptimizer,
    TDOptimizerUpdate,
)
from alberta_framework.core.types import (
    AutostepParamState,
    AutostepState,
    AutoTDIDBDState,
    BatchedLearningResult,
    BatchedMLPResult,
    IDBDState,
    LearnerState,
    LMSState,
    MLPLearnerState,
    MLPParams,
    NormalizerHistory,
    NormalizerTrackingConfig,
    ObGDState,
    Observation,
    Prediction,
    StepSizeHistory,
    StepSizeTrackingConfig,
    Target,
    TDIDBDState,
    TDLearnerState,
    TDTimeStep,
    TimeStep,
    agent_age_s,
    agent_uptime_s,
    create_autotdidbd_state,
    create_obgd_state,
    create_tdidbd_state,
)

# Streams - base
from alberta_framework.streams.base import ScanStream

# Streams - synthetic
from alberta_framework.streams.synthetic import (
    AbruptChangeState,
    AbruptChangeStream,
    AbruptChangeTarget,
    CyclicState,
    CyclicStream,
    CyclicTarget,
    DynamicScaleShiftState,
    DynamicScaleShiftStream,
    PeriodicChangeState,
    PeriodicChangeStream,
    PeriodicChangeTarget,
    RandomWalkState,
    RandomWalkStream,
    RandomWalkTarget,
    ScaleDriftState,
    ScaleDriftStream,
    ScaledStreamState,
    ScaledStreamWrapper,
    SuttonExperiment1State,
    SuttonExperiment1Stream,
    make_scale_range,
)

# Utilities
from alberta_framework.utils.metrics import (
    compare_learners,
    compute_cumulative_error,
    compute_running_mean,
    compute_tracking_error,
    extract_metric,
)
from alberta_framework.utils.timing import Timer, format_duration

# Gymnasium streams (optional)
try:
    from alberta_framework.streams.gymnasium import (
        GymnasiumStream,
        PredictionMode,
        TDStream,
        collect_trajectory,
        learn_from_trajectory,
        learn_from_trajectory_normalized,
        make_epsilon_greedy_policy,
        make_gymnasium_stream,
        make_random_policy,
    )

    _gymnasium_available = True
except ImportError:
    _gymnasium_available = False

__all__ = [
    # Version
    "__version__",
    # Types - Supervised Learning
    "AutostepParamState",
    "AutostepState",
    "BatchedLearningResult",
    "IDBDState",
    "LMSState",
    "LearnerState",
    "NormalizerHistory",
    "AnyNormalizerState",
    "EMANormalizerState",
    "WelfordNormalizerState",
    "NormalizerTrackingConfig",
    "ObGDState",
    "Observation",
    "Prediction",
    "StepSizeHistory",
    "StepSizeTrackingConfig",
    "Target",
    "TimeStep",
    "UpdateResult",
    # Types - MLP
    "BatchedMLPResult",
    "MLPLearnerState",
    "MLPParams",
    "MLPUpdateResult",
    # Types - TD Learning
    "AutoTDIDBDState",
    "TDIDBDState",
    "TDLearnerState",
    "TDTimeStep",
    "TDUpdateResult",
    "agent_age_s",
    "agent_uptime_s",
    "create_obgd_state",
    "create_tdidbd_state",
    "create_autotdidbd_state",
    # Optimizers - Supervised Learning
    "AGCBounding",
    "Autostep",
    "Bounder",
    "IDBD",
    "LMS",
    "ObGD",
    "ObGDBounding",
    "Optimizer",
    # Optimizers - TD Learning
    "AutoTDIDBD",
    "TDIDBD",
    "TDOptimizer",
    "TDOptimizerUpdate",
    # Initializers
    "sparse_init",
    # Normalizers
    "Normalizer",
    "EMANormalizer",
    "WelfordNormalizer",
    # Learners - Supervised Learning
    "LinearLearner",
    "run_learning_loop",
    "run_learning_loop_batched",
    "metrics_to_dicts",
    # Learners - MLP
    "MLPLearner",
    "run_mlp_learning_loop",
    "run_mlp_learning_loop_batched",
    # Learners - Multi-Head MLP
    "BatchedMultiHeadResult",
    "MultiHeadLearningResult",
    "MultiHeadMLPLearner",
    "MultiHeadMLPState",
    "MultiHeadMLPUpdateResult",
    "multi_head_metrics_to_dicts",
    "run_multi_head_learning_loop",
    "run_multi_head_learning_loop_batched",
    # Learners - TD Learning
    "TDLinearLearner",
    "run_td_learning_loop",
    # Streams - protocol
    "ScanStream",
    # Streams - synthetic
    "AbruptChangeState",
    "AbruptChangeStream",
    "AbruptChangeTarget",
    "CyclicState",
    "CyclicStream",
    "CyclicTarget",
    "DynamicScaleShiftState",
    "DynamicScaleShiftStream",
    "PeriodicChangeState",
    "PeriodicChangeStream",
    "PeriodicChangeTarget",
    "RandomWalkState",
    "RandomWalkStream",
    "RandomWalkTarget",
    "ScaleDriftState",
    "ScaleDriftStream",
    "ScaledStreamState",
    "ScaledStreamWrapper",
    "SuttonExperiment1State",
    "SuttonExperiment1Stream",
    # Stream utilities
    "make_scale_range",
    # Utilities
    "compare_learners",
    "compute_cumulative_error",
    "compute_running_mean",
    "compute_tracking_error",
    "extract_metric",
    # Timing
    "Timer",
    "format_duration",
]

# Add Gymnasium exports if available
if _gymnasium_available:
    __all__ += [
        "GymnasiumStream",
        "PredictionMode",
        "TDStream",
        "collect_trajectory",
        "learn_from_trajectory",
        "learn_from_trajectory_normalized",
        "make_epsilon_greedy_policy",
        "make_gymnasium_stream",
        "make_random_policy",
    ]
