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
| 2 | Feature generation and testing | Planned |
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
"""

__version__ = "0.2.0"

# Core types
# Learners
from alberta_framework.core.learners import (
    LinearLearner,
    NormalizedLearnerState,
    NormalizedLinearLearner,
    UpdateResult,
    metrics_to_dicts,
    run_learning_loop,
    run_learning_loop_batched,
    run_normalized_learning_loop,
    run_normalized_learning_loop_batched,
)

# Normalizers
from alberta_framework.core.normalizers import (
    NormalizerState,
    OnlineNormalizer,
    create_normalizer_state,
)

# Optimizers
from alberta_framework.core.optimizers import IDBD, LMS, Autostep, Optimizer
from alberta_framework.core.types import (
    AutostepState,
    BatchedLearningResult,
    BatchedNormalizedResult,
    IDBDState,
    LearnerState,
    LMSState,
    NormalizerHistory,
    NormalizerTrackingConfig,
    Observation,
    Prediction,
    StepSizeHistory,
    StepSizeTrackingConfig,
    Target,
    TimeStep,
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
    # Types
    "AutostepState",
    "BatchedLearningResult",
    "BatchedNormalizedResult",
    "IDBDState",
    "LMSState",
    "LearnerState",
    "NormalizerHistory",
    "NormalizerState",
    "NormalizerTrackingConfig",
    "Observation",
    "Prediction",
    "StepSizeHistory",
    "StepSizeTrackingConfig",
    "Target",
    "TimeStep",
    "UpdateResult",
    # Optimizers
    "Autostep",
    "IDBD",
    "LMS",
    "Optimizer",
    # Normalizers
    "OnlineNormalizer",
    "create_normalizer_state",
    # Learners
    "LinearLearner",
    "NormalizedLearnerState",
    "NormalizedLinearLearner",
    "run_learning_loop",
    "run_learning_loop_batched",
    "run_normalized_learning_loop",
    "run_normalized_learning_loop_batched",
    "metrics_to_dicts",
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
