"""Alberta Framework: Implementation of the Alberta Plan for AI Research.

This framework implements Step 1 of the Alberta Plan: continual supervised
learning with meta-learned step-sizes.

Core Philosophy: Temporal uniformity - every component updates at every time step.

Quick Start:
    >>> import jax.random as jr
    >>> from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop
    >>>
    >>> # Create a non-stationary stream
    >>> stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)
    >>>
    >>> # Create a learner with adaptive step-sizes
    >>> learner = LinearLearner(optimizer=IDBD())
    >>>
    >>> # Run learning loop with scan
    >>> key = jr.key(42)
    >>> state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=key)

Reference: The Alberta Plan for AI Research (Sutton et al.)
"""

__version__ = "0.1.0"

# Core types
# Learners
from alberta_framework.core.learners import (
    LinearLearner,
    NormalizedLearnerState,
    NormalizedLinearLearner,
    UpdateResult,
    metrics_to_dicts,
    run_learning_loop,
    run_normalized_learning_loop,
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
    IDBDState,
    LearnerState,
    LMSState,
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
    RandomWalkState,
    RandomWalkStream,
    RandomWalkTarget,
    SuttonExperiment1State,
    SuttonExperiment1Stream,
)

# Utilities
from alberta_framework.utils.metrics import (
    compare_learners,
    compute_cumulative_error,
    compute_running_mean,
    compute_tracking_error,
    extract_metric,
)

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
    "IDBDState",
    "LMSState",
    "LearnerState",
    "NormalizerState",
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
    "run_normalized_learning_loop",
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
    "RandomWalkState",
    "RandomWalkStream",
    "RandomWalkTarget",
    "SuttonExperiment1State",
    "SuttonExperiment1Stream",
    # Utilities
    "compare_learners",
    "compute_cumulative_error",
    "compute_running_mean",
    "compute_tracking_error",
    "extract_metric",
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
