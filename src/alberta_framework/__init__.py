"""Alberta Framework: Implementation of the Alberta Plan for AI Research.

This framework implements Step 1 of the Alberta Plan: continual supervised
learning with meta-learned step-sizes.

Core Philosophy: Temporal uniformity - every component updates at every time step.

Quick Start:
    >>> from alberta_framework import LinearLearner, IDBD, RandomWalkTarget
    >>>
    >>> # Create a non-stationary stream
    >>> stream = RandomWalkTarget(feature_dim=10, drift_rate=0.001)
    >>>
    >>> # Create a learner with adaptive step-sizes
    >>> learner = LinearLearner(optimizer=IDBD())
    >>>
    >>> # Run learning loop
    >>> state = learner.init(stream.feature_dim)
    >>> for step, timestep in enumerate(stream):
    ...     if step >= 10000:
    ...         break
    ...     result = learner.update(state, timestep.observation, timestep.target)
    ...     state = result.state

Reference: The Alberta Plan for AI Research (Sutton et al.)
"""

__version__ = "0.1.0"

# Core types
# Learners
from alberta_framework.core.learners import (
    LinearLearner,
    NormalizedLearnerState,
    NormalizedLinearLearner,
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
    Target,
    TimeStep,
)

# Streams
from alberta_framework.streams.base import ExperienceStream
from alberta_framework.streams.synthetic import (
    AbruptChangeTarget,
    CyclicTarget,
    RandomWalkTarget,
)

# Gymnasium streams (optional)
try:
    from alberta_framework.streams.gymnasium import (
        GymnasiumStream,
        PredictionMode,
        TDStream,
        make_epsilon_greedy_policy,
        make_gymnasium_stream,
        make_random_policy,
    )

    _gymnasium_available = True
except ImportError:
    _gymnasium_available = False

# Utilities
from alberta_framework.utils.metrics import (
    compare_learners,
    compute_cumulative_error,
    compute_running_mean,
    compute_tracking_error,
    extract_metric,
)

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
    "Target",
    "TimeStep",
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
    # Streams
    "AbruptChangeTarget",
    "CyclicTarget",
    "ExperienceStream",
    "RandomWalkTarget",
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
        "make_epsilon_greedy_policy",
        "make_gymnasium_stream",
        "make_random_policy",
    ]
