"""Core components for the Alberta Framework."""

from alberta_framework.core.learners import LinearLearner, TDLinearLearner, TDUpdateResult
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    TDIDBD,
    AutoTDIDBD,
    Optimizer,
    TDOptimizer,
    TDOptimizerUpdate,
)
from alberta_framework.core.types import (
    AutoTDIDBDState,
    IDBDState,
    LearnerState,
    LMSState,
    Observation,
    Prediction,
    Target,
    TDIDBDState,
    TDLearnerState,
    TDTimeStep,
    TimeStep,
)

__all__ = [
    # Supervised learning
    "IDBD",
    "IDBDState",
    "LMS",
    "LMSState",
    "LearnerState",
    "LinearLearner",
    "Observation",
    "Optimizer",
    "Prediction",
    "Target",
    "TimeStep",
    # TD learning
    "AutoTDIDBD",
    "AutoTDIDBDState",
    "TDIDBD",
    "TDIDBDState",
    "TDLearnerState",
    "TDLinearLearner",
    "TDOptimizer",
    "TDOptimizerUpdate",
    "TDTimeStep",
    "TDUpdateResult",
]
