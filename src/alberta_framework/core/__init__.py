"""Core components for the Alberta Framework."""

from alberta_framework.core.horde import (
    BatchedHordeResult,
    HordeLearner,
    HordeLearningResult,
    HordeUpdateResult,
    run_horde_learning_loop,
    run_horde_learning_loop_batched,
)
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
    DemonType,
    GVFSpec,
    HordeSpec,
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
    create_horde_spec,
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
    # GVF / Horde (Step 3)
    "BatchedHordeResult",
    "DemonType",
    "GVFSpec",
    "HordeLearner",
    "HordeLearningResult",
    "HordeSpec",
    "HordeUpdateResult",
    "create_horde_spec",
    "run_horde_learning_loop",
    "run_horde_learning_loop_batched",
]
