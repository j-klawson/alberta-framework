"""Experience streams for continual learning."""

from alberta_framework.streams.base import ExperienceStream
from alberta_framework.streams.synthetic import (
    AbruptChangeTarget,
    CyclicTarget,
    RandomWalkTarget,
)

__all__ = [
    "AbruptChangeTarget",
    "CyclicTarget",
    "ExperienceStream",
    "RandomWalkTarget",
]

# Gymnasium streams are optional - only export if gymnasium is installed
try:
    from alberta_framework.streams.gymnasium import (
        GymnasiumStream,
        PredictionMode,
        TDStream,
        make_epsilon_greedy_policy,
        make_gymnasium_stream,
        make_random_policy,
    )

    __all__ += [
        "GymnasiumStream",
        "PredictionMode",
        "TDStream",
        "make_epsilon_greedy_policy",
        "make_gymnasium_stream",
        "make_random_policy",
    ]
except ImportError:
    # gymnasium not installed
    pass
