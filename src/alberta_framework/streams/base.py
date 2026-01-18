"""Base protocol for experience streams.

Experience streams generate temporally-uniform experience for continual learning.
Every time step produces a new observation-target pair.
"""

from typing import Iterator, Protocol

from alberta_framework.core.types import TimeStep


class ExperienceStream(Protocol):
    """Protocol for generating temporally-uniform experience.

    An experience stream is an infinite iterator that produces TimeStep
    instances, each containing an observation and target for supervised
    learning tasks.

    The stream should be non-stationary to test continual learning
    capabilities - the underlying target function changes over time.
    """

    def __iter__(self) -> Iterator[TimeStep]:
        """Return iterator over time steps."""
        ...

    def __next__(self) -> TimeStep:
        """Generate the next time step."""
        ...

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        ...
