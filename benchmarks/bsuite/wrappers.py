"""Environment wrappers for Alberta Plan-aligned bsuite benchmarks.

Alberta Plan Step 6: "Episodic problems should be converted to continuing versions."

The ContinuingWrapper converts bsuite's episodic environments into continuing
streams where the agent never sees terminal states. At episode boundaries,
the environment auto-resets and returns a regular mid-step transition with
discount=0 to signal the pseudo-boundary without requiring the agent to
treat it as special.
"""

from __future__ import annotations

import dm_env
import numpy as np
from dm_env import specs


class ContinuingWrapper(dm_env.Environment):  # type: ignore[misc]
    """Wraps an episodic bsuite environment as a continuing stream.

    At episode boundaries, the environment auto-resets but the agent sees
    a normal MID transition. The discount is set to 0.0 at boundaries to
    signal the pseudo-termination without requiring the agent to treat it
    as a terminal state.

    Two modes:
    - ``'continuing'``: Episode boundaries become regular transitions with
      discount=0. Agent state always persists. The agent never sees FIRST
      or LAST step types.
    - ``'standard'``: Pass-through mode for bsuite score compatibility.
      Episodes are episodic but agent state still persists across episodes.

    Parameters
    ----------
    env : dm_env.Environment
        The underlying bsuite environment.
    mode : str
        Either ``'continuing'`` or ``'standard'``. Default: ``'continuing'``.
    continuing_discount : float
        Discount for non-boundary steps in continuing mode. Default: 0.99.
    """

    def __init__(
        self,
        env: dm_env.Environment,
        mode: str = "continuing",
        continuing_discount: float = 0.99,
    ) -> None:
        if mode not in ("continuing", "standard"):
            msg = f"mode must be 'continuing' or 'standard', got {mode!r}"
            raise ValueError(msg)
        self._env = env
        self._mode = mode
        self._continuing_discount = continuing_discount
        self._needs_reset = True

    @property
    def mode(self) -> str:
        """Current wrapper mode."""
        return self._mode

    @property
    def unwrapped(self) -> dm_env.Environment:
        """The underlying unwrapped environment."""
        return self._env

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment.

        In continuing mode, returns a MID timestep (not FIRST) so the
        agent never sees episode boundaries.
        """
        timestep = self._env.reset()
        self._needs_reset = False
        if self._mode == "continuing":
            return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=np.float64(0.0),
                discount=np.float64(self._continuing_discount),
                observation=timestep.observation,
            )
        return timestep

    def step(self, action: int) -> dm_env.TimeStep:
        """Take a step in the environment.

        In continuing mode, episode boundaries are converted to MID
        transitions with discount=0.0.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        dm_env.TimeStep
            The resulting timestep.
        """
        if self._needs_reset:
            return self.reset()

        timestep = self._env.step(action)

        if self._mode == "standard":
            if timestep.last():
                self._needs_reset = True
            return timestep

        # Continuing mode
        if timestep.last():
            # Auto-reset: get new observation, return as MID with discount=0
            new_ts = self._env.reset()
            return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=timestep.reward,
                discount=np.float64(0.0),
                observation=new_ts.observation,
            )
        # Regular mid-step
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=timestep.reward,
            discount=np.float64(self._continuing_discount),
            observation=timestep.observation,
        )

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec of the underlying environment."""
        return self._env.observation_spec()

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec of the underlying environment."""
        return self._env.action_spec()

    def __getattr__(self, name: str) -> object:
        """Forward attribute access to the underlying environment."""
        return getattr(self._env, name)
