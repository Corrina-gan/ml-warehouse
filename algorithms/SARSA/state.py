"""Compact state key for the tabular SARSA agent.

The warehouse observation is 71 integer-valued floats. Most of those
bits are the 3x3 sensor window, and they change whenever the other
robot or a requested shelf moves. Using the raw vector as a table key
would make almost every step a new state, so the table would not
generalise.

The key keeps what a shared Q-table can actually reuse:

- column, row, and facing direction, which locate the robot
- whether it is carrying a shelf
- whether the shelf on its current cell is requested
- whether the cell it faces holds the other robot, a shelf, or a
  requested shelf

The delivery goals are fixed cells, so column and row already tell the
table where a carried shelf can score. Highway cells are fixed by
column and row as well, so that bit is left out. Both robots share one
table. An agent id would only split the same experience in two.

The raw observation cannot name a requested shelf outside the 3x3 window.
attach_task appends the offset to that shelf, and the offset to a goal
only after the shelf has been picked up. The table stores the sign of
each offset, so the same column and row can keep a different action when
the request is somewhere else.

Direction follows rware.warehouse.Direction: 0 up, 1 down, 2 left,
3 right. Up decreases the row, down increases it, left decreases the
column, right increases it.
"""

import numpy as np

from environment.config import OBS_SIZE
from shared.task_features import TASK_EXTRA, TASK_OBS_SIZE

# One sensor cell stores has_agent, a 4-bit direction, has_shelf, shelf_requested.
_CELL_WIDTH = 7
_SENSOR_START = 8
_HERE = 4
# Sensor cells are row-major over offsets dy, dx in {-1, 0, +1}.
# The cell one step ahead of UP, DOWN, LEFT, RIGHT.
_AHEAD_CELL = (1, 7, 3, 5)

STATE_FIELDS = (
    "x",
    "y",
    "direction",
    "carrying",
    "requested_here",
    "agent_ahead",
    "shelf_ahead",
    "requested_ahead",
    "shelf_dx",
    "shelf_dy",
    "goal_dx",
    "goal_dy",
)


def state_key(observation):
    """One robot observation -> a hashable tuple in STATE_FIELDS order."""
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    if obs.shape[0] not in (OBS_SIZE, TASK_OBS_SIZE):
        raise ValueError(
            f"expected an observation of length {OBS_SIZE} or {TASK_OBS_SIZE}, got {obs.shape[0]}"
        )
    if not np.isfinite(obs).all():
        raise ValueError("observation contains NaN or infinity")

    direction_bits = obs[3:7] > 0.5
    if int(direction_bits.sum()) != 1:
        raise ValueError("observation direction is not a one-hot")
    direction = int(np.argmax(direction_bits))

    here = _cell_start(_HERE)
    ahead = _cell_start(_AHEAD_CELL[direction])
    return (
        int(obs[0]),
        int(obs[1]),
        direction,
        _bit(obs[2]),
        _bit(obs[here + 6]),
        _bit(obs[ahead + 0]),
        _bit(obs[ahead + 5]),
        _bit(obs[ahead + 6]),
        *_offset_signs(obs),
    )


def _offset_signs(obs):
    """Sign of the shelf offset, then the goal offset. Zero when absent."""
    if obs.shape[0] < OBS_SIZE + 4:
        return (0, 0, 0, 0)
    return tuple(_sign(obs[OBS_SIZE + index]) for index in range(TASK_EXTRA - 1))


def _sign(value):
    if value > 1e-6:
        return 1
    if value < -1e-6:
        return -1
    return 0


def _cell_start(cell_index):
    return _SENSOR_START + cell_index * _CELL_WIDTH


def _bit(value):
    return int(float(value) > 0.5)
