"""Forward moves the warehouse will reject.

A forward into the map edge stays in place. A forward into a standing shelf
while this robot is carrying a shelf is cancelled. A forward into the cell
the other robot currently occupies is also cancelled unless that robot
leaves on the same step, and a policy that always walks forward never
leaves. The mask removes those forwards from the categorical distribution,
so the robot has to choose a turn, a load, or a wait. It does not point
the robot at a shelf or a goal.

An empty robot may still walk onto a shelf. That move is legal.
"""

import numpy as np
import torch

from environment.config import FORWARD, GRID_COLS, GRID_ROWS

# Same 3x3 layout as the SARSA state key. Ahead of up, down, left, right.
_AHEAD_CELL = (1, 7, 3, 5)
_CELL_WIDTH = 7
_SENSOR_START = 8
_HAS_AGENT = 0
_HAS_SHELF = 5


def blocked_forward(observations):
    """Boolean mask, True where forward cannot leave the cell.

    observations are the unscaled network inputs, shape (..., input dim).
    Column and row are still in grid units.
    """
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape[-1] < _SENSOR_START + 9 * _CELL_WIDTH:
        raise ValueError(f"observation is too short to read the sensor, got {obs.shape[-1]}")
    flat = obs.reshape(-1, obs.shape[-1])
    column = np.rint(flat[:, 0]).astype(np.int32)
    row = np.rint(flat[:, 1]).astype(np.int32)
    direction = np.argmax(flat[:, 3:7], axis=-1)
    wall = (
        ((direction == 0) & (row <= 0))
        | ((direction == 1) & (row >= GRID_ROWS - 1))
        | ((direction == 2) & (column <= 0))
        | ((direction == 3) & (column >= GRID_COLS - 1))
    )
    ahead = np.asarray(_AHEAD_CELL, dtype=np.int32)[direction]
    cell = _SENSOR_START + ahead * _CELL_WIDTH
    rows = np.arange(flat.shape[0])
    shelf_ahead = flat[rows, cell + _HAS_SHELF] > 0.5
    agent_ahead = flat[rows, cell + _HAS_AGENT] > 0.5
    carrying = flat[:, 2] > 0.5
    blocked = wall | (carrying & shelf_ahead) | agent_ahead
    return blocked.reshape(obs.shape[:-1])


def mask_forward_logits(logits, blocked):
    """Set the forward logit to the dtype minimum wherever blocked is true.

    logits: (batch, n_actions) torch tensor. blocked: (batch,) bool tensor
    on the same device. A false mask returns the same tensor.
    """
    if blocked.ndim != 1 or blocked.shape[0] != logits.shape[0]:
        raise ValueError(f"blocked shape {tuple(blocked.shape)} does not match logits {tuple(logits.shape)}")
    if not bool(blocked.any()):
        return logits
    masked = logits.clone()
    masked[blocked, FORWARD] = torch.finfo(logits.dtype).min
    return masked
