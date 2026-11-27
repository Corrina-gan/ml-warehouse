"""Turn the two robot observations into the one network input every algorithm uses.

With parameter sharing, one network serves both robots. An agent-ID one-hot
is appended so the network can tell them apart. Input row i is robot i:
[71 warehouse values | 5 task offsets | one-hot of length N_AGENTS].

The 5 offsets come from shared.task_features. They are attached before the
observation is stored, because the warehouse vector cannot see a requested
shelf outside the 3x3 window or the delivery goals.
"""

import numpy as np

from environment.config import N_AGENTS, OBS_SIZE
from shared.task_features import TASK_OBS_SIZE

INPUT_DIM = TASK_OBS_SIZE + N_AGENTS

# Center cell of the 3x3 sensor: has_agent, 4 direction bits, has_shelf, shelf_requested.
# SARSA reads this same bit as requested_here. The neural nets scale it up,
# because the shelf and goal offsets are all zero exactly when the robot is
# standing on the requested shelf and should toggle load.
REQUESTED_HERE_INDEX = 8 + 4 * 7 + 6

_AGENT_IDS = np.eye(N_AGENTS, dtype=np.float32)


def stack_observations(observations):
    """Tuple of N_AGENTS vectors -> array (N_AGENTS, TASK_OBS_SIZE), float32.

    The warehouse writes the first 71 values. attach_task appends the shelf
    and goal offsets. A NaN or an infinity is rejected here so a broken
    observation never reaches a network.
    """
    stacked = np.stack([np.asarray(obs, dtype=np.float32) for obs in observations])
    if stacked.shape != (N_AGENTS, TASK_OBS_SIZE):
        raise ValueError(f"expected {(N_AGENTS, TASK_OBS_SIZE)}, got {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise ValueError("observation contains NaN or infinity")
    return stacked


def network_input(observations):
    """Prepared state for every algorithm: raw observation plus agent id.

    Coordinates stay in grid units. The registered warehouse does not
    normalise them. Sensor bits are already 0 or 1. The task offsets are
    already attached, and the agent-ID one-hot is appended here.
    """
    return np.concatenate([stack_observations(observations), _AGENT_IDS], axis=1)


def scale_destination_offsets(array, scale, coordinate_scale=1.0, requested_scale=1.0):
    """Rescale the neural input. The stored observation is not modified.

    The four shelf and goal offsets are cell differences divided by the map
    diameter, so one cell is about 0.05 while column and row are in grid
    units. A destination scale of 2 * diameter makes one cell the value 2.

    Column and row are multiplied by coordinate_scale. A scale of 0.1 puts a
    grid coordinate on the same range as a sensor bit, so the direction
    one-hot and the cells ahead of the robot are not drowned by the raw
    column and row.

    The requested-shelf bit on the robot's own cell is multiplied by
    requested_scale. That bit is the pickup cue, and the offsets are zero
    in that state. The carrying flag stays 0 or 1.

    A scale of 1 leaves that group unchanged. SARSA does not call this
    function, and a checkpoint trained without one of these scales loads
    that scale as 1.
    """
    scaled = np.array(array, dtype=np.float32, copy=True)
    destination = float(scale)
    coordinate = float(coordinate_scale)
    requested = float(requested_scale)
    if destination != 1.0:
        scaled[..., OBS_SIZE : OBS_SIZE + 4] *= np.float32(destination)
    if coordinate != 1.0:
        scaled[..., 0:2] *= np.float32(coordinate)
    if requested != 1.0:
        scaled[..., REQUESTED_HERE_INDEX] *= np.float32(requested)
    return scaled


def batch_network_input(observation_tuples):
    """List of T observation tuples -> array (T, N_AGENTS, INPUT_DIM).

    Flatten the first two axes with reshape(-1, INPUT_DIM) to feed a network.
    Row t * N_AGENTS + i is robot i at step t.
    """
    return np.stack([network_input(observations) for observations in observation_tuples])
