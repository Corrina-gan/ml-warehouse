"""Directions the warehouse observation does not contain.

A robot sees its own column, row, facing, and a 3x3 window. A requested
shelf outside that window is invisible, and the delivery goals are not in
the observation at all. The policy cannot learn "walk to that shelf, then
to a goal" from the raw vector.

These five numbers are attached before an action is chosen, and the same
numbers are stored with the transition the update trains on. They are read
from the live warehouse. The environment observation stays 71 values.

While the robot is empty, the first two numbers are the offset to the
nearest requested shelf and the goal offset is 0. After that pickup, the
shelf offset is 0 and the next two numbers point at the nearest delivery
goal. The last number is 1 only while the carried shelf is still requested.
The goal direction is withheld until the pickup, so the features do not
skip that step.

A robot holds one shelf at a time. After a delivery it still carries the
delivered shelf, which nobody is asking for any more, and it cannot pick up
the next request until that shelf is back in storage. While a robot carries
such a shelf, the first two numbers are the offset to the nearest free
storage cell, where a load toggle puts the shelf down. Pointing it at a
requested shelf it cannot lift would lead nowhere.
"""

import numpy as np

from environment.config import N_AGENTS, OBS_SIZE

# Longest manhattan distance on this grid is 10 + 9.
DIAMETER = 19.0
TASK_EXTRA = 5
TASK_OBS_SIZE = OBS_SIZE + TASK_EXTRA


def attach_task(warehouse, observations):
    """Raw or already attached observations -> one vector per robot.

    Recomputes the five numbers from the warehouse, so calling this twice
    does not append them twice.
    """
    raw = []
    for observation in observations:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (OBS_SIZE,) and vector.shape != (TASK_OBS_SIZE,):
            raise ValueError(
                f"expected an observation of length {OBS_SIZE} or {TASK_OBS_SIZE}, got {vector.shape[0]}"
            )
        raw.append(vector[:OBS_SIZE].copy())
    if len(raw) != N_AGENTS:
        raise ValueError(f"expected {N_AGENTS} observations, got {len(raw)}")
    extra = _features(warehouse)
    return tuple(np.concatenate([observation, feature]) for observation, feature in zip(raw, extra))


def _features(warehouse):
    queue = list(warehouse.request_queue)
    queue_ids = {id(shelf) for shelf in queue}
    shelves = tuple((int(shelf.x), int(shelf.y)) for shelf in queue)
    goals = tuple((int(x), int(y)) for x, y in warehouse.goals)
    features = []
    for agent in warehouse.agents:
        carried = agent.carrying_shelf
        carrying_requested = carried is not None and id(carried) in queue_ids
        if carrying_requested and goals:
            dx, dy = _offset(agent.x, agent.y, goals)
            features.append(
                np.asarray((0.0, 0.0, dx / DIAMETER, dy / DIAMETER, 1.0), dtype=np.float32)
            )
        else:
            # Where this robot has to go before it can pick up a request.
            # A loaded robot first has to put its shelf back in storage.
            targets = shelves if carried is None else free_storage_cells(warehouse, agent)
            if targets:
                dx, dy = _offset(agent.x, agent.y, targets)
                features.append(
                    np.asarray((dx / DIAMETER, dy / DIAMETER, 0.0, 0.0, 0.0), dtype=np.float32)
                )
            else:
                features.append(np.zeros(TASK_EXTRA, dtype=np.float32))
    return features


def free_storage_cells(warehouse, agent):
    """Storage cells where this robot could put its shelf down, as (column, row).

    A storage cell is any cell that is not a highway. It is free when no
    shelf stands on it. The shelf the robot is carrying does not count, so
    the cell under a robot that has just lifted a shelf is free.
    """
    carried = agent.carrying_shelf
    taken = {(int(shelf.x), int(shelf.y)) for shelf in warehouse.shelfs if shelf is not carried}
    storage = np.argwhere(~np.asarray(warehouse.highways, dtype=bool))
    return tuple((int(x), int(y)) for y, x in storage if (int(x), int(y)) not in taken)


def _offset(column, row, points):
    target = min(points, key=lambda point: abs(column - point[0]) + abs(row - point[1]))
    return float(target[0] - column), float(target[1] - row)
