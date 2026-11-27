"""Warehouse rules: which of the five actions a robot may take right now.

Every hand-written rule about acting is in this file, and it is shared: any
algorithm may apply it by closing the logits or Q-values of the actions
allowed_actions() marks False. A policy then only chooses among the actions
the rules leave open. The rules read one robot's own observation and the
fixed floor plan, nothing else.

The rules, strongest first:

1. Load handling is automatic.
   An empty-handed robot standing on a requested shelf picks it up.
   A robot holding a shelf it has already delivered puts it down as soon as
   it stands on a storage cell. In every other case TOGGLE_LOAD is closed, so
   a robot never lifts a shelf nobody asked for and never drops a requested
   shelf before the goal.

2. Reach for what is next to you.
   An empty-handed robot with a requested shelf in the cell in front, to
   either side, or behind it steps or turns toward that shelf. A robot holding
   a delivered shelf does the same toward an empty storage cell.

3. No wasted moves.
   NOOP is closed. FORWARD is closed when it would hit the wall, the other
   robot, or, for a loaded robot, a shelf.

What is left for the policy is navigation: which way to walk to find a
requested shelf, how to carry it to the goal, and where to look for an empty
storage cell afterwards.

These rules deliver shelves on their own. RulesOnly below plays them with no
network: walking forward whenever that is allowed already completes
deliveries. That score is the floor. A trained policy that uses these rules
only counts as having learned something by how far it beats it.
"""

import numpy as np

from environment.config import FORWARD, LEFT, N_ACTIONS, NOOP, OBS_SIZE, RIGHT, TOGGLE_LOAD
from environment.environment import make_warehouse
from shared.observation import network_input

# Where each value sits in one robot's observation. rware writes them in this
# order: position, load flag, facing one-hot, highway flag, then nine sensor
# cells of seven values each, row by row around the robot.
X, Y, CARRYING = 0, 1, 2
FACING = slice(3, 7)
FIRST_CELL = 8
CELL_SIZE = 7
HAS_ROBOT, HAS_SHELF, SHELF_REQUESTED = 0, 5, 6
CELL_HERE = 4

# Facing index, as in rware.warehouse.Direction: up, down, left, right.
STEP_X = np.array([0, 0, -1, 1])
STEP_Y = np.array([-1, 1, 0, 0])
SENSOR_CELL = np.array([1, 7, 3, 5])
AFTER_LEFT_TURN = np.array([2, 3, 1, 0])
AFTER_RIGHT_TURN = np.array([3, 2, 0, 1])
OPPOSITE = np.array([1, 0, 3, 2])

# Floor plan of rware-tiny-2ag-v2. Storage cells hold the shelves. Every other
# cell is aisle, where a shelf cannot be put down.
GRID_COLUMNS = 10
GRID_ROWS = 11
STORAGE = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=bool)
STORAGE[1:9, [1, 2, 7, 8]] = True


def check_floor_plan():
    """Raise if STORAGE no longer matches the installed warehouse."""
    env = make_warehouse()
    try:
        aisles = np.asarray(env.unwrapped.highways).astype(bool)
    finally:
        env.close()
    if aisles.shape != STORAGE.shape or not np.array_equal(~aisles, STORAGE):
        raise AssertionError("warehouse_rules.STORAGE does not match the warehouse floor plan")
    return True


def allowed_actions(features):
    """Boolean array (rows, N_ACTIONS). True marks an action the robot may take.

    features: (rows, >= OBS_SIZE), one robot per row, as shared.observation builds it.
    The answer depends on that row alone, so the same call works for one step
    or for a whole episode at once.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < OBS_SIZE:
        raise ValueError(f"expected features of shape (rows, >= {OBS_SIZE}), got {features.shape}")
    rows = np.arange(features.shape[0])
    x = np.rint(features[:, X]).astype(int)
    y = np.rint(features[:, Y]).astype(int)
    facing = features[:, FACING].argmax(axis=1)
    carrying = features[:, CARRYING] > 0.5

    def here(value):
        return features[:, FIRST_CELL + CELL_SIZE * CELL_HERE + value] > 0.5

    def look(direction, value):
        """One sensor value of the neighbouring cell in `direction`, per row."""
        return features[rows, FIRST_CELL + CELL_SIZE * SENSOR_CELL[direction] + value] > 0.5

    def inside(direction):
        next_x = x + STEP_X[direction]
        next_y = y + STEP_Y[direction]
        return (next_x >= 0) & (next_x < GRID_COLUMNS) & (next_y >= 0) & (next_y < GRID_ROWS)

    def empty_storage(direction):
        next_x = np.clip(x + STEP_X[direction], 0, GRID_COLUMNS - 1)
        next_y = np.clip(y + STEP_Y[direction], 0, GRID_ROWS - 1)
        return inside(direction) & STORAGE[next_y, next_x] & ~look(direction, HAS_SHELF)

    # The centre cell describes the shelf the robot is on, or the one it carries.
    empty_handed = ~carrying
    holding_delivered = carrying & ~here(SHELF_REQUESTED)

    # Rule 3: no wasted moves.
    allowed = np.zeros((features.shape[0], N_ACTIONS), dtype=bool)
    allowed[:, LEFT] = True
    allowed[:, RIGHT] = True
    allowed[:, FORWARD] = (
        inside(facing)
        & ~look(facing, HAS_ROBOT)
        & ~(carrying & look(facing, HAS_SHELF))
    )

    # Rule 2: reach for what is next to you. A later line overrides an earlier
    # one, so a target in front wins over one at the side or behind.
    def is_target(direction):
        free = inside(direction) & ~look(direction, HAS_ROBOT)
        shelf_to_fetch = empty_handed & look(direction, SHELF_REQUESTED)
        cell_to_fill = holding_delivered & empty_storage(direction)
        return free & (shelf_to_fetch | cell_to_fill)

    for direction, action in (
        (OPPOSITE[facing], LEFT),
        (AFTER_RIGHT_TURN[facing], RIGHT),
        (AFTER_LEFT_TURN[facing], LEFT),
        (facing, FORWARD),
    ):
        _only(allowed, is_target(direction), action)

    # Rule 1: load handling is automatic.
    pick_up = empty_handed & here(SHELF_REQUESTED)
    put_down = holding_delivered & STORAGE[np.clip(y, 0, GRID_ROWS - 1), np.clip(x, 0, GRID_COLUMNS - 1)]
    _only(allowed, pick_up | put_down, TOGGLE_LOAD)
    return allowed


def _only(allowed, rows, action):
    """In the given rows, close every action except this one."""
    allowed[rows] = False
    allowed[rows, action] = True


class RulesOnly:
    """The rules with no network and no learning.

    forward=True walks forward whenever the rules allow it and otherwise takes
    the first allowed of LEFT, RIGHT. Nothing is random. forward=False picks
    uniformly at random among the allowed actions.

    The forward walk is much the stronger of the two. Its score is the floor a
    trained policy has to beat before it can claim to have learned anything.
    """

    WALK = (FORWARD, LEFT, RIGHT, TOGGLE_LOAD, NOOP)

    def __init__(self, forward=True):
        self.forward = bool(forward)

    def reset_episode(self, _observations=None):
        """Nothing to reset. The rules keep no memory."""

    def choose_actions(self, observations):
        allowed = allowed_actions(network_input(observations))
        if self.forward:
            return tuple(next(action for action in self.WALK if row[action]) for row in allowed)
        return tuple(int(np.random.choice(np.flatnonzero(row))) for row in allowed)
