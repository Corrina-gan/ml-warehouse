"""Training bonus shared by Q-learning, SARSA, REINFORCE, and PPO.

The warehouse pays 1 only when a carried shelf that is in the request
queue sits on a delivery goal. The robot is still carrying it. A goal cell
is a highway, and a shelf cannot be unloaded on a highway, so there is no
separate unload action in the score. The sequence that earns the 1 is:

1. Move to a requested shelf.
2. Toggle load to pick it up.
3. Carry it onto a goal. The warehouse then pays 1 and replaces the request.

That 1 is the delivery reward. The bonus below is added on top of it during
training and is not part of the logged team return or evaluate().

Suggested schedule, per robot:

- Moves closer to a requested shelf: +0.05 per cell.
- Picks up that requested shelf: +0.20. Dropping it again before a delivery
  pays 0.20 back, so toggling the same shelf up and down does not farm the bonus.
- Moves closer to a delivery goal while carrying the requested shelf: +0.05
  per cell.
- Hits an obstacle: -0.05. A forward that stays in place was rejected by a
  wall, a shelf, or another robot. A load toggle is not charged, even when
  it does nothing. Charging a missed toggle removed that action from the
  neural policies, so they walked past a shelf they were standing on.
- Successfully delivers the shelf: the warehouse reward of +1.00, not a
  second copy of that 1 inside this bonus.

Closer-to-shelf and closer-to-goal pay only for a new best distance this
episode. Walking away and back to the same shelf does not pay again, so
circling a shelf is not worth more than picking it up. A step that goes
around an obstacle still pays nothing for distance, and it is not charged.
Charging that detour made the greedy action keep walking into the shelf,
because staying blocked scored better than going around.

Each robot is trained on its own warehouse reward plus its own bonus.
Evaluation still sums the warehouse rewards and ignores the bonus.
"""

from environment.config import FORWARD, N_AGENTS

# Per cell of manhattan progress. One forward changes the distance by at most
# one cell, so a closer step pays 0.05, on the same scale as the delivery of 1.
APPROACH_PER_CELL = 0.05
CARRY_PER_CELL = 0.05
PICKUP = 0.20
OBSTACLE = -0.05


class ProgressRecord:
    """Best distance so far, per robot, for the shelf and for the goal.

    Reset at each episode boundary. A new request queue also clears the
    approach record, because the shelf being chased has changed.
    """

    def __init__(self, n_agents=N_AGENTS):
        self.approach = [None] * int(n_agents)
        self.carry = [None] * int(n_agents)
        self.targets = None

    def reset(self):
        for index in range(len(self.approach)):
            self.approach[index] = None
            self.carry[index] = None
        self.targets = None


def robot_bonuses(before, after, requested_cells, goals, warehouse_team_reward, actions, progress):
    """One progress bonus per robot, excluding the warehouse reward.

    before and after are tuples of (x, y, carrying_requested, carrying).
    requested_cells and goals are (x, y) pairs from before the step.
    actions is one action id per robot, the action that produced after.
    On a delivery the approach term is skipped, because the queue has
    already been replaced. A rejected move is still charged.
    """
    actions = tuple(int(action) for action in actions)
    if len(before) != len(after) or len(before) != len(actions):
        raise ValueError(
            f"expected one snapshot and one action per robot, got {len(before)}, {len(after)}, {len(actions)}"
        )
    if not isinstance(progress, ProgressRecord):
        raise TypeError("robot_bonuses needs the episode ProgressRecord")
    if progress.targets != requested_cells:
        progress.approach = [None] * len(progress.approach)
        progress.targets = requested_cells
    delivered = float(warehouse_team_reward) > 0.0
    return tuple(
        _robot_bonus(previous, current, requested_cells, goals, delivered, action, progress, index)
        for index, (previous, current, action) in enumerate(zip(before, after, actions))
    )


def _robot_bonus(previous, current, requested_cells, goals, delivered, action, progress, index):
    px, py, was_carrying_requested, was_carrying = previous
    cx, cy, _now_carrying_requested, now_carrying = current
    bonus = 0.0
    if _blocked_forward(action, previous, current):
        bonus += OBSTACLE
    if was_carrying_requested:
        if goals:
            bonus += CARRY_PER_CELL * _record_progress(
                progress.carry, index, _nearest(px, py, goals), _nearest(cx, cy, goals)
            )
        if was_carrying and not now_carrying and not delivered:
            bonus -= PICKUP
            progress.carry[index] = None
            progress.approach[index] = None
    elif requested_cells and not delivered:
        bonus += APPROACH_PER_CELL * _record_progress(
            progress.approach, index, _nearest(px, py, requested_cells), _nearest(cx, cy, requested_cells)
        )
        if (not was_carrying) and now_carrying and _now_holds_requested(current):
            bonus += PICKUP
            progress.approach[index] = None
            progress.carry[index] = _nearest(cx, cy, goals) if goals else None
    if delivered:
        progress.approach[index] = None
    return bonus


def _record_progress(record, index, before_distance, after_distance):
    """Cells of improvement beyond the best distance seen so far."""
    best = record[index]
    if best is None:
        best = before_distance
    gained = 0.0
    if after_distance < best:
        gained = best - after_distance
        best = after_distance
    record[index] = best
    return gained


def _blocked_forward(action, previous, current):
    """True when a forward stays in place. A load toggle is not an obstacle."""
    if action != FORWARD:
        return False
    return (previous[0], previous[1]) == (current[0], current[1])


def _now_holds_requested(current):
    return bool(current[2])


def _nearest(column, row, points):
    return min(abs(float(column) - x) + abs(float(row) - y) for x, y in points)
