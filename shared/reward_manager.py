"""Rewards for the shared warehouse: the score, and the reward an algorithm trains on.

The score. step() returns one number per robot. Under RewardType.INDIVIDUAL
the robot that delivers receives 1 and the other receives 0. robot_rewards,
team_reward and reward_is_delivery_unit only read that list. They add no
penalty and no bonus, and shared.evaluation scores every algorithm with
them. Do not change them: a score is real deliveries and nothing else.

The training reward. A delivery is rare. Under random play 5 steps in 50,000
pay a reward (see the Analysis page), so a policy that starts out random sees
about ten rewards in a 100,000-step run and learns nothing from them.
The training reward adds rewards for the steps that lead to a delivery, so a
robot is told it is on the right track before a shelf reaches the goal:

    + LOAD_BONUS      it picks up a requested shelf
    - LOAD_BONUS      it puts that shelf down again before delivering it
    + 1               the warehouse's own delivery reward, unchanged
    + LOAD_BONUS      it puts the shelf back after delivering it, once per delivery
    + PROGRESS_BONUS  for each cell it moves toward the goal while carrying a
                      requested shelf, and the same amount off for each cell
                      it moves away

Picking a shelf up and putting it straight down pays nothing in total, and so
does walking back and forth. The only way to collect reward is the real cycle:
pick up, deliver, put back. A shelf nobody asked for pays nothing either way.

Only training uses the training reward. The environment is not modified.

It is the same for every algorithm. It reads the robots' observations and the
warehouse reward and nothing about the algorithm. Its two amounts are fixed
below and no algorithm has a setting for them. It comes in two forms that
give the same numbers: training_rewards() for a finished episode, and
TrainingReward for one step at a time, which an algorithm that learns during
the episode needs.
"""

import numpy as np

from environment.config import DELIVERY_REWARD, N_AGENTS
from environment.environment import make_warehouse

# The two amounts of the training reward. They are the same for every
# algorithm. Change them here, for everyone, or not at all.
LOAD_BONUS = 0.3
PROGRESS_BONUS = 0.02

# Where the values the training reward reads sit in one robot's observation:
# column, row, load flag, and whether the shelf on the robot's own cell (the
# one it stands on or carries) is requested.
OBS_X, OBS_Y, OBS_CARRYING = 0, 1, 2
OBS_SHELF_REQUESTED_HERE = 8 + 7 * 4 + 6

# The two delivery cells of rware-tiny-2ag-v2, as (column, row).
GOALS = ((4, 10), (5, 10))


def robot_rewards(rewards):
    """One float per robot, in robot order."""
    rewards = tuple(float(reward) for reward in rewards)
    if len(rewards) != N_AGENTS:
        raise ValueError(f"expected {N_AGENTS} rewards, got {len(rewards)}")
    return rewards


def team_reward(rewards):
    """Sum of the per-robot rewards. This is the shared step score.

    A delivery contributes 1. Do not average the two robots.
    """
    return float(sum(robot_rewards(rewards)))


def reward_is_delivery_unit(rewards):
    """True when every robot reward is 0 or the delivery value of 1."""
    return all(reward in (0.0, float(DELIVERY_REWARD)) for reward in robot_rewards(rewards))


def check_goals():
    """Raise if GOALS no longer matches the installed warehouse."""
    env = make_warehouse()
    try:
        goals = tuple(sorted((int(x), int(y)) for x, y in env.unwrapped.goals))
    finally:
        env.close()
    if goals != tuple(sorted(GOALS)):
        raise AssertionError(f"reward_manager.GOALS is {GOALS}, the warehouse has {goals}")
    return True


def cells_to_goal(features):
    """Cells between each robot and the nearest goal, walking along rows and columns."""
    columns = np.minimum(*(np.abs(features[..., OBS_X] - goal_x) for goal_x, _goal_y in GOALS))
    return columns + np.abs(features[..., OBS_Y] - GOALS[0][1])


class TrainingReward:
    """The training reward, one step at a time.

    Call reset() at the start of every episode and step() after every
    environment step. reset() also fits the on_episode_start hook of
    run_episode, iterate_episode and collect_steps.
    """

    def __init__(self):
        self.reset()

    def reset(self, _observations=None):
        # A robot that has delivered still holds the shelf until it puts it down.
        self._holds_delivered = np.zeros(N_AGENTS, dtype=bool)

    def step(self, features, next_features, delivery_rewards):
        """The reward each robot learns from on this step.

        features, next_features: (robots, >= 71), the observations before and
        after the step, as shared.observation.network_input builds them.
        delivery_rewards: (robots,), the warehouse's own reward for the step.
        Returns (robots,).
        """
        features = np.asarray(features, dtype=np.float32)
        next_features = np.asarray(next_features, dtype=np.float32)
        delivery_rewards = np.asarray(delivery_rewards, dtype=np.float32)
        if features.shape != next_features.shape or features.shape[:1] != delivery_rewards.shape:
            raise ValueError("features, next_features, and delivery_rewards must describe the same robots")

        carried = features[:, OBS_CARRYING] > 0.5
        carries = next_features[:, OBS_CARRYING] > 0.5
        was_requested = features[:, OBS_SHELF_REQUESTED_HERE] > 0.5
        is_requested = next_features[:, OBS_SHELF_REQUESTED_HERE] > 0.5
        picked_up = ~carried & carries
        put_down = carried & ~carries

        rewards = delivery_rewards.copy()
        rewards += LOAD_BONUS * (picked_up & is_requested)
        rewards -= LOAD_BONUS * (put_down & was_requested)

        # Putting a delivered shelf down pays once per delivery.
        self._holds_delivered |= delivery_rewards > 0
        rewards += LOAD_BONUS * (put_down & self._holds_delivered)
        self._holds_delivered &= ~put_down

        hauling = carried & was_requested & carries & is_requested
        rewards += PROGRESS_BONUS * hauling * (cells_to_goal(features) - cells_to_goal(next_features))
        return rewards


def training_rewards(features, next_features, delivery_rewards):
    """The training reward for every step of one finished episode.

    features, next_features: (steps, robots, >= 71), the observations before
    and after each step, in order, from the first step of the episode.
    shared.observation.batch_network_input builds them.
    delivery_rewards: (steps, robots), the warehouse's own reward.
    Returns (steps, robots), the same numbers TrainingReward gives step by step.
    """
    features = np.asarray(features, dtype=np.float32)
    next_features = np.asarray(next_features, dtype=np.float32)
    delivery_rewards = np.asarray(delivery_rewards, dtype=np.float32)
    if features.shape != next_features.shape or features.shape[:2] != delivery_rewards.shape:
        raise ValueError("features, next_features, and delivery_rewards must describe the same steps and robots")
    one_step = TrainingReward()
    return np.stack(
        [one_step.step(before, after, paid) for before, after, paid in zip(features, next_features, delivery_rewards)]
    )
