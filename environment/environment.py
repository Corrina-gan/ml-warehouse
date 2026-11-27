"""Shared warehouse for Q-learning, SARSA, REINFORCE, and PPO.

This module only creates the existing ``rware-tiny-2ag-v2`` environment.
It does not change the grid, the transitions, or the reward.
"""

import gymnasium as gym
import rware

from environment.config import (
    ENV_ID,
    GRID_COLS,
    GRID_ROWS,
    MAX_STEPS,
    N_ACTIONS,
    N_AGENTS,
    NOOP,
    OBS_SIZE,
)

# Importing rware registers rware-tiny-2ag-v2 with Gymnasium.
del rware


def make_warehouse(render_mode=None):
    """Return a new ``rware-tiny-2ag-v2`` environment.

    ``reset(seed)`` returns ``(observations, info)``. ``observations`` is
    a tuple of two ``float32`` vectors, each of shape ``(71,)``. ``info``
    is an empty dict.

    ``step(actions)`` takes one action per robot, a tuple of two ints in
    ``0..4``: no-op, forward, left, right, toggle load. It returns
    ``observations, rewards, terminated, truncated, info``. ``rewards`` is
    a list of two numbers. A delivery pays ``1`` to the robot on that goal
    with the requested shelf, and ``0`` to the other robot. The team score
    for a step is the sum of that list.

    The episode ends after 500 steps with ``terminated`` True and
    ``truncated`` False. That flag is the time limit. A delivery does not
    end the episode.
    """
    if render_mode is None:
        return gym.make(ENV_ID)
    return gym.make(ENV_ID, render_mode=render_mode)


def check_contract():
    """Raise if this configuration no longer matches the installed warehouse."""
    env = make_warehouse()
    try:
        warehouse = env.unwrapped
        if env.spec is None or env.spec.id != ENV_ID:
            raise AssertionError(f"environment id is not {ENV_ID}")
        if warehouse.n_agents != N_AGENTS:
            raise AssertionError(f"n_agents is {warehouse.n_agents}, expected {N_AGENTS}")
        if tuple(warehouse.grid_size) != (GRID_ROWS, GRID_COLS):
            raise AssertionError(f"grid_size is {warehouse.grid_size}, expected {(GRID_ROWS, GRID_COLS)}")
        if warehouse.max_steps != MAX_STEPS:
            raise AssertionError(f"max_steps is {warehouse.max_steps}, expected {MAX_STEPS}")
        if len(env.action_space) != N_AGENTS:
            raise AssertionError("action space does not have one Discrete space per robot")
        for space in env.action_space:
            if space.n != N_ACTIONS:
                raise AssertionError(f"action space is Discrete({space.n}), expected {N_ACTIONS}")
        observations, info = env.reset(seed=0)
        if info != {}:
            raise AssertionError(f"reset info is {info}, expected an empty dict")
        if len(observations) != N_AGENTS:
            raise AssertionError(f"reset returned {len(observations)} observations")
        for observation in observations:
            if tuple(observation.shape) != (OBS_SIZE,):
                raise AssertionError(f"observation shape is {observation.shape}")
        _observations, rewards, terminated, truncated, _info = env.step((NOOP, NOOP))
        if len(rewards) != N_AGENTS:
            raise AssertionError(f"reward length is {len(rewards)}")
        if terminated or truncated:
            raise AssertionError("the first step ended the episode")
    finally:
        env.close()
    return True


if __name__ == "__main__":
    env = make_warehouse()
    try:
        for index, (observation_space, action_space) in enumerate(
            zip(env.observation_space, env.action_space), start=1
        ):
            print(f"Robot {index} observation dimensions: {observation_space.shape[0]}")
            print(f"Robot {index} number of actions: {action_space.n}")
    finally:
        env.close()
