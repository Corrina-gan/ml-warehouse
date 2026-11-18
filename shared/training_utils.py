"""Steps of the shared warehouse, stored before the environment moves on.

done ends an episode. is_true_terminal is the bootstrap mask, and it is
false here because the only ending is the 500-step limit.
"""

from dataclasses import dataclass

import numpy as np

from environment.config import MAX_STEPS, N_ACTIONS, N_AGENTS, OBS_SIZE
from shared.reward_manager import robot_rewards, team_reward


def is_true_terminal(step_index, terminated, truncated):
    """No real terminal state exists; the only episode end is the time limit.

    step_index, terminated, and truncated are the facts from this step.
    The environment sets terminated at step 500 and leaves truncated false.
    A delivery does not end the episode. Callers use done to stop or reset,
    and (1 - is_true_terminal(...)) as the bootstrap mask. That mask stays 1,
    so the value target still bootstraps from the next observation.
    """
    del step_index, terminated, truncated
    return False


def as_actions(actions):
    """Return one legal action int per robot."""
    actions = tuple(int(action) for action in actions)
    if len(actions) != N_AGENTS:
        raise ValueError(f"expected {N_AGENTS} actions, got {len(actions)}")
    for action in actions:
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} is outside 0..{N_ACTIONS - 1}")
    return actions


def copy_observations(observations):
    """A detached copy of each robot's observation."""
    copied = tuple(np.array(obs, dtype=np.float32, copy=True) for obs in observations)
    if len(copied) != N_AGENTS or any(obs.shape != (OBS_SIZE,) for obs in copied):
        shapes = tuple(obs.shape for obs in copied)
        raise ValueError(f"expected {N_AGENTS} observations of shape {(OBS_SIZE,)}, got {shapes}")
    return copied


@dataclass(frozen=True)
class Transition:
    """One shared time step."""

    observations: tuple
    actions: tuple
    rewards: tuple
    next_observations: tuple
    terminated: bool
    truncated: bool

    @property
    def team_reward(self):
        return team_reward(self.rewards)

    @property
    def done(self):
        return self.terminated or self.truncated


@dataclass(frozen=True)
class Episode:
    """One episode from the shared warehouse."""

    transitions: tuple
    seed: int | None

    @property
    def length(self):
        return len(self.transitions)

    @property
    def team_return(self):
        """Undiscounted sum of both robots' rewards."""
        return float(sum(step.team_reward for step in self.transitions))

    @property
    def robot_returns(self):
        """Undiscounted return for robot 1, then robot 2."""
        totals = [0.0] * N_AGENTS
        for step in self.transitions:
            for index, reward in enumerate(step.rewards):
                totals[index] += reward
        return tuple(totals)

    @property
    def deliveries(self):
        """Team return rounded to an int. Valid when each reward is 0 or 1."""
        return int(round(self.team_return))


@dataclass(frozen=True)
class Rollout:
    """A fixed number of steps. The last observations continue the next call.

    transitions may cross a time limit. done on a step only means the
    following observations belong to a new episode. Bootstrap with
    is_true_terminal, not with done.
    """

    transitions: tuple
    observations: tuple
    next_seed: int | None


def _reset(env, seed, on_episode_start):
    observations, _info = env.reset(seed=seed)
    observations = copy_observations(observations)
    if on_episode_start is not None:
        on_episode_start(observations)
    return observations


def _one_step(env, observations, choose_actions):
    actions = as_actions(choose_actions(observations))
    next_observations, rewards, terminated, truncated, _info = env.step(actions)
    return Transition(
        observations=observations,
        actions=actions,
        rewards=robot_rewards(rewards),
        next_observations=copy_observations(next_observations),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def iterate_episode(env, choose_actions, seed=None, on_episode_start=None):
    """Yield one transition at a time.

    SARSA updates from the yielded step before the next action is chosen.
    The generator stops when the environment sets done, and also stops at
    MAX_STEPS. done cuts the episode. It is not a terminal bootstrap.
    """
    observations = _reset(env, seed, on_episode_start)
    for _step_index in range(MAX_STEPS):
        transition = _one_step(env, observations, choose_actions)
        yield transition
        if transition.done:
            break
        observations = transition.next_observations


def run_episode(env, choose_actions, seed=None, on_episode_start=None):
    """Play one episode and record every shared transition.

    choose_actions(observations) returns one int per robot.
    on_episode_start(observations) runs after reset, before the first action.
    This stores a finished episode. SARSA should use iterate_episode, and a
    fixed-length PPO rollout should use collect_steps.
    """
    return Episode(
        transitions=tuple(iterate_episode(env, choose_actions, seed, on_episode_start)),
        seed=None if seed is None else int(seed),
    )


def collect_steps(env, choose_actions, n_steps, seed=None, observations=None, on_episode_start=None):
    """Collect n_steps, resetting and continuing after each time limit.

    Pass observations to continue a rollout. Otherwise pass seed, and the
    first reset uses that seed. Each later episode uses the next integer
    seed. The returned observations are the state after the last step, or
    the reset state when that step ended the episode.
    """
    if n_steps < 1:
        raise ValueError("collect_steps needs at least one step")
    if observations is None:
        observations = _reset(env, seed, on_episode_start)
        next_seed = None if seed is None else int(seed) + 1
    else:
        observations = copy_observations(observations)
        next_seed = None if seed is None else int(seed)

    transitions = []
    for _step_index in range(n_steps):
        transition = _one_step(env, observations, choose_actions)
        transitions.append(transition)
        if transition.done:
            observations = _reset(env, next_seed, on_episode_start)
            if next_seed is not None:
                next_seed += 1
        else:
            observations = transition.next_observations
    return Rollout(
        transitions=tuple(transitions),
        observations=observations,
        next_seed=next_seed,
    )
