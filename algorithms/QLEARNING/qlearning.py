"""Tabular Q-learning on the shared warehouse.

Q-learning is off-policy temporal-difference control. The behaviour
policy is epsilon-greedy. After a step leaves (s, a) and arrives at s'
with reward r, the backup uses the best next action, not the action
that policy will execute:

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

The next step draws a fresh epsilon-greedy action. That draw is not
stored, and it does not have to be the action inside the max.

A true terminal step has no next state. Its target is the reward alone:

    Q(s, a) <- Q(s, a) + alpha * (r - Q(s, a))

This warehouse never produces that case. The only ending is the 500-step
limit, and is_true_terminal stays false there, so the last step still
bootstraps from the max over the next state.

The table and the compact state key are the ones SARSA uses. The two
algorithms differ in the action the target reads. Both robots write the
shared table. Each backup uses that robot's own learning reward: its
warehouse reward plus its own stage bonus.
"""

from pathlib import Path

import numpy as np

from algorithms.QLEARNING.config import QLearningSettings, scheduled_epsilon
from algorithms.SARSA.sarsa import QTable
from algorithms.SARSA.state import STATE_FIELDS, state_key
from environment.config import LEARNING_SIGNAL, N_ACTIONS, N_AGENTS, PARAM_SHARING


def q_learning_target(reward, next_max_q, gamma, terminal):
    """Scalar TD target. A terminal step drops the next-state term."""
    reward = float(reward)
    if terminal:
        return reward
    return reward + float(gamma) * float(next_max_q)


class QLearningAgent:
    """Shared Q-table, epsilon-greedy behaviour, and greedy evaluation."""

    def __init__(self, settings=None, seed=0):
        if not PARAM_SHARING:
            raise ValueError("this Q-learning agent uses one shared table for both robots")
        if LEARNING_SIGNAL != "team":
            raise ValueError(f"Q-learning is wired for the team reward, got {LEARNING_SIGNAL!r}")

        self.settings = QLearningSettings() if settings is None else settings
        self.q = QTable(N_ACTIONS, self.settings.initial_q)
        self.rng = np.random.default_rng(int(seed))
        self.completed_episodes = 0
        self.epsilon = scheduled_epsilon(self.settings, 0)

    def choose_actions(self, observations):
        """Epsilon-greedy action for each robot. Training calls this."""
        return tuple(self._epsilon_greedy(state_key(observation)) for observation in observations)

    def act_greedy(self, observations):
        """Argmax of Q for each robot. evaluate() requires this policy."""
        return tuple(
            self.q.greedy_action(state_key(observation), self.rng, deterministic=True)
            for observation in observations
        )

    def backup(self, observations, actions, reward, next_observations=None):
        """Q-learning backup. Pass the next observations on a continuing step.

        Omit them when the step is a true terminal. The target is then
        the reward alone. The behaviour action for the next step is not
        an argument: the target reads max over the next row.
        """
        terminal = next_observations is None
        if not terminal and len(next_observations) != N_AGENTS:
            raise ValueError(f"expected {N_AGENTS} next observations")
        return self._backup_each(observations, actions, reward, terminal, next_observations)

    def finish_episode(self):
        """Apply the epsilon schedule after the episode has ended."""
        self.completed_episodes += 1
        self.epsilon = scheduled_epsilon(self.settings, self.completed_episodes)
        return self.epsilon

    def save(self, path):
        """Write the visited rows. Greedy playback only needs keys and values."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        states = list(self.q.rows.keys())
        if states:
            keys = np.asarray(states, dtype=np.int16)
            values = np.stack([self.q.rows[state] for state in states])
        else:
            keys = np.zeros((0, len(STATE_FIELDS)), dtype=np.int16)
            values = np.zeros((0, N_ACTIONS), dtype=np.float64)
        np.savez_compressed(
            path,
            keys=keys,
            values=np.asarray(values, dtype=np.float64),
            epsilon=np.float64(self.epsilon),
            completed_episodes=np.int64(self.completed_episodes),
        )
        return path

    def load(self, path):
        """Replace the table with rows saved by save()."""
        data = np.load(path)
        self.q.rows = {}
        keys = data["keys"]
        values = data["values"]
        if keys.shape[0] != values.shape[0]:
            raise ValueError("saved Q-learning keys and values have different lengths")
        for key, row in zip(keys, values):
            state = tuple(int(part) for part in key)
            self.q.rows[state] = np.asarray(row, dtype=np.float64).copy()
        self.epsilon = float(data["epsilon"])
        self.completed_episodes = int(data["completed_episodes"])
        return self

    def _epsilon_greedy(self, state):
        if float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return self.q.greedy_action(state, self.rng, deterministic=False)

    def _backup_each(self, observations, actions, reward, terminal, next_observations):
        if len(observations) != N_AGENTS or len(actions) != N_AGENTS:
            raise ValueError(f"expected {N_AGENTS} observations and actions")
        errors = []
        alpha = self.settings.alpha
        gamma = self.settings.gamma
        for robot in range(N_AGENTS):
            state = state_key(observations[robot])
            action = int(actions[robot])
            robot_reward = _robot_reward(reward, robot)
            if terminal:
                target = q_learning_target(robot_reward, 0.0, gamma, terminal=True)
            else:
                next_state = state_key(next_observations[robot])
                # Best stored action, including when the behaviour policy
                # will execute a different one.
                next_q = self.q.max_value(next_state)
                target = q_learning_target(robot_reward, next_q, gamma, terminal=False)
            errors.append(self.q.update(state, action, target, alpha))
        return tuple(errors)


def _robot_reward(reward, robot):
    """Scalar reward, or the entry that belongs to this robot."""
    if isinstance(reward, (int, float)):
        return float(reward)
    return float(reward[robot])
