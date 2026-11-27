"""Tabular SARSA on the shared warehouse.

SARSA is on-policy temporal-difference control. After a step leaves
(s, a) and arrives at s' with reward r, the behaviour policy picks one
next action a' with epsilon-greedy. The backup is

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))

a' is then the action executed on the next step. The action inside the
backup and the action that is carried forward are the same draw.

A true terminal step has no next action. Its target is the reward alone:

    Q(s, a) <- Q(s, a) + alpha * (r - Q(s, a))

This warehouse never produces that case. The only ending is the 500-step
limit, and is_true_terminal stays false there, so the last step still
bootstraps from the action epsilon-greedy would take. That action is not
executed, because the episode is over.

Both robots write the shared table. The reward in the backup is the team
reward, the sum of the two robot rewards. A delivery pays 1 to the robot
on the goal and 0 to the other, and the sum gives both rows the same
learning signal.
"""

from pathlib import Path

import numpy as np

from algorithms.SARSA.config import SARSASettings, scheduled_epsilon
from algorithms.SARSA.state import STATE_FIELDS, state_key
from environment.config import LEARNING_SIGNAL, N_ACTIONS, N_AGENTS, PARAM_SHARING


def sarsa_target(reward, next_q, gamma, terminal):
    """Scalar TD target. A terminal step drops the next-state term."""
    reward = float(reward)
    if terminal:
        return reward
    return reward + float(gamma) * float(next_q)


class QTable:
    """Q(s, a) stored sparsely. A missing state reads as initial_q."""

    def __init__(self, n_actions, initial_q):
        if n_actions < 1:
            raise ValueError("the table needs at least one action")
        self.n_actions = int(n_actions)
        self.initial_q = float(initial_q)
        self.rows = {}

    def __len__(self):
        return len(self.rows)

    def get(self, state, action):
        row = self.rows.get(state)
        if row is None:
            return self.initial_q
        return float(row[action])

    def max_value(self, state):
        row = self.rows.get(state)
        if row is None:
            return self.initial_q
        return float(row.max())

    def update(self, state, action, target, alpha):
        """Apply one tabular step. Returns the TD error target - Q(s, a)."""
        row = self.rows.get(state)
        if row is None:
            row = np.full(self.n_actions, self.initial_q, dtype=np.float64)
            self.rows[state] = row
        action = int(action)
        td_error = float(target) - float(row[action])
        row[action] += float(alpha) * td_error
        return td_error

    def greedy_action(self, state, rng, deterministic):
        """An action whose Q is maximal.

        Training breaks ties at random, so equal values do not collapse
        onto action 0. Evaluation breaks ties by the lowest action id,
        which keeps the greedy policy repeatable.
        """
        row = self.rows.get(state)
        if row is None:
            if deterministic:
                return 0
            return int(rng.integers(0, self.n_actions))
        best = np.flatnonzero(row == row.max())
        if deterministic or len(best) == 1:
            return int(best[0])
        return int(rng.choice(best))


class SARSAAgent:
    """Shared Q-table, epsilon-greedy behaviour, and greedy evaluation."""

    def __init__(self, settings=None, seed=0):
        if not PARAM_SHARING:
            raise ValueError("this SARSA agent uses one shared table for both robots")
        if LEARNING_SIGNAL != "team":
            raise ValueError(f"SARSA is wired for the team reward, got {LEARNING_SIGNAL!r}")

        self.settings = SARSASettings() if settings is None else settings
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

    def backup(self, observations, actions, reward, next_observations=None, next_actions=None):
        """SARSA backup. Pass the next state and the chosen next action.

        Omit both when the step is a true terminal. The target is then
        the reward alone.
        """
        terminal = next_observations is None and next_actions is None
        if terminal:
            return self._backup_each(observations, actions, reward, terminal=True)
        if next_observations is None or next_actions is None:
            raise ValueError("a non-terminal SARSA backup needs the chosen next action")
        if len(next_actions) != N_AGENTS or len(next_observations) != N_AGENTS:
            raise ValueError(f"expected {N_AGENTS} next actions and observations")
        return self._backup_each(
            observations,
            actions,
            reward,
            terminal=False,
            next_observations=next_observations,
            next_actions=next_actions,
        )

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
            raise ValueError("saved SARSA keys and values have different lengths")
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

    def _backup_each(self, observations, actions, reward, terminal, next_observations=None, next_actions=None):
        if len(observations) != N_AGENTS or len(actions) != N_AGENTS:
            raise ValueError(f"expected {N_AGENTS} observations and actions")
        errors = []
        alpha = self.settings.alpha
        gamma = self.settings.gamma
        for robot in range(N_AGENTS):
            state = state_key(observations[robot])
            action = int(actions[robot])
            if terminal:
                target = sarsa_target(_robot_reward(reward, robot), 0.0, gamma, terminal=True)
            else:
                next_state = state_key(next_observations[robot])
                next_action = int(next_actions[robot])
                # Q of the action that was selected, including when another
                # action in this row has a larger value.
                next_q = self.q.get(next_state, next_action)
                target = sarsa_target(_robot_reward(reward, robot), next_q, gamma, terminal=False)
            errors.append(self.q.update(state, action, target, alpha))
        return tuple(errors)


def _robot_reward(reward, robot):
    """Scalar team reward, or the entry that belongs to this robot."""
    if isinstance(reward, (int, float)):
        return float(reward)
    return float(reward[robot])
