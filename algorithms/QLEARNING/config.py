"""Q-learning settings for the shared warehouse.

Gamma, the step budget, and the run ids come from environment.config.
Those stay the same for Q-learning, SARSA, REINFORCE, and PPO. The
values below belong to Q-learning only.

The exploration schedule is the same family SARSA uses. Q-learning is
off-policy, so the schedule only chooses the behaviour action. The
backup always reads the best action in the next state, whatever this
schedule drew.
"""

import math
from dataclasses import asdict, dataclass

from algorithms.SARSA.config import EPSILON_SCHEDULES, scheduled_epsilon
from environment.config import EVAL_EVERY_STEPS, GAMMA, MAX_STEPS, TOTAL_ENV_STEPS

__all__ = ["EPSILON_SCHEDULES", "QLearningSettings", "scheduled_epsilon"]


@dataclass(frozen=True)
class QLearningSettings:
    """One Q-learning run. Defaults match the shared comparison protocol."""

    gamma: float = GAMMA
    # Tabular step size. 1 replaces the stored value with the target.
    alpha: float = 0.1
    epsilon_start: float = 1.0
    # Used by the decaying schedule. Constant ignores it. GLIE ignores it
    # because that schedule is required to approach 0.
    epsilon_min: float = 0.05
    epsilon_schedule: str = "decaying"
    # Unseen states read as this value. 0 is the usual tabular start.
    initial_q: float = 0.0
    total_env_steps: int = TOTAL_ENV_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS

    def __post_init__(self):
        if self.epsilon_schedule not in EPSILON_SCHEDULES:
            names = ", ".join(EPSILON_SCHEDULES)
            raise ValueError(f"epsilon_schedule must be one of {names}, got {self.epsilon_schedule!r}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start}")
        if not 0.0 <= self.epsilon_min <= 1.0:
            raise ValueError(f"epsilon_min must be in [0, 1], got {self.epsilon_min}")
        if self.epsilon_schedule == "decaying" and self.epsilon_min > self.epsilon_start:
            raise ValueError("epsilon_min cannot be above epsilon_start")
        if not math.isfinite(self.initial_q):
            raise ValueError("initial_q must be finite")
        if self.total_env_steps < 1:
            raise ValueError("Q-learning needs at least one environment step")
        if self.eval_every_steps < 0:
            raise ValueError("eval_every_steps cannot be negative")

    def planned_episodes(self):
        """Episodes the decaying schedule spreads itself across.

        Every warehouse episode is MAX_STEPS long, so the shared step
        budget is this many episodes when it divides evenly.
        """
        return max(1, math.ceil(self.total_env_steps / MAX_STEPS))

    def as_dict(self):
        values = asdict(self)
        values["planned_episodes"] = self.planned_episodes()
        return values
