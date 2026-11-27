"""SARSA settings for the shared warehouse.

Gamma, the step budget, and the run ids come from environment.config.
Those stay the same for Q-learning, SARSA, REINFORCE, and PPO. The values
below belong to SARSA only.

The exploration schedule is chosen here. Three schedules match the
training remark:

- constant: keep epsilon fixed for every episode.
- decaying: reduce epsilon over the planned episodes, and stop at a
  positive minimum so exploration does not shut off.
- glie: epsilon falls toward 0. In the limit every action keeps being
  tried, which is the theoretical GLIE exploration rule. A finite
  episode budget cannot visit every state-action pair forever; the
  schedule only has that shape.
"""

import math
from dataclasses import asdict, dataclass

from environment.config import EVAL_EVERY_STEPS, GAMMA, MAX_STEPS, TOTAL_ENV_STEPS

EPSILON_SCHEDULES = ("constant", "decaying", "glie")


@dataclass(frozen=True)
class SARSASettings:
    """One SARSA run. Defaults match the shared comparison protocol."""

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
            raise ValueError("SARSA needs at least one environment step")
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


def scheduled_epsilon(settings, completed_episodes):
    """Epsilon for an episode, after `completed_episodes` have finished.

    The training loop applies this at the end of an episode. Episode 0
    therefore uses completed_episodes=0, which is the starting epsilon
    for every schedule.
    """
    completed = int(completed_episodes)
    if completed < 0:
        raise ValueError("completed_episodes cannot be negative")
    schedule = settings.epsilon_schedule
    if schedule == "constant":
        return float(settings.epsilon_start)
    if schedule == "glie":
        # 1, 1/2, 1/3, ... when epsilon_start is 1. The harmonic series
        # diverges, so the chance of trying a non-greedy action never
        # sums to a finite number. The limit of the sequence is 0.
        return float(settings.epsilon_start) / (completed + 1)
    fraction = min(1.0, completed / settings.planned_episodes())
    if fraction >= 1.0:
        return float(settings.epsilon_min)
    start = float(settings.epsilon_start)
    floor = float(settings.epsilon_min)
    # The linear step can undershoot the floor by a rounding error.
    return max(floor, start + (floor - start) * fraction)
