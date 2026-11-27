"""REINFORCE settings for the shared warehouse.

Gamma, the hidden-layer widths, the step budget, and the run ids come from
environment.config. Those stay the same for Q-learning, SARSA, REINFORCE,
and PPO. The values below belong to REINFORCE only.
"""

from dataclasses import asdict, dataclass

from environment.config import (
    EVAL_EVERY_STEPS,
    GAMMA,
    HIDDEN,
    TOTAL_ENV_STEPS,
)
from shared.task_features import DIAMETER


@dataclass(frozen=True)
class REINFORCESettings:
    """One REINFORCE run. Defaults match the shared comparison protocol."""

    gamma: float = GAMMA
    hidden: tuple = HIDDEN
    # REINFORCE needs finished episodes. One update uses this many of them,
    # so 100,000 steps give 200 updates.
    episodes_per_update: int = 1
    # The learning rate falls along half a cosine to a tenth of this value.
    learning_rate: float = 2e-3
    # Learning rate of the baseline network. 0 switches the baseline off,
    # and the update then weights each action by its plain return.
    baseline_rate: float = 3e-3
    # Passes over the batch when fitting the baseline to the returns.
    baseline_epochs: int = 4
    baseline_minibatch_size: int = 500
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    # The entropy bonus falls in a straight line from start to end across the
    # step budget, so the policy explores early and commits late. Evaluation
    # is greedy.
    entropy_coef_start: float = 0.03
    entropy_coef_end: float = 0.001
    # Starting bias on the FORWARD logit. An untrained policy then mostly
    # walks, so it covers the warehouse and meets the shared rewards from the
    # first episode. 0 starts from equal probabilities instead.
    forward_bias: float = 2.0
    # Input scales, the same three PPO uses. The task offsets are stored as
    # cell-difference / DIAMETER, so one cell is about 0.05 while column and
    # row are in grid units. See shared.observation.scale_destination_offsets.
    destination_scale: float = 2.0 * DIAMETER
    coordinate_scale: float = 0.1
    requested_scale: float = 10.0
    # Replay: keep the episodes with the most deliveries and learn from them
    # again after every update. A good episode is too valuable to use only once.
    replay_size: int = 32
    replay_updates: int = 6
    replay_episodes: int = 4
    replay_coef: float = 3.0
    total_env_steps: int = TOTAL_ENV_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS

    def __post_init__(self):
        # The hyperparameter search passes hidden as a list.
        object.__setattr__(self, "hidden", tuple(int(width) for width in self.hidden))
        if self.episodes_per_update < 1:
            raise ValueError(f"episodes_per_update must be at least 1, got {self.episodes_per_update}")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.baseline_rate < 0.0:
            raise ValueError("baseline_rate cannot be negative")
        if min(self.entropy_coef_start, self.entropy_coef_end, self.replay_coef) < 0.0:
            raise ValueError("entropy and replay coefficients cannot be negative")
        if min(self.replay_size, self.replay_updates, self.replay_episodes) < 0:
            raise ValueError("replay settings cannot be negative")

    def as_dict(self):
        values = asdict(self)
        values["hidden"] = list(self.hidden)
        return values
