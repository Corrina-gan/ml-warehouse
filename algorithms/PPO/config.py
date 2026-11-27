"""PPO settings for the shared warehouse.

Gamma, the hidden-layer widths, the step budget, and the run ids come from
environment.config. Those stay the same for Q-learning, SARSA, REINFORCE,
and PPO. The values below belong to PPO only.
"""

from dataclasses import asdict, dataclass

from environment.config import (
    EVAL_EVERY_STEPS,
    GAMMA,
    HIDDEN,
    MAX_STEPS,
    TOTAL_ENV_STEPS,
)
from shared.task_features import DIAMETER


@dataclass(frozen=True)
class PPOSettings:
    """One PPO run. Defaults match the shared comparison protocol."""

    gamma: float = GAMMA
    hidden: tuple = HIDDEN
    # One rollout is one 500-step episode when collection starts on a reset.
    # collect_steps keeps going if a rollout begins in the middle of an episode.
    rollout_steps: int = MAX_STEPS
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    # 500 steps * 2 robots = 1000 samples, so this divides the full rollout.
    minibatch_size: int = 250
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    # High at the start, then lower. A flat 0.02 collapsed the distribution
    # while a missed toggle was still penalised, and that penalty removed
    # load from every state. After that penalty was removed, a schedule that
    # ended at 0.01 left the entropy near 1.3: the greedy action was whichever
    # logit was slightly ahead, usually a turn or a wait, in every state.
    # The run starts at 0.02 and falls to 0.001 so the policy can commit to
    # the action that matches the observation.
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.001
    # The four destination offsets are stored as cell-difference / DIAMETER,
    # so one cell is about 0.05, while column and row in the same vector are
    # in grid units. Multiplying those four numbers by 2 * DIAMETER makes one
    # cell of error the value 2. Column and row are multiplied by
    # coordinate_scale, and the requested-shelf bit on the current cell by
    # requested_scale. The carrying flag stays 0 or 1. SARSA does not use
    # these scales. A checkpoint that does not name a scale was trained at 1.
    destination_scale: float = 2.0 * DIAMETER
    coordinate_scale: float = 0.1
    requested_scale: float = 10.0
    # Drop forward from the distribution when the warehouse would cancel it:
    # the map edge, the other robot's current cell, or a standing shelf
    # while this robot is carrying one. A checkpoint trained without the
    # field plays every action, including a forward into a wall.
    mask_blocked_forward: bool = True
    # Multiplies the mean squared error. The critic term in the loss is 0.5 * MSE.
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_env_steps: int = TOTAL_ENV_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS

    def as_dict(self):
        values = asdict(self)
        values["hidden"] = list(self.hidden)
        return values
