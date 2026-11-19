"""PPO settings for the shared warehouse.

Gamma, the hidden-layer widths, the step budget, and the run ids come from
environment.config. Those stay the same for SARSA, REINFORCE, and PPO.
The values below belong to PPO only.
"""

from dataclasses import asdict, dataclass

from environment.config import (
    EVAL_EVERY_STEPS,
    GAMMA,
    HIDDEN,
    MAX_STEPS,
    TOTAL_ENV_STEPS,
)


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
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_env_steps: int = TOTAL_ENV_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS

    def as_dict(self):
        values = asdict(self)
        values["hidden"] = list(self.hidden)
        return values
