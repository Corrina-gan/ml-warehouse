"""REINFORCE settings for the shared warehouse.

Gamma, the hidden-layer widths, the step budget, and the run ids come from
environment.config. Those stay the same for SARSA, REINFORCE, and PPO.
The values below belong to REINFORCE only.
"""

from dataclasses import asdict, dataclass

from environment.config import (
    EVAL_EVERY_STEPS,
    GAMMA,
    HIDDEN,
    TOTAL_ENV_STEPS,
)


@dataclass(frozen=True)
class ReinforceSettings:
    """One REINFORCE run. Defaults match the shared comparison protocol."""

    gamma: float = GAMMA
    hidden: tuple = HIDDEN
    # Starting bias on the FORWARD logit. Before any training the greedy
    # policy then walks forward whenever the rules allow it and turns at
    # walls. That walk already delivers shelves, so training starts from a
    # policy that sees rewards. 0 starts from equal probabilities instead.
    forward_bias: float = 2.0
    # REINFORCE needs finished episodes. One update uses this many of them,
    # so 100,000 steps give 100 updates.
    episodes_per_update: int = 2
    # The learning rate falls along half a cosine to a tenth of this value.
    learning_rate: float = 2e-3
    baseline_learning_rate: float = 3e-3
    # Passes over the batch when fitting the baseline to the returns.
    baseline_epochs: int = 4
    baseline_minibatch_size: int = 500
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    # Entropy bonus at the start. It falls in a straight line to zero, so the
    # policy explores early and commits late. Evaluation is greedy.
    entropy_coef: float = 0.03
    # Replay: keep the episodes with the most deliveries and learn from them
    # again after every update. A good episode is too valuable to use only once.
    replay_size: int = 32
    replay_updates: int = 6
    replay_episodes: int = 4
    replay_coef: float = 3.0
    total_env_steps: int = TOTAL_ENV_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS

    def __post_init__(self):
        if self.episodes_per_update < 1:
            raise ValueError(f"episodes_per_update must be at least 1, got {self.episodes_per_update}")
        if self.learning_rate <= 0.0 or self.baseline_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.entropy_coef < 0.0 or self.replay_coef < 0.0:
            raise ValueError("entropy_coef and replay_coef cannot be negative")
        if min(self.replay_size, self.replay_updates, self.replay_episodes) < 0:
            raise ValueError("replay settings cannot be negative")

    def as_dict(self):
        values = asdict(self)
        values["hidden"] = list(self.hidden)
        return values
