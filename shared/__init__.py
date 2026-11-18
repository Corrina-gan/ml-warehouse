"""Shared training helpers. Algorithm files import these instead of copying them."""

from shared.evaluation import evaluate, format_score
from shared.observation import (
    INPUT_DIM,
    batch_network_input,
    network_input,
    stack_observations,
)
from shared.reward_manager import robot_rewards, team_reward
from shared.seeding import first_episode_seed, set_seed
from shared.training_utils import (
    Episode,
    Rollout,
    Transition,
    collect_steps,
    is_true_terminal,
    iterate_episode,
    run_episode,
)

__all__ = [
    "INPUT_DIM",
    "Episode",
    "Rollout",
    "Transition",
    "batch_network_input",
    "collect_steps",
    "evaluate",
    "first_episode_seed",
    "format_score",
    "is_true_terminal",
    "iterate_episode",
    "network_input",
    "robot_rewards",
    "run_episode",
    "set_seed",
    "stack_observations",
    "team_reward",
]
