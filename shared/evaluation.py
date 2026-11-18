"""Score a policy on a fresh warehouse. Every algorithm should report this result."""

import random

import numpy as np

from environment.config import (
    ENV_ID,
    EVAL_DETERMINISTIC,
    EVAL_EPISODES,
    EVAL_SEEDS,
    FORWARD,
    GAMMA,
    N_AGENTS,
)
from environment.environment import make_warehouse
from shared.reward_manager import reward_is_delivery_unit
from shared.training_utils import run_episode


def _seed_text(seeds):
    if list(seeds) == list(range(seeds[0], seeds[0] + len(seeds))):
        if len(seeds) == 1:
            return str(seeds[0])
        return f"{seeds[0]}..{seeds[-1]}"
    return ", ".join(str(seed) for seed in seeds)


def seed_action_rng(seed):
    """Seed the RNGs a policy might sample actions from.

    env.reset(seed) does not seed these. numpy, the random module, and
    torch each get the episode seed when that library is installed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def evaluate(choose_actions, episodes=EVAL_EPISODES, seeds=None, on_episode_start=None):
    """Score a policy on the shared warehouse.

    Uses EVAL_SEEDS unless seeds is passed. Those seeds are reserved, so
    do not train on them. The env is a new warehouse, so this does not reset an
    environment used for training. The compared number is team_return_mean.
    success_rate is the share of episodes with at least one delivery.
    blocked_forward_rate is the share of forward actions that stay in place.

    EVAL_DETERMINISTIC is true, so choose_actions must be greedy: argmax
    of Q, or argmax of the action logits. After reset, each episode seeds
    the action RNGs, which keeps a sampled action repeatable.
    """
    if seeds is None:
        seeds = EVAL_SEEDS[:episodes]
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("evaluate needs at least one seed")

    env = make_warehouse()
    try:
        played = []
        for seed in seeds:

            def start(observations, episode_seed=seed, follow=on_episode_start):
                seed_action_rng(episode_seed)
                if follow is not None:
                    follow(observations)

            played.append(
                run_episode(
                    env,
                    choose_actions,
                    seed=seed,
                    on_episode_start=start,
                )
            )
    finally:
        env.close()

    team_returns = [episode.team_return for episode in played]
    lengths = [episode.length for episode in played]
    robot_returns = [episode.robot_returns for episode in played]
    measurable = all(
        reward_is_delivery_unit(step.rewards)
        for episode in played
        for step in episode.transitions
    )
    deliveries = [episode.deliveries for episode in played] if measurable else None
    success_rate = None
    if deliveries is not None:
        success_rate = float(np.mean([count > 0 for count in deliveries]))
    return {
        "env_id": ENV_ID,
        "gamma": GAMMA,
        "deterministic": EVAL_DETERMINISTIC,
        "episodes": len(played),
        "seeds": seeds,
        "team_return_mean": float(np.mean(team_returns)),
        "team_return_std": float(np.std(team_returns, ddof=1)) if len(team_returns) > 1 else 0.0,
        "length_mean": float(np.mean(lengths)),
        "deliveries_mean": None if deliveries is None else float(np.mean(deliveries)),
        "success_rate": success_rate,
        "blocked_forward_rate": _blocked_forward_rate(played),
        "robot_return_mean": tuple(
            float(np.mean([returns[index] for returns in robot_returns]))
            for index in range(N_AGENTS)
        ),
        "episode_team_returns": tuple(float(value) for value in team_returns),
        "episode_lengths": tuple(int(value) for value in lengths),
        "episode_deliveries": None if deliveries is None else tuple(int(value) for value in deliveries),
        "episode_robot_returns": tuple(
            tuple(float(value) for value in returns) for returns in robot_returns
        ),
    }


def _blocked_forward_rate(played):
    """Share of forward actions that left that robot in the same cell.

    A blocked forward pays 0, the same as a step that does not deliver, so
    the reward alone cannot show a wasted move. This rate is the navigation
    figure every algorithm reports from the same evaluation episodes.
    """
    attempts = 0
    blocked = 0
    for episode in played:
        for step in episode.transitions:
            for robot, action in enumerate(step.actions):
                if action != FORWARD:
                    continue
                attempts += 1
                before = step.observations[robot]
                after = step.next_observations[robot]
                if before[0] == after[0] and before[1] == after[1]:
                    blocked += 1
    if attempts == 0:
        return 0.0
    return float(blocked / attempts)


def format_score(score):
    """One-line report. Every algorithm should print this from evaluate()."""
    robots = ", ".join(f"{value:.2f}" for value in score["robot_return_mean"])
    deliveries = score["deliveries_mean"]
    delivery_text = "n/a" if deliveries is None else f"{deliveries:.2f}"
    success = score["success_rate"]
    success_text = "n/a" if success is None else f"{success:.0%}"
    return (
        f"{score['env_id']}: team return {score['team_return_mean']:.2f} "
        f"± {score['team_return_std']:.2f} "
        f"over {score['episodes']} episodes "
        f"(seeds {_seed_text(score['seeds'])}, "
        f"success rate {success_text}, "
        f"mean deliveries {delivery_text}, "
        f"blocked forwards {score['blocked_forward_rate']:.1%}, "
        f"robot returns {robots})"
    )
