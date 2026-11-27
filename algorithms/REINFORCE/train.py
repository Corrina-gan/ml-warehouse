"""Train REINFORCE on rware-tiny-2ag-v2 and score it with the shared evaluation.

Run from the project root:

    python -m algorithms.REINFORCE.train

That trains seeds 0, 1, and 2 for 100,000 environment steps each, which is 200
episodes of 500 steps. The greedy policy is scored every 10,000 steps on the
held-out evaluation seeds. The number to compare with Q-learning, SARSA, and
PPO is team_return_mean.

The files go to results/official/reinforce. That is where the app pages, the
model selection, and the ensemble read the default 100,000-step run of every
algorithm. They have the same names and columns as the PPO files.
"""

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from algorithms.REINFORCE.config import REINFORCESettings
from algorithms.REINFORCE.reinforce import REINFORCEAgent
from environment.config import EVAL_EPISODES, MAX_STEPS, N_AGENTS, SEEDS
from environment.environment import check_contract, make_warehouse
from shared.episode_debug import write_episode_debug
from shared.evaluation import evaluate, format_score
from shared.seeding import first_episode_seed, set_seed
from shared.task_features import DIAMETER
from shared.training_utils import Episode, iterate_episode, run_episode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "official" / "reinforce"


def train_seed(run_seed, settings, eval_episodes, output_dir, eval_seeds=None, debug=True):
    """Train one run id and write its logs, weights, and final score.

    eval_seeds scores this run somewhere other than the official evaluation
    seeds. debug writes the per-episode trace. The hyperparameter search
    turns the trace off.
    """
    set_seed(run_seed)
    agent = REINFORCEAgent(settings)
    env = make_warehouse()
    directory = Path(output_dir) / f"seed_{run_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    training_rows = []
    eval_rows = []
    steps = 0
    episode_index = 0
    seed = first_episode_seed(run_seed)
    next_eval = settings.eval_every_steps
    last_eval_step = None
    score = None

    try:
        while steps < settings.total_env_steps:
            # REINFORCE learns from whole episodes, so collect complete ones.
            steps_before = steps
            episodes = []
            while len(episodes) < settings.episodes_per_update and steps < settings.total_env_steps:
                episode = _play_episode(env, agent, seed, settings.total_env_steps - steps)
                episodes.append(episode)
                steps += episode.length
                seed += 1
                episode_index += 1
                if debug:
                    write_episode_debug(
                        directory,
                        episode_index,
                        episode.transitions,
                        keep_trace=episode_index == 1 or steps >= settings.total_env_steps,
                    )

            metrics = agent.update(episodes, env_steps=steps_before)
            # One row per episode, like the SARSA and PPO logs, so
            # team_reward_sum covers the same 500 steps in every chart.
            reached = steps_before
            for episode in episodes:
                reached += episode.length
                row = {"env_steps": reached, "seed": run_seed, "team_reward_sum": episode.team_return, **metrics}
                training_rows.append(row)
                _print_update(row, settings.total_env_steps)

            if settings.eval_every_steps > 0 and steps >= next_eval and eval_episodes > 0:
                score = _evaluate(agent, eval_episodes, eval_seeds)
                last_eval_step = steps
                eval_rows.append(_eval_row(steps, score))
                print(format_score(score), flush=True)
                while next_eval <= steps:
                    next_eval += settings.eval_every_steps
    finally:
        env.close()

    agent.save(directory / "reinforce.pt")
    if eval_episodes > 0 and last_eval_step != steps:
        score = _evaluate(agent, eval_episodes, eval_seeds)
        eval_rows.append(_eval_row(steps, score))
        print(format_score(score), flush=True)

    _write_csv(directory / "training_log.csv", training_rows)
    _write_csv(directory / "eval_log.csv", eval_rows)
    _write_protocol(directory / "protocol.json", run_seed, settings, eval_episodes, score)
    if score is not None:
        _write_final_score(directory, score)
    return score


def _play_episode(env, agent, seed, step_limit):
    """One sampled episode. It is cut short only when the step budget ends inside it."""
    if step_limit >= MAX_STEPS:
        return run_episode(env, agent.choose_actions, seed=seed, on_episode_start=agent.reset_episode)
    transitions = []
    for transition in iterate_episode(env, agent.choose_actions, seed=seed, on_episode_start=agent.reset_episode):
        transitions.append(transition)
        if len(transitions) >= step_limit:
            break
    return Episode(transitions=tuple(transitions), seed=seed)


def _evaluate(agent, episodes, seeds=None):
    """Score the greedy policy without changing the training random state.

    evaluate() seeds the action RNGs for each test episode. Training samples
    the next episode from the run seed's generator, so that generator is
    restored afterwards. seeds replaces the official evaluation seeds when
    a search is scoring a trial.
    """
    state = _capture_rng()
    try:
        chosen = None if seeds is None else tuple(int(seed) for seed in seeds)[:episodes]
        return evaluate(agent.act_greedy, episodes=episodes, seeds=chosen, on_episode_start=agent.reset_episode)
    finally:
        _restore_rng(state)


def _capture_rng():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [generator.clone() for generator in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _print_update(row, total_steps):
    print(
        f"steps {row['env_steps']}/{total_steps} "
        f"policy {row['policy_loss']:.4f} "
        f"baseline {row['baseline_loss']:.4f} "
        f"entropy {row['entropy']:.3f} "
        f"coef {row['entropy_coef']:.3f} "
        f"replay {row['replay_episodes']} "
        f"team reward {row['team_reward_sum']:.0f}",
        flush=True,
    )


def _eval_row(steps, score):
    return {
        "env_steps": steps,
        "team_return_mean": score["team_return_mean"],
        "team_return_std": score["team_return_std"],
        "success_rate": score["success_rate"],
        "deliveries_mean": score["deliveries_mean"],
        "blocked_forward_rate": score["blocked_forward_rate"],
        "length_mean": score["length_mean"],
        "robot_return_mean": " ".join(f"{value:.6f}" for value in score["robot_return_mean"]),
    }


def _write_final_score(directory, score):
    summary_path = directory / "final_summary.csv"
    episodes_path = directory / "final_episodes.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in (
            "env_id",
            "episodes",
            "team_return_mean",
            "team_return_std",
            "success_rate",
            "deliveries_mean",
            "blocked_forward_rate",
            "length_mean",
            "robot_return_mean",
        ):
            value = score[key]
            if isinstance(value, tuple):
                value = " ".join(f"{item:.6f}" for item in value)
            writer.writerow([key, value])

    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["episode", "seed", "team_return", "deliveries", "length"]
        header.extend(f"robot_{index}_return" for index in range(1, N_AGENTS + 1))
        writer.writerow(header)
        deliveries = score["episode_deliveries"]
        for index, seed in enumerate(score["seeds"]):
            row = [
                index,
                seed,
                score["episode_team_returns"][index],
                "" if deliveries is None else deliveries[index],
                score["episode_lengths"][index],
            ]
            row.extend(score["episode_robot_returns"][index])
            writer.writerow(row)


def _write_protocol(path, run_seed, settings, eval_episodes, score):
    protocol = {
        "algorithm": "REINFORCE",
        "run_seed": run_seed,
        "first_episode_seed": first_episode_seed(run_seed),
        "learning_signal": (
            "each robot's warehouse reward plus its own stage bonus from "
            "shared.learning_reward: +0.05 per cell closer to a requested shelf, "
            "+0.20 for picking that shelf up, +0.05 per cell closer to a goal while "
            "carrying it, +0.05 per cell closer to a free storage cell while carrying "
            "a shelf nobody is asking for, +0.20 for putting that shelf down there, "
            "-0.20 for picking up a shelf nobody asked for, and -0.05 when a forward "
            "is blocked. Closer pays only for a new best distance. A delivery is the "
            "warehouse reward of 1. The logged team reward and the evaluation score "
            "are the warehouse sum only."
        ),
        "observation": (
            "71 warehouse values plus the shared task offsets: to the nearest "
            "requested shelf, to the nearest goal once that shelf is carried, and to "
            "the nearest free storage cell while the robot carries a shelf nobody is "
            "asking for. REINFORCE multiplies the four offset numbers by "
            f"{settings.destination_scale:g} before the network, so one cell of error "
            f"has size {settings.destination_scale / DIAMETER:g}. Column and row are "
            f"multiplied by {settings.coordinate_scale:g}, and the requested-shelf "
            f"bit on the robot's own cell by {settings.requested_scale:g}. "
            "PPO uses the same three scales."
        ),
        "parameter_sharing": True,
        "policy": (
            "categorical policy over 5 actions, shared by both robots. The second "
            "hidden layer is a GRU, so the policy carries a memory through the episode. "
            f"The FORWARD logit starts at {settings.forward_bias:g}. "
            f"The entropy coefficient falls linearly from {settings.entropy_coef_start:g} "
            f"to {settings.entropy_coef_end:g} across the step budget. "
            "Greedy evaluation is argmax. Forward is removed from the distribution "
            "when it would hit the map edge, the other robot, or a standing shelf "
            "while this robot is carrying one."
        ),
        "baseline": (
            "switched off in this run, so each action is weighted by its plain return"
            if settings.baseline_rate <= 0.0
            else (
                "separate value network, same hidden widths as the policy, fitted to the "
                "Monte Carlo return. It only reduces variance."
            )
        ),
        "update": (
            f"One update per {settings.episodes_per_update} finished episode(s). Each action "
            "is weighted by its discounted return minus the baseline. The return is summed "
            "to the end of the episode and nothing is bootstrapped. The episodes with the "
            "most deliveries are replayed after each update."
        ),
        "settings": settings.as_dict(),
        "eval_episodes": eval_episodes,
        "final_team_return_mean": None if score is None else score["team_return_mean"],
    }
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison(path, scores):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "team_return_mean",
                "team_return_std",
                "success_rate",
                "deliveries_mean",
                "blocked_forward_rate",
            ]
        )
        for seed, score in scores:
            writer.writerow(
                [
                    seed,
                    score["team_return_mean"],
                    score["team_return_std"],
                    score["success_rate"],
                    score["deliveries_mean"],
                    score["blocked_forward_rate"],
                ]
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train REINFORCE on the shared warehouse.")
    parser.add_argument("--seed", type=int, default=None, help="One run id from SEEDS. Omit to run every seed.")
    parser.add_argument("--steps", type=int, default=None, help="Environment steps per seed.")
    parser.add_argument("--eval-every", type=int, default=None, help="Score the greedy policy after this many steps.")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    changes = {}
    if args.steps is not None:
        changes["total_env_steps"] = args.steps
    if args.eval_every is not None:
        changes["eval_every_steps"] = args.eval_every
    settings = replace(REINFORCESettings(), **changes)
    if settings.total_env_steps < 1:
        raise ValueError("REINFORCE needs at least one environment step")
    if args.eval_episodes < 0:
        raise ValueError("eval episodes cannot be negative")

    # One thread. The networks are too small for more threads to be faster,
    # and a run then does not depend on how many cores the machine has.
    torch.set_num_threads(1)
    run_seeds = SEEDS if args.seed is None else (args.seed,)
    check_contract()
    args.output.mkdir(parents=True, exist_ok=True)
    scores = []
    for run_seed in run_seeds:
        print(f"REINFORCE run seed {run_seed}, first episode seed {first_episode_seed(run_seed)}")
        score = train_seed(run_seed, settings, args.eval_episodes, args.output)
        if score is not None:
            scores.append((run_seed, score))
    if scores:
        _write_comparison(args.output / "comparison.csv", scores)
        mean_return = float(np.mean([score["team_return_mean"] for _seed, score in scores]))
        print(f"mean team return across {len(scores)} seed(s): {mean_return:.2f}")
    return scores


if __name__ == "__main__":
    main()
