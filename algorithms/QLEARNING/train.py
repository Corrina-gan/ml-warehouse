"""Train tabular Q-learning on rware-tiny-2ag-v2 and score it with the shared evaluation.

Run from the project root:

    python -m algorithms.QLEARNING.train

That trains seeds 0, 1, and 2 for 100,000 environment steps each. The greedy
policy is scored every 10,000 steps on the held-out evaluation seeds. The
number to compare with SARSA, REINFORCE, and PPO is team_return_mean.

One episode follows the Q-learning control loop:

1. Reset the warehouse and read the initial observations.
2. Choose each robot's action with epsilon-greedy and execute it.
3. If the step is a true terminal, back up the reward alone and end the episode.
4. Otherwise back up reward + gamma * max over Q(next state, ·).
5. While the episode continues, choose the next action again with epsilon-greedy.
   That action is a new draw. It is not the action inside the max.
6. When the episode ends, apply the epsilon schedule.
7. Stop once the environment-step budget is spent. Otherwise reset.

is_true_terminal is false at the 500-step limit, so step 3 does not run on
this warehouse. The last step still uses step 4's backup. done only stops
the episode. It does not remove the next-state term.
"""

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from algorithms.QLEARNING.config import EPSILON_SCHEDULES, QLearningSettings
from algorithms.QLEARNING.qlearning import QLearningAgent
from algorithms.SARSA.state import STATE_FIELDS
from environment.config import EVAL_EPISODES, N_AGENTS, SEEDS
from environment.environment import check_contract, make_warehouse
from shared.episode_debug import write_episode_debug
from shared.evaluation import evaluate, format_score
from shared.seeding import first_episode_seed, set_seed
from shared.training_utils import is_true_terminal, iterate_episode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qlearning"


def run_q_learning_episode(env, agent, seed, step_limit):
    """Play one episode and update the table after every step.

    Each action is chosen when it is executed. The backup does not keep
    that action for the next step, because the target uses the max over
    the next state rather than the behaviour action.
    """
    if step_limit < 1:
        raise ValueError("an episode needs at least one step")

    length = 0
    team_reward_sum = 0.0
    abs_td = []
    epsilon = agent.epsilon
    transitions = []
    for step_index, transition in enumerate(iterate_episode(env, agent.choose_actions, seed=seed)):
        transitions.append(transition)
        reward = transition.learning_rewards
        team_reward_sum += transition.team_reward
        length += 1
        true_terminal = is_true_terminal(step_index, transition.terminated, transition.truncated)

        if true_terminal:
            errors = agent.backup(transition.observations, transition.actions, reward)
        else:
            errors = agent.backup(
                transition.observations,
                transition.actions,
                reward,
                transition.next_observations,
            )
        abs_td.extend(abs(error) for error in errors)
        if transition.done or length >= step_limit:
            break

    agent.finish_episode()
    mean_abs_td = 0.0 if not abs_td else float(sum(abs_td) / len(abs_td))
    return {
        "length": length,
        "team_reward_sum": float(team_reward_sum),
        "epsilon": float(epsilon),
        "mean_abs_td": mean_abs_td,
        "q_states": len(agent.q),
        "transitions": tuple(transitions),
    }


def train_seed(run_seed, settings, eval_episodes, output_dir, eval_seeds=None, debug=True):
    """Train one run id and write its logs, table, and final score.

    eval_seeds scores this run somewhere other than the official evaluation
    seeds. debug writes the per-episode trace. The hyperparameter search
    turns the trace off.
    """
    set_seed(run_seed)
    agent = QLearningAgent(settings, seed=run_seed)
    env = make_warehouse()
    directory = output_dir / f"seed_{run_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    training_rows = []
    eval_rows = []
    steps = 0
    episode = 0
    seed = first_episode_seed(run_seed)
    last_eval_step = None
    score = None

    try:
        while steps < settings.total_env_steps:
            remaining = settings.total_env_steps - steps
            stats = run_q_learning_episode(env, agent, seed=seed, step_limit=remaining)
            steps += stats["length"]
            episode += 1
            if debug:
                write_episode_debug(
                    directory,
                    episode,
                    stats["transitions"],
                    keep_trace=episode == 1 or steps >= settings.total_env_steps,
                )
            seed += 1
            row = {
                "env_steps": steps,
                "seed": run_seed,
                "episode": episode,
                "team_reward_sum": stats["team_reward_sum"],
                "epsilon": stats["epsilon"],
                "mean_abs_td": stats["mean_abs_td"],
                "q_states": stats["q_states"],
            }
            training_rows.append(row)
            _print_episode(row, settings.total_env_steps)

            if _due(steps, settings.eval_every_steps) and eval_episodes > 0:
                score = _evaluate(agent, eval_episodes, eval_seeds)
                last_eval_step = steps
                eval_rows.append(_eval_row(steps, score))
                print(format_score(score), flush=True)
    finally:
        env.close()

    agent.save(directory / "qlearning.npz")
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


def _due(steps, eval_every):
    return eval_every > 0 and steps % eval_every == 0


def _evaluate(agent, episodes, seeds=None):
    """Score the greedy table without changing the training draws.

    evaluate() reseeds the global RNGs. Action draws for epsilon-greedy
    come from the agent's own generator, which those calls do not touch.
    The global state is restored so anything else in the process stays put.
    seeds replaces the official evaluation seeds when a search is scoring
    a trial.
    """
    state = _capture_rng()
    try:
        chosen = None if seeds is None else tuple(int(seed) for seed in seeds)[:episodes]
        return evaluate(agent.act_greedy, episodes=episodes, seeds=chosen)
    finally:
        _restore_rng(state)


def _capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state()}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _print_episode(row, total_steps):
    print(
        f"episode {row['episode']} steps {row['env_steps']}/{total_steps} "
        f"epsilon {row['epsilon']:.3f} "
        f"td {row['mean_abs_td']:.4f} "
        f"states {row['q_states']} "
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
        "algorithm": "Q-learning",
        "run_seed": run_seed,
        "first_episode_seed": first_episode_seed(run_seed),
        "learning_signal": (
            "each robot's warehouse reward plus its own stage bonus: +0.05 per cell "
            "closer to a requested shelf, +0.20 for picking that shelf up, +0.05 per "
            "cell closer to a goal while carrying it, and -0.05 when a forward is "
            "blocked. A missed load toggle is not charged. Closer pays only for a "
            "new best distance, so circling a shelf does not pay again. A delivery is the warehouse "
            "reward of 1. A successful step that does not get closer is not penalised. "
            "The logged team reward and the evaluation score are the warehouse sum only."
        ),
        "observation": (
            "71 warehouse values plus the offset to the nearest requested shelf. "
            "After that shelf is picked up, the offset switches to the nearest goal. "
            "The goal offset is zero before the pickup."
        ),
        "parameter_sharing": True,
        "representation": (
            "one shared Q-table. Each robot is a compact key: "
            + ", ".join(STATE_FIELDS)
            + ". Greedy evaluation is argmax, with ties broken by the lowest action id."
        ),
        "update": (
            "On a continuing step, Q(s, a) moves toward reward + gamma * max_a' Q(s', a'). "
            "The behaviour policy then draws a fresh epsilon-greedy action for s'. "
            "That draw is not the action inside the max. On a true terminal, the target "
            "is the reward alone. is_true_terminal stays false at the 500-step limit, so "
            "the last step bootstraps and the episode then resets."
        ),
        "exploration": settings.epsilon_schedule,
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
    parser = argparse.ArgumentParser(description="Train Q-learning on the shared warehouse.")
    parser.add_argument("--seed", type=int, default=None, help="One run id from SEEDS. Omit to run every seed.")
    parser.add_argument("--steps", type=int, default=None, help="Environment steps per seed.")
    parser.add_argument("--eval-every", type=int, default=None, help="Score the greedy policy after this many steps.")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--epsilon-schedule", choices=EPSILON_SCHEDULES, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Tabular step size.")
    parser.add_argument("--epsilon", type=float, default=None, help="Starting epsilon.")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Floor used by the decaying schedule.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    changes = {}
    if args.steps is not None:
        changes["total_env_steps"] = args.steps
    if args.eval_every is not None:
        changes["eval_every_steps"] = args.eval_every
    if args.epsilon_schedule is not None:
        changes["epsilon_schedule"] = args.epsilon_schedule
    if args.alpha is not None:
        changes["alpha"] = args.alpha
    if args.epsilon is not None:
        changes["epsilon_start"] = args.epsilon
    if args.epsilon_min is not None:
        changes["epsilon_min"] = args.epsilon_min
    settings = replace(QLearningSettings(), **changes)
    if args.eval_episodes < 0:
        raise ValueError("eval episodes cannot be negative")

    run_seeds = SEEDS if args.seed is None else (args.seed,)
    check_contract()
    args.output.mkdir(parents=True, exist_ok=True)
    scores = []
    for run_seed in run_seeds:
        print(
            f"Q-learning run seed {run_seed}, first episode seed {first_episode_seed(run_seed)}, "
            f"schedule {settings.epsilon_schedule}"
        )
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
