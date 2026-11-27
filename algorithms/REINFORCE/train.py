"""Train REINFORCE on rware-tiny-2ag-v2 and score it with the shared evaluation.

Run from the project root:

    python -m algorithms.REINFORCE.train

That trains seeds 0, 1, and 2 for 100,000 environment steps each, which is 200
episodes of 500 steps. The greedy policy is scored every 10,000 steps on the
held-out evaluation seeds. The number to compare with SARSA and PPO is
team_return_mean. The files in results/reinforce/seed_N have the same names
and columns as the PPO ones.

When all three seeds are trained it also writes two checks:

    results/reinforce/ensemble/        the three policies voting together,
                                       same 10 evaluation episodes
    results/reinforce/unseen_maps.csv  200 more unseen episodes for each seed,
                                       for the ensemble, and for the warehouse
                                       rules played with no learning at all

The no-learning rows matter. The rules in shared/warehouse_rules.py deliver
shelves on their own when a robot simply walks forward, so a trained policy
only counts as learning by how far it beats those rows.

    python -m algorithms.REINFORCE.train --summary-only

rebuilds those two checks from the saved weights without training again.
"""

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from algorithms.REINFORCE.config import ReinforceSettings
from algorithms.REINFORCE.reinforce import PolicyEnsemble, ReinforceAgent, load_agent
from environment.config import EVAL_EPISODES, EVAL_SEED_START, MAX_STEPS, N_AGENTS, SEEDS
from environment.environment import check_contract, make_warehouse
from shared.evaluation import evaluate, format_score
from shared.reward_manager import check_goals
from shared.warehouse_rules import RulesOnly, check_floor_plan
from shared.seeding import first_episode_seed, set_seed
from shared.training_utils import run_episode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "reinforce"

# Extra test episodes for unseen_maps.csv. They start one million seeds after
# EVAL_SEEDS, so no training run, EDA episode, or evaluation episode uses them.
UNSEEN_SEED_START = EVAL_SEED_START + 1_000_000
UNSEEN_EPISODES = 200


def train_seed(run_seed, settings, eval_episodes, output_dir):
    """Train one run id and write its logs, weights, and final score."""
    set_seed(run_seed)
    agent = ReinforceAgent(settings)
    env = make_warehouse()
    directory = output_dir / f"seed_{run_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    training_rows = []
    eval_rows = []
    steps = 0
    seed = first_episode_seed(run_seed)
    next_eval = settings.eval_every_steps
    last_eval_step = None
    score = None

    try:
        while steps < settings.total_env_steps:
            # REINFORCE learns from whole episodes, so collect complete ones.
            episodes = []
            for _ in range(settings.episodes_per_update):
                if steps >= settings.total_env_steps:
                    break
                episode = run_episode(
                    env,
                    agent.choose_actions,
                    seed=seed,
                    on_episode_start=agent.reset_episode,
                )
                episodes.append(episode)
                steps += episode.length
                seed += 1

            metrics = agent.update(episodes, progress=steps / settings.total_env_steps)
            row = {"env_steps": steps, "seed": run_seed, **metrics}
            training_rows.append(row)
            _print_update(row, settings.total_env_steps)

            if settings.eval_every_steps > 0 and steps >= next_eval and eval_episodes > 0:
                score = _evaluate(agent, eval_episodes)
                last_eval_step = steps
                eval_rows.append(_eval_row(steps, score))
                print(format_score(score), flush=True)
                while next_eval <= steps:
                    next_eval += settings.eval_every_steps
    finally:
        env.close()

    agent.save(directory / "reinforce.pt")
    if eval_episodes > 0 and last_eval_step != steps:
        score = _evaluate(agent, eval_episodes)
        eval_rows.append(_eval_row(steps, score))
        print(format_score(score), flush=True)

    _write_csv(directory / "training_log.csv", training_rows)
    _write_csv(directory / "eval_log.csv", eval_rows)
    _write_protocol(directory / "protocol.json", run_seed, settings, eval_episodes, score)
    if score is not None:
        _write_final_score(directory, score)
    return score


def summarise(output_dir, eval_episodes, unseen_episodes):
    """Score every saved seed, the ensemble, and the rules-only floor.

    Reads results/reinforce/seed_N/reinforce.pt. Nothing here trains.
    """
    paths = [output_dir / f"seed_{run_seed}" / "reinforce.pt" for run_seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"train every seed before the summary, missing: {missing}")

    # A separate agent per use: the ensemble shares the action that is played,
    # so its members must not be the ones scored on their own.
    scores = [(run_seed, _score(load_agent(path), eval_episodes)) for run_seed, path in zip(SEEDS, paths)]
    _write_comparison(output_dir / "comparison.csv", scores)
    mean_return = float(np.mean([score["team_return_mean"] for _seed, score in scores]))
    print(f"mean team return across {len(scores)} seed(s): {mean_return:.2f}")

    ensemble = PolicyEnsemble([load_agent(path) for path in paths])
    ensemble_score = _score(ensemble, eval_episodes)
    (output_dir / "ensemble").mkdir(parents=True, exist_ok=True)
    _write_final_score(output_dir / "ensemble", ensemble_score)
    print(f"ensemble of {len(paths)} seeds: {format_score(ensemble_score)}")

    if unseen_episodes < 1:
        return
    unseen_seeds = tuple(range(UNSEEN_SEED_START, UNSEEN_SEED_START + unseen_episodes))
    walker = RulesOnly(forward=True)
    wanderer = RulesOnly(forward=False)
    players = [
        ("rules only and walking forward (no learning)", walker.choose_actions, walker.reset_episode),
        ("rules only and random moves (no learning)", wanderer.choose_actions, wanderer.reset_episode),
    ]
    for run_seed, path in zip(SEEDS, paths):
        agent = load_agent(path)
        players.append((f"REINFORCE seed {run_seed}", agent.act_greedy, agent.reset_episode))
    players.append((f"REINFORCE ensemble of {len(paths)} seeds", ensemble.act_greedy, ensemble.reset_episode))

    rows = []
    for label, choose_actions, reset in players:
        score = evaluate(choose_actions, seeds=unseen_seeds, on_episode_start=reset)
        low, high = _bootstrap_interval(score["episode_team_returns"])
        rows.append(
            {
                "policy": label,
                "episodes": score["episodes"],
                "team_return_mean": score["team_return_mean"],
                "ci95_low": low,
                "ci95_high": high,
                "success_rate": score["success_rate"],
                "blocked_forward_rate": score["blocked_forward_rate"],
            }
        )
        print(f"{label}: {score['team_return_mean']:.2f} [{low:.2f}, {high:.2f}] over {score['episodes']} unseen episodes")
    _write_csv(output_dir / "unseen_maps.csv", rows)


def _score(player, episodes):
    return evaluate(player.act_greedy, episodes=episodes, on_episode_start=player.reset_episode)


def _bootstrap_interval(values, resamples=2000):
    """95% interval for the mean: resample the episodes with replacement."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _evaluate(agent, episodes):
    """Score the greedy policy without changing the training random state.

    evaluate() seeds the action RNGs for each test episode. Training samples
    the next episode from the run seed's generator, so that generator is
    restored afterwards.
    """
    state = _capture_rng()
    try:
        return _score(agent, episodes)
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
            "Each robot learns from the shared training reward, training_rewards() in "
            "shared/reward_manager.py: the warehouse delivery reward plus bonuses for "
            "picking up a requested shelf, moving it toward the goal, and putting it "
            "back. This differs from LEARNING_SIGNAL in environment.config. Evaluation "
            "still counts real deliveries only."
        ),
        "parameter_sharing": True,
        "policy": (
            "categorical policy over 5 actions, shared by both robots, hidden widths HIDDEN "
            "with a GRU as the second layer, greedy argmax at evaluation"
        ),
        "baseline": "separate value network for V(s) of the training return, fitted to Monte Carlo returns",
        "bootstrap": "none. Returns are summed to the end of the 500-step episode.",
        "action_rules": (
            "shared/warehouse_rules.py removes actions before the policy chooses, and the "
            "policy starts with a bias toward FORWARD (settings.forward_bias). "
            "results/reinforce/unseen_maps.csv reports those rules with no learning as the floor."
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
    parser.add_argument("--unseen-episodes", type=int, default=UNSEEN_EPISODES, help="Episodes in unseen_maps.csv. 0 skips it.")
    parser.add_argument("--summary-only", action="store_true", help="Skip training and rebuild the summary from saved weights.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    changes = {}
    if args.steps is not None:
        changes["total_env_steps"] = args.steps
    if args.eval_every is not None:
        changes["eval_every_steps"] = args.eval_every
    settings = replace(ReinforceSettings(), **changes)
    if settings.total_env_steps < MAX_STEPS:
        raise ValueError(f"REINFORCE needs at least one whole episode, {MAX_STEPS} steps")
    if args.eval_episodes < 0 or args.unseen_episodes < 0:
        raise ValueError("episode counts cannot be negative")

    # The networks are small, so one thread is the fastest, and it makes a
    # run with the same seed repeat exactly.
    torch.set_num_threads(1)
    check_contract()
    check_floor_plan()
    check_goals()
    args.output.mkdir(parents=True, exist_ok=True)

    run_seeds = SEEDS if args.seed is None else (args.seed,)
    scores = []
    if not args.summary_only:
        for run_seed in run_seeds:
            print(f"REINFORCE run seed {run_seed}, first episode seed {first_episode_seed(run_seed)}")
            score = train_seed(run_seed, settings, args.eval_episodes, args.output)
            if score is not None:
                scores.append((run_seed, score))

    # The summary needs all three seeds, so a single --seed run stops here.
    if (args.summary_only or args.seed is None) and args.eval_episodes > 0:
        summarise(args.output, args.eval_episodes, args.unseen_episodes)
    return scores


if __name__ == "__main__":
    main()
