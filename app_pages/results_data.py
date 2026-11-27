"""Official 100,000-step scores for Q-learning, SARSA, REINFORCE, and PPO.

The charts read saved files only. A run counts when it uses the shared
warehouse, the shared step budget, and the shared evaluation seeds.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from environment.config import (
    ENV_ID,
    EVAL_SEEDS,
    GAMMA,
    HIDDEN,
    SEEDS,
    TOTAL_ENV_STEPS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

# Initial 100,000-step runs with the protocol defaults. A retrained policy in
# results/final_runs replaces that algorithm's bar after hyperparameter search.
ALGORITHMS = (
    {"name": "Q-learning", "folder": "official/qlearning", "final": "final_runs/qlearning"},
    {"name": "SARSA", "folder": "official/sarsa", "final": "final_runs/sarsa"},
    {"name": "REINFORCE", "folder": "official/reinforce", "final": "final_runs/reinforce"},
    {"name": "PPO", "folder": "official/ppo", "final": "final_runs/ppo"},
)
TIME_KEYS = ("elapsed_seconds", "train_seconds", "duration_seconds")
MOVING_AVERAGE = 10


def build_report():
    """Scores, episode returns, and training curves for the results page.

    A finished retrain in results/final_runs replaces that algorithm's
    initial official run. Until then the page keeps the default-setting run.
    """
    algorithms = []
    for item in ALGORITHMS:
        retrained = _load_algorithm(item["name"], item["final"])
        if retrained["status"] == "official":
            algorithms.append(retrained)
        else:
            algorithms.append(_load_algorithm(item["name"], item["folder"]))
    return {
        "algorithms": algorithms,
        "random": _load_random(),
        "long_ppo": _load_long_ppo(),
    }


def judgments(report):
    """Names for the final evaluation, computed from the official runs."""
    official = [item for item in report["algorithms"] if item["status"] == "official"]
    missing = [item["name"] for item in report["algorithms"] if item["status"] == "missing"]
    excluded = [item for item in report["algorithms"] if item["status"] == "excluded"]
    if not official:
        return {
            "best_name": "No completed run",
            "best_detail": "None of the algorithms has a saved 100,000-step result.",
            "consistent_name": "No completed run",
            "consistent_detail": "Consistency needs the evaluation episodes from a finished run.",
            "fastest_name": "Not recorded",
            "fastest_detail": "Training time is not compared until a finished run records it.",
            "conclusion": _conclusion(report, official, missing, excluded, None, None, None, False),
        }

    best_value = max(item["mean_return"] for item in official)
    best = [item["name"] for item in official if item["mean_return"] == best_value]
    consistent_value = min(item["return_std"] for item in official)
    consistent = [item["name"] for item in official if item["return_std"] == consistent_value]
    flat_zero = best_value == 0 and consistent_value == 0
    timed = [item for item in official if item["train_seconds"] is not None]
    fastest = None
    fastest_value = None
    if timed:
        fastest_value = min(item["train_seconds"] for item in timed)
        fastest = [item["name"] for item in timed if item["train_seconds"] == fastest_value]

    return {
        "best_name": _join(best),
        "best_detail": _best_detail(report, best_value),
        "consistent_name": _join(consistent),
        "consistent_detail": _consistent_detail(consistent_value, flat_zero),
        "fastest_name": "Not recorded" if fastest is None else _join(fastest),
        "fastest_detail": _fastest_detail(official, timed, fastest_value),
        "conclusion": _conclusion(
            report,
            official,
            missing,
            excluded,
            best,
            best_value,
            consistent_value,
            flat_zero,
        ),
    }


def _load_algorithm(name, folder):
    root = RESULTS / folder
    if not root.is_dir() or not (root / "comparison.csv").is_file():
        return {"name": name, "folder": folder, "status": "missing", "reason": ""}

    problems = []
    episodes = []
    training = []
    times = []
    seed_rows = []
    for seed in SEEDS:
        directory = root / f"seed_{seed}"
        protocol_path = directory / "protocol.json"
        summary_path = directory / "final_summary.csv"
        episode_path = directory / "final_episodes.csv"
        if not protocol_path.is_file() or not summary_path.is_file() or not episode_path.is_file():
            problems.append(f"seed {seed} is missing its final score")
            continue
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        problems.extend(_protocol_problems(protocol, seed))
        summary = _read_summary(summary_path)
        if summary.get("env_id") != ENV_ID:
            problems.append(f"seed {seed} scored {summary.get('env_id', 'an unknown environment')}")
        played = pd.read_csv(episode_path)
        if list(played["seed"].astype(int)) != list(EVAL_SEEDS):
            problems.append(f"seed {seed} used different evaluation seeds")
        played = played.copy()
        played["algorithm"] = name
        played["run_seed"] = seed
        episodes.append(played)
        seed_rows.append(
            {
                "Algorithm": name,
                "Run seed": seed,
                "Mean team return": float(summary["team_return_mean"]),
                "Success rate": float(summary["success_rate"]),
            }
        )
        recorded = _recorded_seconds(protocol)
        if recorded is not None:
            times.append(recorded)
        log_path = directory / "training_log.csv"
        if log_path.is_file():
            log = pd.read_csv(log_path)
            log["algorithm"] = name
            log["run_seed"] = seed
            training.append(log)

    if problems or len(episodes) != len(SEEDS):
        reason = "; ".join(problems) if problems else "a run seed is missing"
        return {"name": name, "folder": folder, "status": "excluded", "reason": reason}

    played = pd.concat(episodes, ignore_index=True)
    returns = played["team_return"].astype(float)
    successes = played["deliveries"].astype(float) > 0
    curve = pd.concat(training, ignore_index=True) if training else None
    return {
        "name": name,
        "folder": folder,
        "status": "official",
        "reason": "",
        "steps": TOTAL_ENV_STEPS,
        "mean_return": float(returns.mean()),
        "success_rate": float(successes.mean()),
        "return_std": float(returns.std(ddof=1)),
        "episode_count": int(len(played)),
        "train_seconds": None if not times else float(np.mean(times)),
        "timed_seeds": len(times),
        "episodes": played,
        "seed_rows": seed_rows,
        "training": curve,
        "nonzero_rollouts": _nonzero_rollouts(curve),
        "rollout_count": 0 if curve is None else int(len(curve)),
    }


def _protocol_problems(protocol, seed):
    settings = protocol.get("settings") or {}
    steps = settings.get("total_env_steps", protocol.get("total_env_steps"))
    gamma = settings.get("gamma", protocol.get("gamma"))
    hidden = settings.get("hidden", protocol.get("hidden"))
    problems = []
    if steps != TOTAL_ENV_STEPS:
        shown = "an unrecorded step count" if steps is None else f"{int(steps):,} environment steps"
        problems.append(f"seed {seed} used {shown}")
    if gamma is not None and float(gamma) != float(GAMMA):
        problems.append(f"seed {seed} used discount {gamma}")
    if hidden is not None and tuple(hidden) != tuple(HIDDEN):
        problems.append(f"seed {seed} used hidden layers {tuple(hidden)}")
    return problems


def _recorded_seconds(protocol):
    settings = protocol.get("settings") or {}
    for key in TIME_KEYS:
        if key in protocol:
            return float(protocol[key])
        if key in settings:
            return float(settings[key])
    return None


def _nonzero_rollouts(curve):
    if curve is None or "team_reward_sum" not in curve.columns:
        return None
    counts = {}
    for seed, group in curve.groupby("run_seed"):
        counts[int(seed)] = int((group["team_reward_sum"].astype(float) > 0).sum())
    return counts


def _load_random():
    episode_path = RESULTS / "baseline" / "random_policy_episodes.csv"
    if not episode_path.is_file():
        return None
    played = pd.read_csv(episode_path)
    if list(played["seed"].astype(int)) != list(EVAL_SEEDS):
        return None
    returns = played["team_return"].astype(float)
    successes = played["deliveries"].astype(float) > 0
    return {
        "name": "Random",
        "mean_return": float(returns.mean()),
        "success_rate": float(successes.mean()),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "episode_count": int(len(played)),
        "episodes": played,
    }


def _load_long_ppo():
    root = RESULTS / "ppo_long"
    comparison = root / "comparison.csv"
    if not comparison.is_file():
        return None
    steps = set()
    for seed in SEEDS:
        protocol_path = root / f"seed_{seed}" / "protocol.json"
        if not protocol_path.is_file():
            return None
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        settings = protocol.get("settings") or {}
        steps.add(settings.get("total_env_steps"))
    if len(steps) != 1:
        return None
    step_count = steps.pop()
    if step_count in (None, TOTAL_ENV_STEPS):
        return None
    scores = pd.read_csv(comparison)
    return {
        "steps": int(step_count),
        "mean_return": float(scores["team_return_mean"].astype(float).mean()),
    }


def _read_summary(path):
    frame = pd.read_csv(path)
    values = {}
    for row in frame.itertuples(index=False):
        text = str(row.value)
        try:
            values[str(row.metric)] = float(text)
        except ValueError:
            values[str(row.metric)] = text
    values["env_id"] = _raw_metric(frame, "env_id")
    return values


def _raw_metric(frame, metric):
    match = frame.loc[frame["metric"] == metric, "value"]
    if match.empty:
        return None
    return str(match.iloc[0])


def _best_detail(report, best_value):
    detail = f"Mean team return {best_value:.2f} on the shared evaluation seeds."
    random_floor = report["random"]
    if random_floor is not None and random_floor["mean_return"] == best_value:
        detail += " The random policy has the same mean."
    return detail


def _consistent_detail(consistent_value, flat_zero):
    if flat_zero:
        return "Every evaluation episode returned 0, so the standard deviation is 0."
    return f"Sample standard deviation of episode team return is {consistent_value:.2f}."


def _conclusion(report, official, missing, excluded, best, best_value, consistent_value, flat_zero):
    sentences = []
    if missing:
        noun = _join(missing)
        verb = "has" if len(missing) == 1 else "have"
        sentences.append(
            f"{noun} {verb} no saved run on the 100,000-step protocol, so the ranking leaves "
            f"{'it' if len(missing) == 1 else 'them'} out."
        )
    for item in excluded:
        sentences.append(f"{item['name']} was left out of the charts: {item['reason']}.")
    if official:
        sentences.append(
            f"{_join(item['name'] for item in official)} finished seeds 0, 1, and 2 on {ENV_ID} "
            f"for {TOTAL_ENV_STEPS:,} environment steps and was scored greedily on evaluation "
            f"seeds {EVAL_SEEDS[0]} to {EVAL_SEEDS[-1]}."
        )
        sentences.append(
            f"The highest mean team return is {best_value:.2f} ({_join(best)}). "
            f"The smallest spread of episode returns is {consistent_value:.2f}."
        )
        if flat_zero:
            sentences.append(
                "That spread is 0 because every saved evaluation episode returned 0."
            )
    random_floor = report["random"]
    if random_floor is not None and official:
        sentences.append(
            f"The uniform random policy scores {random_floor['mean_return']:.2f} "
            f"on those same {random_floor['episode_count']} episodes."
        )
    long_run = report["long_ppo"]
    if long_run is not None:
        sentences.append(
            f"A separate PPO run of {long_run['steps']:,} steps scores "
            f"{long_run['mean_return']:.2f}. That larger budget stays out of these charts."
        )
    if official and any(item["train_seconds"] is None for item in official):
        sentences.append(
            "The result files do not record wall-clock training time, so the fastest algorithm "
            "is not named."
        )
    return " ".join(sentences)


def _fastest_detail(official, timed, fastest_value):
    if not timed:
        return "The official result files do not record wall-clock training time."
    shown = f"Mean training time {fastest_value / 60:.1f} minutes."
    missing_time = [item["name"] for item in official if item["train_seconds"] is None]
    if missing_time:
        verb = "has" if len(missing_time) == 1 else "have"
        return shown + f" {_join(missing_time)} {verb} no recorded time."
    return shown


def _join(names):
    names = list(names)
    if not names:
        return "None"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def best_saved_policy(report):
    """The saved algorithm with the highest evaluation mean team return.

    This reads the same official runs as the results page. A tie keeps the
    higher success rate, then the smaller spread of episode returns, then
    the listed algorithm order. Returns None when no completed run is saved.
    """
    official = [item for item in report["algorithms"] if item["status"] == "official"]
    if not official:
        return None
    order = {item["name"]: index for index, item in enumerate(ALGORITHMS)}
    ranked = sorted(
        official,
        key=lambda item: (
            -float(item["mean_return"]),
            -float(item["success_rate"]),
            float(item["return_std"]),
            order.get(item["name"], len(order)),
        ),
    )
    winner = ranked[0]
    best_return = float(winner["mean_return"])
    tied = [item["name"] for item in official if float(item["mean_return"]) == best_return]
    return {
        "name": winner["name"],
        "mean_return": best_return,
        "success_rate": float(winner["success_rate"]),
        "return_std": float(winner["return_std"]),
        "tied": tied,
    }
