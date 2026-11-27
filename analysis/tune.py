"""Search each algorithm on three training seeds, then score the winner.

The official 100,000-step runs stay where they are. This module does not
replace them. For each algorithm it:

1. Uses that algorithm's own 10 configurations. Trial 0 is the default.
2. Trains every configuration on run seeds 0, 1, and 2.
3. Records the mean and sample standard deviation of the validation scores.
4. Selects the highest mean. A tie prefers the smaller spread, then the
   earlier trial.
5. Scores the three saved policies on five held-out episode seeds.
   Those seeds are not the validation seeds, and they are not used to choose
   the configuration. Run seeds stay inside SEEDS, so this step does not
   train on seeds 3, 4, or 5.

Run from the project root:

    python -m analysis.tune
    python -m analysis.tune --algorithm sarsa --stage search
    python -m analysis.tune --algorithm ppo --stage final
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

from algorithms.PPO.config import PPOSettings
from algorithms.PPO.ppo import load_agent as load_ppo
from algorithms.PPO.train import train_seed as train_ppo
from algorithms.QLEARNING.config import QLearningSettings
from algorithms.QLEARNING.qlearning import QLearningAgent
from algorithms.QLEARNING.train import train_seed as train_q_learning
from algorithms.REINFORCE.config import REINFORCESettings
from algorithms.REINFORCE.reinforce import load_agent as load_reinforce
from algorithms.REINFORCE.train import train_seed as train_reinforce
from algorithms.SARSA.config import SARSASettings
from algorithms.SARSA.sarsa import SARSAAgent
from algorithms.SARSA.train import train_seed as train_sarsa
from environment.config import EVAL_EPISODES, EVAL_EVERY_STEPS, EVAL_SEEDS, SEEDS, TOTAL_ENV_STEPS
from environment.environment import check_contract
from shared.evaluation import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tuning"
FINAL_RUNS = PROJECT_ROOT / "results" / "final_runs"

# Away from training episode seeds and from EVAL_SEEDS.
VALIDATION_SEED_START = 2_000_000
VALIDATION_SEEDS = tuple(range(VALIDATION_SEED_START, VALIDATION_SEED_START + 10))
# Five official episode seeds. Selection uses VALIDATION_SEEDS, not these.
FINAL_EVAL_SEEDS = EVAL_SEEDS[:5]


def _load_qlearning(path):
    agent = QLearningAgent()
    agent.load(path)
    return agent


def _load_sarsa(path):
    agent = SARSAAgent()
    agent.load(path)
    return agent

# Trial 0 is the current default, so a tie can keep it. Each later trial
# changes one setting. Ten trials, and the same keys on every row.
SEARCH = {
    "sarsa": {
        "name": "SARSA",
        "settings": SARSASettings,
        "train": train_sarsa,
        "checkpoint": "sarsa.npz",
        "load": _load_sarsa,
        "trials": [
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.05, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.2, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.3, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.01, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.1, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "glie"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 0.2, "epsilon_min": 0.05, "epsilon_schedule": "constant"},
            {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.995, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
        ],
    },
    "qlearning": {
        "name": "Q-learning",
        "settings": QLearningSettings,
        "train": train_q_learning,
        "checkpoint": "qlearning.npz",
        "load": _load_qlearning,
        "trials": [
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.05, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.2, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.3, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.01, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.1, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "glie"},
            {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 0.2, "epsilon_min": 0.05, "epsilon_schedule": "constant"},
            {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
            {"alpha": 0.1, "gamma": 0.995, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_schedule": "decaying"},
        ],
    },
    "reinforce": {
        "name": "REINFORCE",
        "settings": REINFORCESettings,
        "train": train_reinforce,
        "checkpoint": "reinforce.pt",
        "load": load_reinforce,
        "trials": [
            {"learning_rate": 3e-4, "gamma": 0.99, "entropy_coef_end": 0.001, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 1e-4, "gamma": 0.99, "entropy_coef_end": 0.001, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 1e-3, "gamma": 0.99, "entropy_coef_end": 0.001, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.95, "entropy_coef_end": 0.001, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.995, "entropy_coef_end": 0.001, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.99, "entropy_coef_end": 0.0005, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.99, "entropy_coef_end": 0.005, "hidden": [128, 128], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.99, "entropy_coef_end": 0.001, "hidden": [64, 64], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.99, "entropy_coef_end": 0.001, "hidden": [256, 256], "baseline_rate": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.99, "entropy_coef_end": 0.001, "hidden": [128, 128], "baseline_rate": 0.0},
        ],
    },
    "ppo": {
        "name": "PPO",
        "settings": PPOSettings,
        "train": train_ppo,
        "checkpoint": "ppo.pt",
        "load": load_ppo,
        "trials": [
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 1e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 1e-3, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 250, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 500, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 125, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 250, "epochs": 8, "clip_eps": 0.2, "entropy_coef_end": 0.001},
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.1, "entropy_coef_end": 0.001},
            {"learning_rate": 3e-4, "gamma": 0.99, "rollout_steps": 500, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.005},
            {"learning_rate": 3e-4, "gamma": 0.95, "rollout_steps": 500, "minibatch_size": 250, "epochs": 4, "clip_eps": 0.2, "entropy_coef_end": 0.001},
        ],
    },
}


def trials_for(algorithm):
    """The algorithm's own trials. Trial 0 is its current default."""
    spec = SEARCH[algorithm]
    if "trials" in spec:
        return [dict(trial) for trial in spec["trials"]]
    grid = spec["grid"]
    keys = list(grid)
    trials = []
    for values in product(*(grid[key] for key in keys)):
        trials.append({key: value for key, value in zip(keys, values)})
    return trials


def run_search(algorithm, steps, run_seed, eval_episodes, output_dir, max_trials=None):
    """Train every configuration on seeds 0, 1, and 2, then choose the winner."""
    del run_seed
    _check_search(algorithm, steps, SEEDS[0], eval_episodes)
    spec = SEARCH[algorithm]
    root = Path(output_dir) / algorithm
    root.mkdir(parents=True, exist_ok=True)
    planned = trials_for(algorithm)
    if max_trials is not None:
        planned = planned[: max(0, int(max_trials))]
    validation = VALIDATION_SEEDS[:eval_episodes]
    finished = []
    for index, tuned in enumerate(planned):
        record = _one_trial(spec, index, tuned, steps, validation, root)
        finished.append(record)
        _write_trials(root / "trials.csv", finished)
        print(
            f"{spec['name']} trial {index}: mean {record['team_return_mean']:.2f} "
            f"std {record['team_return_std']:.2f} across seeds {list(SEEDS)}",
            flush=True,
        )
    best = _select(finished)
    payload = _selection_payload(spec["name"], best, steps, validation)
    (root / "best.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"{spec['name']} selected trial {payload['trial']} {payload['tuned']} "
        f"mean {best['team_return_mean']:.2f} std {best['team_return_std']:.2f}",
        flush=True,
    )
    return payload


def run_final(algorithm, steps, output_dir, eval_episodes=len(FINAL_EVAL_SEEDS)):
    """Score the selected policies on five episode seeds the search did not use."""
    del steps
    if algorithm not in SEARCH:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if eval_episodes != len(FINAL_EVAL_SEEDS):
        eval_episodes = len(FINAL_EVAL_SEEDS)
    root = Path(output_dir) / algorithm
    best_path = root / "best.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"{best_path} is missing. Run the search before the final evaluation.")
    selected = json.loads(best_path.read_text(encoding="utf-8"))
    spec = SEARCH[algorithm]
    trial = int(selected["trial"])
    trial_dir = root / f"trial_{trial:02d}"
    held_out = FINAL_EVAL_SEEDS[:eval_episodes]
    overlap = set(held_out) & set(selected.get("validation_seeds") or [])
    if overlap:
        raise RuntimeError(f"final seeds overlap the validation seeds: {sorted(overlap)}")
    scores = []
    for run_seed in SEEDS:
        path = trial_dir / f"seed_{run_seed}" / spec["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(f"missing trained policy: {path}")
        print(
            f"{spec['name']} final evaluation, training seed {run_seed}, "
            f"episode seeds {held_out[0]} to {held_out[-1]}",
            flush=True,
        )
        agent = spec["load"](path)
        score = evaluate(agent.act_greedy, episodes=len(held_out), seeds=held_out)
        scores.append({"seed": int(run_seed), **_public_score(score)})
    mean_return, spread = _mean_std([item["team_return_mean"] for item in scores])
    report = {
        "algorithm": spec["name"],
        "stage": "final",
        "steps": int(selected["steps"]),
        "trial": trial,
        "tuned": selected["tuned"],
        "training_seeds": list(SEEDS),
        "evaluation_seeds": list(held_out),
        "per_seed": scores,
        "team_return_mean": mean_return,
        "team_return_std": spread,
    }
    final_dir = root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (root / "final.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_comparison(final_dir / "comparison.csv", scores)
    print(
        f"{spec['name']} final team return {mean_return:.2f} ± {spread:.2f} "
        f"across {len(scores)} training seeds",
        flush=True,
    )
    return report


def run_retrain(algorithm, search_dir, steps=TOTAL_ENV_STEPS, eval_episodes=EVAL_EPISODES):
    """Retrain the selected configuration on seeds 0, 1, and 2.

    The policies are written to results/final_runs. This does not replace
    results/official. The score uses the official evaluation episode seeds,
    which the search did not use to choose the configuration.
    """
    if algorithm not in SEARCH:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if steps < 1:
        raise ValueError("final training needs at least one environment step")
    spec = SEARCH[algorithm]
    best_path = Path(search_dir) / algorithm / "best.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"{best_path} is missing. Select a configuration before retraining.")
    selected = json.loads(best_path.read_text(encoding="utf-8"))
    tuned = _settings_ready(selected["tuned"])
    out = FINAL_RUNS / algorithm
    out.mkdir(parents=True, exist_ok=True)
    eval_every = EVAL_EVERY_STEPS if steps % EVAL_EVERY_STEPS == 0 else 0
    settings = replace(
        spec["settings"](),
        total_env_steps=int(steps),
        eval_every_steps=eval_every,
        **tuned,
    )
    scores = []
    for run_seed in SEEDS:
        if _retrain_done(out / f"seed_{run_seed}", spec["checkpoint"], steps, tuned):
            print(f"{spec['name']} retrain seed {run_seed} already finished", flush=True)
            score = _score_saved(spec, out / f"seed_{run_seed}" / spec["checkpoint"], eval_episodes)
        else:
            print(f"{spec['name']} retrain seed {run_seed}, {steps} steps", flush=True)
            score = spec["train"](run_seed, settings, eval_episodes, out, eval_seeds=None, debug=False)
        if score is None:
            raise RuntimeError(f"{spec['name']} seed {run_seed} did not return a score")
        scores.append({"seed": int(run_seed), **_public_score(score)})
    mean_return, spread = _mean_std([item["team_return_mean"] for item in scores])
    report = {
        "algorithm": spec["name"],
        "stage": "retrained",
        "steps": int(steps),
        "trial": int(selected["trial"]),
        "tuned": selected["tuned"],
        "training_seeds": list(SEEDS),
        "evaluation_seeds": list(scores[0]["seeds"]) if scores else list(EVAL_SEEDS[:eval_episodes]),
        "per_seed": scores,
        "team_return_mean": mean_return,
        "team_return_std": spread,
        "source": str(best_path),
    }
    (out / "retrain.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_comparison(out / "comparison.csv", scores)
    print(
        f"{spec['name']} retrained team return {mean_return:.2f} ± {spread:.2f} "
        f"in {out}",
        flush=True,
    )
    return report


def _settings_ready(tuned):
    ready = _json_ready(tuned)
    if "hidden" in ready:
        ready["hidden"] = tuple(int(width) for width in ready["hidden"])
    return ready


def _retrain_done(directory, checkpoint, steps, tuned):
    protocol_path = directory / "protocol.json"
    if not protocol_path.is_file() or not (directory / checkpoint).is_file():
        return False
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    saved = protocol.get("settings") or {}
    if int(saved.get("total_env_steps", -1)) != int(steps):
        return False
    for key, value in tuned.items():
        current = saved.get(key)
        if isinstance(value, tuple):
            if tuple(current or []) != value:
                return False
        elif isinstance(value, float):
            if current is None or abs(float(current) - value) > 1e-12:
                return False
        elif current != value:
            return False
    return True


def _score_saved(spec, path, episodes):
    agent = spec["load"](path)
    return evaluate(agent.act_greedy, episodes=episodes, seeds=EVAL_SEEDS[:episodes])


def load_tuning(output_dir=DEFAULT_OUTPUT):
    """Read whatever the search and the final training have saved."""
    root = Path(output_dir)
    algorithms = []
    for key, spec in SEARCH.items():
        directory = root / key
        algorithms.append(
            {
                "key": key,
                "name": spec["name"],
                "trials": _read_trials(directory / "trials.csv"),
                "best": _read_json(directory / "best.json"),
                "final": _read_json(directory / "final.json"),
            }
        )
    return algorithms


def _check_search(algorithm, steps, run_seed, eval_episodes):
    if algorithm not in SEARCH:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if steps < 1:
        raise ValueError("a trial needs at least one environment step")
    if int(run_seed) not in SEEDS:
        raise ValueError(f"search seed must be one of {SEEDS}, got {run_seed}")
    if eval_episodes < 1:
        raise ValueError("validation needs at least one episode")
    if eval_episodes > len(VALIDATION_SEEDS):
        raise ValueError(f"validation has {len(VALIDATION_SEEDS)} seeds")
    overlap = set(VALIDATION_SEEDS) & set(EVAL_SEEDS)
    if overlap:
        raise RuntimeError(f"validation seeds overlap the official evaluation seeds: {sorted(overlap)}")


def _one_trial(spec, index, tuned, steps, validation, root):
    """Train one configuration on seeds 0, 1, and 2. A finished seed is not repeated."""
    directory = root / f"trial_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trial.json"
    ready = _json_ready(tuned)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    same = (
        saved is not None
        and saved.get("tuned") == ready
        and int(saved.get("steps", -1)) == int(steps)
    )
    if same and saved.get("complete") is True:
        print(f"{spec['name']} trial {index} already finished", flush=True)
        return saved
    kept = []
    if same:
        for row in saved.get("per_seed") or []:
            seed = int(row["seed"])
            checkpoint = directory / f"seed_{seed}" / spec["checkpoint"]
            if seed in SEEDS and checkpoint.is_file():
                kept.append(row)
    done = {int(row["seed"]) for row in kept}
    settings = replace(spec["settings"](), total_env_steps=int(steps), eval_every_steps=0, **tuned)
    print(f"{spec['name']} trial {index}: {tuned}", flush=True)
    for run_seed in SEEDS:
        if run_seed in done:
            print(f"{spec['name']} trial {index} seed {run_seed} already finished", flush=True)
            continue
        print(f"{spec['name']} trial {index} seed {run_seed}, {steps} steps", flush=True)
        score = spec["train"](
            run_seed,
            settings,
            len(validation),
            directory,
            eval_seeds=validation,
            debug=False,
        )
        if score is None:
            raise RuntimeError(f"{spec['name']} trial {index} seed {run_seed} did not return a score")
        reward, deliveries = _training_totals(directory / f"seed_{run_seed}" / "training_log.csv")
        kept.append(
            {
                "seed": int(run_seed),
                "training_reward_sum": reward,
                "training_deliveries": deliveries,
                **_public_score(score),
            }
        )
        kept.sort(key=lambda row: int(row["seed"]))
        path.write_text(
            json.dumps(_trial_record(spec, index, steps, ready, validation, kept, complete=False), indent=2) + "\n",
            encoding="utf-8",
        )
    record = _trial_record(spec, index, steps, ready, validation, kept, complete=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _trial_record(spec, index, steps, tuned, validation, per_seed, complete):
    returns = [float(row["team_return_mean"]) for row in per_seed]
    mean_return, spread = _mean_std(returns)
    return {
        "trial": index,
        "algorithm": spec["name"],
        "steps": int(steps),
        "run_seeds": [int(row["seed"]) for row in per_seed],
        "validation_seeds": list(validation),
        "tuned": tuned,
        "per_seed": per_seed,
        "team_return_mean": mean_return,
        "team_return_std": spread,
        "success_rate": _mean_or_none(row.get("success_rate") for row in per_seed),
        "deliveries_mean": _mean_or_none(row.get("deliveries_mean") for row in per_seed),
        "blocked_forward_rate": _mean_or_none(row.get("blocked_forward_rate") for row in per_seed),
        "training_reward_sum": float(sum(row["training_reward_sum"] for row in per_seed)),
        "training_deliveries": int(sum(row["training_deliveries"] for row in per_seed)),
        "complete": bool(complete) and [int(row["seed"]) for row in per_seed] == list(SEEDS),
    }


def _mean_std(values):
    """Mean and sample standard deviation. One value has spread 0."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("a mean needs at least one score")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance ** 0.5


def _mean_or_none(values):
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _public_score(score):
    return {
        "team_return_mean": float(score["team_return_mean"]),
        "team_return_std": float(score["team_return_std"]),
        "success_rate": None if score["success_rate"] is None else float(score["success_rate"]),
        "deliveries_mean": None if score["deliveries_mean"] is None else float(score["deliveries_mean"]),
        "blocked_forward_rate": float(score["blocked_forward_rate"]),
        "seeds": [int(seed) for seed in score["seeds"]],
    }


def _training_totals(path):
    reward = 0.0
    deliveries = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["team_reward_sum"])
            reward += value
            if value > 0.0:
                deliveries += 1
    return reward, deliveries


def _select(finished):
    """Highest mean validation return, then the smaller spread, then the earlier trial."""
    if not finished:
        raise ValueError("the search finished no trials")
    return min(
        finished,
        key=lambda record: (
            -float(record["team_return_mean"]),
            float(record["team_return_std"]),
            int(record["trial"]),
        ),
    )


def _selection_payload(name, best, steps, validation):
    return {
        "algorithm": name,
        "stage": "selected",
        "steps": int(steps),
        "run_seeds": list(SEEDS),
        "validation_seeds": list(validation),
        "final_evaluation_seeds": list(FINAL_EVAL_SEEDS),
        "tuned": best["tuned"],
        "trial": int(best["trial"]),
        "per_seed": best["per_seed"],
        "team_return_mean": float(best["team_return_mean"]),
        "team_return_std": float(best["team_return_std"]),
        "success_rate": best["success_rate"],
        "deliveries_mean": best["deliveries_mean"],
        "blocked_forward_rate": float(best["blocked_forward_rate"]),
        "training_reward_sum": float(best["training_reward_sum"]),
        "training_deliveries": int(best["training_deliveries"]),
        "selection": (
            "Highest mean greedy team return across training seeds 0, 1, and 2 "
            "on the validation episode seeds. A tie prefers the smaller sample "
            "standard deviation, then the earlier trial. Trial 0 is the protocol "
            "defaults. The five final evaluation seeds are not used here."
        ),
    }


def _write_trials(path, records):
    keys = []
    fixed = (
        "trial",
        "team_return_mean",
        "team_return_std",
        "seed_0",
        "seed_1",
        "seed_2",
        "success_rate",
        "deliveries_mean",
        "blocked_forward_rate",
        "training_reward_sum",
        "training_deliveries",
    )
    for record in records:
        flat = _flat_trial(record)
        for key in fixed:
            if key in flat and key not in keys:
                keys.append(key)
        for key in record["tuned"]:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            flat = _flat_trial(record)
            row = {key: flat.get(key, "") for key in keys}
            row.update(record["tuned"])
            writer.writerow(row)


def _flat_trial(record):
    flat = dict(record)
    for row in record.get("per_seed") or []:
        flat[f"seed_{int(row['seed'])}"] = row["team_return_mean"]
    return flat


def _write_comparison(path, scores):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seed", "team_return_mean", "team_return_std", "success_rate", "deliveries_mean", "blocked_forward_rate"]
        )
        for score in scores:
            writer.writerow(
                [
                    score["seed"],
                    score["team_return_mean"],
                    score["team_return_std"],
                    score["success_rate"],
                    score["deliveries_mean"],
                    score["blocked_forward_rate"],
                ]
            )


def _json_ready(tuned):
    ready = {}
    for key, value in tuned.items():
        if isinstance(value, float):
            ready[key] = float(value)
        elif isinstance(value, int):
            ready[key] = int(value)
        elif isinstance(value, (list, tuple)):
            ready[key] = [int(item) if isinstance(item, int) else item for item in value]
        else:
            ready[key] = value
    return ready


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trials(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune Q-learning, SARSA, REINFORCE, and PPO on the shared warehouse.")
    parser.add_argument("--algorithm", choices=("qlearning", "sarsa", "reinforce", "ppo", "all"), default="all")
    parser.add_argument("--stage", choices=("search", "final", "retrain", "all"), default="all")
    parser.add_argument("--steps", type=int, default=20_000, help="Environment steps for each search trial.")
    parser.add_argument(
        "--final-steps",
        type=int,
        default=None,
        help="Environment steps for the retraining. Defaults to the search budget.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Kept for older commands. The search always trains seeds 0, 1, and 2.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Validation episodes for each training seed. The final score uses five other episode seeds.")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    names = list(SEARCH) if args.algorithm == "all" else [args.algorithm]
    final_steps = args.steps if args.final_steps is None else args.final_steps
    if args.stage in ("search", "all"):
        check_contract()
    for name in names:
        if args.stage in ("search", "all"):
            run_search(name, args.steps, args.seed, args.eval_episodes, args.output, args.max_trials)
        if args.stage in ("final", "all"):
            check_contract()
            run_final(name, final_steps, args.output)
        if args.stage == "retrain":
            check_contract()
            run_retrain(name, args.output, steps=TOTAL_ENV_STEPS if args.final_steps is None else args.final_steps)
    return load_tuning(args.output)


if __name__ == "__main__":
    main()
