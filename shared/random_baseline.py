"""Score a uniform random policy on the held-out evaluation seeds.

This is the comparison floor for Q-learning, SARSA, REINFORCE, and PPO. It does not
train, and it does not use the EDA seeds 0..99.

Run from the project root:

    python -m shared.random_baseline
"""

import csv
import json
from pathlib import Path

import numpy as np

from environment.config import (
    ENV_ID,
    EVAL_SEEDS,
    N_ACTIONS,
    N_AGENTS,
    SEEDS,
    TRAIN_SEED_STRIDE,
)
from environment.environment import check_contract
from shared.evaluation import evaluate, format_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline"


def random_actions(observations):
    """One independent uniform action per robot. The episode seed drives this."""
    del observations
    return tuple(int(np.random.randint(N_ACTIONS)) for _ in range(N_AGENTS))


def save_baseline(score):
    """Write the summary, one row per episode, and the split used for training."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "random_policy_summary.csv"
    episodes_path = OUTPUT_DIR / "random_policy_episodes.csv"
    protocol_path = OUTPUT_DIR / "protocol.json"

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

    protocol = {
        "env_id": ENV_ID,
        "policy": "uniform random, one action drawn independently per robot",
        "state_representation": (
            "71 raw float32 observation values plus a 2-value agent-ID one-hot. "
            "No imputation and no scaling. Coordinates stay in grid units."
        ),
        "learning_signal": "sum of the two robot rewards",
        "eda_seeds": "0..99, exploration of the environment only, not the test set",
        "training_run_seeds": list(SEEDS),
        "training_episode_seed_starts": [seed * TRAIN_SEED_STRIDE for seed in SEEDS],
        "evaluation_seeds": list(score["seeds"]),
        "compared_metric": "team_return_mean",
        "also_reported": ["success_rate", "deliveries_mean", "blocked_forward_rate"],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return summary_path, episodes_path, protocol_path


def main():
    check_contract()
    score = evaluate(random_actions)
    paths = save_baseline(score)
    print(format_score(score))
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
