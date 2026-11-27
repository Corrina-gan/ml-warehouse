"""Read the saved runs and shortlist which learners are worth refining.

Part A asks for several learning methods, one comparison, an account of
what drives the scores, and a shortlist. This module reads saved
100,000-step logs for Q-learning, SARSA, REINFORCE, and PPO. It does not
train them. A learner with no log stays in the approach table and stays
out of the score ranking.

The official score is the mean team return of the greedy policy on
evaluation seeds 10000000 to 10000009. When that score separates the
learners, the shortlist follows it. A tie at zero falls back to training
deliveries, which only show that the behaviour policy reached the reward.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from environment.config import N_ACTIONS, SEEDS, TOTAL_ENV_STEPS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

# Uniform categorical entropy. A learned policy near this value has not
# separated the five actions, so greedy argmax is a thin preference.
UNIFORM_ENTROPY = math.log(N_ACTIONS)

ALGORITHMS = (
    {
        "name": "Q-learning",
        "folder": "official/qlearning",
        "final": "final_runs/qlearning",
        "family": "Off-policy temporal difference",
        "stores": "One shared Q-table on a compact state",
        "exploration": "Epsilon-greedy, decaying from 1.0 to 0.05",
        "readable": True,
    },
    {
        "name": "SARSA",
        "folder": "official/sarsa",
        "final": "final_runs/sarsa",
        "family": "On-policy temporal difference",
        "stores": "One shared Q-table on a compact state",
        "exploration": "Epsilon-greedy, decaying from 1.0 to 0.05",
        "readable": True,
    },
    {
        "name": "REINFORCE",
        "folder": "official/reinforce",
        "final": "final_runs/reinforce",
        "family": "Episodic policy gradient",
        "stores": "One shared categorical policy, no critic",
        "exploration": "Sampled actions plus an entropy bonus",
        "readable": False,
    },
    {
        "name": "PPO",
        "folder": "official/ppo",
        "final": "final_runs/ppo",
        "family": "Actor-critic",
        "stores": "Shared actor and a value function",
        "exploration": "Sampled actions, clipped update, entropy bonus",
        "readable": False,
    },
)


def build_selection(results=RESULTS):
    """Tables and remarks for the saved official runs.

    Returns plain data and sentences computed from the files. The sentences
    change when a new log is saved, so the page does not keep a handwritten
    score that can drift from the CSVs.
    """
    per_seed = _per_seed(results)
    if per_seed.empty:
        return {
            "approaches": _approaches(),
            "per_seed": per_seed,
            "summary": pd.DataFrame(),
            "errors": pd.DataFrame(),
            "shortlist": [],
            "remarks": ["No official training log was found under results/."],
        }

    summary = _summarise(per_seed)
    errors = _errors(per_seed, summary)
    names, remarks = _shortlist(summary)
    return {
        "approaches": _approaches(),
        "per_seed": per_seed,
        "summary": summary,
        "errors": errors,
        "shortlist": names,
        "remarks": remarks,
    }


def _approaches():
    return pd.DataFrame(
        [
            {
                "Algorithm": item["name"],
                "Learning approach": item["family"],
                "What is stored": item["stores"],
                "How it explores": item["exploration"],
            }
            for item in ALGORITHMS
        ]
    )


def _per_seed(results):
    rows = []
    for item in ALGORITHMS:
        for seed in SEEDS:
            final_dir = results / item["final"] / f"seed_{seed}"
            directory = final_dir if (final_dir / "training_log.csv").is_file() else results / item["folder"] / f"seed_{seed}"
            log_path = directory / "training_log.csv"
            eval_path = directory / "eval_log.csv"
            if not log_path.is_file() or not eval_path.is_file():
                continue
            log = pd.read_csv(log_path)
            evaluated = pd.read_csv(eval_path)
            reward = log["team_reward_sum"].astype(float)
            final = evaluated.iloc[-1]
            entropy = _tail_mean(log, "entropy")
            rows.append(
                {
                    "Algorithm": item["name"],
                    "Run seed": int(seed),
                    "Readable action values": item["readable"],
                    "Training deliveries": int((reward > 0).sum()),
                    "Training reward sum": float(reward.sum()),
                    "Greedy team return": float(final["team_return_mean"]),
                    "Blocked forward rate": float(final["blocked_forward_rate"]),
                    "Eval checkpoints": int(len(evaluated)),
                    "Checkpoints with a delivery": int(
                        (evaluated["team_return_mean"].astype(float) > 0).sum()
                    ),
                    "Final entropy": entropy,
                    "Final epsilon": _last(log, "epsilon"),
                }
            )
    return pd.DataFrame(rows)


def _tail_mean(log, column, count=5):
    if column not in log.columns or log.empty:
        return None
    values = log[column].astype(float).tail(count)
    return float(values.mean())


def _last(log, column):
    if column not in log.columns or log.empty:
        return None
    return float(log[column].iloc[-1])


def _summarise(per_seed):
    rows = []
    for name, group in per_seed.groupby("Algorithm", sort=False):
        entropy = group["Final entropy"].dropna()
        rows.append(
            {
                "Algorithm": name,
                "Greedy team return": float(group["Greedy team return"].mean()),
                "Training deliveries": int(group["Training deliveries"].sum()),
                "Seeds with a training delivery": int((group["Training deliveries"] > 0).sum()),
                "Run seeds": int(len(group)),
                "Mean blocked forward rate": float(group["Blocked forward rate"].mean()),
                "Blocked forward spread": float(group["Blocked forward rate"].max() - group["Blocked forward rate"].min()),
                "Final entropy": None if entropy.empty else float(entropy.mean()),
                "Readable action values": bool(group["Readable action values"].iloc[0]),
                "Checkpoints with a delivery": int(group["Checkpoints with a delivery"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _errors(per_seed, summary):
    """Name the failure that the files actually show."""
    rows = []
    greedy = summary["Greedy team return"].astype(float)
    if len(greedy) and float(greedy.max()) == 0.0:
        rows.append(
            {
                "Error": "No held-out delivery",
                "Where it shows": "Final greedy evaluation of every run seed",
                "What the files show": (
                    "Mean team return is 0.00 and no evaluation checkpoint "
                    "recorded a delivery."
                ),
                "What to change next": (
                    "The behaviour policy, not the scorekeeper. A new metric "
                    "would not create a delivery the greedy policy missed."
                ),
            }
        )

    found = summary.loc[summary["Training deliveries"] > 0, "Algorithm"].tolist()
    if found and float(greedy.max()) == 0.0:
        rows.append(
            {
                "Error": "Training discoveries are not repeated",
                "Where it shows": ", ".join(found),
                "What the files show": (
                    "Exploration collected a team reward during training. "
                    "The greedy policy scored on the held-out seeds did not."
                ),
                "What to change next": (
                    "Keep more of the action that collected the reward when "
                    "exploration is turned off, or give that action a state "
                    "the table can match again."
                ),
            }
        )

    stuck = per_seed.loc[per_seed["Blocked forward rate"] >= 0.5]
    if not stuck.empty:
        detail = []
        for row in stuck.to_dict(orient="records"):
            detail.append(
                f"{row['Algorithm']} seed {int(row['Run seed'])} blocks "
                f"{row['Blocked forward rate']:.0%} of forward moves"
            )
        rows.append(
            {
                "Error": "Greedy forward into an obstacle",
                "Where it shows": "; ".join(detail),
                "What the files show": (
                    "A blocked forward stays in place and pays 0, so it looks "
                    "like any other step that does not deliver."
                ),
                "What to change next": (
                    "The sensor already shows the cell ahead. The other actions "
                    "still lose to forward, so the update has to raise those "
                    "actions after a blocked step instead of treating it as "
                    "just another zero reward."
                ),
            }
        )

    uniform = summary.dropna(subset=["Final entropy"])
    uniform = uniform.loc[uniform["Final entropy"] >= 0.95 * UNIFORM_ENTROPY]
    if not uniform.empty:
        shown = ", ".join(
            f"{row['Algorithm']} entropy {row['Final entropy']:.3f}"
            for row in uniform.to_dict(orient="records")
        )
        rows.append(
            {
                "Error": "Actions are not separated",
                "Where it shows": shown,
                "What the files show": (
                    f"A uniform choice over {N_ACTIONS} actions has entropy "
                    f"{UNIFORM_ENTROPY:.3f}. The saved policies are still near that."
                ),
                "What to change next": (
                    "The gradient rarely sees a delivery, so the entropy bonus "
                    "keeps the policy spread out. Greedy argmax then follows "
                    "a very small difference in the logits."
                ),
            }
        )
    return pd.DataFrame(rows)


def _shortlist(summary):
    """Pick learners when the official score does not separate them.

    Effectiveness is the greedy team return. When that is tied, a learner
    stays on the shortlist only if every run seed found a delivery while
    exploring. Order those by how many deliveries they found, then prefer
    the learner whose action values can be read. PPO can remain in the
    comparison without being the first model to refine.
    """
    remarks = []
    if summary.empty:
        return [], ["Nothing is shortlisted because no run was loaded."]

    best = float(summary["Greedy team return"].max())
    leaders = summary.loc[summary["Greedy team return"] == best, "Algorithm"].tolist()
    remarks.append(
        f"After {TOTAL_ENV_STEPS:,} environment steps, the best greedy team "
        f"return is {best:.2f} ({_join(leaders)})."
    )
    if best == 0.0 and len(leaders) == len(summary):
        remarks.append(
            "That score matches across the trained policies, so effectiveness "
            "on the held-out seeds does not choose a winner."
        )

    if best > 0.0:
        ordered = summary.sort_values(
            ["Greedy team return", "Training deliveries", "Readable action values"],
            ascending=[False, False, False],
        )
        names = ordered["Algorithm"].tolist()
        delivery_bits = [
            f"{row['Algorithm']} {int(row['Training deliveries'])}"
            for row in summary.to_dict(orient="records")
        ]
        remarks.append(
            "Training deliveries, counted from the warehouse reward while "
            f"the behaviour policy was exploring, are {_join(delivery_bits)}."
        )
        remarks.append(f"The shortlist follows that greedy return: {_join(names)}.")
        return names, remarks

    covered = summary.loc[
        summary["Seeds with a training delivery"] == summary["Run seeds"]
    ].copy()
    covered = covered.sort_values(
        ["Training deliveries", "Readable action values"],
        ascending=[False, False],
    )
    names = covered["Algorithm"].tolist()

    delivery_bits = [
        f"{row['Algorithm']} {int(row['Training deliveries'])}"
        for row in summary.to_dict(orient="records")
    ]
    remarks.append(
        "Training deliveries, counted only while the behaviour policy was "
        f"exploring, are {_join(delivery_bits)}."
    )
    missed = summary.loc[
        summary["Seeds with a training delivery"] < summary["Run seeds"], "Algorithm"
    ].tolist()
    if names:
        remarks.append(
            f"The shortlist is {_join(names)}. Each of those learners found a "
            "delivery on every run seed. The order follows the training "
            "delivery count. A readable Q-table is preferred when those counts match."
        )
    else:
        remarks.append(
            "No learner found a delivery on every run seed, so the shortlist is empty."
        )
    if missed:
        remarks.append(
            f"{_join(missed)} stays in the comparison and is not first for "
            "refinement: at least one run seed finished without a team reward."
        )
    remarks.append(
        "None of the shortlisted greedy policies delivered on the held-out "
        "seeds, so the shortlist is about who found the reward while exploring, "
        "not about who already generalises."
    )
    return names, remarks


def _join(names):
    names = [str(name) for name in names]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"
