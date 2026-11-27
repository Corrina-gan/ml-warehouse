"""Score a majority vote of the saved greedy policies.

Part B asks for a combination of models and a test on episodes the
learners did not train on. The members are the Q-learning table, the SARSA
table, the REINFORCE policy, and the PPO actor. A seed votes with the
checkpoints that exist, and it needs at least two. The vote asks each one
for a greedy action and keeps the majority. A tie takes the lowest action
id. That is the same tie break the tabular agents use at evaluation, so
the combined policy is repeatable.

The vote does not retrain and does not change the reward. If the policies
agree on a poor action, the vote repeats it. A disagreement is the only
place the combination can leave a path that one member would not take
alone. The episodes are the shared evaluation seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from algorithms.PPO.ppo import load_agent as load_ppo
from algorithms.QLEARNING.qlearning import QLearningAgent
from algorithms.REINFORCE.reinforce import load_agent as load_reinforce
from algorithms.SARSA.sarsa import SARSAAgent
from environment.config import EVAL_EPISODES, EVAL_SEEDS, N_AGENTS, SEEDS
from shared.evaluation import evaluate, format_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "ensemble"

MEMBERS = (
    ("Q-learning", "qlearning", "qlearning.npz"),
    ("SARSA", "sarsa", "sarsa.npz"),
    ("REINFORCE", "reinforce", "reinforce.pt"),
    ("PPO", "ppo", "ppo.pt"),
)


class VotingPolicy:
    """Greedy majority vote. Ties break to the lowest action id."""

    def __init__(self, agents):
        if len(agents) < 2:
            raise ValueError("a vote needs at least two policies")
        self.agents = list(agents)
        self.decisions = 0
        self.disagreements = 0
        self.majorities = 0
        self.splits = 0

    def act_greedy(self, observations):
        votes = [agent.act_greedy(observations) for agent in self.agents]
        chosen = []
        for robot in range(N_AGENTS):
            actions = [int(vote[robot]) for vote in votes]
            self.decisions += 1
            if len(set(actions)) > 1:
                self.disagreements += 1
            counts = {}
            for action in actions:
                counts[action] = counts.get(action, 0) + 1
            best = max(counts.values())
            if best == 1:
                self.splits += 1
            elif best < len(actions):
                self.majorities += 1
            winners = [action for action, count in counts.items() if count == best]
            chosen.append(min(winners))
        return tuple(chosen)


def _listed(names):
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def load_members(run_seed, results):
    """Saved policies for one run seed, in member order.

    A missing checkpoint is skipped. These are the same saved runs the
    comparison page reads.
    """
    agents = []
    present = []
    missing = []
    for name, folder, filename in MEMBERS:
        path = _member_path(results, run_seed, folder, filename)
        if path is None:
            missing.append(name)
            continue
        agents.append(_load(folder, path))
        present.append(name)
    if len(agents) < 2:
        looked = ", ".join(
            str(results / folder / f"seed_{run_seed}" / filename)
            for _name, folder, filename in MEMBERS
        )
        raise FileNotFoundError(
            f"seed {run_seed} has {len(agents)} saved {'policy' if len(agents) == 1 else 'policies'}; "
            f"a vote needs two. Looked in {looked}"
        )
    return agents, present, missing


def _member_path(results, run_seed, folder, filename):
    # The vote uses the retrained policy when results/final_runs has it.
    # Otherwise it uses the initial default-setting run in results/official.
    candidates = [
        results / "final_runs" / folder / f"seed_{run_seed}" / filename,
        results / "official" / folder / f"seed_{run_seed}" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load(folder, path):
    if folder.endswith("qlearning"):
        agent = QLearningAgent()
        agent.load(path)
        return agent
    if folder.endswith("sarsa"):
        agent = SARSAAgent()
        agent.load(path)
        return agent
    if "reinforce" in folder:
        return load_reinforce(path)
    if folder.endswith("ppo"):
        return load_ppo(path)
    raise ValueError(f"unknown policy folder {folder}")


def vote_seed(run_seed, results, episodes=EVAL_EPISODES):
    """Greedy vote on the shared evaluation seeds. Returns the score and the vote counts."""
    agents, present, missing = load_members(run_seed, results)
    policy = VotingPolicy(agents)
    score = evaluate(policy.act_greedy, episodes=episodes, seeds=EVAL_SEEDS[:episodes])
    score["run_seed"] = int(run_seed)
    score["decisions"] = policy.decisions
    score["disagreements"] = policy.disagreements
    score["disagreement_rate"] = (
        0.0 if policy.decisions == 0 else policy.disagreements / policy.decisions
    )
    score["majorities"] = policy.majorities
    score["splits"] = policy.splits
    score["majority_rate"] = (
        0.0 if policy.decisions == 0 else policy.majorities / policy.decisions
    )
    score["split_rate"] = 0.0 if policy.decisions == 0 else policy.splits / policy.decisions
    score["members"] = present
    score["missing_members"] = missing
    return score


def write_results(scores, output):
    output.mkdir(parents=True, exist_ok=True)
    comparison = output / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "team_return_mean",
                "team_return_std",
                "success_rate",
                "deliveries_mean",
                "blocked_forward_rate",
                "disagreement_rate",
                "majority_rate",
                "split_rate",
                "decisions",
                "disagreements",
                "majorities",
                "splits",
            ]
        )
        for score in scores:
            writer.writerow(
                [
                    score["run_seed"],
                    score["team_return_mean"],
                    score["team_return_std"],
                    score["success_rate"],
                    score["deliveries_mean"],
                    score["blocked_forward_rate"],
                    score["disagreement_rate"],
                    score["majority_rate"],
                    score["split_rate"],
                    score["decisions"],
                    score["disagreements"],
                    score["majorities"],
                    score["splits"],
                ]
            )
    members = scores[0]["members"]
    missing = scores[0]["missing_members"]
    combination = (
        f"Majority vote of the {_listed(members)} greedy actions saved for the same "
        "run seed. A tie takes the lowest action id."
    )
    if missing:
        pronoun = "it" if len(missing) == 1 else "them"
        verb = "has" if len(missing) == 1 else "have"
        combination += (
            f" {_listed(missing)} {verb} no checkpoint, so this vote leaves {pronoun} out."
        )
    payload = {
        "combination": combination,
        "members": members,
        "missing_members": missing,
        "evaluation_seeds": list(EVAL_SEEDS[: scores[0]["episodes"]]),
        "seeds": [
            {
                "run_seed": score["run_seed"],
                "team_return_mean": score["team_return_mean"],
                "success_rate": score["success_rate"],
                "blocked_forward_rate": score["blocked_forward_rate"],
                "disagreement_rate": score["disagreement_rate"],
                "majority_rate": score["majority_rate"],
                "split_rate": score["split_rate"],
                "decisions": score["decisions"],
                "disagreements": score["disagreements"],
                "majorities": score["majorities"],
                "splits": score["splits"],
            }
            for score in scores
        ],
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the greedy majority vote.")
    parser.add_argument("--seed", type=int, default=None, help="One run id from SEEDS. Omit to vote every seed.")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run_seeds = SEEDS if args.seed is None else (args.seed,)
    scores = []
    for run_seed in run_seeds:
        score = vote_seed(run_seed, args.results, args.episodes)
        print(f"voting {_listed(score['members'])} from run seed {run_seed}")
        if score["missing_members"]:
            print(f"no checkpoint for {_listed(score['missing_members'])}")
        print(format_score(score))
        print(
            f"disagreement on {score['disagreements']} of {score['decisions']} "
            f"robot decisions ({score['disagreement_rate']:.1%}); "
            f"majority {score['majority_rate']:.1%}, split {score['split_rate']:.1%}"
        )
        scores.append(score)
    if scores:
        path = write_results(scores, args.output)
        print(f"wrote {path}")
    return scores


if __name__ == "__main__":
    main()
