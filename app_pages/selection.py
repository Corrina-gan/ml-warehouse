"""Shortlist and combined policy, computed from the saved runs."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.selection import build_selection

ENSEMBLE = Path(__file__).resolve().parents[1] / "results" / "ensemble"

st.caption(
    "Q-learning, SARSA, REINFORCE, and PPO stay on the same warehouse and the "
    "same 100,000-step protocol. This page reads the latest saved runs: "
    "official Q-learning, SARSA, REINFORCE, and PPO runs. A finished retrain in results/final_runs replaces that algorithm. "
    "It does not retrain them. The combined policy is a majority vote of "
    "the saved greedy actions, scored on the same held-out seeds."
)


@st.cache_data(show_spinner="Reading the saved runs", max_entries=2)
def _selection():
    # Q-learning stays in the approach table when its 100,000-step log is absent.
    return build_selection()


report = _selection()
section = st.segmented_control(
    "Selection section",
    ["Approaches", "Robustness", "Errors", "Shortlist", "Combined policy"],
    default="Approaches",
    key="selection_section",
)

if section == "Approaches":
    st.dataframe(report["approaches"], hide_index=True, width="stretch")
    st.caption(
        "Q-learning and SARSA are tabular temporal difference. REINFORCE is an "
        "episodic policy gradient, and PPO is an actor-critic. They share the team reward, "
        "the discount, the step budget, and the evaluation seeds."
    )

elif section == "Robustness":
    summary = report["summary"]
    if summary.empty:
        st.info("No saved run was found to compare.", icon=":material/info:")
    else:
        shown = summary[
            [
                "Algorithm",
                "Greedy team return",
                "Training deliveries",
                "Seeds with a training delivery",
                "Mean blocked forward rate",
                "Blocked forward spread",
            ]
        ].copy()
        shown["Greedy team return"] = shown["Greedy team return"].map(lambda value: f"{value:.2f}")
        shown["Mean blocked forward rate"] = shown["Mean blocked forward rate"].map(
            lambda value: f"{value:.0%}"
        )
        shown["Blocked forward spread"] = shown["Blocked forward spread"].map(
            lambda value: f"{value:.0%}"
        )
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption(
            "Greedy team return is the held-out score. Training deliveries "
            "count episodes or rollouts where exploration received a team "
            "reward. Blocked forward spread is the gap between the highest "
            "and lowest run seed."
        )
        with st.expander("Per-seed scores"):
            detail = report["per_seed"][
                [
                    "Algorithm",
                    "Run seed",
                    "Training deliveries",
                    "Training reward sum",
                    "Greedy team return",
                    "Blocked forward rate",
                    "Final entropy",
                    "Final epsilon",
                ]
            ].copy()
            st.dataframe(detail, hide_index=True, width="stretch")

elif section == "Errors":
    errors = report["errors"]
    if errors.empty:
        st.info("The saved files did not match a listed error.", icon=":material/info:")
    else:
        st.dataframe(errors, hide_index=True, width="stretch")
        st.caption(
            "Each row is computed from the training logs and the greedy "
            "evaluation logs. A blocked forward pays 0, so the team return "
            "alone hides that waste."
        )

elif section == "Shortlist":
    names = report["shortlist"]
    with st.container(border=True):
        st.markdown("**Shortlisted for refinement**")
        st.markdown(f"**{', '.join(names) if names else 'None'}**")
    for remark in report["remarks"]:
        st.write(remark)
    st.caption(
        "The official score stays the greedy team return. When that score "
        "is tied, the shortlist keeps a learner only if every run seed found "
        "a delivery while exploring."
    )

else:
    comparison = ENSEMBLE / "comparison.csv"
    summary_path = ENSEMBLE / "summary.json"
    if not comparison.is_file() or not summary_path.is_file():
        st.info(
            "The combined policy has not been scored yet. From the project "
            "root, run `python -m analysis.ensemble`.",
            icon=":material/info:",
        )
    else:
        scores = pd.read_csv(comparison)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mean_return = float(scores["team_return_mean"].mean())
        mean_disagreement = float(scores["disagreement_rate"].mean())
        with st.container(horizontal=True):
            st.metric("Mean team return", f"{mean_return:.2f}", border=True)
            st.metric("Success rate", f"{float(scores['success_rate'].mean()):.0%}", border=True)
            st.metric("2-1 majority", f"{float(scores['majority_rate'].mean()):.0%}", border=True)
            st.metric("Three-way split", f"{float(scores['split_rate'].mean()):.0%}", border=True)
        st.caption(summary["combination"])
        shown = scores.rename(
            columns={
                "seed": "Run seed",
                "team_return_mean": "Team return",
                "success_rate": "Success rate",
                "blocked_forward_rate": "Blocked forward rate",
                "disagreement_rate": "Disagreement",
                "majority_rate": "2-1 majority",
                "split_rate": "Three-way split",
            }
        )
        st.dataframe(
            shown[
                [
                    "Run seed",
                    "Team return",
                    "Success rate",
                    "Blocked forward rate",
                    "Disagreement",
                    "2-1 majority",
                    "Three-way split",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Team return": st.column_config.NumberColumn(format="%.2f"),
                "Success rate": st.column_config.NumberColumn(format="%.0%"),
                "Blocked forward rate": st.column_config.NumberColumn(format="%.0%"),
                "Disagreement": st.column_config.NumberColumn(format="%.0%"),
                "2-1 majority": st.column_config.NumberColumn(format="%.0%"),
                "Three-way split": st.column_config.NumberColumn(format="%.0%"),
            },
        )
        split_rate = float(scores["split_rate"].mean())
        if mean_return == 0.0:
            st.write(
                "The vote scores 0.00 on the held-out seeds. "
                f"The policies disagreed on {mean_disagreement:.0%} of robot decisions, "
                f"and {split_rate:.0%} of decisions were a three-way split. "
                "A split keeps the lowest action id, so it does not blend the three choices. "
                "The 2-1 votes did not produce a delivery either."
            )
        else:
            st.write(
                f"The vote mean team return is {mean_return:.2f}. "
                f"The members disagreed on {mean_disagreement:.0%} of robot decisions."
            )
