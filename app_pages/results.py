"""Results and comparison for the shared 100,000-step warehouse protocol."""

import altair as alt
import pandas as pd
import streamlit as st

from app_pages.results_data import MOVING_AVERAGE, build_report, judgments
from environment.config import ENV_ID, EVAL_EPISODES, EVAL_SEEDS, SEEDS, TOTAL_ENV_STEPS

CHART_ORDER = ["Q-learning", "SARSA", "REINFORCE", "PPO", "Random"]
CHART_COLORS = {
    "Q-learning": "#E45756",
    "SARSA": "#4C78A8",
    "REINFORCE": "#54A24B",
    "PPO": "#F58518",
    "Random": "#9D9D9D",
}

report = build_report()
official = [item for item in report["algorithms"] if item["status"] == "official"]
missing = [item["name"] for item in report["algorithms"] if item["status"] == "missing"]
findings = judgments(report)

st.caption(
    f"Same warehouse ({ENV_ID}), same budget ({TOTAL_ENV_STEPS:,} environment steps), "
    f"and the same {EVAL_EPISODES} greedy evaluation episodes "
    f"(seeds {EVAL_SEEDS[0]} to {EVAL_SEEDS[-1]}). "
    f"Run seeds are {', '.join(str(seed) for seed in SEEDS)}. "
    "Team return is the sum of the two robot rewards. "
    "These bars use results/official: the first 100,000-step runs with the default settings, including REINFORCE. A retrained policy in results/final_runs replaces that algorithm when the search has chosen its settings. "
    "The 1,000,000-step PPO run is extra evidence and is not a bar in these charts."
)

def _training_section(runs):
    curves = [item for item in runs if item["training"] is not None and len(item["training"])]
    st.subheader("A. Training performance")
    if not curves:
        st.info(
            "No official run has a training log yet. Reward, the moving average, and the "
            "policy and value losses appear here when a 100,000-step log is saved.",
            icon=":material/info:",
        )
        return

    frame = _average_seeds(pd.concat([item["training"] for item in curves], ignore_index=True))
    if "team_reward_sum" in frame.columns:
        reward = frame.rename(
            columns={
                "algorithm": "Algorithm",
                "env_steps": "Environment steps",
                "team_reward_sum": "Team reward",
                "moving_average": "Moving average",
            }
        )
        present = [name for name in CHART_ORDER if name in set(reward["Algorithm"])]
        color = alt.Color(
            "Algorithm:N",
            sort=present,
            scale=_color_scale(present),
            legend=None if len(present) == 1 else alt.Legend(title=None),
        )
        raw = (
            alt.Chart(reward)
            .mark_circle(size=28, opacity=0.45)
            .encode(
                x=alt.X("Environment steps:Q", title="Environment steps"),
                y=alt.Y("Team reward:Q", title="Mean team reward per rollout"),
                color=color,
                tooltip=[
                    alt.Tooltip("Algorithm:N", title="Algorithm"),
                    alt.Tooltip("Environment steps:Q", title="Environment steps", format=","),
                    alt.Tooltip("Team reward:Q", title="Mean team reward", format=".2f"),
                    alt.Tooltip("Moving average:Q", title=f"{MOVING_AVERAGE}-rollout average", format=".2f"),
                ],
            )
        )
        average = alt.Chart(reward).mark_line().encode(
            x="Environment steps:Q",
            y=alt.Y("Moving average:Q"),
            color=color,
        )
        st.altair_chart((raw + average).properties(height=280), width="stretch")
        st.caption(
            f"Each point is the mean team reward across seeds {', '.join(str(seed) for seed in SEEDS)}. "
            f"The line is a {MOVING_AVERAGE}-rollout moving average of that mean. "
            + _reward_caption(curves)
        )

    loss_frame = _loss_frame(frame)
    if loss_frame is None:
        return
    loss_charts = [
        ("Policy loss", "Policy loss") if "Policy loss" in loss_frame.columns else None,
        ("Value loss", "Value loss") if "Value loss" in loss_frame.columns else None,
    ]
    loss_charts = [item for item in loss_charts if item is not None]
    if loss_charts:
        columns = st.columns(len(loss_charts))
        for column, (title, field) in zip(columns, loss_charts):
            with column:
                st.markdown(f"**{title}**")
                series = loss_frame[["Environment steps", field, "Algorithm"]]
                single = series["Algorithm"].nunique() == 1
                st.line_chart(
                    series,
                    x="Environment steps",
                    y=field,
                    color=CHART_COLORS[series["Algorithm"].iloc[0]] if single else "Algorithm",
                    height=260,
                )
        st.caption(
            f"Policy loss and value loss are the mean across seeds {', '.join(str(seed) for seed in SEEDS)}."
        )


def _metrics_section(runs, random_floor):
    st.subheader("B. Performance metrics")
    st.caption(
        "Average reward is the mean team return over every official evaluation episode. "
        "Success rate is the share of those episodes with at least one delivery. "
        "The random policy is the untrained floor on the same seeds, not a fourth learner."
    )
    scored = _scored_frame(runs, random_floor)
    episodes = _episode_frame(runs, random_floor)
    reward_top = max(1.0, float(scored["Mean team return"].max()) if len(scored) else 1.0)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Average reward**")
            st.altair_chart(
                _bar(scored, "Mean team return", "Mean team return", reward_top, ".2f"),
                width="stretch",
            )
    with right:
        with st.container(border=True):
            st.markdown("**Success rate**")
            st.altair_chart(
                _bar(scored, "Success rate", "Success rate", 1.0, ".0%", axis_format="%"),
                width="stretch",
            )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Average training time**")
            timed = scored.dropna(subset=["Training minutes"])
            if timed.empty:
                st.info(
                    "None of the official result files record wall-clock training time, "
                    "so this chart has no bars.",
                    icon=":material/schedule:",
                )
            else:
                st.altair_chart(
                    _bar(timed, "Training minutes", "Minutes", None, ".1f"),
                    width="stretch",
                )
    with right:
        with st.container(border=True):
            st.markdown("**Reward distribution**")
            if episodes.empty:
                st.info("No evaluation episodes are saved yet.", icon=":material/info:")
            else:
                present = [name for name in CHART_ORDER if name in set(episodes["Algorithm"])]
                returns = episodes["Team return"].astype(float)
                top = max(1.0, float(returns.max()))
                chart = (
                    alt.Chart(episodes)
                    .mark_boxplot(extent="min-max", size=36)
                    .encode(
                        x=alt.X("Algorithm:N", sort=present, title=None),
                        y=alt.Y(
                            "Team return:Q",
                            title="Episode team return",
                            scale=alt.Scale(domain=[0, top]),
                        ),
                        color=alt.Color(
                            "Algorithm:N",
                            sort=present,
                            scale=_color_scale(present),
                            legend=None,
                        ),
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, width="stretch")
            st.caption(
                "Each mark is one held-out episode. The box is the distribution across "
                "those episodes for every finished run."
            )


def _comparison_section(report, runs):
    st.subheader("C. Algorithm comparison")
    st.caption(
        "Every row is held to the same environment, the same 100,000 environment steps, "
        "the same three run seeds, and the same evaluation episodes. "
        "Random did not train. It is the floor for the reward and success charts."
    )
    rows = []
    for item in report["algorithms"]:
        if item["status"] == "official":
            rows.append(
                {
                    "Algorithm": item["name"],
                    "Result": "100,000-step run",
                    "Mean team return": f"{item['mean_return']:.2f}",
                    "Success rate": f"{item['success_rate']:.0%}",
                    "Return std": f"{item['return_std']:.2f}",
                    "Training time": _time_text(item["train_seconds"]),
                }
            )
        elif item["status"] == "missing":
            rows.append(
                {
                    "Algorithm": item["name"],
                    "Result": "No saved run",
                    "Mean team return": "—",
                    "Success rate": "—",
                    "Return std": "—",
                    "Training time": "—",
                }
            )
        else:
            rows.append(
                {
                    "Algorithm": item["name"],
                    "Result": item["reason"],
                    "Mean team return": "—",
                    "Success rate": "—",
                    "Return std": "—",
                    "Training time": "—",
                }
            )
    random_floor = report["random"]
    if random_floor is not None:
        rows.append(
            {
                "Algorithm": "Random",
                "Result": "Untrained floor, same evaluation seeds",
                "Mean team return": f"{random_floor['mean_return']:.2f}",
                "Success rate": f"{random_floor['success_rate']:.0%}",
                "Return std": f"{random_floor['return_std']:.2f}",
                "Training time": "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    seed_rows = [row for item in runs for row in item["seed_rows"]]
    if seed_rows:
        with st.expander("Per-seed evaluation scores"):
            detail = pd.DataFrame(seed_rows)
            detail["Success rate"] = detail["Success rate"].map(lambda value: f"{value:.0%}")
            detail["Mean team return"] = detail["Mean team return"].map(lambda value: f"{value:.2f}")
            st.dataframe(detail, hide_index=True, width="stretch")


def _evaluation_section(findings):
    st.subheader("D. Final evaluation")
    with st.container(horizontal=True):
        _finding("Best-performing algorithm", findings["best_name"], findings["best_detail"])
        _finding("Most consistent algorithm", findings["consistent_name"], findings["consistent_detail"])
        _finding("Fastest training algorithm", findings["fastest_name"], findings["fastest_detail"])
    with st.container(border=True):
        st.markdown("**Overall conclusion**")
        st.write(findings["conclusion"])


def _finding(label, name, detail):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{name}**")
        st.caption(detail)


def _scored_frame(runs, random_floor):
    rows = []
    for item in runs:
        rows.append(
            {
                "Algorithm": item["name"],
                "Mean team return": item["mean_return"],
                "Success rate": item["success_rate"],
                "Training minutes": None
                if item["train_seconds"] is None
                else item["train_seconds"] / 60,
            }
        )
    if random_floor is not None:
        rows.append(
            {
                "Algorithm": "Random",
                "Mean team return": random_floor["mean_return"],
                "Success rate": random_floor["success_rate"],
                "Training minutes": None,
            }
        )
    return pd.DataFrame(rows)


def _episode_frame(runs, random_floor):
    frames = []
    for item in runs:
        played = item["episodes"][
            ["algorithm", "seed", "team_return"]
        ].rename(
            columns={
                "algorithm": "Algorithm",
                "seed": "Evaluation seed",
                "team_return": "Team return",
            }
        )
        frames.append(played)
    if random_floor is not None:
        played = random_floor["episodes"][["seed", "team_return"]].rename(
            columns={"seed": "Evaluation seed", "team_return": "Team return"}
        )
        played["Algorithm"] = "Random"
        frames.append(played)
    if not frames:
        return pd.DataFrame(columns=["Algorithm", "Evaluation seed", "Team return"])
    return pd.concat(frames, ignore_index=True)


def _bar(frame, column, title, top, text_format, axis_format=None):
    present = [name for name in CHART_ORDER if name in set(frame["Algorithm"])]
    y_kwargs = {"title": title}
    if top is not None:
        y_kwargs["scale"] = alt.Scale(domain=[0, top])
    if axis_format:
        y_kwargs["axis"] = alt.Axis(format=axis_format)
    y_field = alt.Y(f"{column}:Q", **y_kwargs)
    encoding = dict(
        x=alt.X("Algorithm:N", sort=present, title=None),
        y=y_field,
        color=alt.Color("Algorithm:N", sort=present, scale=_color_scale(present), legend=None),
        tooltip=[
            alt.Tooltip("Algorithm:N", title="Algorithm"),
            alt.Tooltip(f"{column}:Q", title=title, format=text_format),
        ],
    )
    bars = alt.Chart(frame).mark_bar().encode(**encoding)
    labels = (
        alt.Chart(frame)
        .mark_text(dy=-8, clip=False)
        .encode(text=alt.Text(f"{column}:Q", format=text_format), **encoding)
    )
    return (bars + labels).properties(height=280)


def _color_scale(present):
    return alt.Scale(domain=present, range=[CHART_COLORS[name] for name in present])


def _average_seeds(frame):
    """One row per algorithm and step, averaged over the run seeds."""
    metrics = [
        column
        for column in ("team_reward_sum", "policy_loss", "value_loss")
        if column in frame.columns
    ]
    averaged = (
        frame.groupby(["algorithm", "env_steps"], as_index=False)[metrics]
        .mean()
        .sort_values(["algorithm", "env_steps"])
    )
    if "team_reward_sum" in averaged.columns:
        averaged["moving_average"] = averaged.groupby("algorithm")["team_reward_sum"].transform(
            lambda series: series.rolling(MOVING_AVERAGE, min_periods=1).mean()
        )
    return averaged


def _loss_frame(frame):
    if "policy_loss" not in frame.columns and "value_loss" not in frame.columns:
        return None
    renamed = frame.rename(
        columns={
            "algorithm": "Algorithm",
            "env_steps": "Environment steps",
            "policy_loss": "Policy loss",
            "value_loss": "Value loss",
        }
    )
    columns = ["Environment steps", "Algorithm"]
    if "Policy loss" in renamed.columns:
        columns.append("Policy loss")
    if "Value loss" in renamed.columns:
        columns.append("Value loss")
    return renamed[columns]


def _reward_caption(runs):
    parts = []
    for item in runs:
        counts = item["nonzero_rollouts"]
        if not counts:
            continue
        hits = [
            f"seed {seed} on {count} rollout{'s' if count != 1 else ''}"
            for seed, count in sorted(counts.items())
            if count
        ]
        quiet = [f"Seed {seed}" for seed, count in sorted(counts.items()) if not count]
        sentence = f"{item['name']}: "
        if hits:
            sentence += "team reward above zero for " + ", ".join(hits)
        else:
            sentence += "no rollout had a team reward above zero"
        if hits and quiet:
            sentence += ". " + ", ".join(quiet) + " had none"
        parts.append(sentence)
    if not parts:
        return "The training logs have no rollout reward column."
    return " ".join(part + "." for part in parts)


def _time_text(seconds):
    if seconds is None:
        return "—"
    return f"{seconds / 60:.1f} min"


def _names(names):
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


if missing:
    st.warning(
        f"{_names(missing)} {'has' if len(missing) == 1 else 'have'} no saved result on this "
        "protocol. The charts and the final evaluation use the completed runs. "
        "A score of zero is a finished run. A missing algorithm is left blank.",
        icon=":material/info:",
    )

_training_section(official)
_metrics_section(official, report["random"])
_comparison_section(report, official)
_evaluation_section(findings)
