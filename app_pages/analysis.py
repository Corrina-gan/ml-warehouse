import altair as alt
import streamlit as st

from app_pages.data import action_outcomes, load_csv, results_ready, summary

if not results_ready():
    st.warning(
        "The analysis files are not in results/eda yet. From the project root, run "
        "`python -m eda.warehouse_eda`.",
        icon=":material/info:",
    )
    st.stop()

stats = summary()
baseline = load_csv("random_baseline.csv")
mean_return = stats["mean_reward"]

st.caption(
    "rware-tiny-2ag-v2, measured as installed. 100 random episodes, seeds 0 to 99. "
    "The reward was not changed. Team return is the sum of the two robot rewards."
)

with st.container(horizontal=True):
    st.metric(
        "Mean team return",
        f"{mean_return:.2f}",
        border=True,
        chart_data=baseline["reward"].tolist(),
        chart_type="bar",
    )
    st.metric(
        "Episodes with a delivery",
        f"{stats['delivery_success_rate']:.0%}",
        border=True,
    )
    st.metric("Steps in every episode", f"{stats['mean_episode_length']:.0f}", border=True)
    st.metric(
        "Steps that paid a reward",
        f"{stats['positive_reward_steps']:.0f} / {stats['total_steps']:.0f}",
        border=True,
    )

section = st.segmented_control(
    "Analysis section",
    ["Overview", "Observations", "Actions"],
    default="Overview",
    key="analysis_section",
)

if section == "Overview":
    st.info(
        "Five episodes scored 1 and the other 95 scored 0. A delivery pays 1 to one "
        "robot and does not end the episode. At step 500 the warehouse sets terminated "
        "to true and truncated to false.",
        icon=":material/info:",
    )
    returns = (
        alt.Chart(baseline)
        .mark_bar()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("reward:Q", title="Team return", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("episode:Q", title="Episode"),
                alt.Tooltip("seed:Q", title="Seed"),
                alt.Tooltip("reward:Q", title="Team return", format=".0f"),
                alt.Tooltip("robot_1_reward:Q", title="Robot 1", format=".0f"),
                alt.Tooltip("robot_2_reward:Q", title="Robot 2", format=".0f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(returns, width="stretch")
    st.caption(
        f"Median team return is {stats['median_reward']:.0f}. "
        f"Sample standard deviation is {stats['std_reward_sample']:.3f}. "
        f"Mean robot returns are {stats['mean_robot_1_reward']:.2f} and "
        f"{stats['mean_robot_2_reward']:.2f}."
    )

elif section == "Observations":
    st.markdown(
        "Each robot receives its own 71-value `float32` vector. The values are "
        "integers. The declared box bounds are infinite, so they do not describe "
        "the real range. An empty sensor cell still writes the facing-up direction bits."
    )
    components = load_csv("observation_components.csv")
    self_features = components.loc[
        components["group"] == "self", ["index", "component", "meaning"]
    ]
    st.dataframe(self_features, hide_index=True, width="stretch")
    ranges = load_csv("observation_ranges.csv")
    self_ranges = ranges.loc[ranges["index"] <= 7].copy()
    self_ranges["component"] = self_features["component"].to_numpy()
    st.dataframe(
        self_ranges.rename(
            columns={
                "index": "Index",
                "component": "Field",
                "observed_min": "Observed minimum",
                "observed_max": "Observed maximum",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Indexes 8 through 70 are a 3 by 3 sensor. Each cell has 7 values: "
        "has agent, four direction bits, has shelf, and shelf requested. "
        "Column x was observed from 0 to 9. Row y was observed from 0 to 10."
    )

else:
    st.markdown(
        "Each robot chooses one of 5 actions. Turns never change the cell. "
        "A blocked forward move stays in place and pays 0, the same as a step that does not deliver."
    )
    outcomes = action_outcomes()
    chart = (
        alt.Chart(outcomes)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort="-x", title="Outcome"),
            x=alt.X("count:Q", title="Count over 100,000 robot actions"),
            tooltip=[
                alt.Tooltip("label:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")
    actions = load_csv("action_descriptions.csv")
    st.dataframe(
        actions.rename(columns={"action_id": "ID", "name": "Action", "effect": "Effect"}),
        hide_index=True,
        width="stretch",
    )
