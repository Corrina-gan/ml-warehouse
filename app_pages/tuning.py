"""Search results, the selected settings, and the held-out evaluation."""

import pandas as pd
import streamlit as st

from analysis.tune import FINAL_EVAL_SEEDS, PROJECT_ROOT, VALIDATION_SEEDS, load_tuning
from environment.config import SEEDS, TOTAL_ENV_STEPS

SEPARATE = PROJECT_ROOT / "results" / "tuning_multiseed"
LABELS = {
    "trial": "Trial",
    "team_return_mean": "Mean validation return",
    "team_return_std": "Standard deviation",
    "seed_0": "Seed 0",
    "seed_1": "Seed 1",
    "seed_2": "Seed 2",
    "success_rate": "Success rate",
    "deliveries_mean": "Mean deliveries",
    "blocked_forward_rate": "Blocked forward rate",
    "training_reward_sum": "Training reward",
    "training_deliveries": "Training deliveries",
    "alpha": "Step size",
    "gamma": "Discount",
    "epsilon_start": "Starting epsilon",
    "epsilon_min": "Epsilon floor",
    "epsilon_schedule": "Epsilon schedule",
    "learning_rate": "Learning rate",
    "entropy_coef_end": "Final entropy coefficient",
    "hidden": "Network size",
    "baseline_rate": "Baseline rate",
    "rollout_steps": "Rollout steps",
    "minibatch_size": "Minibatch size",
    "epochs": "Epochs",
    "clip_eps": "Clip range",
    "selected": "Selected",
}
INTEGERS = {"trial", "training_deliveries", "minibatch_size", "epochs", "rollout_steps"}
# seed columns are means, so they stay in NUMBERS.
NUMBERS = {
    "team_return_mean",
    "team_return_std",
    "seed_0",
    "seed_1",
    "seed_2",
    "success_rate",
    "deliveries_mean",
    "blocked_forward_rate",
    "training_reward_sum",
    "alpha",
    "gamma",
    "epsilon_start",
    "epsilon_min",
    "learning_rate",
    "entropy_coef_end",
    "baseline_rate",
    "clip_eps",
}

st.caption(
    "Q-learning, SARSA, REINFORCE, and PPO each have 10 configurations. "
    "Every configuration is trained on seeds 0, 1, and 2. "
    "This page reads results/tuning_multiseed. "
    f"The mean and standard deviation use validation seeds {VALIDATION_SEEDS[0]} to {VALIDATION_SEEDS[4]}. "
    f"The selected policies are scored on episode seeds {FINAL_EVAL_SEEDS[0]} to {FINAL_EVAL_SEEDS[-1]}."
)


@st.cache_data(show_spinner="Reading the tuning logs", ttl=15, max_entries=2)
def _tuning():
    return load_tuning(SEPARATE)


def _setting_text(tuned):
    parts = []
    for key, value in tuned.items():
        if isinstance(value, float):
            shown = f"{value:g}"
        elif isinstance(value, list):
            shown = "x".join(str(item) for item in value)
        else:
            shown = str(value)
        parts.append(f"{key}={shown}")
    return ", ".join(parts)


report = _tuning()
section = st.segmented_control(
    "Tuning section",
    ["Workflow", "Trials", "Final evaluation"],
    default="Workflow",
    key="tuning_section",
    required=True,
)

if section == "Workflow":
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Step": "1. Initial training",
                    "What it uses": f"Protocol defaults, {TOTAL_ENV_STEPS:,} steps, seeds {', '.join(str(seed) for seed in SEEDS)}",
                    "Score": "Official evaluation seeds, already saved",
                },
                {
                    "Step": "2. Hyperparameter search",
                    "What it uses": "10 configurations for that algorithm, each trained on seeds 0, 1, and 2",
                    "Score": "Validation episodes, then the mean and standard deviation",
                },
                {
                    "Step": "3. Select the winner",
                    "What it uses": "Highest mean. A tie prefers the smaller standard deviation, then trial 0",
                    "Score": f"Validation seeds {VALIDATION_SEEDS[0]} to {VALIDATION_SEEDS[4]}",
                },
                {
                    "Step": "4. Final evaluation",
                    "What it uses": "The three saved policies for the selected configuration",
                    "Score": f"Episode seeds {FINAL_EVAL_SEEDS[0]} to {FINAL_EVAL_SEEDS[-1]}",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Q-learning and SARSA vary step size, discount, epsilon, and the epsilon schedule. "
        "REINFORCE varies learning rate, discount, entropy, network size, and the baseline. "
        "PPO varies learning rate, rollout length, minibatch size, epochs, clip range, entropy, and discount. "
        "Trial 0 is that algorithm's current default. Each later trial changes one setting."
    )

elif section == "Trials":
    names = [item["name"] for item in report]
    choice = st.segmented_control(
        "Algorithm",
        names,
        default=names[0],
        key="tuning_algorithm",
        required=True,
    )
    item = next(entry for entry in report if entry["name"] == choice)
    trials = item["trials"]
    if not trials:
        st.info(
            f"No {choice} trial has been written yet. The search saves a row after each one finishes.",
            icon=":material/info:",
        )
    else:
        frame = pd.DataFrame(trials)
        best = item["best"] or {}
        selected_trial = int(best["trial"]) if best else -1
        frame["selected"] = frame["trial"].astype(int).eq(selected_trial)
        for column in list(frame.columns):
            if column not in INTEGERS and column not in NUMBERS:
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = converted.astype("Int64") if column in INTEGERS else converted
        order = [column for column in LABELS if column in frame.columns]
        shown = frame[order].rename(columns=LABELS)
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "Success rate": st.column_config.NumberColumn("Success rate", format="percent"),
                "Blocked forward rate": st.column_config.NumberColumn("Blocked forward rate", format="percent"),
            },
        )
        if best:
            st.caption(
                f"Selected trial {int(best['trial'])}: {best['tuned']}. "
                f"Validation team return {float(best['team_return_mean']):.2f} "
                f"after {int(best['steps']):,} steps on run seed {int(best['run_seed'])}."
            )

elif section == "Final evaluation":
    finished = [item for item in report if item["final"]]
    if not finished:
        st.info(
            "The retrained policies have not been scored on the official seeds yet.",
            icon=":material/info:",
        )
    else:
        with st.container(horizontal=True):
            for item in finished:
                final = item["final"]
                st.metric(
                    item["name"],
                    f"{float(final['team_return_mean']):.2f}",
                    border=True,
                )
        rows = []
        for item in report:
            final = item["final"]
            if not final:
                continue
            for seed_row in final["per_seed"]:
                rows.append(
                    {
                        "Algorithm": item["name"],
                        "Run seed": int(seed_row["seed"]),
                        "Team return": float(seed_row["team_return_mean"]),
                        "Success rate": seed_row["success_rate"],
                        "Mean deliveries": seed_row["deliveries_mean"],
                        "Blocked forward rate": float(seed_row["blocked_forward_rate"]),
                        "Settings": _setting_text(final["tuned"]),
                    }
                )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Success rate": st.column_config.NumberColumn("Success rate", format="percent"),
                "Blocked forward rate": st.column_config.NumberColumn("Blocked forward rate", format="percent"),
            },
        )
        st.caption(
            "Each row is one training seed scored on the five held-out episode seeds. "
            "The search did not use these episodes to choose the settings."
        )
