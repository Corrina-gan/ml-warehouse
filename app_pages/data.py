"""Load the saved warehouse analysis. Pages do not re-run the 100 episodes."""

from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eda"

OUTCOME_LABELS = {
    "noop": "No-op",
    "left": "Turn left",
    "right": "Turn right",
    "forward_moved": "Forward, moved",
    "forward_boundary": "Forward, boundary",
    "forward_blocked_shelf": "Forward, shelf block",
    "forward_blocked_collision": "Forward, collision",
    "toggle_pickup": "Toggle, pickup",
    "toggle_drop": "Toggle, drop",
    "toggle_empty": "Toggle, empty cell",
    "toggle_highway_kept": "Toggle, kept on highway",
}


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def results_ready() -> bool:
    return (RESULTS / "statistical_summary.csv").exists() and (
        RESULTS / "random_baseline.csv"
    ).exists()


def summary() -> dict[str, float | str]:
    frame = load_csv("statistical_summary.csv")
    values: dict[str, float | str] = {}
    for row in frame.itertuples(index=False):
        text = str(row.value)
        try:
            values[str(row.metric)] = float(text)
        except ValueError:
            values[str(row.metric)] = text
    return values


def action_outcomes() -> pd.DataFrame:
    frame = load_csv("blocked_movement_summary.csv")
    frame = frame[frame["outcome"].isin(OUTCOME_LABELS)].copy()
    frame["label"] = frame["outcome"].map(OUTCOME_LABELS)
    return frame
