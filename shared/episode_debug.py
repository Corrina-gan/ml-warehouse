"""Per-episode trace of what each robot actually did.

Every episode appends one summary row. A full step trace is written for the
first episode, the last episode, and any episode that delivers a shelf.
The warehouse environment is not modified.
"""

import csv
import json
from pathlib import Path

import numpy as np

from environment.config import ACTION_NAMES, FORWARD, OBS_SIZE, TOGGLE_LOAD
from shared.observation import REQUESTED_HERE_INDEX

_DIRECTIONS = ("up", "down", "left", "right")
_SUMMARY_FIELDS = (
    "episode",
    "steps",
    "team_reward",
    "shaped_reward",
    "deliveries",
    "pickups",
    "pickup_attempts",
    "blocked_forwards",
    "forward",
    "left",
    "right",
    "toggle_load",
    "noop",
    "on_requested_shelf",
    "toggle_on_requested",
    "mean_toggle_probability",
)


def write_episode_debug(directory, episode_index, transitions, probabilities=None, keep_trace=False):
    """Append the episode summary and, when requested, the step trace."""
    directory = Path(directory)
    debug_dir = directory / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    steps = _step_rows(transitions, probabilities)
    summary = _summary(episode_index, steps)
    _append_summary(debug_dir / "episode_summaries.csv", summary)
    if keep_trace or episode_index == 1 or summary["deliveries"] > 0:
        path = debug_dir / "traces" / f"episode_{int(episode_index):04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "steps": steps}, indent=2) + "\n", encoding="utf-8")
    return summary


def _step_rows(transitions, probabilities):
    rows = []
    for index, transition in enumerate(transitions):
        probs = None
        if probabilities is not None and index < len(probabilities) and probabilities[index] is not None:
            probs = np.asarray(probabilities[index], dtype=np.float64)
        for robot in range(len(transition.actions)):
            before = np.asarray(transition.observations[robot], dtype=np.float64).reshape(-1)
            after = np.asarray(transition.next_observations[robot], dtype=np.float64).reshape(-1)
            action = int(transition.actions[robot])
            carrying_before = bool(before[2] > 0.5)
            carrying_after = bool(after[2] > 0.5)
            carrying_requested = before.shape[0] > OBS_SIZE + 4 and bool(before[OBS_SIZE + 4] > 0.5)
            requested_here = before.shape[0] > REQUESTED_HERE_INDEX and bool(before[REQUESTED_HERE_INDEX] > 0.5)
            blocked = action == FORWARD and float(before[0]) == float(after[0]) and float(before[1]) == float(after[1])
            warehouse = float(transition.rewards[robot])
            shaped = float(transition.learning_rewards[robot])
            row = {
                "t": index,
                "robot": robot,
                "x": int(before[0]),
                "y": int(before[1]),
                "orientation": _orientation(before),
                "action": ACTION_NAMES[action],
                "carrying": carrying_before,
                "carrying_requested": carrying_requested,
                "requested_here": requested_here and not carrying_before,
                "pickup_attempted": action == TOGGLE_LOAD and not carrying_before,
                "pickup_successful": (not carrying_before) and carrying_after,
                "delivery_attempted": carrying_requested and action == FORWARD,
                "delivery_successful": warehouse > 0.0,
                "warehouse_reward": warehouse,
                "shaped_reward": shaped,
                "blocked": blocked,
            }
            if probs is not None and robot < probs.shape[0]:
                for name, value in zip(ACTION_NAMES, probs[robot]):
                    row[f"p_{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}"] = float(value)
            rows.append(row)
    return rows


def _summary(episode_index, steps):
    counts = {name: 0 for name in ("noop", "forward", "left", "right", "toggle_load")}
    key = {
        "No-op": "noop",
        "Forward": "forward",
        "Left": "left",
        "Right": "right",
        "Toggle load": "toggle_load",
    }
    toggle_probability = []
    for row in steps:
        counts[key[row["action"]]] += 1
        if "p_toggle_load" in row:
            toggle_probability.append(row["p_toggle_load"])
    return {
        "episode": int(episode_index),
        "steps": len({row["t"] for row in steps}) if steps else 0,
        "team_reward": float(sum(row["warehouse_reward"] for row in steps)),
        "shaped_reward": float(sum(row["shaped_reward"] for row in steps)),
        "deliveries": int(sum(1 for row in steps if row["delivery_successful"])),
        "pickups": int(sum(1 for row in steps if row["pickup_successful"])),
        "pickup_attempts": int(sum(1 for row in steps if row["pickup_attempted"])),
        "blocked_forwards": int(sum(1 for row in steps if row["blocked"])),
        "on_requested_shelf": int(sum(1 for row in steps if row["requested_here"])),
        "toggle_on_requested": int(
            sum(1 for row in steps if row["requested_here"] and row["action"] == "Toggle load")
        ),
        **counts,
        "mean_toggle_probability": ""
        if not toggle_probability
        else float(sum(toggle_probability) / len(toggle_probability)),
    }


def _append_summary(path, summary):
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: summary[field] for field in _SUMMARY_FIELDS})


def _orientation(observation):
    bits = observation[3:7] > 0.5
    if int(bits.sum()) != 1:
        return "unknown"
    return _DIRECTIONS[int(np.argmax(bits))]
