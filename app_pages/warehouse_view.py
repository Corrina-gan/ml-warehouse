"""Draw both robots on the shared warehouse and step through one episode."""

import base64
import io
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from rware.warehouse import Direction

from environment.config import ACTION_NAMES, MAX_STEPS, N_ACTIONS, N_AGENTS, SEEDS
from environment.environment import make_warehouse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPO_RESULTS = PROJECT_ROOT / "results" / "ppo"

CELL = 48
ROBOT_COLORS = ((255, 140, 0), (30, 136, 229))
FLOOR = (248, 248, 248)
AISLE = (230, 230, 230)
GRID = (190, 190, 190)
SHELF = (72, 61, 139)
REQUESTED = (0, 128, 128)
GOAL = (60, 60, 60)
CARRY = (255, 255, 255)
FACING = (20, 20, 20)
DIRECTION_NAMES = ("up", "down", "left", "right")


def show_warehouse():
    """Algorithm, seed, play, and step controls for one episode."""
    st.session_state.setdefault("warehouse_step", 0)
    st.session_state.setdefault("warehouse_playing", False)
    st.session_state.setdefault("warehouse_seed", 0)
    choices = _algorithm_choices()
    st.session_state.setdefault("warehouse_algorithm", choices[0][0])

    def reset_view():
        st.session_state["warehouse_step"] = 0
        st.session_state["warehouse_playing"] = False

    labels = {key: label for key, label, _mtime in choices}
    if st.session_state["warehouse_algorithm"] not in labels:
        st.session_state["warehouse_algorithm"] = choices[0][0]
    st.selectbox(
        "Algorithm",
        [key for key, _label, _mtime in choices],
        format_func=labels.get,
        key="warehouse_algorithm",
        on_change=reset_view,
        help="Random takes a uniform action. PPO plays the greedy policy from the 100,000-step run.",
    )
    st.number_input(
        "Seed",
        min_value=0,
        max_value=99,
        key="warehouse_seed",
        on_change=reset_view,
        help="Same seeds as the analysis. Seed 0 is the sample episode.",
    )
    algorithm = st.session_state["warehouse_algorithm"]
    weights_mtime = next(mtime for key, _label, mtime in choices if key == algorithm)
    try:
        episode = record_episode(int(st.session_state["warehouse_seed"]), algorithm, weights_mtime)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        st.error("PPO needs PyTorch in the environment that runs this app.")
        return
    length = len(episode["frames"]) - 1
    st.session_state["warehouse_step"] = min(int(st.session_state["warehouse_step"]), length)

    with st.container(horizontal=True):
        st.button(
            "Pause" if st.session_state["warehouse_playing"] else "Play",
            type="primary",
            icon=":material/pause:" if st.session_state["warehouse_playing"] else ":material/play_arrow:",
            on_click=_toggle_play,
        )
        st.button("Step", icon=":material/skip_next:", on_click=_step_once, args=(length,))

    if st.session_state["warehouse_playing"]:
        _playing(episode)
    else:
        _paused(episode)


@st.fragment(run_every=0.05)
def _playing(episode):
    length = len(episode["frames"]) - 1
    nxt = min(int(st.session_state.get("warehouse_step", 0)) + 1, length)
    st.session_state["warehouse_step"] = nxt
    if nxt >= length:
        st.session_state["warehouse_playing"] = False
        st.rerun()
    _draw(episode)


@st.fragment
def _paused(episode):
    _draw(episode)


def _toggle_play():
    st.session_state["warehouse_playing"] = not st.session_state.get("warehouse_playing", False)


def _step_once(length):
    st.session_state["warehouse_playing"] = False
    st.session_state["warehouse_step"] = min(int(st.session_state.get("warehouse_step", 0)) + 1, length)


def _draw(episode):
    length = len(episode["frames"]) - 1
    st.slider("Step", min_value=0, max_value=length, key="warehouse_step")
    step = int(st.session_state["warehouse_step"])
    frame = episode["frames"][step]
    st.html(_image_html(render(episode["layout"], frame)))
    st.caption(
        f"{_algorithm_label(st.session_state['warehouse_algorithm'])}. "
        f"Seed {st.session_state['warehouse_seed']}, step {step} of {length}. "
        "Orange is robot 1. Blue is robot 2. The black mark is the direction they face."
    )
    lines = []
    for index, agent in enumerate(frame["agents"]):
        carrying = "carrying a shelf" if agent["carrying"] else "not carrying"
        lines.append(
            f"Robot {index + 1} is at column {agent['x']}, row {agent['y']}, "
            f"facing {DIRECTION_NAMES[agent['dir']]}, {carrying}."
        )
    if frame["actions"] is None:
        lines.append("This is the start of the episode.")
    else:
        action_text = ", ".join(
            f"robot {index + 1} {ACTION_NAMES[action].lower()}"
            for index, action in enumerate(frame["actions"])
        )
        lines.append(f"Last actions: {action_text}.")
    st.caption(" ".join(lines))


def _algorithm_choices():
    """Random, plus each official 100k PPO run whose weights are on disk."""
    choices = [("random", "Random", 0)]
    for seed in SEEDS:
        path = PPO_RESULTS / f"seed_{seed}" / "ppo.pt"
        if path.is_file():
            choices.append((f"ppo:{path.resolve()}", f"PPO, seed {seed}", path.stat().st_mtime_ns))
    return choices


def _algorithm_label(algorithm):
    for key, label, _mtime in _algorithm_choices():
        if key == algorithm:
            return label
    return "Random"


@st.cache_resource(show_spinner="Loading the PPO policy")
def _load_ppo(path):
    from algorithms.PPO.ppo import load_agent

    return load_agent(path)


@st.cache_data(show_spinner="Placing the robots in the warehouse")
def record_episode(seed, algorithm, weights_mtime):
    """One episode under the selected policy. Frames store positions only."""
    del weights_mtime
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    env = make_warehouse()
    try:
        observations, _info = env.reset(seed=int(seed))
        warehouse = env.unwrapped
        layout = {
            "rows": int(warehouse.grid_size[0]),
            "cols": int(warehouse.grid_size[1]),
            "highways": warehouse.highways.astype(bool).tolist(),
            "goals": [(int(x), int(y)) for x, y in warehouse.goals],
        }
        frames = [_snapshot(warehouse, None)]
        rng = np.random.default_rng(int(seed))
        agent = None if algorithm == "random" else _load_ppo(algorithm.removeprefix("ppo:"))
        for _step in range(MAX_STEPS):
            if agent is None:
                actions = tuple(int(action) for action in rng.integers(0, N_ACTIONS, size=N_AGENTS))
            else:
                actions = agent.act_greedy(observations)
            observations, rewards, terminated, truncated, _info = env.step(actions)
            frames.append(_snapshot(warehouse, actions, rewards))
            if terminated or truncated:
                break
        return {"layout": layout, "frames": frames}
    finally:
        env.close()


def _snapshot(warehouse, actions, rewards=None):
    requested = set(warehouse.request_queue)
    return {
        "agents": [
            {
                "x": int(agent.x),
                "y": int(agent.y),
                "dir": int(agent.dir.value),
                "carrying": agent.carrying_shelf is not None,
            }
            for agent in warehouse.agents
        ],
        "shelves": [
            {
                "x": int(shelf.x),
                "y": int(shelf.y),
                "requested": shelf in requested,
            }
            for shelf in warehouse.shelfs
        ],
        "actions": None if actions is None else tuple(int(action) for action in actions),
        "rewards": None if rewards is None else tuple(float(reward) for reward in rewards),
    }


def _image_html(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<img src="data:image/png;base64,'
        f'{encoded}" alt="Warehouse grid with robot 1 in orange and robot 2 in blue" '
        'style="max-width:100%; height:auto;">'
    )


def render(layout, frame):
    """RGB image of the grid, shelves, goals, and both robots."""
    rows = layout["rows"]
    cols = layout["cols"]
    image = np.full((rows * CELL, cols * CELL, 3), FLOOR, dtype=np.uint8)
    highways = layout["highways"]
    for y in range(rows):
        for x in range(cols):
            if highways[y][x]:
                _fill(image, x, y, AISLE, 1)
            y0 = y * CELL
            x0 = x * CELL
            image[y0 : y0 + 1, x0 : x0 + CELL] = GRID
            image[y0 : y0 + CELL, x0 : x0 + 1] = GRID
    for goal_x, goal_y in layout["goals"]:
        _fill(image, goal_x, goal_y, GOAL, 4)
    for shelf in frame["shelves"]:
        color = REQUESTED if shelf["requested"] else SHELF
        _fill(image, shelf["x"], shelf["y"], color, 8)
    for index, agent in enumerate(frame["agents"]):
        _fill(image, agent["x"], agent["y"], ROBOT_COLORS[index], 6)
        _paint_facing(image, agent["x"], agent["y"], Direction(agent["dir"]))
        if agent["carrying"]:
            _fill(image, agent["x"], agent["y"], CARRY, 18)
    return image


def _fill(image, x, y, color, pad):
    y0 = y * CELL + pad
    x0 = x * CELL + pad
    y1 = (y + 1) * CELL - pad
    x1 = (x + 1) * CELL - pad
    image[y0:y1, x0:x1] = color


def _paint_facing(image, x, y, direction):
    y0 = y * CELL + 8
    x0 = x * CELL + 8
    y1 = (y + 1) * CELL - 8
    x1 = (x + 1) * CELL - 8
    cx0 = x * CELL + CELL // 2 - 4
    cx1 = x * CELL + CELL // 2 + 4
    cy0 = y * CELL + CELL // 2 - 4
    cy1 = y * CELL + CELL // 2 + 4
    if direction == Direction.UP:
        image[y0 : y0 + 8, cx0:cx1] = FACING
    elif direction == Direction.DOWN:
        image[y1 - 8 : y1, cx0:cx1] = FACING
    elif direction == Direction.LEFT:
        image[cy0:cy1, x0 : x0 + 8] = FACING
    else:
        image[cy0:cy1, x1 - 8 : x1] = FACING
