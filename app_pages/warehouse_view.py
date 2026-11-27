"""Draw both robots on the shared warehouse and step through one episode."""

import base64
import io
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from rware.warehouse import Direction

from environment.config import ACTION_NAMES, MAX_STEPS, SEEDS
from environment.environment import make_warehouse
from shared.evaluation import seed_action_rng
from shared.task_features import attach_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = {
    "Q-learning": ("qlearning", "qlearning.npz"),
    "SARSA": ("sarsa", "sarsa.npz"),
    "REINFORCE": ("reinforce", "reinforce.pt"),
    "PPO": ("ppo", "ppo.pt"),
}

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


ALGORITHMS = ("Q-learning", "SARSA", "REINFORCE", "PPO")
SPEEDS = ("Slow", "Normal", "Fast")


def show_warehouse():
    """Algorithm, speed, seed, play, and step controls for one episode."""
    st.session_state.setdefault("warehouse_step", 0)
    st.session_state.setdefault("warehouse_playing", False)
    st.session_state.setdefault("warehouse_seed", 0)
    st.session_state.setdefault("warehouse_algorithm", "PPO")
    st.session_state.setdefault("warehouse_speed", "Normal")
    if st.session_state["warehouse_algorithm"] not in ALGORITHMS:
        st.session_state["warehouse_algorithm"] = "PPO"
    if st.session_state["warehouse_speed"] not in SPEEDS:
        st.session_state["warehouse_speed"] = "Normal"

    def reset_view():
        st.session_state["warehouse_step"] = 0
        st.session_state["warehouse_playing"] = False
        st.session_state["warehouse_pending_steps"] = 0
        _close_runners(st.session_state.get("warehouse_runners"))
        st.session_state["warehouse_runners"] = None

    with st.container(border=True):
        st.subheader("B. Algorithm selection")
        st.caption(
            "The same episode plays twice: the left grid is the untrained policy, "
            "and the right grid is the greedy policy after 100,000 steps."
        )
        st.segmented_control(
            "Select algorithm",
            ALGORITHMS,
            key="warehouse_algorithm",
            required=True,
            width="stretch",
            on_change=reset_view,
            help="Before training samples the untrained policy. After training plays the saved greedy policy.",
        )
        st.segmented_control(
            "Simulation speed",
            SPEEDS,
            key="warehouse_speed",
            required=True,
            width="stretch",
            help="Slow, normal, and fast change how quickly the episode advances.",
        )
        with st.container(horizontal=True):
            st.metric("Selected algorithm", st.session_state["warehouse_algorithm"], border=True)
            st.metric("Simulation speed", st.session_state["warehouse_speed"], border=True)
    st.number_input(
        "Episode seed",
        min_value=0,
        max_value=99,
        key="warehouse_seed",
        on_change=reset_view,
        help="The testing episode. Seed 0 is the sample episode.",
    )
    algorithm = st.session_state["warehouse_algorithm"]
    checkpoint = _checkpoint(algorithm)
    if checkpoint is None:
        st.info(
            f"{algorithm} has no saved 100,000-step policy, so the after-training grid cannot play.",
            icon=":material/info:",
        )
        return
    training_seed, weights_path = checkpoint
    st.session_state["warehouse_train_seed"] = training_seed
    episode_seed = int(st.session_state["warehouse_seed"])
    key = (algorithm, episode_seed, str(weights_path), weights_path.stat().st_mtime_ns)
    runners = st.session_state.get("warehouse_runners")
    if runners is not None and runners["key"] != key:
        _close_runners(runners)
        runners = None
        st.session_state["warehouse_runners"] = None
        st.session_state["warehouse_step"] = 0
        st.session_state["warehouse_playing"] = False
    pending = int(st.session_state.pop("warehouse_pending_steps", 0))
    if st.session_state["warehouse_playing"] or pending or runners is not None:
        if runners is None:
            try:
                runners = _open_runners(algorithm, episode_seed, str(weights_path), key)
            except ModuleNotFoundError as exc:
                if exc.name != "torch":
                    raise
                st.error(f"{algorithm} needs PyTorch in the environment that runs this app.")
                return
            st.session_state["warehouse_runners"] = runners
        for _ in range(pending):
            if not _runners_finished(runners):
                _advance_runners(runners)
            st.session_state["warehouse_step"] = _runner_length(runners)
    if runners is None:
        opening = _opening_episode(episode_seed)
        before = after = opening
    else:
        before = runners["before"].episode()
        after = runners["after"].episode()
    length = min(len(before["frames"]), len(after["frames"])) - 1
    st.session_state["warehouse_step"] = min(int(st.session_state["warehouse_step"]), length)

    playing = st.session_state["warehouse_playing"]
    step_now = int(st.session_state["warehouse_step"])
    with st.container(horizontal=True):
        st.button(
            "Play",
            type="primary",
            icon=":material/play_arrow:",
            disabled=playing,
            on_click=_play,
            args=(length,),
        )
        st.button(
            "Pause",
            icon=":material/pause:",
            disabled=not playing,
            on_click=_pause,
        )
        st.button(
            "Reset",
            icon=":material/replay:",
            disabled=step_now == 0 and not playing,
            on_click=_reset,
        )
        st.button("Step", icon=":material/skip_next:", on_click=_step_once, args=(length,))

    if st.session_state["warehouse_playing"]:
        _players()[st.session_state["warehouse_speed"]](before, after)
    else:
        _paused(before, after)


def _tick(before, after):
    runners = st.session_state.get("warehouse_runners")
    step = int(st.session_state.get("warehouse_step", 0))
    if runners is not None and step >= _runner_length(runners) and not _runners_finished(runners):
        _advance_runners(runners)
        before = runners["before"].episode()
        after = runners["after"].episode()
    length = _episode_length(before, after)
    nxt = min(step + 1, length)
    st.session_state["warehouse_step"] = nxt
    if runners is not None and _runners_finished(runners) and nxt >= length:
        st.session_state["warehouse_playing"] = False
        st.rerun()
    _draw(before, after)


@st.fragment(run_every=0.4)
def _playing_slow(before, after):
    _tick(before, after)


@st.fragment(run_every=0.15)
def _playing_normal(before, after):
    _tick(before, after)


@st.fragment(run_every=0.05)
def _playing_fast(before, after):
    _tick(before, after)


def _players():
    return {"Slow": _playing_slow, "Normal": _playing_normal, "Fast": _playing_fast}


@st.fragment
def _paused(before, after):
    _draw(before, after)


def _episode_length(before, after):
    return min(len(before["frames"]), len(after["frames"])) - 1


def _play(length):
    if length > 0 and int(st.session_state.get("warehouse_step", 0)) >= length:
        st.session_state["warehouse_step"] = 0
    st.session_state["warehouse_playing"] = True


def _pause():
    st.session_state["warehouse_playing"] = False


def _reset():
    st.session_state["warehouse_step"] = 0
    st.session_state["warehouse_playing"] = False


def _step_once(length):
    st.session_state["warehouse_playing"] = False
    step = int(st.session_state.get("warehouse_step", 0))
    if step < length:
        st.session_state["warehouse_step"] = step + 1
    else:
        st.session_state["warehouse_pending_steps"] = 1


def _draw(before, after):
    length = _episode_length(before, after)
    st.slider("Step", min_value=0, max_value=max(length, 1), key="warehouse_step")
    step = min(int(st.session_state["warehouse_step"]), length)
    st.subheader("C. Live simulation")
    st.caption(
        f"{st.session_state['warehouse_algorithm']}, episode seed {st.session_state['warehouse_seed']}, "
        f"step {step} of {length}. "
        f"{'Press play to run the episode. ' if length == 0 else ''}"
        "Both grids start from that same seed. "
        "Orange is robot 1. Blue is robot 2. The black mark is the direction they face."
    )
    left, right = st.columns(2)
    with left:
        _panel(
            "Before training",
            before,
            step,
            "Untrained policy. Q-learning and SARSA pick an action at random. "
            "PPO and REINFORCE sample the initial network.",
        )
    with right:
        _panel(
            "After training",
            after,
            step,
            f"Greedy policy after 100,000 steps, training seed {st.session_state.get('warehouse_train_seed', 0)}.",
        )


def _panel(title, episode, step, blurb):
    frame = episode["frames"][step]
    total_reward, deliveries = _score_so_far(episode, step)
    robot_1, robot_2 = frame["agents"]
    st.markdown(f"**{title}**")
    st.caption(blurb)
    with st.container(horizontal=True):
        st.metric("Reward", _count_text(total_reward), border=True)
        st.metric("Deliveries", deliveries, border=True)
    with st.container(horizontal=True):
        st.metric("Robot 1", f"{robot_1['x']}, {robot_1['y']}", help="Column, row.", border=True)
        st.metric("Robot 2", f"{robot_2['x']}, {robot_2['y']}", help="Column, row.", border=True)
    st.html(_image_html(render(episode["layout"], frame), title))
    st.caption(_frame_caption(frame))


def _frame_caption(frame):
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
    return " ".join(lines)


def _score_so_far(episode, step):
    """Team reward and deliveries from the start through this step."""
    total = 0.0
    deliveries = 0
    for frame in episode["frames"][1 : step + 1]:
        for reward in frame["rewards"] or ():
            total += float(reward)
            if reward > 0:
                deliveries += 1
    return total, deliveries


def _count_text(value):
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.2f}"


def _checkpoint(algorithm):
    """The trained policy. A retrain in final_runs is used when it exists."""
    folder, filename = POLICY_FILES[algorithm]
    roots = (
        PROJECT_ROOT / "results" / "final_runs" / folder,
        PROJECT_ROOT / "results" / "official" / folder,
    )
    for root in roots:
        for seed in SEEDS:
            path = root / f"seed_{seed}" / filename
            if path.is_file():
                return seed, path
    return None


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_agent(algorithm, path):
    """The greedy policy. Models stay in memory; episode frames do not."""
    if algorithm == "Q-learning":
        from algorithms.QLEARNING.qlearning import QLearningAgent

        agent = QLearningAgent()
        agent.load(path)
        return agent
    if algorithm == "SARSA":
        from algorithms.SARSA.sarsa import SARSAAgent

        agent = SARSAAgent()
        agent.load(path)
        return agent
    if algorithm == "REINFORCE":
        from algorithms.REINFORCE.reinforce import load_agent

        return load_agent(path)
    if algorithm == "PPO":
        from algorithms.PPO.ppo import load_agent

        return load_agent(path)
    raise ValueError(f"unknown algorithm {algorithm}")


def _untrained_agent(algorithm, seed):
    """A policy with no learned weights. Training starts from this agent."""
    if algorithm == "Q-learning":
        from algorithms.QLEARNING.qlearning import QLearningAgent

        return QLearningAgent(seed=int(seed))
    if algorithm == "SARSA":
        from algorithms.SARSA.sarsa import SARSAAgent

        return SARSAAgent(seed=int(seed))
    if algorithm == "REINFORCE":
        from algorithms.REINFORCE.reinforce import REINFORCEAgent

        return REINFORCEAgent()
    if algorithm == "PPO":
        from algorithms.PPO.ppo import PPOAgent

        return PPOAgent()
    raise ValueError(f"unknown algorithm {algorithm}")


@st.cache_data(show_spinner=False, max_entries=8)
def _opening_episode(seed):
    """The starting grid only. The episode runs later, one step at a time."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    env = make_warehouse()
    try:
        env.reset(seed=int(seed))
        warehouse = env.unwrapped
        return {"layout": _layout(warehouse), "frames": [_snapshot(warehouse, None)]}
    finally:
        env.close()


def _open_runners(algorithm, seed, weights_path, key):
    before = _start_runner(algorithm, seed, weights_path, "before")
    try:
        after = _start_runner(algorithm, seed, weights_path, "after")
    except Exception:
        before.close()
        raise
    return {"key": key, "before": before, "after": after}


def _start_runner(algorithm, seed, weights_path, stage):
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    env = make_warehouse()
    observations, _info = env.reset(seed=int(seed))
    seed_action_rng(int(seed))
    warehouse = env.unwrapped
    if stage == "before":
        choose = _untrained_agent(algorithm, seed).choose_actions
    else:
        choose = _load_agent(algorithm, weights_path).act_greedy
    return _Runner(env, choose, observations, _layout(warehouse), [_snapshot(warehouse, None)])


def _layout(warehouse):
    return {
        "rows": int(warehouse.grid_size[0]),
        "cols": int(warehouse.grid_size[1]),
        "highways": warehouse.highways.astype(bool).tolist(),
        "goals": [(int(x), int(y)) for x, y in warehouse.goals],
    }


class _Runner:
    """One warehouse kept open so play can advance a single step."""

    def __init__(self, env, choose, observations, layout, frames):
        self.env = env
        self.choose = choose
        self.observations = observations
        self.layout = layout
        self.frames = frames
        self.done = False

    def episode(self):
        return {"layout": self.layout, "frames": self.frames}

    def advance(self):
        if self.done:
            return
        warehouse = self.env.unwrapped
        actions = self.choose(attach_task(warehouse, self.observations))
        self.observations, rewards, terminated, truncated, _info = self.env.step(actions)
        self.frames.append(_snapshot(warehouse, actions, rewards))
        if terminated or truncated or len(self.frames) - 1 >= MAX_STEPS:
            self.done = True

    def close(self):
        self.env.close()


def _advance_runners(runners):
    runners["before"].advance()
    runners["after"].advance()


def _runners_finished(runners):
    return runners["before"].done and runners["after"].done


def _runner_length(runners):
    return min(len(runners["before"].frames), len(runners["after"].frames)) - 1


def _close_runners(runners):
    if not runners:
        return
    runners["before"].close()
    runners["after"].close()


@st.cache_data(show_spinner=False, max_entries=16)
def record_episode(algorithm, seed, weights_path, weights_mtime, stage):
    """One episode. Before training samples the untrained policy. After training is greedy."""
    del weights_mtime
    if stage not in ("before", "after"):
        raise ValueError(f"stage must be before or after, got {stage}")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    env = make_warehouse()
    try:
        observations, _info = env.reset(seed=int(seed))
        seed_action_rng(int(seed))
        warehouse = env.unwrapped
        layout = {
            "rows": int(warehouse.grid_size[0]),
            "cols": int(warehouse.grid_size[1]),
            "highways": warehouse.highways.astype(bool).tolist(),
            "goals": [(int(x), int(y)) for x, y in warehouse.goals],
        }
        frames = [_snapshot(warehouse, None)]
        if stage == "before":
            agent = _untrained_agent(algorithm, seed)
            choose = agent.choose_actions
        else:
            agent = _load_agent(algorithm, weights_path)
            choose = agent.act_greedy
        for _step in range(MAX_STEPS):
            actions = choose(attach_task(warehouse, observations))
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


def _image_html(array, title):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<img src="data:image/png;base64,'
        f'{encoded}" alt="{title}: warehouse grid with robot 1 in orange and robot 2 in blue" '
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
        color = REQUESTED if shelf["requested"] or shelf.get("highlight") else SHELF
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
