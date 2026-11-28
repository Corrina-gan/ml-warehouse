"""Draw both robots on the shared warehouse and step through one episode."""

import time
import warnings
from html import escape
from pathlib import Path

import streamlit as st

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

ROBOT_COLORS = ("#ff8c00", "#1e88e5")
FLOOR = "#f8f8f8"
AISLE = "#e6e6e6"
GRID = "#bebebe"
SHELF = "#483d8b"
REQUESTED = "#008080"
GOAL = "#3c3c3c"
FACING = "#141414"
DIRECTION_NAMES = ("up", "down", "left", "right")
# Degrees clockwise from up, in the same order.
DIRECTION_ANGLES = (0, 180, 270, 90)


ALGORITHMS = ("Q-learning", "SARSA", "REINFORCE", "PPO")
SPEEDS = ("Slow", "Normal", "Fast")
# Seconds for one step, and the steps one refresh of the grids covers. A
# refresh takes Streamlit about 0.2 seconds, so Normal and Fast send several
# steps at once and the browser moves the robots through them.
PLAYBACK = {"Slow": (0.4, 1), "Normal": (0.15, 3), "Fast": (0.05, 8)}
# Streamlit drops what it has not sent yet when the next refresh starts, and
# that was always the picture drawn last. Waiting this long sends it first.
SEND_PAUSE = 0.03


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


def _tick(before, after, speed):
    seconds, steps = PLAYBACK[speed]
    runners = st.session_state.get("warehouse_runners")
    step = int(st.session_state.get("warehouse_step", 0))
    if runners is not None:
        while _runner_length(runners) < step + steps and not _runners_finished(runners):
            _advance_runners(runners)
        before = runners["before"].episode()
        after = runners["after"].episode()
    nxt = min(step + steps, _episode_length(before, after))
    st.session_state["warehouse_step"] = nxt
    _draw(before, after, nxt - step, seconds)
    time.sleep(SEND_PAUSE)
    if nxt == step:
        # Nothing was left to play: the refresh before this one showed the last steps.
        st.session_state["warehouse_playing"] = False
        st.rerun()


def _refresh(speed):
    """Seconds between two refreshes at this speed."""
    seconds, steps = PLAYBACK[speed]
    return seconds * steps


@st.fragment(run_every=_refresh("Slow"))
def _playing_slow(before, after):
    _tick(before, after, "Slow")


@st.fragment(run_every=_refresh("Normal"))
def _playing_normal(before, after):
    _tick(before, after, "Normal")


@st.fragment(run_every=_refresh("Fast"))
def _playing_fast(before, after):
    _tick(before, after, "Fast")


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


def _draw(before, after, steps=0, seconds=0.0):
    """Both grids at the current step. With steps above 0 they play that many steps up to it."""
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
    st.html(LEGEND)
    left, right = st.columns(2)
    with left:
        _panel(
            "Before training",
            before,
            step,
            "Untrained policy. Q-learning and SARSA pick an action at random. "
            "PPO and REINFORCE sample the initial network.",
            steps,
            seconds,
        )
    with right:
        _panel(
            "After training",
            after,
            step,
            f"Greedy policy after 100,000 steps, training seed {st.session_state.get('warehouse_train_seed', 0)}.",
            steps,
            seconds,
        )


def _panel(title, episode, step, blurb, steps=0, seconds=0.0):
    frames = episode["frames"]
    frame = frames[step]
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
    st.html(_image_html(render(episode["layout"], frame, frames[:step], steps, seconds), title))
    st.caption(_frame_caption(frame))


def _frame_caption(frame):
    lines = []
    for index, agent in enumerate(frame["agents"]):
        lines.append(
            f"Robot {index + 1} is at column {agent['x']}, row {agent['y']}, "
            f"facing {DIRECTION_NAMES[agent['dir']]}, {_load_text(frame, agent)}."
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


def _load_text(frame, agent):
    if not agent["carrying"]:
        return "not carrying"
    # A carried shelf is stored on its robot's cell.
    for shelf in frame["shelves"]:
        if (shelf["x"], shelf["y"]) == (agent["x"], agent["y"]):
            if _asked_for(shelf):
                return "carrying a requested shelf"
            return "carrying a shelf that is not requested"
    return "carrying a shelf"


def _asked_for(shelf):
    return bool(shelf["requested"] or shelf.get("highlight"))


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


# The picture is HTML, so the browser itself can move the robots between two
# refreshes. Every part is one cell big, and --x and --y are its column and row.
STYLE = (
    f"<style>.wv {{ --aisle: {AISLE}; --grid: {GRID}; --floor: {FLOOR}; --goal: {GOAL}; "
    f"--facing: {FACING}; --asked: {REQUESTED}; }}"
    """
.wv { position: relative; width: min(100%, 30rem); aspect-ratio: var(--c) / var(--r); overflow: hidden;
  border-radius: .4rem; background: var(--aisle);
  background-image: linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: calc(100% / var(--c)) calc(100% / var(--r)); }
.wv > * { position: absolute; box-sizing: border-box; width: calc(100% / var(--c)); height: calc(100% / var(--r));
  left: calc(var(--x) * 100% / var(--c)); top: calc(var(--y) * 100% / var(--r)); }
.wv > .floor { background: var(--floor); border: solid var(--grid); border-width: 1px 0 0 1px; }
.wv > .goal { background: var(--goal); transform: scale(.83); }
.wv > .shelf { border-radius: 15%; }
.wv > .robot { z-index: 2; filter: drop-shadow(0 0 1px #fff) drop-shadow(0 0 1px #fff); }
.wv > .robot > .body { position: absolute; inset: 15%; border-radius: 50%; background: var(--k);
  box-shadow: 0 0 0 1px #0006; }
.wv > .robot > .nose { position: absolute; left: 34%; top: 1%; width: 32%; height: 24%;
  background: var(--facing); clip-path: polygon(50% 0, 100% 100%, 0 100%); }
.wv > .plus { z-index: 4; display: flex; align-items: center; justify-content: center; border-radius: 50%;
  background: var(--asked); color: #fff; font-weight: 700; font-size: 1.1rem; line-height: 1; }
</style>"""
)
LEGEND = (
    """<style>
.wv-key { display: flex; flex-wrap: wrap; gap: .3rem 1.5rem; font-size: .875rem; opacity: .75; }
.wv-key > span { display: inline-flex; align-items: center; gap: .45rem; }
.wv-key i { display: grid; place-items: center; flex: none; width: 1.35rem; height: 1.35rem;
  border-radius: 50%; color: #fff; font: 700 .6rem/1 sans-serif; }
.wv-key b { width: .6rem; height: .6rem; border-radius: 15%; box-shadow: 0 0 0 .1rem #fff; }
</style>"""
    '<div class="wv-key">'
    f'<span><i style="background:{ROBOT_COLORS[0]}"><b style="background:{REQUESTED}"></b></i>'
    "carrying a requested shelf to a goal</span>"
    f'<span><i style="background:{ROBOT_COLORS[0]}"><b style="background:{SHELF}"></b></i>'
    "carrying a shelf nobody asks for: a delivered one goes back to storage</span>"
    f'<span><i style="background:{REQUESTED}">+1</i>a delivery</span>'
    "</div>"
)


def _image_html(drawing, title):
    label = escape(f"{title}: warehouse grid with robot 1 in orange and robot 2 in blue")
    return f'{STYLE}<div role="img" aria-label="{label}">{drawing}</div>'


def render(layout, frame, before=(), steps=0, seconds=0.0):
    """The grid as HTML: floor, goals, shelves, and both robots.

    before holds every earlier frame of the episode, oldest first. With
    steps above 0 the picture starts that many steps back and plays forward
    to frame, taking that many seconds for each step. Otherwise it is still.
    """
    steps = min(int(steps), len(before)) if seconds > 0 else 0
    # A +1 stays over its goal for this many steps after the delivery.
    linger = max(2, round(0.8 / seconds)) if steps else 2
    recent = [*before[max(0, len(before) - steps - linger) :], frame]
    start = len(recent) - 1 - steps
    played = recent[start:]
    rules = []
    parts = []

    def add(kind, x, y, looks, inner="", fixed=""):
        rule, style = _motion(looks, seconds)
        if rule:
            rules.append(rule)
        parts.append(f'<div class="{kind}" style="--x:{x};--y:{y};{fixed}{style}">{inner}</div>')

    for y, row in enumerate(layout["highways"]):
        for x, aisle in enumerate(row):
            if not aisle:
                add("floor", x, y, [""])
    for x, y in layout["goals"]:
        add("goal", x, y, [""])

    for index, robot in enumerate(frame["agents"]):
        path = [earlier["agents"][index] for earlier in played]
        angles = _angles([point["dir"] for point in path])
        add(
            "robot",
            robot["x"],
            robot["y"],
            [
                f"transform:translate({(point['x'] - robot['x']) * 100}%,{(point['y'] - robot['y']) * 100}%) "
                f"rotate({angle}deg)"
                for point, angle in zip(path, angles)
            ],
            '<div class="body"></div><div class="nose"></div>',
            f"--k:{ROBOT_COLORS[index]};",
        )

    # Shelves come after the robots, so a shelf stays in view while it is lifted or put down.
    loaded = [{(robot["x"], robot["y"]) for robot in earlier["agents"] if robot["carrying"]} for earlier in played]
    for index, shelf in enumerate(frame["shelves"]):
        path = [earlier["shelves"][index] for earlier in played]
        add(
            "shelf",
            shelf["x"],
            shelf["y"],
            [
                _shelf_look(
                    point["x"] - shelf["x"],
                    point["y"] - shelf["y"],
                    (point["x"], point["y"]) in cells,
                    _asked_for(point),
                )
                for point, cells in zip(path, loaded)
            ],
        )

    for index, earlier in enumerate(recent):
        ages = range(start - index, len(recent) - index)
        for robot, reward in zip(earlier["agents"], earlier["rewards"] or ()):
            if reward > 0 and any(0 <= age < linger for age in ages):
                add("plus", robot["x"], robot["y"], [_plus_look(age, linger) for age in ages], "+1")

    keyframes = f"<style>{''.join(rules)}</style>" if rules else ""
    # data-step makes every refresh a new picture for the browser, so its motion always starts again.
    return (
        f'{keyframes}<div class="wv" style="--c:{layout["cols"]};--r:{layout["rows"]}" '
        f'data-step="{len(before)}">{"".join(parts)}</div>'
    )


def _shelf_look(dx, dy, carried, asked_for):
    """CSS for one shelf. On a robot it is drawn small, on top, with a white rim."""
    color = REQUESTED if asked_for else SHELF
    if carried:
        size = "scale(.4);box-shadow:0 0 0 .3rem #fff;z-index:3"
    else:
        size = "scale(.667);box-shadow:0 0 0 0 #fff;z-index:1"
    return f"background-color:{color};transform:translate({dx * 100}%,{dy * 100}%) {size}"


def _plus_look(age, linger):
    """CSS for a +1: it grows in on the delivery step, then rises and fades."""
    if age < 0:
        return "opacity:0;transform:translateY(-75%) scale(.2)"
    share = min(age / linger, 1.0)
    return f"opacity:{1 - share:g};transform:translateY({-75 - 45 * share:g}%) scale(.7)"


def _angles(directions):
    """Degrees for each facing, counted so that every turn goes the short way round."""
    angles = [DIRECTION_ANGLES[directions[0]]]
    for direction in directions[1:]:
        turn = (DIRECTION_ANGLES[direction] - angles[-1] + 180) % 360 - 180
        angles.append(angles[-1] + turn)
    return angles


def _motion(looks, seconds):
    """Turn one look per step into CSS: the @keyframes rule, if anything changes, and the style.

    The style ends on the last look. A rule is named after its own text, so
    the two grids on the page can share one but never pick up each other's.
    """
    last = looks[-1]
    if all(look == last for look in looks):
        return "", last
    span = len(looks) - 1
    stops = "".join(f"{100 * index / span:g}%{{{look}}}" for index, look in enumerate(looks))
    name = f"wv{hash(stops) & 0xFFFFFFFF:x}"
    return f"@keyframes {name}{{{stops}}}", f"{last};animation:{name} {seconds * span:g}s linear both"
