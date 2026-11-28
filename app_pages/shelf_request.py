"""Ask for one shelf by ID and play a saved policy."""

import time

import streamlit as st

from app_pages.results_data import best_saved_policy, build_report
from app_pages.warehouse_view import (
    PLAYBACK,
    SEND_PAUSE,
    SPEEDS,
    _checkpoint,
    _count_text,
    _frame_caption,
    _image_html,
    _load_agent,
    _refresh,
    _score_so_far,
    render,
)
from environment.config import MAX_STEPS
from shared.chosen_shelf import play_chosen_shelf
from shared.shelf_catalog import parse_shelf_code, read_shelves, shelf_code

CHOICES = ("Best", "Q-learning", "SARSA", "REINFORCE", "PPO")


def show_shelf_request():
    """Shelf ID, algorithm, playback, and the delivery result."""
    st.session_state.setdefault("shelf_submitted", False)
    st.session_state.setdefault("shelf_request_code", "")
    st.session_state.setdefault("shelf_step", 0)
    st.session_state.setdefault("shelf_playing", False)
    st.session_state.setdefault("shelf_request_error", "")

    catalog = _catalog()
    valid_ids = {row["id"] for row in catalog}
    last_code = catalog[-1]["code"] if catalog else "S000"

    with st.container(border=True):
        st.subheader("Shelf request")
        st.caption(
            "Enter a shelf ID from this warehouse. The request queue becomes that "
            "one shelf. Both robots then use the selected greedy policy. Saved "
            "models and the random request mode are left as they are."
        )
        st.segmented_control(
            "Select algorithm",
            CHOICES,
            default="Best",
            key="shelf_algorithm",
            width="stretch",
            on_change=_reset_playback,
        )
        st.segmented_control(
            "Simulation speed",
            SPEEDS,
            default="Normal",
            key="shelf_speed",
            width="stretch",
        )
        st.text_input(
            "Shelf ID",
            value="S001",
            key="shelf_id_input",
            placeholder="S001",
            help=f"IDs on this grid run from S001 to {last_code}. S001, s001, and 1 are the same shelf.",
        )
        st.number_input(
            "Episode seed",
            min_value=0,
            max_value=99,
            value=0,
            key="shelf_episode_seed",
            help="Places the two robots. The shelf ID does not change with the seed.",
        )
        st.button(
            "Request this shelf",
            type="primary",
            icon=":material/inventory_2:",
            on_click=_submit_shelf,
            args=(valid_ids, last_code),
            key="shelf_submit",
        )

    error = st.session_state.get("shelf_request_error") or ""
    if error:
        st.error(error)

    with st.expander("Shelf IDs on this grid"):
        st.dataframe(
            [
                {"Shelf ID": row["code"], "Column": row["column"], "Row": row["row"]}
                for row in catalog
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "These IDs are the shelf objects created by rware-tiny-2ag-v2. "
            "There is no parcel list and no external dataset."
        )

    if not st.session_state["shelf_submitted"]:
        st.info(
            "Request a shelf ID to run the delivery. Random mode still draws its own queue.",
            icon=":material/info:",
        )
        return

    choice = st.session_state["shelf_algorithm"]
    algorithm, best = _resolve(choice)
    if algorithm is None:
        st.info(
            "Best needs a saved evaluation, and none was found.",
            icon=":material/info:",
        )
        return
    if best is not None:
        _best_caption(best)

    checkpoint = _checkpoint(algorithm)
    if checkpoint is None:
        st.info(
            f"{algorithm} has no saved 100,000-step policy, so this request cannot play.",
            icon=":material/info:",
        )
        return
    training_seed, weights_path = checkpoint
    code = st.session_state["shelf_request_code"]
    episode_seed = int(st.session_state["shelf_episode_seed"])
    try:
        episode = _record(
            algorithm,
            code,
            episode_seed,
            str(weights_path),
            weights_path.stat().st_mtime_ns,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        st.error(f"{algorithm} needs PyTorch in the environment that runs this app.")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    _outcome(episode, algorithm, training_seed, episode_seed)
    length = len(episode["frames"]) - 1
    st.session_state["shelf_step"] = min(int(st.session_state["shelf_step"]), length)
    playing = st.session_state["shelf_playing"]
    step_now = int(st.session_state["shelf_step"])
    with st.container(horizontal=True):
        st.button(
            "Play",
            type="primary",
            icon=":material/play_arrow:",
            disabled=playing,
            on_click=_play,
            args=(length,),
            key="shelf_play",
        )
        st.button(
            "Pause",
            icon=":material/pause:",
            disabled=not playing,
            on_click=_pause,
            key="shelf_pause",
        )
        st.button(
            "Reset",
            icon=":material/replay:",
            disabled=step_now == 0 and not playing,
            on_click=_reset_playback,
            key="shelf_reset",
        )
        st.button(
            "Step",
            icon=":material/skip_next:",
            on_click=_step_once,
            args=(length,),
            key="shelf_step_button",
        )
    if st.session_state["shelf_playing"]:
        _players()[st.session_state["shelf_speed"]](episode)
    else:
        _paused(episode)


def _submit_shelf(valid_ids, last_code):
    code = _accepted_code(st.session_state.get("shelf_id_input", ""), valid_ids)
    if code is None:
        entered = st.session_state.get("shelf_id_input", "")
        st.session_state["shelf_request_error"] = (
            f"Enter a shelf ID from S001 to {last_code}. {entered!r} is not one of them."
        )
        return
    st.session_state["shelf_request_error"] = ""
    st.session_state["shelf_submitted"] = True
    st.session_state["shelf_request_code"] = code
    _reset_playback()


def _accepted_code(text, valid_ids):
    number = parse_shelf_code(text)
    if number is None or number not in valid_ids:
        return None
    return shelf_code(number)


def _resolve(choice):
    if choice != "Best":
        return choice, None
    best = best_saved_policy(_evaluation())
    if best is None:
        return None, None
    return best["name"], best


def _best_caption(best):
    names = best["tied"]
    if len(names) == 1:
        sentence = (
            f"Best is {best['name']}. Its mean team return on the saved evaluation "
            f"is {best['mean_return']:.2f}, with a success rate of {best['success_rate']:.0%}."
        )
    else:
        listed = ", ".join(names)
        sentence = (
            f"{listed} share the highest mean team return of {best['mean_return']:.2f}. "
            f"The controller uses {best['name']}."
        )
    st.caption(sentence + " The saved weights are only read.")


@st.cache_data(show_spinner=False, max_entries=2)
def _evaluation():
    return build_report()


@st.cache_data(show_spinner=False, max_entries=2)
def _catalog():
    return read_shelves()


@st.cache_data(show_spinner=False, max_entries=16)
def _record(algorithm, code, seed, weights_path, weights_mtime):
    del weights_mtime
    agent = _load_agent(algorithm, weights_path)
    return play_chosen_shelf(agent.act_greedy, code, seed)


def _outcome(episode, algorithm, training_seed, episode_seed):
    shelf = episode["shelf"]
    status = "Delivered" if episode["delivered"] else "Not delivered"
    with st.container(horizontal=True):
        st.metric("Shelf", shelf["code"], border=True)
        st.metric("Algorithm", algorithm, border=True)
        st.metric("Reward", _count_text(episode["reward"]), border=True)
        st.metric("Steps", episode["steps"], border=True)
        st.metric("Completion time", _seconds(episode["seconds"]), border=True)
        st.metric("Result", status, border=True)
    if episode["delivered"]:
        detail = (
            f"{shelf['code']} started at column {shelf['column']}, row {shelf['row']} "
            f"and reached a goal in {episode['steps']} steps."
        )
    else:
        detail = (
            f"{shelf['code']} started at column {shelf['column']}, row {shelf['row']} "
            f"and was still not on a goal after {episode['steps']} steps."
        )
    st.caption(
        f"{detail} Completion time is the wall-clock time of that run. "
        f"Playback speed does not change it. Training seed {training_seed}, "
        f"episode seed {episode_seed}. The limit is {MAX_STEPS} steps. "
        "No extra training is required: the saved policy already follows the "
        "nearest shelf in the request queue, and this request puts one shelf there."
    )


def _seconds(value):
    return f"{float(value):.2f} s"


def _reset_playback():
    st.session_state["shelf_step"] = 0
    st.session_state["shelf_playing"] = False


def _play(length):
    if int(st.session_state.get("shelf_step", 0)) >= length:
        st.session_state["shelf_step"] = 0
    st.session_state["shelf_playing"] = True


def _pause():
    st.session_state["shelf_playing"] = False


def _step_once(length):
    st.session_state["shelf_playing"] = False
    st.session_state["shelf_step"] = min(int(st.session_state.get("shelf_step", 0)) + 1, length)


def _tick(episode, speed):
    seconds, steps = PLAYBACK[speed]
    length = len(episode["frames"]) - 1
    step = int(st.session_state.get("shelf_step", 0))
    nxt = min(step + steps, length)
    st.session_state["shelf_step"] = nxt
    _draw(episode, nxt - step, seconds)
    time.sleep(SEND_PAUSE)
    if nxt == step:
        # Nothing was left to play: the refresh before this one showed the last steps.
        st.session_state["shelf_playing"] = False
        st.rerun()


@st.fragment(run_every=_refresh("Slow"))
def _playing_slow(episode):
    _tick(episode, "Slow")


@st.fragment(run_every=_refresh("Normal"))
def _playing_normal(episode):
    _tick(episode, "Normal")


@st.fragment(run_every=_refresh("Fast"))
def _playing_fast(episode):
    _tick(episode, "Fast")


def _players():
    return {"Slow": _playing_slow, "Normal": _playing_normal, "Fast": _playing_fast}


@st.fragment
def _paused(episode):
    _draw(episode)


def _draw(episode, steps=0, seconds=0.0):
    length = len(episode["frames"]) - 1
    st.slider("Step", min_value=0, max_value=max(length, 1), key="shelf_step")
    step = min(int(st.session_state["shelf_step"]), length)
    frame = episode["frames"][step]
    total_reward, deliveries = _score_so_far(episode, step)
    highlighted = next(shelf for shelf in frame["shelves"] if shelf.get("highlight"))
    st.subheader("Delivery")
    st.caption(
        f"{episode['shelf']['code']} is the teal shelf, now at column {highlighted['x']}, "
        f"row {highlighted['y']}. Step {step} of {length}. "
        f"Reward so far {_count_text(total_reward)}, deliveries {deliveries}. "
        "Orange is robot 1. Blue is robot 2."
    )
    st.html(
        _image_html(
            render(episode["layout"], frame, episode["frames"][:step], steps, seconds),
            episode["shelf"]["code"],
        )
    )
    st.caption(_frame_caption(frame))
