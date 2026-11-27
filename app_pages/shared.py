import streamlit as st

from environment.config import (
    ENV_ID,
    EVAL_EPISODES,
    EVAL_SEEDS,
    GAMMA,
    HIDDEN,
    LEARNING_SIGNAL,
    MAX_STEPS,
    N_ACTIONS,
    N_AGENTS,
    OBS_SIZE,
    SEEDS,
    TOTAL_ENV_STEPS,
)
from shared import INPUT_DIM

st.caption(
    "Q-learning, SARSA, REINFORCE, and PPO import these modules. They do not each create a "
    "different warehouse or a different score."
)

with st.container(horizontal=True):
    st.metric("Environment", ENV_ID, border=True)
    st.metric("Episode length", MAX_STEPS, border=True)
    st.metric("Network input", INPUT_DIM, border=True)
    st.metric("Discount", GAMMA, border=True)

st.subheader("What to import")
st.markdown(
    """
```python
from environment import make_warehouse
from shared import (
    evaluate,
    iterate_episode,
    network_input,
    run_episode,
    collect_steps,
    team_reward,
)
```

`make_warehouse()` returns `rware-tiny-2ag-v2` with its registered settings. The warehouse observation is 71 values. `attach_task` adds the offset to the requested shelf, and the offset to a goal after that shelf is picked up. `network_input` then appends an agent-ID one-hot. `team_reward` is the sum of the two robot rewards, not the average. The training update also adds a stage bonus: +0.05 per cell closer to the requested shelf, +0.20 for picking it up, +0.05 per cell closer to a goal while carrying it, and -0.05 for a blocked forward. A missed load toggle is not charged, and circling back to a shelf does not pay the closer bonus again. A delivery is the warehouse reward of 1. Evaluation scores the warehouse sum only.
"""
)

st.subheader("How each algorithm steps")
st.dataframe(
    [
        {
            "Algorithm": "Q-learning",
            "Loop": "iterate_episode",
            "Why": "Update after each step from the best action in the next state.",
        },
        {
            "Algorithm": "SARSA",
            "Loop": "iterate_episode",
            "Why": "Update after each step, before the next action.",
        },
        {
            "Algorithm": "REINFORCE",
            "Loop": "run_episode",
            "Why": "The return is the sum of team rewards over the whole episode.",
        },
        {
            "Algorithm": "PPO",
            "Loop": "collect_steps",
            "Why": "A fixed rollout can cross the 500-step limit and keep going.",
        },
    ],
    hide_index=True,
    width="stretch",
)

st.subheader("Same comparison")
st.markdown(
    f"`evaluate()` plays {EVAL_EPISODES} greedy episodes on seeds "
    f"{EVAL_SEEDS[0]} to {EVAL_SEEDS[-1]}. The compared number is `team_return_mean`. "
    f"It also reports `success_rate` (episodes with a delivery) and "
    f"`blocked_forward_rate` (forward moves that stay in the same cell). "
    f"Training runs use seeds {', '.join(str(seed) for seed in SEEDS)} and "
    f"{TOTAL_ENV_STEPS:,} environment steps. Hidden layers are {HIDDEN}. "
    f"The learning signal is `{LEARNING_SIGNAL}`."
)
st.caption(
    "The random policy on those same evaluation seeds is the comparison floor. "
    "It is saved in results/baseline and is not a trained agent."
)
st.caption(
    f"{N_AGENTS} robots, {OBS_SIZE} observation values, {N_ACTIONS} actions. "
    "At the step limit, `is_true_terminal()` stays false, so a value function still bootstraps. "
    "REINFORCE does not bootstrap. A delivery does not end the episode."
)
