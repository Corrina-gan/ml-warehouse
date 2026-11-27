"""Play one saved policy against a single user-selected shelf.

reset() still draws the usual random request queue. This module replaces
that queue with one shelf from the same warehouse, then stops when that
shelf reaches a goal. The environment's own step replaces a delivered
shelf with a new random request; that replacement is cleared so the
playback stays on the shelf the user asked for. The warehouse source
and the random-request path are unchanged.
"""

import time
import warnings

from environment.config import MAX_STEPS
from environment.environment import make_warehouse
from shared.evaluation import seed_action_rng
from shared.shelf_catalog import find_shelf, shelf_code
from shared.task_features import attach_task


def play_chosen_shelf(choose, shelf_text, seed):
    """Greedy episode whose only request is the shelf named by shelf_text.

    ``choose`` takes the task-augmented observations and returns one action
    per robot. The episode ends when that shelf is delivered or at 500 steps.
    """
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"gymnasium\.utils\.passive_env_checker",
    )
    started = time.perf_counter()
    env = make_warehouse()
    try:
        env.reset(seed=int(seed))
        seed_action_rng(int(seed))
        warehouse = env.unwrapped
        target = find_shelf(warehouse, shelf_text)
        if target is None:
            known = shelf_code(warehouse.shelfs[-1].id) if warehouse.shelfs else "S000"
            raise ValueError(f"{shelf_text} is not a shelf on this grid. The last id is {known}.")
        warehouse.request_queue = [target]
        target_id = int(target.id)
        origin = {
            "id": target_id,
            "code": shelf_code(target_id),
            "column": int(target.x),
            "row": int(target.y),
        }
        # reset() already observed the random queue. Rebuild so the first
        # action sees only the selected shelf.
        observations = tuple(warehouse._make_obs(agent) for agent in warehouse.agents)
        layout = {
            "rows": int(warehouse.grid_size[0]),
            "cols": int(warehouse.grid_size[1]),
            "highways": warehouse.highways.astype(bool).tolist(),
            "goals": [(int(x), int(y)) for x, y in warehouse.goals],
        }
        frames = [_snapshot(warehouse, None, target_id)]
        team_reward = 0.0
        delivered = False
        steps = 0
        for step_index in range(1, MAX_STEPS + 1):
            actions = choose(attach_task(warehouse, observations))
            observations, rewards, terminated, truncated, _info = env.step(actions)
            team = float(sum(rewards))
            team_reward += team
            steps = step_index
            if team > 0.0:
                delivered = True
                warehouse.request_queue.clear()
            frames.append(_snapshot(warehouse, actions, target_id, rewards))
            if delivered or terminated or truncated:
                break
        return {
            "layout": layout,
            "frames": frames,
            "delivered": delivered,
            "steps": steps,
            "reward": team_reward,
            "seconds": time.perf_counter() - started,
            "shelf": origin,
        }
    finally:
        env.close()


def _snapshot(warehouse, actions, target_id, rewards=None):
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
                "highlight": int(shelf.id) == int(target_id),
            }
            for shelf in warehouse.shelfs
        ],
        "actions": None if actions is None else tuple(int(action) for action in actions),
        "rewards": None if rewards is None else tuple(float(reward) for reward in rewards),
    }
