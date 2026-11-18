"""Facts about the installed ``rware-tiny-2ag-v2`` warehouse.

These match the environment registered by
https://github.com/semitable/robotic-warehouse. Algorithm files should
import them instead of hard-coding a different warehouse.
"""

ENV_ID = "rware-tiny-2ag-v2"

N_AGENTS = 2
OBS_SIZE = 71
N_ACTIONS = 5

# Registered episode length. A delivery does not end the episode.
MAX_STEPS = 500

# Same integers as rware.warehouse.Action.
NOOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
TOGGLE_LOAD = 4
ACTION_NAMES = ("No-op", "Forward", "Left", "Right", "Toggle load")

# RewardType.INDIVIDUAL: the delivering robot receives 1, the other receives 0.
DELIVERY_REWARD = 1.0

# Discount used when an algorithm learns from these rewards. Not an environment attribute.
GAMMA = 0.99

# evaluate() plays these episodes. Report that result for every algorithm.
EVAL_EPISODES = 10
# Kept away from the EDA seeds 0..99 and from training episode seeds.
EVAL_SEED_START = 10_000_000
EVAL_SEEDS = tuple(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_EPISODES))
EVAL_DETERMINISTIC = True

# Compare algorithms by environment steps, on the same run ids.
TOTAL_ENV_STEPS = 100_000
EVAL_EVERY_STEPS = 10_000
SEEDS = (0, 1, 2)

# Run `s` starts its episode seeds at s * TRAIN_SEED_STRIDE.
TRAIN_SEED_STRIDE = 100_000

# One network shape and one learning signal for every algorithm.
HIDDEN = (128, 128)
PARAM_SHARING = True
LEARNING_SIGNAL = "team"
