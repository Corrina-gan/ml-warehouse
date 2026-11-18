"""One place to seed every random number generator, and to pick training seeds."""

import random

import numpy as np

from environment.config import EVAL_SEEDS, SEEDS, TRAIN_SEED_STRIDE


def set_seed(seed):
    """Seed python, numpy and torch (if installed) for one training run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_episode_seed(run_seed):
    """First environment seed for a run. Pass this as ``seed`` to collect_steps.

    Later episodes count up from here, so run ``s`` uses
    s * TRAIN_SEED_STRIDE and never touches another run or EVAL_SEEDS.
    """
    run_seed = int(run_seed)
    if run_seed not in SEEDS:
        raise ValueError(f"run seed {run_seed} is not in SEEDS {SEEDS}")
    start = run_seed * TRAIN_SEED_STRIDE
    if start in EVAL_SEEDS:
        raise AssertionError("training seeds overlap evaluation seeds")
    return start
