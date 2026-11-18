"""Turn the two robot observations into the one network input every algorithm uses.

With parameter sharing, one network serves both robots. An agent-ID one-hot
is appended so the network can tell them apart. Input row i is robot i:
[71 observation values | one-hot of length N_AGENTS].
"""

import numpy as np

from environment.config import N_AGENTS, OBS_SIZE

INPUT_DIM = OBS_SIZE + N_AGENTS

_AGENT_IDS = np.eye(N_AGENTS, dtype=np.float32)


def stack_observations(observations):
    """Tuple of N_AGENTS vectors -> array (N_AGENTS, OBS_SIZE), float32.

    The warehouse writes every value itself. There is no missing entry to
    impute and no row to drop. A NaN or an infinity is rejected here so a
    broken observation never reaches a network.
    """
    stacked = np.stack([np.asarray(obs, dtype=np.float32) for obs in observations])
    if stacked.shape != (N_AGENTS, OBS_SIZE):
        raise ValueError(f"expected {(N_AGENTS, OBS_SIZE)}, got {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise ValueError("observation contains NaN or infinity")
    return stacked


def network_input(observations):
    """Prepared state for every algorithm: raw observation plus agent id.

    Coordinates stay in grid units. The registered warehouse does not
    normalise them, and scaling only x and y would make this vector differ
    from the environment the EDA measured. Sensor bits are already 0 or 1.
    The agent-ID one-hot is the only field added on this side of the step.
    """
    return np.concatenate([stack_observations(observations), _AGENT_IDS], axis=1)


def batch_network_input(observation_tuples):
    """List of T observation tuples -> array (T, N_AGENTS, INPUT_DIM).

    Flatten the first two axes with reshape(-1, INPUT_DIM) to feed a network.
    Row t * N_AGENTS + i is robot i at step t.
    """
    return np.stack([network_input(observations) for observations in observation_tuples])
