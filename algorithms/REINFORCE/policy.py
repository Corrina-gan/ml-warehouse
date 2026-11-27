"""Policy and baseline networks for REINFORCE.

The policy reads one robot's prepared observation and returns logits for the
five warehouse actions. The same weights serve both robots. The agent-ID
one-hot is already in the input, so the shared policy can still tell the
robots apart.

Its hidden widths are HIDDEN, like the PPO actor, and its input is the same
INPUT_DIM values. The one difference is the second hidden layer, which is a
GRU. A GRU keeps a memory vector from one step to the next. A robot sees only
the eight cells around it, so it has to search for shelves, and evaluation is
greedy. A greedy policy without memory repeats the same action whenever it
sees the same cells, and walks in a loop. With memory it does not.

The baseline is a separate network for V(s), the same shape as the PPO critic.
Choosing an action does not call it.
"""

import numpy as np
from torch import nn

from environment.config import FORWARD, HIDDEN, N_ACTIONS
from shared.observation import INPUT_DIM


def _orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _hidden_widths(hidden):
    hidden = tuple(int(width) for width in hidden)
    if not hidden or any(width < 1 for width in hidden):
        raise ValueError(f"hidden widths must be positive, got {hidden}")
    return hidden


class Policy(nn.Module):
    """A tanh layer, then a GRU, then a categorical policy head."""

    def __init__(self, hidden=HIDDEN, forward_bias=0.0):
        super().__init__()
        widths = _hidden_widths(hidden)
        if len(widths) != 2:
            raise ValueError(f"the recurrent policy needs exactly two hidden widths, got {widths}")
        self.encoder = nn.Sequential(
            _orthogonal(nn.Linear(INPUT_DIM, widths[0]), gain=np.sqrt(2)),
            nn.Tanh(),
        )
        self.memory = nn.GRU(widths[0], widths[1], batch_first=True)
        for name, parameter in self.memory.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.constant_(parameter, 0.0)
        # Small policy weights keep the initial logits near zero, so the
        # starting policy is decided by the bias alone: forward_bias on the
        # FORWARD logit, zero on the others.
        self.policy = _orthogonal(nn.Linear(widths[1], N_ACTIONS), gain=0.01)
        nn.init.constant_(self.policy.bias[FORWARD : FORWARD + 1], float(forward_bias))

    def forward(self, features, memory=None):
        """features: (robots, steps, INPUT_DIM) -> logits (robots, steps, N_ACTIONS), memory.

        Pass the returned memory back in to continue the same episode one
        step at a time. Leave it out to start from an empty memory.
        """
        if features.ndim != 3 or features.shape[2] != INPUT_DIM:
            raise ValueError(f"expected features of shape (robots, steps, {INPUT_DIM}), got {tuple(features.shape)}")
        remembered, memory = self.memory(self.encoder(features), memory)
        return self.policy(remembered), memory


class Baseline(nn.Module):
    """Hidden layers of shape HIDDEN, then a scalar value head."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        blocks = []
        in_features = INPUT_DIM
        for width in _hidden_widths(hidden):
            blocks.append(_orthogonal(nn.Linear(in_features, width), gain=np.sqrt(2)))
            blocks.append(nn.Tanh())
            in_features = width
        self.trunk = nn.Sequential(*blocks)
        self.value = _orthogonal(nn.Linear(in_features, 1), gain=1.0)

    def forward(self, features):
        """features: (..., INPUT_DIM) -> V(s) with shape (...)."""
        if features.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected features ending in {INPUT_DIM} values, got {tuple(features.shape)}")
        return self.value(self.trunk(features)).squeeze(-1)
