"""Actor for PPO. The actor is the policy.

It reads one robot's prepared observation and returns logits for the five
warehouse actions. The same weights serve both robots. The agent-ID one-hot
is already in the input, so the shared policy can still tell the robots apart.

The critic is a different network. Choosing an action does not call it.
"""

import numpy as np
from torch import nn

from environment.config import HIDDEN, N_ACTIONS
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


class Actor(nn.Module):
    """Hidden layers of shape HIDDEN, then a categorical policy head."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        blocks = []
        in_features = INPUT_DIM
        for width in _hidden_widths(hidden):
            blocks.append(_orthogonal(nn.Linear(in_features, width), gain=np.sqrt(2)))
            blocks.append(nn.Tanh())
            in_features = width
        self.trunk = nn.Sequential(*blocks)
        # Small policy weights keep the initial logits near zero, so each of the
        # five actions starts with probability about 1/5.
        self.policy = _orthogonal(nn.Linear(in_features, N_ACTIONS), gain=0.01)

    def forward(self, features):
        """features: (batch, INPUT_DIM) -> action logits (batch, N_ACTIONS)."""
        if features.ndim != 2 or features.shape[1] != INPUT_DIM:
            raise ValueError(f"expected features of shape (batch, {INPUT_DIM}), got {tuple(features.shape)}")
        return self.policy(self.trunk(features))
