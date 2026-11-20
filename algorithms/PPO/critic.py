"""Critic for PPO. The critic estimates the value V(s).

V(s) is the expected discounted team return from this robot's observation.
The actor chooses the action. The critic does not. Its value is the baseline
in the advantage, so a delivery is a large advantage when the critic did not
already expect that return. The critic is trained by squared error against
the return produced by generalized advantage estimation.

The input matches the actor: 71 observation values plus the agent-ID one-hot.
The same critic weights serve both robots.
"""

import numpy as np
from torch import nn

from environment.config import HIDDEN
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


class Critic(nn.Module):
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
        """features: (batch, INPUT_DIM) -> V(s) with shape (batch,)."""
        if features.ndim != 2 or features.shape[1] != INPUT_DIM:
            raise ValueError(f"expected features of shape (batch, {INPUT_DIM}), got {tuple(features.shape)}")
        return self.value(self.trunk(features)).squeeze(-1)
