"""REINFORCE on the shared warehouse.

REINFORCE is the Monte Carlo policy gradient. The policy plays a whole episode
by sampling its actions. Afterwards every action is weighted by its return,
the discounted reward that followed it, and the policy is moved toward the
actions with high returns. Nothing is bootstrapped: a return is summed to the
end of the 500-step episode, and no value estimate stands in for reward that
has not arrived. Q-learning, SARSA, and PPO all bootstrap.

One policy is shared by both robots. Each robot learns from its own
transition.learning_rewards, the warehouse reward plus the stage bonus in
shared.learning_reward. That is the same reward Q-learning, SARSA, and PPO
train on, and this agent has no setting that changes it. It also sees the
same observation, and shared.action_mask closes the same blocked forward
moves. The saved evaluation score counts deliveries only.

The textbook update alone learns too slowly for 200 episodes, so this agent
adds five things:

1. Forward start. The policy begins with a preference for walking forward
   (settings.forward_bias), so it covers the warehouse and meets the shared
   rewards from the first episode.
2. Baseline. A value network predicts the return of a state. The update uses
   return minus prediction, which is far less noisy than the return alone.
   The baseline is fitted to the returns themselves, so it bootstraps nothing.
3. Entropy bonus. It keeps the policy trying different actions early on, and
   fades so the policy is decisive when it is scored greedily.
4. Replay. The episodes with the most deliveries are kept and learned from
   again after every update.
5. Memory. The policy network keeps a memory vector between steps, see
   policy.py.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from algorithms.REINFORCE.config import REINFORCESettings
from algorithms.REINFORCE.policy import Baseline, Policy
from environment.config import FORWARD, MAX_STEPS, N_ACTIONS, NOOP, PARAM_SHARING
from shared.action_mask import blocked_forward, mask_forward_logits
from shared.observation import (
    INPUT_DIM,
    batch_network_input,
    network_input,
    scale_destination_offsets,
)


def discounted_returns(rewards, gamma):
    """G_t = r_t + gamma * G_(t+1), worked backwards from the last step.

    rewards: (steps, columns). Each column is discounted on its own.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 2 or rewards.shape[0] < 1:
        raise ValueError(f"expected (steps, columns) rewards, got {rewards.shape}")
    if not np.isfinite(rewards).all():
        raise ValueError("rewards contain NaN or infinity")
    returns = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[step] + gamma * running
        returns[step] = running
    return returns


@dataclass(frozen=True)
class EpisodeData:
    """One finished episode as tensors. The first axis is the robot, the second the step."""

    features: torch.Tensor  # (robots, steps, INPUT_DIM), scaled for the networks
    actions: torch.Tensor  # (robots, steps)
    blocked: torch.Tensor  # (robots, steps), True where the shared mask closed FORWARD
    returns: torch.Tensor  # (robots, steps), return of each robot's learning reward
    changed: torch.Tensor  # (robots, steps), 1 where the action changed that robot's state
    deliveries: float  # shelves the team really delivered in the episode


class REINFORCEAgent:
    """Samples actions for training and plays greedy actions for evaluate().

    The policy remembers things during an episode, so call reset_episode()
    before each one. run_episode() and evaluate() take it as on_episode_start.
    A caller that only has act_greedy, such as the app or the ensemble vote,
    is covered too: the agent notices a new episode by itself.
    """

    def __init__(self, settings=None):
        if not PARAM_SHARING:
            raise ValueError("this REINFORCE agent uses one shared network for both robots")

        self.settings = REINFORCESettings() if settings is None else settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy(self.settings.hidden, self.settings.forward_bias).to(self.device)
        self.baseline = Baseline(self.settings.hidden).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.settings.learning_rate,
            eps=self.settings.adam_eps,
        )
        # A rate of 0 means no baseline. The optimizer still needs a positive rate to exist.
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(),
            lr=self.settings.baseline_rate or 1e-3,
            eps=self.settings.adam_eps,
        )
        self.replay = []
        self.reset_episode()

    def reset_episode(self, _observations=None):
        """Forget the last episode: empty memory."""
        self._memory = None
        self._steps = 0
        self._position = None

    def choose_actions(self, observations):
        """Sample one action per robot. run_episode calls this during training."""
        actions = Categorical(logits=self._step_logits(observations)).sample()
        return tuple(int(action) for action in actions.tolist())

    def act_greedy(self, observations):
        """Argmax of the action logits. evaluate() requires this greedy policy."""
        actions = torch.argmax(self._step_logits(observations), dim=-1)
        return tuple(int(action) for action in actions.tolist())

    def entropy_coefficient(self, env_steps):
        """Linear fall from entropy_coef_start at step 0 to entropy_coef_end at the budget."""
        start = float(self.settings.entropy_coef_start)
        end = float(self.settings.entropy_coef_end)
        return start + (end - start) * self._progress(env_steps)

    def update(self, episodes, env_steps=0):
        """One REINFORCE update on finished episodes. Returns the mean losses.

        env_steps is how far through the run these episodes start. It selects
        the learning rate and the entropy coefficient for this update.
        """
        if not episodes:
            raise ValueError("REINFORCE needs at least one finished episode")
        settings = self.settings
        batch = [self._prepare(episode) for episode in episodes]
        features = torch.cat([data.features for data in batch])
        actions = torch.cat([data.actions for data in batch])
        blocked = torch.cat([data.blocked for data in batch])
        returns = torch.cat([data.returns for data in batch])

        # Large steps and a strong entropy bonus early, small steps and a weak
        # one late. The learning rate follows half a cosine down to a tenth.
        progress = self._progress(env_steps)
        learning_rate = settings.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in self.policy_optimizer.param_groups:
            group["lr"] = learning_rate
        entropy_coef = self.entropy_coefficient(env_steps)

        # Advantage: how much better the episode went from here than expected.
        advantages = self._advantages(features, returns)
        if float(returns.std()) > 1e-6:
            # Same scale in every batch. A batch with no reward at all is left
            # as it is, because rescaling would turn baseline error into a signal.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # The REINFORCE step: raise the log-probability of each action in
        # proportion to its advantage.
        self.policy.train()
        distribution = self._distribution(features, blocked)
        policy_loss = -(distribution.log_prob(actions) * advantages).mean()
        entropy = distribution.entropy().mean()
        self._step(self.policy_optimizer, self.policy, policy_loss - entropy_coef * entropy)

        baseline_loss = self._fit_baseline(features, returns)
        self._remember(batch)
        self._replay(entropy_coef)

        return {
            "policy_loss": float(policy_loss.item()),
            "baseline_loss": baseline_loss,
            "entropy": float(entropy.item()),
            "entropy_coef": float(entropy_coef),
            "replay_episodes": len(self.replay),
        }

    def _progress(self, env_steps):
        total = float(self.settings.total_env_steps)
        if total <= 0.0:
            raise ValueError("total_env_steps must be positive")
        return min(max(float(env_steps), 0.0), total) / total

    def _step_logits(self, observations):
        """Logits for this step, (N_AGENTS, N_ACTIONS), with a blocked forward closed.

        Advances the memory by one step, so it is called once per environment step.
        """
        raw = network_input(observations)
        # A robot moves at most one cell per step, and an episode has
        # MAX_STEPS steps. A bigger jump, or that many steps, means a new
        # episode has started, so it begins with an empty memory even when
        # the caller has no reset_episode() hook.
        position = raw[:, :2].copy()
        jumped = self._position is not None and float(np.abs(position - self._position).sum(axis=1).max()) > 1.0
        if jumped or self._steps >= MAX_STEPS:
            self.reset_episode()
        self._position = position
        self._steps += 1
        blocked = torch.as_tensor(blocked_forward(raw), device=self.device)
        self.policy.eval()
        with torch.no_grad():
            # Each robot is a sequence of one step, continued from its memory.
            step = torch.as_tensor(self._scaled(raw), device=self.device).unsqueeze(1)
            logits, self._memory = self.policy(step, self._memory)
        return mask_forward_logits(logits.squeeze(1), blocked)

    def _scaled(self, features):
        """The network input: the shared observation with the three input scales applied."""
        return scale_destination_offsets(
            features,
            self.settings.destination_scale,
            self.settings.coordinate_scale,
            self.settings.requested_scale,
        )

    def _prepare(self, episode):
        """Turn one shared Episode into tensors and its returns."""
        transitions = episode.transitions
        if not transitions:
            raise ValueError("REINFORCE received an empty episode")
        features = batch_network_input([step.observations for step in transitions])
        next_features = batch_network_input([step.next_observations for step in transitions])
        actions = np.asarray([step.actions for step in transitions], dtype=np.int64)
        delivery_rewards = np.asarray([step.rewards for step in transitions], dtype=np.float32)
        # The shared learning reward. This agent has no setting that changes it.
        rewards = np.asarray([step.learning_rewards for step in transitions], dtype=np.float32)

        # Where the shared mask closed FORWARD. The update has to use the same
        # closed action the policy saw when it chose.
        blocked = blocked_forward(features)
        if (blocked & (actions == FORWARD)).any():
            raise ValueError("the episode contains a forward move the shared mask had closed")
        # Position, load, facing and highway flag are the first eight values.
        changed = (np.abs(next_features[..., :8] - features[..., :8]) > 0.5).any(axis=-1) & (actions != NOOP)

        def by_robot(array):
            return torch.as_tensor(np.ascontiguousarray(np.swapaxes(array, 0, 1)), device=self.device)

        return EpisodeData(
            features=by_robot(self._scaled(features)),
            actions=by_robot(actions),
            blocked=by_robot(blocked),
            returns=by_robot(discounted_returns(rewards, self.settings.gamma)),
            changed=by_robot(changed.astype(np.float32)),
            deliveries=float(delivery_rewards.sum()),
        )

    def _distribution(self, features, blocked):
        logits, _memory = self.policy(features)
        masked = mask_forward_logits(logits.reshape(-1, N_ACTIONS), blocked.reshape(-1))
        return Categorical(logits=masked.reshape(logits.shape))

    def _advantages(self, features, returns):
        """Return minus the baseline's prediction. The plain return when the baseline is off."""
        if self.settings.baseline_rate <= 0.0:
            return returns
        with torch.no_grad():
            return returns - self.baseline(features)

    def _step(self, optimizer, network, loss):
        if not torch.isfinite(loss):
            raise RuntimeError(f"REINFORCE loss is not finite: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), self.settings.max_grad_norm)
        optimizer.step()

    def _fit_baseline(self, features, returns):
        """Squared error of V(s) against the Monte Carlo return."""
        settings = self.settings
        if settings.baseline_rate <= 0.0:
            return 0.0
        flat_features = features.reshape(-1, INPUT_DIM)
        flat_returns = returns.reshape(-1)
        samples = flat_features.shape[0]
        total = 0.0
        updates = 0
        self.baseline.train()
        for _epoch in range(settings.baseline_epochs):
            order = torch.randperm(samples, device=self.device)
            for start in range(0, samples, settings.baseline_minibatch_size):
                index = order[start : start + settings.baseline_minibatch_size]
                loss = (self.baseline(flat_features[index]) - flat_returns[index]).pow(2).mean()
                self._step(self.baseline_optimizer, self.baseline, loss)
                total += float(loss.item())
                updates += 1
        self.baseline.eval()
        return total / max(updates, 1)

    def _remember(self, batch):
        """Keep the replay_size episodes with the most real deliveries."""
        self.replay.extend(data for data in batch if data.deliveries > 0)
        if len(self.replay) > self.settings.replay_size:
            self.replay.sort(key=lambda data: data.deliveries)
            self.replay = self.replay[len(self.replay) - self.settings.replay_size :]

    def _replay(self, entropy_coef):
        """Learn again from stored episodes, where they went better than expected."""
        settings = self.settings
        for _update in range(settings.replay_updates):
            if not self.replay or settings.replay_episodes < 1:
                return
            # Episodes with more deliveries are drawn more often.
            deliveries = np.asarray([data.deliveries for data in self.replay], dtype=np.float64)
            count = min(settings.replay_episodes, len(self.replay))
            picks = np.random.choice(len(self.replay), size=count, replace=False, p=deliveries / deliveries.sum())
            chosen = [self.replay[int(pick)] for pick in picks]
            features = torch.cat([data.features for data in chosen])
            actions = torch.cat([data.actions for data in chosen])
            blocked = torch.cat([data.blocked for data in chosen])
            returns = torch.cat([data.returns for data in chosen])
            changed = torch.cat([data.changed for data in chosen])
            # Only steps that went better than expected, and only actions
            # that changed something, are imitated.
            weights = self._advantages(features, returns).clamp(min=0.0) * changed
            self.policy.train()
            distribution = self._distribution(features, blocked)
            imitation = -(distribution.log_prob(actions) * weights).sum() / (weights > 1e-8).sum().clamp(min=1)
            loss = settings.replay_coef * imitation - 0.5 * entropy_coef * distribution.entropy().mean()
            self._step(self.policy_optimizer, self.policy, loss)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "baseline": self.baseline.state_dict(),
                "settings": self.settings.as_dict(),
            },
            path,
        )
        return path

    def load(self, path):
        checkpoint = _read_checkpoint(path, self.device)
        self._load_weights(checkpoint)
        return checkpoint

    def _load_weights(self, checkpoint):
        self.policy.load_state_dict(checkpoint["policy"])
        self.baseline.load_state_dict(checkpoint["baseline"])
        self.reset_episode()


def load_agent(path):
    """Rebuild the policy and the baseline from reinforce.pt and load their weights."""
    checkpoint = _read_checkpoint(path, map_location="cpu")
    known = set(REINFORCESettings.__dataclass_fields__)
    settings = {key: value for key, value in dict(checkpoint["settings"]).items() if key in known}
    agent = REINFORCEAgent(REINFORCESettings(**settings))
    agent._load_weights(checkpoint)
    return agent


def _read_checkpoint(path, map_location):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
