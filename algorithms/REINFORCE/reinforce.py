"""REINFORCE on the shared warehouse.

REINFORCE is the Monte Carlo policy gradient. The policy plays a whole episode
by sampling its actions. Afterwards every action is weighted by its return,
the discounted reward that followed it, and the policy is moved toward the
actions with high returns. Nothing is bootstrapped: a return is summed to the
end of the 500-step episode, and no value estimate stands in for reward that
has not arrived. PPO and SARSA both bootstrap.

One policy is shared by both robots. Each robot learns from its own reward.

The textbook update alone scores zero here, so this agent adds six things.
Each one was measured by training with it and without it.

1. Training reward. shared/reward_manager.py pays for the steps that lead to
   a delivery, so the policy is told it is on the right track before a shelf
   reaches the goal. It is the shared training reward, not one of this agent's own.
2. Rules. shared/warehouse_rules.py closes the actions that make no sense in
   a warehouse. The policy chooses among the rest. They are shared rules too.
3. Forward start. The policy begins with a preference for walking forward
   (settings.forward_bias), so it moves around and meets rewards from the
   first episode.
4. Baseline. A value network predicts the return of a state. The update uses
   return minus prediction, which is far less noisy than the return alone.
   The baseline is fitted to the returns themselves, so it bootstraps nothing.
5. Entropy bonus. It keeps the policy trying different actions early on, and
   fades to zero so the policy is decisive when it is scored greedily.
6. Replay and memory. The episodes with the most deliveries are kept and
   learned from again after every update. The policy network keeps a memory
   vector between steps, see policy.py.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from algorithms.REINFORCE.config import ReinforceSettings
from algorithms.REINFORCE.policy import Baseline, Policy
from environment.config import MAX_STEPS, N_ACTIONS, NOOP, PARAM_SHARING
from shared.observation import INPUT_DIM, batch_network_input, network_input
from shared.reward_manager import training_rewards
from shared.warehouse_rules import allowed_actions

# Logit given to an action the rules have closed. Its probability becomes zero.
CLOSED = -1e9


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

    features: torch.Tensor  # (robots, steps, INPUT_DIM)
    actions: torch.Tensor  # (robots, steps)
    allowed: torch.Tensor  # (robots, steps, N_ACTIONS), what the shared rules left open
    returns: torch.Tensor  # (robots, steps), return of each robot's training reward
    changed: torch.Tensor  # (robots, steps), 1 where the action changed that robot's state
    deliveries: float  # shelves the team really delivered in the episode


class ReinforceAgent:
    """Samples actions for training and plays greedy actions for evaluate().

    The policy remembers things during an episode, so call reset_episode()
    before each one. run_episode() and evaluate() take it as on_episode_start.
    """

    def __init__(self, settings=None):
        if not PARAM_SHARING:
            raise ValueError("this REINFORCE agent uses one shared network for both robots")

        self.settings = ReinforceSettings() if settings is None else settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy(self.settings.hidden, self.settings.forward_bias).to(self.device)
        self.baseline = Baseline(self.settings.hidden).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.settings.learning_rate,
            eps=self.settings.adam_eps,
        )
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(),
            lr=self.settings.baseline_learning_rate,
            eps=self.settings.adam_eps,
        )
        self.replay = []
        self.reset_episode()

    def reset_episode(self, _observations=None):
        """Forget the last episode: empty memory."""
        self._memory = None
        self._steps = 0

    def choose_actions(self, observations):
        """Sample one action per robot. run_episode calls this during training."""
        actions = Categorical(logits=self.step_logits(observations)).sample()
        return tuple(int(action) for action in actions.tolist())

    def act_greedy(self, observations):
        """Argmax of the action logits. evaluate() requires this greedy policy."""
        actions = torch.argmax(self.step_logits(observations), dim=-1)
        return tuple(int(action) for action in actions.tolist())

    def step_logits(self, observations):
        """Logits for this step, (N_AGENTS, N_ACTIONS), closed actions at CLOSED.

        Advances the memory by one step, so call it once per environment step.
        """
        # An episode is always MAX_STEPS long, so a caller that forgot
        # reset_episode() still starts the next episode with an empty memory.
        if self._steps >= MAX_STEPS:
            self.reset_episode()
        self._steps += 1
        features = network_input(observations)
        allowed = torch.as_tensor(allowed_actions(features), device=self.device)
        self.policy.eval()
        with torch.no_grad():
            # Each robot is a sequence of one step, continued from its memory.
            step = torch.as_tensor(features, device=self.device).unsqueeze(1)
            logits, self._memory = self.policy(step, self._memory)
        return logits.squeeze(1).masked_fill(~allowed, CLOSED)

    def update(self, episodes, progress):
        """One REINFORCE update on finished episodes. Returns the mean losses.

        progress is the share of the step budget used so far, from 0 to 1.
        """
        if not episodes:
            raise ValueError("REINFORCE needs at least one finished episode")
        settings = self.settings
        progress = float(np.clip(progress, 0.0, 1.0))
        batch = [self._prepare(episode) for episode in episodes]
        features = torch.cat([data.features for data in batch])
        actions = torch.cat([data.actions for data in batch])
        allowed = torch.cat([data.allowed for data in batch])
        returns = torch.cat([data.returns for data in batch])

        # Large steps and a strong entropy bonus early, small steps and none
        # late. The learning rate follows half a cosine down to a tenth.
        learning_rate = settings.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in self.policy_optimizer.param_groups:
            group["lr"] = learning_rate
        entropy_coef = settings.entropy_coef * (1.0 - progress)

        # Advantage: how much better the episode went from here than expected.
        with torch.no_grad():
            advantages = returns - self.baseline(features)
        if float(returns.std()) > 1e-6:
            # Same scale in every batch. A batch with no reward at all is left
            # as it is, because rescaling would turn baseline error into a signal.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # The REINFORCE step: raise the log-probability of each action in
        # proportion to its advantage.
        self.policy.train()
        distribution = self._distribution(features, allowed)
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
            "replay_episodes": len(self.replay),
            "team_reward_sum": float(sum(data.deliveries for data in batch)),
        }

    def _prepare(self, episode):
        """Turn one shared Episode into tensors and its returns."""
        transitions = episode.transitions
        if not transitions:
            raise ValueError("REINFORCE received an empty episode")
        features = batch_network_input([step.observations for step in transitions])
        next_features = batch_network_input([step.next_observations for step in transitions])
        actions = np.asarray([step.actions for step in transitions], dtype=np.int64)
        delivery_rewards = np.asarray([step.rewards for step in transitions], dtype=np.float32)

        # What the rules allowed at each step. The update has to use the same
        # closed actions the policy saw when it chose.
        allowed = allowed_actions(features.reshape(-1, INPUT_DIM)).reshape(*actions.shape, N_ACTIONS)
        if not np.take_along_axis(allowed, actions[..., None], axis=2).all():
            raise ValueError("the episode contains an action the rules do not allow")

        # The shared training reward. This agent has no setting that changes it.
        rewards = training_rewards(features, next_features, delivery_rewards)
        # Position, load, facing and highway flag are the first eight values.
        changed = (np.abs(next_features[..., :8] - features[..., :8]) > 0.5).any(axis=-1) & (actions != NOOP)

        def by_robot(array):
            return torch.as_tensor(np.ascontiguousarray(np.swapaxes(array, 0, 1)), device=self.device)

        return EpisodeData(
            features=by_robot(features),
            actions=by_robot(actions),
            allowed=by_robot(allowed),
            returns=by_robot(discounted_returns(rewards, self.settings.gamma)),
            changed=by_robot(changed.astype(np.float32)),
            deliveries=float(delivery_rewards.sum()),
        )

    def _distribution(self, features, allowed):
        logits, _memory = self.policy(features)
        return Categorical(logits=logits.masked_fill(~allowed, CLOSED))

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
        """Learn again from stored episodes, where they beat the baseline."""
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
            allowed = torch.cat([data.allowed for data in chosen])
            returns = torch.cat([data.returns for data in chosen])
            changed = torch.cat([data.changed for data in chosen])
            with torch.no_grad():
                # Only steps that went better than the baseline expected, and
                # only actions that changed something, are imitated.
                weights = (returns - self.baseline(features)).clamp(min=0.0) * changed
            self.policy.train()
            distribution = self._distribution(features, allowed)
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


class PolicyEnsemble:
    """Several trained agents voting on one action per robot.

    Every member reads the same observations and keeps its own memory. Their
    action probabilities are averaged and the most likely action is played.
    """

    def __init__(self, agents):
        self.agents = list(agents)
        if not self.agents:
            raise ValueError("an ensemble needs at least one agent")

    def reset_episode(self, _observations=None):
        for agent in self.agents:
            agent.reset_episode()

    def act_greedy(self, observations):
        probabilities = torch.stack(
            [torch.softmax(agent.step_logits(observations), dim=-1) for agent in self.agents]
        ).mean(dim=0)
        return tuple(int(action) for action in torch.argmax(probabilities, dim=-1).tolist())


def load_agent(path):
    """Rebuild the policy and the baseline from reinforce.pt and load their weights."""
    checkpoint = _read_checkpoint(path, map_location="cpu")
    raw = dict(checkpoint["settings"])
    raw["hidden"] = tuple(raw["hidden"])
    agent = ReinforceAgent(ReinforceSettings(**raw))
    agent._load_weights(checkpoint)
    return agent


def _read_checkpoint(path, map_location):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
