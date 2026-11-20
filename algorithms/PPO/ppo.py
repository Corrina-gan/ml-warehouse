"""Proximal Policy Optimization on the shared warehouse.

PPO is an actor-critic method. The actor is the policy that chooses each
robot's action. The critic is a separate network that estimates V(s), the
discounted team return. The advantage is the return minus that value, and the
clipped update moves only the actor. The value loss moves only the critic.

Each environment step gives both robots the team reward, which is the sum of
the two individual rewards. A delivery by one robot is therefore a learning
signal for both. One actor and one critic are shared by both robots.

A rollout comes from collect_steps. done on a transition means the next
stored observation belongs to a new episode. is_true_terminal stays false at
the 500-step limit, so the value target still bootstraps from the observation
step() returned. The advantage trace stops at done, so the new episode does
not leak into the old one.

The policy loss is the clipped surrogate. The value loss is squared error on
the return. An entropy bonus keeps probability on unused actions. Deliveries
are rare under a random policy, and a collapsed categorical distribution
would stop searching for them.
"""

from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from algorithms.PPO.actor import Actor
from algorithms.PPO.config import PPOSettings
from algorithms.PPO.critic import Critic
from environment.config import LEARNING_SIGNAL, N_AGENTS, PARAM_SHARING
from shared.observation import INPUT_DIM, batch_network_input
from shared.training_utils import is_true_terminal


def generalized_advantages(rewards, values, next_values, done, true_terminal, gamma, gae_lambda):
    """Advantage and return for one rollout.

    rewards, values, next_values: (T, N_AGENTS)
    done, true_terminal: (T,) with 1 when that fact is true

    next_values[t] is V(next_observations) of that same transition. At a time
    limit this is the final warehouse state, not the reset state of the next
    episode. true_terminal scales that bootstrap. done stops the trace.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    done = np.asarray(done, dtype=np.float32)
    true_terminal = np.asarray(true_terminal, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards, values, and next_values must share a shape")
    if rewards.ndim != 2:
        raise ValueError(f"expected (T, N_AGENTS) rewards, got {rewards.shape}")
    if done.shape != (rewards.shape[0],) or true_terminal.shape != done.shape:
        raise ValueError("done and true_terminal must have one flag per step")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all() or not np.isfinite(next_values).all():
        raise ValueError("advantage inputs contain NaN or infinity")

    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    trace = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(steps - 1, -1, -1):
        bootstrap = (1.0 - true_terminal[t]) * next_values[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        carry = 1.0 - done[t]
        trace = delta + gamma * gae_lambda * carry * trace
        advantages[t] = trace
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _normalize(advantages):
    """Zero mean and unit scale for the policy loss. Returns keep the raw advantage."""
    centered = advantages - advantages.mean()
    scale = advantages.std()
    if not np.isfinite(scale) or scale < 1e-8:
        return centered.astype(np.float32)
    return (centered / (scale + 1e-8)).astype(np.float32)


def _rollout_targets(rollout):
    """Team reward for both robots, plus the done and bootstrap flags."""
    transitions = rollout.transitions
    if not transitions:
        raise ValueError("PPO needs at least one transition")

    steps = len(transitions)
    rewards = np.zeros((steps, N_AGENTS), dtype=np.float32)
    done = np.zeros(steps, dtype=np.float32)
    true_terminal = np.zeros(steps, dtype=np.float32)
    actions = np.zeros((steps, N_AGENTS), dtype=np.int64)
    episode_step = 0
    for t, step in enumerate(transitions):
        # The same team sum is the reward for both rows. The delivering robot
        # already received 1 from the warehouse; the other received 0. Summing
        # credits the shared policy for the delivery.
        rewards[t] = step.team_reward
        done[t] = float(step.done)
        true_terminal[t] = float(is_true_terminal(episode_step, step.terminated, step.truncated))
        actions[t] = step.actions
        episode_step = 0 if step.done else episode_step + 1

    observations = batch_network_input(step.observations for step in transitions)
    next_observations = batch_network_input(step.next_observations for step in transitions)
    return observations, next_observations, actions, rewards, done, true_terminal


class PPOAgent:
    """Collects stochastic actions for training and greedy actions for evaluate()."""

    def __init__(self, settings=None):
        if not PARAM_SHARING:
            raise ValueError("this PPO agent uses one shared network for both robots")
        if LEARNING_SIGNAL != "team":
            raise ValueError(f"PPO is wired for the team reward, got {LEARNING_SIGNAL!r}")

        self.settings = PPOSettings() if settings is None else settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.settings.hidden).to(self.device)
        self.critic = Critic(self.settings.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            self._parameters(),
            lr=self.settings.learning_rate,
            eps=self.settings.adam_eps,
        )

    def choose_actions(self, observations):
        """Sample one action per robot. collect_steps calls this during training."""
        return self._actions(observations, deterministic=False)

    def act_greedy(self, observations):
        """Argmax of the action logits. evaluate() requires this greedy policy."""
        return self._actions(observations, deterministic=True)

    def _actions(self, observations, deterministic):
        features = self._features(observations)
        with torch.no_grad():
            logits = self.actor(features)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = Categorical(logits=logits).sample()
        return tuple(int(action) for action in actions.tolist())

    def update(self, rollout):
        """One PPO update on a collect_steps rollout. Returns the mean losses."""
        observations, next_observations, actions, rewards, done, true_terminal = _rollout_targets(rollout)
        with torch.no_grad():
            flat_obs = self._flat(observations)
            flat_next = self._flat(next_observations)
            logits = self.actor(flat_obs)
            values = self.critic(flat_obs)
            next_values = self.critic(flat_next)
            old_log_probs = Categorical(logits=logits).log_prob(self._actions_tensor(actions))

        steps, agents = rewards.shape
        values_np = values.detach().cpu().numpy().reshape(steps, agents)
        next_values_np = next_values.detach().cpu().numpy().reshape(steps, agents)
        advantages, returns = generalized_advantages(
            rewards,
            values_np,
            next_values_np,
            done,
            true_terminal,
            gamma=self.settings.gamma,
            gae_lambda=self.settings.gae_lambda,
        )
        policy_advantages = _normalize(advantages)

        features = flat_obs.detach()
        actions_t = self._actions_tensor(actions)
        old_log_probs = old_log_probs.detach()
        advantages_t = torch.as_tensor(policy_advantages.reshape(-1), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=self.device)

        samples = features.shape[0]
        batch_size = min(self.settings.minibatch_size, samples)
        if batch_size < 1:
            raise ValueError("PPO minibatch is empty")

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0
        self.actor.train()
        self.critic.train()
        for _epoch in range(self.settings.epochs):
            order = torch.randperm(samples, device=self.device)
            for start in range(0, samples, batch_size):
                index = order[start : start + batch_size]
                loss, metrics = self._loss(
                    features[index],
                    actions_t[index],
                    old_log_probs[index],
                    advantages_t[index],
                    returns_t[index],
                )
                if not np.isfinite(metrics["loss"]):
                    raise RuntimeError(f"PPO loss is not finite: {metrics}")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._parameters(), self.settings.max_grad_norm)
                self.optimizer.step()
                for key in totals:
                    totals[key] += metrics[key]
                updates += 1

        result = {key: value / updates for key, value in totals.items()}
        result["team_reward_sum"] = float(rewards[:, 0].sum())
        result["explained_variance"] = _explained_variance(values_np.reshape(-1), returns.reshape(-1))
        return result

    def _loss(self, features, actions, old_log_probs, advantages, returns):
        # Separate networks: the clipped surrogate reaches the actor, and the
        # squared error on the return reaches the critic.
        logits = self.actor(features)
        values = self.critic(features)
        distribution = Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - self.settings.clip_eps, 1.0 + self.settings.clip_eps) * advantages
        # The paper maximises the smaller of the two surrogates. Adam minimises,
        # so the policy term is negated. value_loss is the mean squared error.
        # value_coef (0.5) scales it, so the critic term is 0.5 * MSE.
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = (values - returns).pow(2).mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss + self.settings.value_coef * value_loss - self.settings.entropy_coef * entropy
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.settings.clip_eps).float().mean()
        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
        }
        return loss, metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "settings": self.settings.as_dict(),
            },
            path,
        )
        return path

    def load(self, path):
        checkpoint = _read_checkpoint(path, self.device)
        self._load_weights(checkpoint)
        return checkpoint

    def _parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def _load_weights(self, checkpoint):
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def _features(self, observations):
        array = batch_network_input((observations,))[0]
        return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=self.device)

    def _flat(self, observations):
        steps, agents, width = observations.shape
        if agents != N_AGENTS or width != INPUT_DIM:
            raise ValueError(f"expected observations {(steps, N_AGENTS, INPUT_DIM)}, got {observations.shape}")
        flat = np.ascontiguousarray(observations).reshape(steps * agents, width)
        return torch.as_tensor(flat, dtype=torch.float32, device=self.device)

    def _actions_tensor(self, actions):
        flat = np.asarray(actions, dtype=np.int64).reshape(-1)
        return torch.as_tensor(flat, dtype=torch.int64, device=self.device)


def load_agent(path):
    """Rebuild the actor and the critic from ppo.pt and load their weights."""
    checkpoint = _read_checkpoint(path, map_location="cpu")
    raw = dict(checkpoint["settings"])
    raw["hidden"] = tuple(raw["hidden"])
    agent = PPOAgent(PPOSettings(**raw))
    agent._load_weights(checkpoint)
    return agent


def _read_checkpoint(path, map_location):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _explained_variance(values, returns):
    variance = float(np.var(returns))
    if variance < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / variance)
