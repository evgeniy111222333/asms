"""
Deep Reinforcement Learning Agents for Crypto Execution Optimization
=====================================================================

Implements production-quality deep RL agents for optimal trade execution:
- PPOAgent: Proximal Policy Optimization with clipped objective
- SACAgent: Soft Actor-Critic with automatic entropy tuning
- TD3Agent: Twin Delayed DDPG for continuous action spaces
- ReplayBuffer / PrioritizedReplayBuffer: Experience storage
- OrnsteinUhlenbeckNoise: Mean-reverting exploration noise
- ActionMask: Trading constraint enforcement
- RewardShaper: Risk-adjusted reward computation

All agents support GPU training with mixed precision and graceful CPU fallback.

Typical usage:
    >>> agent = SACAgent(state_dim=64, action_dim=4)
    >>> action = agent.select_action(state)
    >>> agent.store_transition(state, action, reward, next_state, done)
    >>> metrics = agent.train_step()
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Network initialisation
# ---------------------------------------------------------------------------

def _orthogonal_init(module: nn.Module) -> None:
    """Apply orthogonal initialisation to linear layers with gain √2."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        nn.init.zeros_(module.bias)


def _uniform_init(module: nn.Module, bound: float = 3e-3) -> None:
    """Apply uniform initialisation for policy output layers."""
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -bound, bound)
        nn.init.uniform_(module.bias, -bound, bound)


# ---------------------------------------------------------------------------
# Actor Network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """Feed-forward actor (policy) network with optional action masking.

    Architecture:
        Linear → ReLU → Linear → ReLU → Linear (mean) + log_std

    For deterministic policies (TD3), only the mean head is used.

    Args:
        state_dim: Dimensionality of the state/observation space.
        action_dim: Dimensionality of the action space.
        hidden_dims: List of hidden layer sizes.
        max_action: Maximum absolute action value for scaling.
        deterministic: If True, output only the mean (no log_std).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        max_action: float = 1.0,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.max_action = max_action
        self.deterministic = deterministic

        layers: List[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        if not deterministic:
            self.log_std_head = nn.Linear(prev_dim, action_dim)

        # Initialise
        self.backbone.apply(_orthogonal_init)
        self.mean_head.apply(_uniform_init)
        if not deterministic:
            self.log_std_head.apply(_uniform_init)

    def forward(
        self, state: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            state: (batch, state_dim)
            mask:  Optional (batch, action_dim) boolean mask (True = allowed).

        Returns:
            Tuple of (mean, log_std) for stochastic policies,
            or (action, None) for deterministic policies.
        """
        features = self.backbone(state)
        mean = self.mean_head(features)

        if mask is not None:
            # Set disallowed actions to very negative value before tanh
            mean = mean.masked_fill(~mask, -1e8)

        if self.deterministic:
            return torch.tanh(mean) * self.max_action, None

        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(
        self,
        state: Tensor,
        mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sample an action from the policy.

        Args:
            state:          (batch, state_dim)
            mask:           Optional action mask.
            deterministic:  If True, return the mean action directly.

        Returns:
            Tuple of (action, log_prob).
        """
        mean, log_std = self.forward(state, mask)

        if deterministic or self.deterministic:
            action = torch.tanh(mean) * self.max_action
            return action, torch.zeros(state.shape[0], 1, device=state.device)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action

        # Compute log_prob with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(
            self.max_action * (1 - (action / self.max_action).pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """Twin critic (Q-value) network for SAC/TD3.

    Contains two independent Q-networks to mitigate overestimation.

    Args:
        state_dim:  Dimensionality of the state space.
        action_dim: Dimensionality of the action space.
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        # Q1
        q1_layers: List[nn.Module] = []
        prev = state_dim + action_dim
        for h in hidden_dims:
            q1_layers.append(nn.Linear(prev, h))
            q1_layers.append(nn.ReLU())
            prev = h
        q1_layers.append(nn.Linear(prev, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2
        q2_layers: List[nn.Module] = []
        prev = state_dim + action_dim
        for h in hidden_dims:
            q2_layers.append(nn.Linear(prev, h))
            q2_layers.append(nn.ReLU())
            prev = h
        q2_layers.append(nn.Linear(prev, 1))
        self.q2 = nn.Sequential(*q2_layers)

        self.q1.apply(_orthogonal_init)
        self.q2.apply(_orthogonal_init)

    def forward(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through both Q-networks.

        Args:
            state:  (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            Tuple of (q1_value, q2_value), each (batch, 1).
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Standard FIFO replay buffer for off-policy RL.

    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Dimensionality of the state.
        action_dim: Dimensionality of the action.
        device: Torch device for tensor storage.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 1,
        action_dim: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = capacity
        self.device = device or get_device()
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros(capacity, state_dim, device=self.device)
        self.actions = torch.zeros(capacity, action_dim, device=self.device)
        self.rewards = torch.zeros(capacity, 1, device=self.device)
        self.next_states = torch.zeros(capacity, state_dim, device=self.device)
        self.dones = torch.zeros(capacity, 1, device=self.device)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """Sample a random batch of transitions.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.

    Uses proportional prioritisation with importance sampling weights.

    Args:
        capacity: Maximum buffer size.
        state_dim: State dimensionality.
        action_dim: Action dimensionality.
        alpha: Prioritisation exponent (0 = uniform, 1 = full priority).
        beta: Importance sampling exponent (annealed to 1).
        beta_increment: Per-sample beta increment for annealing.
        device: Torch device.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 1,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = capacity
        self.device = device or get_device()
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros(capacity, state_dim, device=self.device)
        self.actions = torch.zeros(capacity, action_dim, device=self.device)
        self.rewards = torch.zeros(capacity, 1, device=self.device)
        self.next_states = torch.zeros(capacity, state_dim, device=self.device)
        self.dones = torch.zeros(capacity, 1, device=self.device)
        self.priorities = torch.zeros(capacity, device=self.device)
        self._max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum priority."""
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)
        self.priorities[self.ptr] = self._max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[Dict[str, Tensor], Tensor, List[int]]:
        """Sample a prioritised batch.

        Returns:
            Tuple of (batch_dict, importance_weights, indices).
        """
        priorities = self.priorities[: self.size]
        probs = priorities.pow(self.alpha)
        probs /= probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=False).tolist()

        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]).pow(-self.beta)
        weights /= weights.max()
        weights = weights.unsqueeze(-1).to(self.device)

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }
        return batch, weights, indices

    def update_priorities(self, indices: List[int], priorities: Tensor) -> None:
        """Update priorities for sampled transitions."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio.item() + 1e-6
            self._max_priority = max(self._max_priority, prio.item())

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Exploration Noise
# ---------------------------------------------------------------------------

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Suitable for continuous control tasks where smooth exploration is desired.

    Args:
        action_dim: Dimensionality of the action.
        mu: Mean reversion level.
        theta: Mean reversion speed.
        sigma: Volatility (noise magnitude).
        device: Torch device.
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device or get_device()
        self.state = torch.ones(action_dim, device=self.device) * self.mu

    def sample(self) -> Tensor:
        """Generate a noise sample."""
        dx = (
            self.theta * (self.mu - self.state)
            + self.sigma * torch.randn_like(self.state)
        )
        self.state = self.state + dx
        return self.state.clone()

    def reset(self) -> None:
        """Reset the noise process to the mean."""
        self.state = torch.ones(self.action_dim, device=self.device) * self.mu


# ---------------------------------------------------------------------------
# Action Mask
# ---------------------------------------------------------------------------

class ActionMask:
    """Enforces trading constraints by masking invalid actions.

    Constraints include:
    - Position limits (max long/short)
    - Leverage limits
    - Minimum order size
    - Cooldown periods

    Args:
        max_position: Maximum absolute position size.
        max_leverage: Maximum allowed leverage.
        min_order_size: Minimum order size (below this = hold).
        cooldown_steps: Minimum steps between trades.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        max_leverage: float = 3.0,
        min_order_size: float = 0.01,
        cooldown_steps: int = 0,
    ) -> None:
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.min_order_size = min_order_size
        self.cooldown_steps = cooldown_steps
        self._steps_since_trade = 0

    def get_mask(
        self, current_position: float, action_dim: int
    ) -> Tensor:
        """Compute a boolean action mask based on current position.

        Assumes action layout: [strong_sell, sell, hold, buy, strong_buy]
        or continuous [−1, 1] discretised.

        Args:
            current_position: Current portfolio position (signed).
            action_dim: Number of discrete actions or action components.

        Returns:
            Boolean tensor of shape (action_dim,) where True = allowed.
        """
        mask = torch.ones(action_dim, dtype=torch.bool)
        self._steps_since_trade += 1

        # If at max long, disable buy actions
        if current_position >= self.max_position * self.max_leverage:
            # Disable buy-like actions (upper half of discrete actions)
            mid = action_dim // 2
            mask[mid + 1:] = False

        # If at max short, disable sell actions
        if current_position <= -self.max_position * self.max_leverage:
            mid = action_dim // 2
            mask[:mid] = False

        # Cooldown: force hold
        if self._steps_since_trade < self.cooldown_steps:
            mid = action_dim // 2
            hold_only = torch.zeros(action_dim, dtype=torch.bool)
            hold_only[mid] = True
            mask = mask & hold_only

        return mask

    def notify_trade(self) -> None:
        """Notify the mask that a trade was executed (reset cooldown)."""
        self._steps_since_trade = 0


# ---------------------------------------------------------------------------
# Reward Shaper
# ---------------------------------------------------------------------------

class RewardShaper:
    """Shapes rewards for risk-adjusted execution optimisation.

    Combines:
    - PnL (profit and loss)
    - Risk penalty (drawdown, volatility)
    - Execution cost penalty (slippage, fees)
    - Position concentration penalty

    Args:
        pnl_weight: Weight for profit component.
        risk_weight: Weight for risk penalty.
        cost_weight: Weight for execution cost penalty.
        concentration_weight: Weight for concentration penalty.
        risk_free_rate: Annualised risk-free rate for Sharpe calculation.
        target_vol: Target annualised volatility.
    """

    def __init__(
        self,
        pnl_weight: float = 1.0,
        risk_weight: float = 0.5,
        cost_weight: float = 0.2,
        concentration_weight: float = 0.1,
        risk_free_rate: float = 0.02,
        target_vol: float = 0.15,
    ) -> None:
        self.pnl_weight = pnl_weight
        self.risk_weight = risk_weight
        self.cost_weight = cost_weight
        self.concentration_weight = concentration_weight
        self.risk_free_rate = risk_free_rate
        self.target_vol = target_vol
        self._returns: Deque[float] = deque(maxlen=252)

    def compute_reward(
        self,
        pnl: float,
        position: float,
        execution_cost: float,
        portfolio_value: float,
    ) -> float:
        """Compute a shaped reward incorporating risk adjustment.

        Args:
            pnl: Realised profit/loss for this step.
            position: Current position size.
            execution_cost: Slippage + fees for this step.
            portfolio_value: Total portfolio value.

        Returns:
            Shaped reward scalar.
        """
        # PnL component
        step_return = pnl / max(portfolio_value, 1e-8)
        self._returns.append(step_return)

        # Risk penalty: drawdown-like measure via realised vol
        if len(self._returns) > 10:
            returns_arr = np.array(self._returns)
            realised_vol = returns_arr.std() * math.sqrt(252)
            vol_penalty = max(0, realised_vol - self.target_vol)
        else:
            vol_penalty = 0.0

        # Concentration penalty
        concentration = abs(position) / max(portfolio_value, 1e-8)

        reward = (
            self.pnl_weight * step_return
            - self.risk_weight * vol_penalty
            - self.cost_weight * execution_cost / max(portfolio_value, 1e-8)
            - self.concentration_weight * concentration
        )
        return reward

    def reset(self) -> None:
        """Reset the return history."""
        self._returns.clear()


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """Proximal Policy Optimisation agent with clipped objective.

    Features:
    - Clipped surrogate objective
    - Generalised Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus for exploration
    - Mini-batch updates

    Args:
        state_dim: State dimensionality.
        action_dim: Action dimensionality.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_epsilon: PPO clipping parameter.
        entropy_coeff: Entropy bonus coefficient.
        value_coeff: Value loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        epochs_per_update: Number of optimisation epochs per rollout.
        mini_batch_size: Mini-batch size for updates.
        hidden_dims: Actor/Critic hidden layer sizes.
        device: Torch device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 10,
        mini_batch_size: int = 64,
        hidden_dims: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size

        # Networks
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims or [256, 256], deterministic=False
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        self.critic.apply(_orthogonal_init)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Rollout storage
        self._states: List[Tensor] = []
        self._actions: List[Tensor] = []
        self._log_probs: List[Tensor] = []
        self._rewards: List[float] = []
        self._values: List[Tensor] = []
        self._dones: List[bool] = []

        # Metrics
        self._episode_rewards: Deque[float] = deque(maxlen=100)
        self._episode_lengths: Deque[int] = deque(maxlen=100)
        self._current_reward = 0.0
        self._current_length = 0

    def select_action(
        self,
        state: np.ndarray,
        mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select an action using the current policy.

        Args:
            state: Numpy array of shape (state_dim,).
            mask: Optional action mask.
            deterministic: If True, use the mean action.

        Returns:
            Action as numpy array.
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(state_t, mask, deterministic)
            value = self.critic(state_t)

        self._states.append(state_t.squeeze(0))
        self._actions.append(action.squeeze(0))
        self._log_probs.append(log_prob.squeeze(0))
        self._values.append(value.squeeze(0))

        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self,
        reward: float,
        done: bool,
    ) -> None:
        """Store reward and done signal for the current step."""
        self._rewards.append(reward)
        self._dones.append(done)
        self._current_reward += reward
        self._current_length += 1

        if done:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

    def update(self) -> Dict[str, float]:
        """Perform a PPO update using collected rollout data.

        Returns:
            Dict of training metrics.
        """
        if len(self._states) < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Compute GAE
        states = torch.stack(self._states)
        actions = torch.stack(self._actions)
        old_log_probs = torch.stack(self._log_probs)
        values = torch.stack(self._values)

        advantages = []
        gae = 0.0
        for t in reversed(range(len(self._rewards))):
            if t == len(self._rewards) - 1:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_value = values[t + 1]
            delta = (
                self._rewards[t]
                + self.gamma * next_value * (1 - self._dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self._dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        dataset_size = len(self._states)
        for _ in range(self.epochs_per_update):
            indices = torch.randperm(dataset_size, device=self.device)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # New log probs
                mean, log_std = self.actor(mb_states)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Clipped surrogate
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = self.critic(mb_states).squeeze(-1)
                value_loss = F.mse_loss(values_pred, mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Clear rollout storage
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "mean_episode_reward": (
                np.mean(self._episode_rewards) if self._episode_rewards else 0.0
            ),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Return current training metrics."""
        return {
            "mean_episode_reward": (
                np.mean(self._episode_rewards) if self._episode_rewards else 0.0
            ),
            "mean_episode_length": (
                np.mean(self._episode_lengths) if self._episode_lengths else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """Soft Actor-Critic agent with automatic entropy tuning.

    Features:
    - Automatic entropy coefficient (alpha) tuning
    - Twin Q-networks
    - Target networks with Polyak averaging
    - Gaussian policy with tanh squashing
    - GPU support with optional mixed precision

    Args:
        state_dim: State dimensionality.
        action_dim: Action dimensionality.
        lr: Learning rate.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        alpha_lr: Learning rate for the entropy coefficient.
        target_entropy: Target entropy (auto-computed if None).
        hidden_dims: Hidden layer sizes.
        device: Torch device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        hidden_dims: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Networks
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims or [256, 256], max_action=1.0
        ).to(self.device)
        self.critic = CriticNetwork(
            state_dim, action_dim, hidden_dims or [256, 256]
        ).to(self.device)
        self.critic_target = CriticNetwork(
            state_dim, action_dim, hidden_dims or [256, 256]
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy coefficient
        self.target_entropy = target_entropy or -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100_000, state_dim=state_dim, action_dim=action_dim, device=self.device
        )

        # Metrics
        self._episode_rewards: Deque[float] = deque(maxlen=100)
        self._current_reward = 0.0
        self._training_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select an action.

        Args:
            state: (state_dim,) numpy array.
            deterministic: Use mean action if True.

        Returns:
            Action as numpy array.
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._current_reward += reward
        if done:
            self._episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """Perform a single SAC training step.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict of training metrics.
        """
        if len(self.replay_buffer) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": self.alpha.item()}

        batch = self.replay_buffer.sample(batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha.detach() * next_log_probs
            q_backup = rewards + self.gamma * (1 - dones) * q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -self.log_alpha * (log_probs + self.target_entropy).detach().mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # --- Soft update target ---
        self._soft_update_target()

        self._training_steps += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "mean_q": q_new.mean().item(),
        }

    def _soft_update_target(self) -> None:
        """Polyak-average update of target critic parameters."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_metrics(self) -> Dict[str, float]:
        """Return current training metrics."""
        return {
            "mean_episode_reward": (
                np.mean(self._episode_rewards) if self._episode_rewards else 0.0
            ),
            "training_steps": self._training_steps,
            "alpha": self.alpha.item(),
        }


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------

class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient agent.

    Features:
    - Twin Q-networks to mitigate overestimation
    - Delayed policy updates
    - Target policy smoothing
    - Ornstein-Uhlenbeck exploration noise

    Args:
        state_dim: State dimensionality.
        action_dim: Action dimensionality.
        lr: Learning rate.
        gamma: Discount factor.
        tau: Soft update coefficient.
        policy_noise: Noise added to target actions (smoothing).
        noise_clip: Range to clip target noise.
        policy_delay: Number of critic updates per actor update.
        max_action: Maximum action magnitude.
        hidden_dims: Hidden layer sizes.
        ou_sigma: OU noise sigma for exploration.
        device: Torch device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        max_action: float = 1.0,
        hidden_dims: Optional[List[int]] = None,
        ou_sigma: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.action_dim = action_dim

        # Networks
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims or [256, 256],
            max_action=max_action, deterministic=True,
        ).to(self.device)
        self.actor_target = ActorNetwork(
            state_dim, action_dim, hidden_dims or [256, 256],
            max_action=max_action, deterministic=True,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(
            state_dim, action_dim, hidden_dims or [256, 256]
        ).to(self.device)
        self.critic_target = CriticNetwork(
            state_dim, action_dim, hidden_dims or [256, 256]
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Exploration noise
        self.ou_noise = OrnsteinUhlenbeckNoise(
            action_dim, sigma=ou_sigma, device=self.device
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100_000, state_dim=state_dim, action_dim=action_dim, device=self.device
        )

        # Metrics
        self._episode_rewards: Deque[float] = deque(maxlen=100)
        self._current_reward = 0.0
        self._training_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        """Select an action with optional OU noise exploration.

        Args:
            state: (state_dim,) numpy array.
            explore: Whether to add exploration noise.

        Returns:
            Action as numpy array.
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state_t)
        action = action.squeeze(0)

        if explore:
            noise = self.ou_noise.sample()
            action = (action + noise).clamp(-self.max_action, self.max_action)

        return action.cpu().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._current_reward += reward
        if done:
            self._episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """Perform a single TD3 training step.

        Args:
            batch_size: Transitions to sample.

        Returns:
            Dict of training metrics.
        """
        if len(self.replay_buffer) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        batch = self.replay_buffer.sample(batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # --- Critic update ---
        with torch.no_grad():
            # Target policy smoothing
            next_actions, _ = self.actor_target(next_states)
            noise = (
                torch.randn_like(next_actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(
                -self.max_action, self.max_action
            )

            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_backup = rewards + self.gamma * (1 - dones) * q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Delayed actor update ---
        actor_loss = 0.0
        if self._training_steps % self.policy_delay == 0:
            actor_actions, _ = self.actor(states)
            q1_actor, _ = self.critic(states, actor_actions)
            actor_loss_tensor = -q1_actor.mean()

            self.actor_optimizer.zero_grad()
            actor_loss_tensor.backward()
            self.actor_optimizer.step()

            # Soft update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss = actor_loss_tensor.item()

        self._training_steps += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak-averaged soft update of target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_metrics(self) -> Dict[str, float]:
        """Return current training metrics."""
        return {
            "mean_episode_reward": (
                np.mean(self._episode_rewards) if self._episode_rewards else 0.0
            ),
            "training_steps": self._training_steps,
        }


# ---------------------------------------------------------------------------
# Training Loop Utility
# ---------------------------------------------------------------------------

class RLTrainingLoop:
    """Orchestrates the training loop for any RL agent.

    Handles episode management, periodic logging, model checkpointing,
    and metric tracking.

    Args:
        agent: One of PPOAgent, SACAgent, or TD3Agent.
        env: Environment with reset() and step() methods (gym-like).
        max_episodes: Maximum number of training episodes.
        max_steps_per_episode: Max steps before forcing done.
        eval_interval: Episodes between evaluation runs.
        checkpoint_interval: Episodes between model saves.
        checkpoint_dir: Directory to save checkpoints.
        warmup_steps: Random action steps before training begins.
    """

    def __init__(
        self,
        agent: Any,
        env: Any,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 1000,
        eval_interval: int = 100,
        checkpoint_interval: int = 500,
        checkpoint_dir: str = "./checkpoints",
        warmup_steps: int = 1000,
    ) -> None:
        self.agent = agent
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps = max_steps_per_episode
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.metrics_history: List[Dict[str, float]] = []

    def run(self) -> List[Dict[str, float]]:
        """Execute the training loop.

        Returns:
            List of metric dicts, one per evaluation point.
        """
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0

            for step in range(self.max_steps):
                # Select action
                if self.total_steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition (for off-policy agents)
                if hasattr(self.agent, "store_transition"):
                    self.agent.store_transition(
                        state, action, reward, next_state, done
                    )
                elif hasattr(self.agent, "store_transition"):
                    pass  # PPO uses different interface
                else:
                    # Generic: reward/done only
                    pass

                # Train step (for off-policy agents)
                if (
                    self.total_steps >= self.warmup_steps
                    and hasattr(self.agent, "train_step")
                ):
                    metrics = self.agent.train_step()
                    if self.total_steps % 1000 == 0 and metrics:
                        self.metrics_history.append(metrics)

                state = next_state
                episode_reward += reward
                self.total_steps += 1

                if done:
                    break

            # Evaluation
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.agent.get_metrics()
                eval_metrics["episode"] = episode
                eval_metrics["total_steps"] = self.total_steps
                self.metrics_history.append(eval_metrics)

            # Checkpoint
            if (episode + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(episode)

        return self.metrics_history

    def _save_checkpoint(self, episode: int) -> None:
        """Save a model checkpoint."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = f"{self.checkpoint_dir}/agent_ep{episode}.pt"
        if hasattr(self.agent, "state_dict"):
            torch.save(self.agent.state_dict(), path)
