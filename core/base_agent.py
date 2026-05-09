"""
Base PPO agent class for continuous action spaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
import logging

from .config import PPOConfig

logger = logging.getLogger(__name__)


class ActorCriticNet(nn.Module):
    """
    Base Actor-Critic network with shared feature layers.
    Supports both discrete and continuous action spaces.
    Accepts either uniform hidden_dim+num_layers or custom hidden_dims list.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.0,
        action_space: str = 'discrete',
        hidden_dims: Optional[list] = None,
        activation: str = 'ReLU'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space

        # Select activation function
        act_cls = nn.ELU if activation == 'ELU' else nn.ReLU

        # Determine layer sizes
        if hidden_dims is not None:
            # Custom architecture: e.g. [256, 128, 64]
            layer_sizes = [state_dim] + list(hidden_dims)
        else:
            # Uniform architecture: hidden_dim repeated num_layers times
            layer_sizes = [state_dim] + [hidden_dim] * num_layers

        # Build shared feature layers
        shared_layers = []
        for i in range(len(layer_sizes) - 1):
            shared_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            shared_layers.append(act_cls())

        self.shared_layers = nn.Sequential(*shared_layers)

        # Last shared layer size for heads
        last_dim = layer_sizes[-1]

        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            act_cls(),
            nn.Linear(last_dim, action_dim)
        )

        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            act_cls(),
            nn.Linear(last_dim, 1)
        )

    def forward(self, state: torch.Tensor, use_softmax: bool = True, apply_softmax: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor
            use_softmax: Whether to apply softmax to actor output (for discrete actions)
            apply_softmax: Explicit flag to force softmax application (for ONNX export)

        Returns:
            Tuple of (action_output, value)
        """
        shared_features = self.shared_layers(state)
        actor_output = self.actor_head(shared_features)
        value = self.critic_head(shared_features)

        # Use explicit flag or action_space check
        should_apply_softmax = apply_softmax or (use_softmax and self.action_space == 'discrete')
        if should_apply_softmax:
            actor_output = torch.softmax(actor_output, dim=-1)

        return actor_output, value

    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get raw logits (no softmax) for discrete actions."""
        shared_features = self.shared_layers(state)
        return self.actor_head(shared_features)

    def get_action_mean(self, state: torch.Tensor) -> torch.Tensor:
        """Get action mean for continuous actions."""
        shared_features = self.shared_layers(state)
        return self.actor_head(shared_features)


class BasePPOAgent(ABC):
    """
    Abstract base class for PPO agents.
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, config: PPOConfig, device: Optional[str] = None):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.network_layers
        self.dropout_rate = config.dropout_rate
        self.action_space_type = config.action_space

        # PPO hyperparameters
        self.gamma = config.gamma
        self.lam = config.lam
        self.epsilon = config.epsilon
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.initial_entropy_coef = config.entropy_coef
        self.entropy_coef = config.entropy_coef

        # Entropy decay settings
        self.entropy_decay = config.entropy_decay
        self.min_entropy_coef = config.min_entropy_coef

        # Gradient clipping
        self.use_gradient_clipping = config.use_gradient_clipping
        self.gradient_clip_value = config.gradient_clip_value

        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Mixed precision training (AMP) — only on CUDA device
        _requested_amp = config.use_amp if hasattr(config, 'use_amp') else False
        self.use_amp = _requested_amp and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Initialize network (to be implemented by subclasses)
        self.actor_critic = None
        self.optimizer = None

        # Learning rate scheduler
        self.use_lr_scheduler = config.use_lr_scheduler
        self.scheduler = None

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Select an action given the current state.

        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action selection

        Returns:
            Tuple of (action, log_prob, value)
        """
        pass

    @abstractmethod
    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        old_log_probs: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[float, float]:
        """
        Update the agent using PPO.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            old_log_probs: List of old log probabilities
            values: List of state values
            dones: List of done flags

        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        pass

    @abstractmethod
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        pass

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> List[float]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags

        Returns:
            List of advantage values
        """
        advantages = []
        gae = 0

        # Add dummy value for next state
        values = values + [0]

        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                gae = delta + self.gamma * self.lam * gae

            advantages.append(gae)  # O(1) append instead of O(n) insert

        advantages.reverse()  # O(n) single reverse
        return advantages

    def save_weights(self, path: str):
        """Save model weights."""
        state_dict = self.actor_critic.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model weights saved to {path}")

    def load_weights(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(state_dict)
        logger.info(f"Model weights loaded from {path}")

    def set_train_mode(self):
        """Set model to training mode."""
        self.actor_critic.train()

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.actor_critic.eval()

    def decay_entropy(self):
        """Decay entropy coefficient for exploration-exploitation balance."""
        self.entropy_coef = max(
            self.min_entropy_coef,
            self.entropy_coef * self.entropy_decay
        )


class ContinuousPPOAgent(BasePPOAgent):
    """
    PPO Agent for continuous action spaces using Gaussian distribution.
    Used for humanoid robot scenario.
    """

    def __init__(self, config: PPOConfig, device: Optional[str] = None):
        super().__init__(config, device)

        # Initialize actor-critic network
        self.actor_critic = ActorCriticNet(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout_rate,
            action_space='continuous',
            hidden_dims=getattr(config, 'hidden_dims', None),
            activation=getattr(config, 'activation', 'ReLU')
        ).to(self.device)

        # Learnable log standard deviation for actions (on same device as network)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), config.init_log_std, device=self.device),
            requires_grad=True
        )

        # Clamping bounds for log_std
        self.min_log_std = config.min_log_std
        self.max_log_std = config.max_log_std

        self.optimizer = optim.Adam(
            list(self.actor_critic.parameters()) + [self.log_std],
            lr=config.learning_rate,
            eps=1e-5
        )

        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config.lr_decay_rate
            )

        logger.info(f"Initialized ContinuousPPOAgent on {self.device}")

        # Running reward normalization stats
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Select an action using Gaussian distribution.

        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action selection (mean action)

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_mean, value = self.actor_critic(state_tensor, use_softmax=False)

            # Clamp log std
            log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)

            if deterministic:
                action = action_mean
                log_prob = torch.zeros(1, device=self.device)
            else:
                dist = Normal(action_mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_mean, values = self.actor_critic(states, use_softmax=False)

        # Clamp log std
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).expand_as(action_mean)

        dist = Normal(action_mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        old_log_probs: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[float, float]:
        """
        Update the agent using PPO.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            old_log_probs: List of old log probabilities
            values: List of state values
            dones: List of done flags

        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Compute GAE advantages
        advantages = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        returns = advantages + values

        # PPO update
        actor_losses = []
        critic_losses = []

        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, value_pred, entropy = self.evaluate_actions(
                    batch_states,
                    batch_actions
                )

                # Compute ratio
                ratio = torch.exp(torch.clamp(
                    log_probs - batch_old_log_probs,
                    -10, 10
                ))

                # PPO clipped loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = nn.MSELoss()(value_pred, batch_returns)

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

                # Optimize with mixed precision support
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.use_gradient_clipping:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor_critic.parameters()) + [self.log_std],
                            self.gradient_clip_value
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor_critic.parameters()) + [self.log_std],
                            self.gradient_clip_value
                        )
                    self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        if self.use_lr_scheduler:
            self.scheduler.step()

        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)

        logger.debug(f"PPO update completed. Actor Loss: {avg_actor_loss:.4f}, "
                     f"Critic Loss: {avg_critic_loss:.4f}")

        return avg_actor_loss, avg_critic_loss

    def save_weights(self, path: str):
        """Save model weights including log_std parameter."""
        state_dict = self.actor_critic.state_dict()
        state_dict['log_std'] = self.log_std.data.cpu()
        torch.save(state_dict, path)
        logger.info(f"Model weights (including log_std) saved to {path}")

    def load_weights(self, path: str):
        """Load model weights including log_std parameter."""
        state_dict = torch.load(path, map_location=self.device)

        log_std = None
        network_state_dict = {}
        for k, v in state_dict.items():
            if k == 'log_std':
                log_std = v
            else:
                network_state_dict[k] = v

        self.actor_critic.load_state_dict(network_state_dict)

        if log_std is not None:
            self.log_std.data = log_std.to(self.device)
            logger.info(f"Loaded log_std: {self.log_std}")
        else:
            logger.warning("log_std not found in checkpoint, using initialized value")

        logger.info(f"Model weights loaded from {path}")

    # ── Convenience helpers ─────────

    def take_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action and return only the action vector."""
        action, _, _ = self.select_action(state)
        return action

    def select_actions_batch(
        self,
        states: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch action selection — single GPU forward pass for N states.

        Args:
            states: (N, state_dim) numpy array
            deterministic: Use mean action if True

        Returns:
            actions:   (N, action_dim) numpy array
            log_probs: (N,) numpy array
            values:    (N,) numpy array
        """
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_mean, values = self.actor_critic(state_tensor, use_softmax=False)
            log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)

            if deterministic:
                actions = action_mean
                log_probs = torch.zeros(len(states), device=self.device)
            else:
                dist = Normal(action_mean, std)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(dim=-1)

        return (actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.squeeze(-1).cpu().numpy())

    def get_action_mean_std(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get action mean and std for a single state."""
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_mean, _ = self.actor_critic(state_tensor, use_softmax=False)
            log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)

        return action_mean.squeeze(0).cpu().numpy(), std.cpu().numpy()

    def compute_bootstrap_values(self, states: np.ndarray) -> np.ndarray:
        """Compute value predictions for a batch of states (for GAE bootstrapping)."""
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, values = self.actor_critic(state_tensor, use_softmax=False)
        return values.squeeze(-1).cpu().numpy()

    def evaluate_batch(self, states: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of states, returning (action_mean, values)."""
        state_tensor = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            mean, values = self.actor_critic(state_tensor, use_softmax=False)

        return mean, values

    def update_from_arrays(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        old_log_probs: np.ndarray,
        values_arr: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Optimized PPO update accepting pre-converted numpy arrays.

        Arrays can be (N,) or (N, D) shaped.  Uses torch.as_tensor for
        zero-copy when the arrays are already contiguous float32.
        """
        advantages = self._compute_gae_array(rewards, values_arr, dones, self.gamma, self.lam)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values_t = torch.as_tensor(values_arr, dtype=torch.float32, device=self.device)
        returns = advantages + values_t

        actor_losses = []
        critic_losses = []

        for _ in range(self.epochs):
            indices = torch.randperm(len(states_t), device=self.device)

            for start in range(0, len(states_t), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                log_probs, value_pred, entropy = self.evaluate_actions(
                    states_t[batch_idx], actions_t[batch_idx]
                )

                ratio = torch.exp(torch.clamp(
                    log_probs - old_log_probs_t[batch_idx], -10, 10
                ))
                batch_adv = advantages[batch_idx]

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(value_pred, returns[batch_idx])

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.use_gradient_clipping:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor_critic.parameters()) + [self.log_std],
                            self.gradient_clip_value
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor_critic.parameters()) + [self.log_std],
                            self.gradient_clip_value
                        )
                    self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        if self.use_lr_scheduler:
            self.scheduler.step()

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    def update_from_arrays_with_advantages(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        old_log_probs: np.ndarray,
        values_arr: np.ndarray,
        dones: np.ndarray,
        advantages: np.ndarray,
    ) -> Tuple[float, float]:
        """
        PPO update with pre-computed advantages (avoids redundant GAE computation).
        Includes reward normalization and value function clipping for stability.
        """
        # Update running reward stats
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        self._reward_count += 1
        decay = 0.99
        self._reward_running_mean = decay * self._reward_running_mean + (1 - decay) * batch_mean
        self._reward_running_var = decay * self._reward_running_var + (1 - decay) * batch_var

        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values_t = torch.as_tensor(values_arr, dtype=torch.float32, device=self.device)
        returns = advantages_t + values_t

        actor_losses = []
        critic_losses = []

        for _ in range(self.epochs):
            indices = torch.randperm(len(states_t), device=self.device)

            for start in range(0, len(states_t), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                log_probs, value_pred, entropy = self.evaluate_actions(
                    states_t[batch_idx], actions_t[batch_idx]
                )

                ratio = torch.exp(torch.clamp(
                    log_probs - old_log_probs_t[batch_idx], -10, 10
                ))
                batch_adv = advantages_t[batch_idx]

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function clipping
                values_clipped = values_t[batch_idx] + torch.clamp(
                    value_pred - values_t[batch_idx], -self.epsilon, self.epsilon
                )
                critic_loss = 0.5 * torch.max(
                    (value_pred - returns[batch_idx]).pow(2),
                    (values_clipped - returns[batch_idx]).pow(2),
                ).mean()

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.use_gradient_clipping:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor_critic.parameters()) + [self.log_std],
                            self.gradient_clip_value
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor_critic.parameters()) + [self.log_std],
                            self.gradient_clip_value
                        )
                    self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        if self.use_lr_scheduler:
            self.scheduler.step()

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    @staticmethod
    def _compute_gae_array(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.999,
        lam: float = 0.95,
    ) -> np.ndarray:
        """Vectorized GAE computation on numpy arrays."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for i in range(n - 1, -1, -1):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                next_val = values[i + 1] if i + 1 < n else 0.0
                delta = rewards[i] + gamma * next_val - values[i]
                gae = delta + gamma * lam * gae
            advantages[i] = gae

        return advantages

    @staticmethod
    def _compute_gae_per_env(
        rewards_2d: np.ndarray,
        values_2d: np.ndarray,
        dones_2d: np.ndarray,
        bootstrap_values: np.ndarray,
        gamma: float = 0.999,
        lam: float = 0.95,
    ) -> np.ndarray:
        """
        Compute GAE per-env with proper bootstrap values.

        Args:
            rewards_2d: (steps, envs) reward array
            values_2d: (steps, envs) value array
            dones_2d: (steps, envs) bool done array
            bootstrap_values: (envs,) value of state after last collected step
            gamma: discount factor
            lam: GAE lambda

        Returns:
            (steps, envs) advantage array
        """
        steps, envs = rewards_2d.shape
        advantages = np.zeros((steps, envs), dtype=np.float32)

        for e in range(envs):
            gae = 0.0
            for t in range(steps - 1, -1, -1):
                if dones_2d[t, e]:
                    delta = rewards_2d[t, e] - values_2d[t, e]
                    gae = delta
                else:
                    next_val = bootstrap_values[e] if t == steps - 1 else values_2d[t + 1, e]
                    delta = rewards_2d[t, e] + gamma * next_val - values_2d[t, e]
                    gae = delta + gamma * lam * gae
                advantages[t, e] = gae

        return advantages
