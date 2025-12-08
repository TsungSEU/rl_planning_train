#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Navigation Planner
Implements PPO algorithm for path planning optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import yaml
import argparse
import os
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import environment
try:
    from environment import PathPlanningEnvironment, SimplePathPlanningEnv
except ImportError:
    # Create simple environment locally if import fails
    class SimplePathPlanningEnv:
        def __init__(self, width=20, height=20):
            self.width = width
            self.height = height
            self.action_space = 4
            self.observation_space = 24  # Updated to match specification
            self.actions = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
            
        def reset(self, start_pos=None, goal_pos=None):
            self.agent_pos = np.array([0.0, 0.0])
            self.goal_pos = np.array([float(self.width-1), float(self.height-1)])
            return self._get_state()
            
        def _get_state(self):
            """Return extended state representation"""
            # Simple state with all zeros except position
            state = np.zeros(24)
            state[0] = self.agent_pos[0] / (self.width - 1)  # norm_lat
            state[1] = self.agent_pos[1] / (self.height - 1)  # norm_lon
            return state
            
        def step(self, action):
            dx, dy = self.actions[action]
            new_pos = self.agent_pos + [dx, dy]
            new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
            self.agent_pos = new_pos
            
            distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
            reward = -distance_to_goal * 0.1
            done = False
            if distance_to_goal < 0.5:
                reward += 10.0
                done = True
            reward -= 0.1
            info = {'distance_to_goal': distance_to_goal}
            return self._get_state(), reward, done, info


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2, dropout_rate=0.0):
        super(ActorCritic, self).__init__()
        
        # Build shared layers with configurable depth
        shared_layers = []
        shared_layers.append(nn.Linear(state_dim, hidden_dim))
        shared_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            shared_layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Actor head (policy) - outputs logits, not probabilities
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # No softmax here - output raw logits as required by specification
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        shared_features = self.shared_layers(state)
        logits = self.actor_head(shared_features)  # Raw logits
        value = self.critic_head(shared_features)
        return logits, value  # Return logits, not probabilities


class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dim = config['hidden_dim']
        
        # Network parameters
        self.num_layers = config.get('network_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.0)
        
        # Initialize networks
        self.actor_critic = ActorCritic(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim,
            self.num_layers,
            self.dropout_rate
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config['learning_rate'])
        
        # Learning rate scheduler
        self.use_lr_scheduler = config.get('use_lr_scheduler', False)
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=config.get('lr_decay_rate', 0.99)
            )
        
        # Hyperparameters
        self.gamma = config['gamma']  # discount factor
        self.lam = config['lam']      # GAE lambda
        self.epsilon = config['epsilon']  # PPO clip parameter
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Gradient clipping
        self.use_gradient_clipping = config.get('use_gradient_clipping', False)
        self.gradient_clip_value = config.get('gradient_clip_value', 0.5)
        
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        
        # Add a dummy value for next state
        values = values + [0]
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                gae = delta + self.gamma * self.lam * gae
                
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self, states, actions, rewards, old_log_probs, values, dones):
        """
        Perform PPO update
        """
        # Compute advantages
        advantages = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        returns = advantages + values
        
        # PPO update
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                logits, value = self.actor_critic(batch_states)
                # Use logits directly for Categorical distribution
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                
                # New log probabilities
                new_log_probs = dist.log_prob(batch_actions)
                
                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.gradient_clip_value)
                    
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                
        # Update learning rate
        if self.use_lr_scheduler:
            self.scheduler.step()
                
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        
        logger.debug(f"PPO update completed. Actor Loss: {avg_actor_loss:.4f}, "
                     f"Critic Loss: {avg_critic_loss:.4f}")
        return avg_actor_loss, avg_critic_loss


def load_training_config(config_path):
    """
    Load training configuration from YAML file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for navigation planning')
    parser.add_argument('--config', type=str, default='config/advanced_ppo_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--weights-path', type=str, default='models/ppo_weights.pth',
                        help='Path to save trained weights')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes (overrides config)')
    parser.add_argument('--env-type', type=str, default='simple',
                        choices=['simple', 'complex'],
                        help='Type of environment for training')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Save model every N episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Override episodes if specified
    if args.episodes is not None:
        config['episodes'] = args.episodes
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Initialize environment
    if args.env_type == 'simple':
        env = SimplePathPlanningEnv(
            width=config.get('env_width', 20), 
            height=config.get('env_height', 20)
        )
    else:
        env = PathPlanningEnvironment(
            width=config.get('env_width', 20), 
            height=config.get('env_height', 20), 
            obstacle_ratio=config.get('obstacle_ratio', 0.2)
        )
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    logger.info("Starting PPO training...")
    logger.info(f"Training for {config['episodes']} episodes")
    logger.info(f"Using environment type: {args.env_type}")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # Training loop
    for episode in range(config['episodes']):
        # Collect trajectory data
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Collect episode trajectory
        for step in range(config.get('max_steps', 200)):
            # Convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Get action probabilities and value
            logits, value = trainer.actor_critic(state_tensor)
            # Use logits directly for Categorical distribution
            dist = Categorical(logits=logits)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store data
            states.append(state.copy())
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action.item())
            state = next_state
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                success_count += 1
                break
        
        # Update agent
        if len(states) > 0:
            actor_loss, critic_loss = trainer.update(states, actions, rewards, log_probs, values, dones)
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if episode % config.get('log_interval', 100) == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            success_rate = success_count / max(episode, 1)  # Avoid division by zero
            logger.info(f"Episode {episode}/{config['episodes']}, "
                       f"Avg Reward: {avg_reward:.2f}, "
                       f"Avg Length: {avg_length:.2f}, "
                       f"Success Rate: {success_rate:.2%}")
        
        # Save model periodically
        if episode > 0 and episode % args.save_interval == 0:
            checkpoint_path = models_dir / f"ppo_weights_ep_{episode}.pth"
            torch.save(trainer.actor_critic.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final weights
    torch.save(trainer.actor_critic.state_dict(), args.weights_path)
    logger.info(f"Model weights saved to {args.weights_path}")
    
    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_count': success_count
    }
    
    metrics_path = models_dir / "training_metrics.npy"
    np.save(metrics_path, metrics)
    logger.info(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()