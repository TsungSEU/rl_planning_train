#!/usr/bin/env python3
"""
Hyperparameter tuning script for PPO path planning
Performs grid search over key hyperparameters
"""

import numpy as np
import torch
import yaml
import argparse
import logging
from pathlib import Path
import itertools
import json
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SimplePathPlanningEnv
from planner_rl_train import PPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_hyperparams(config):
    """
    Evaluate a set of hyperparameters by training a model and measuring performance.
    
    Args:
        config: Configuration dictionary with hyperparameters
        
    Returns:
        Average reward achieved with these hyperparameters
    """
    # Initialize environment
    env = SimplePathPlanningEnv(width=10, height=10)  # Smaller env for faster tuning
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Metrics
    episode_rewards = []
    success_count = 0
    
    # Reduced training for hyperparameter tuning
    num_episodes = min(config.get('episodes', 1000), 1000)
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset(start_pos=(0, 0), goal_pos=(9, 9))
        episode_reward = 0
        done = False
        
        # Collect episode trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        for step in range(config.get('max_steps', 100)):
            # Convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Get action probabilities and value
            logits, value = trainer.actor_critic(state_tensor)
            # Use logits directly for Categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            
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
            
            if done:
                success_count += 1
                break
        
        # Update agent
        if len(states) > 0:
            trainer.update(states, actions, rewards, log_probs, values, dones)
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
    
    # Return average reward as performance metric
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    logger.info(f"Hyperparameters: LR={config['learning_rate']}, "
                f"Gamma={config['gamma']}, Entropy={config.get('entropy_coef', 0.01)}")
    logger.info(f"Performance: Avg Reward={avg_reward:.2f}, Success Rate={success_rate:.2%}")
    
    return avg_reward


def grid_search(base_config, param_grid):
    """
    Perform grid search over hyperparameters.
    
    Args:
        base_config: Base configuration dictionary
        param_grid: Dictionary mapping parameter names to lists of values
        
    Returns:
        Best hyperparameters and corresponding performance
    """
    # Generate all combinations of hyperparameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    best_params = None
    best_performance = -np.inf
    
    results = []
    
    logger.info(f"Starting grid search over {len(combinations)} combinations...")
    
    for combo in combinations:
        # Create configuration with current combination
        config = base_config.copy()
        for name, value in zip(param_names, combo):
            config[name] = value
        
        # Evaluate hyperparameters
        performance = evaluate_hyperparams(config)
        
        # Store results
        result = {name: value for name, value in zip(param_names, combo)}
        result['performance'] = performance
        results.append(result)
        
        # Update best parameters
        if performance > best_performance:
            best_performance = performance
            best_params = config.copy()
        
        logger.info(f"Completed evaluation. Performance: {performance:.2f}")
    
    return best_params, results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for PPO path planning')
    parser.add_argument('--config', type=str, default='config/advanced_ppo_config.yaml',
                        help='Base configuration file')
    parser.add_argument('--output', type=str, default='hyperparameter_tuning_results.json',
                        help='Path to save tuning results')
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [0.0001, 0.0003, 0.001, 0.003],
        'gamma': [0.99, 0.995, 0.999],
        'entropy_coef': [0.001, 0.01, 0.02, 0.05]
    }
    
    logger.info("Starting hyperparameter tuning...")
    best_params, results = grid_search(base_config, param_grid)
    
    # Save results
    tuning_results = {
        'best_params': best_params,
        'all_results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    logger.info(f"Tuning completed. Results saved to {args.output}")
    logger.info(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()