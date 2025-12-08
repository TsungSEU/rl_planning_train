#!/usr/bin/env python3
"""
Evaluation script for trained PPO navigation planner
Loads a trained model and evaluates its performance on path planning tasks
"""

import numpy as np
import torch
import yaml
import argparse
import logging
import os
import json
from pathlib import Path
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PathPlanningEnvironment, SimplePathPlanningEnv
from planner_rl_train import ActorCritic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, config_path: str, num_episodes: int = 100, 
                   env_type: str = 'simple', render: bool = False, 
                   save_results: bool = True) -> dict:
    """
    Evaluate a trained model on the path planning environment.
    
    Args:
        model_path: Path to the trained model weights
        config_path: Path to the training configuration file
        num_episodes: Number of evaluation episodes
        env_type: Type of environment ('simple' or 'complex')
        render: Whether to render the environment
        save_results: Whether to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    if env_type == 'simple':
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
    
    # Initialize model
    model = ActorCritic(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config.get('network_layers', 2),
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    # Evaluation loop
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Episode loop
        while not done and steps < config.get('max_steps', 200):
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                logits, _ = model(state_tensor)
                # Use logits directly for action selection (argmax)
                action = torch.argmax(logits).item()  # Greedy action selection
                
            # Execute action
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if requested (only for first few episodes)
            if render and episode < 3:
                print(f"Episode {episode}, Step {steps}: Position={state[:2]}, "  # Only show position part
                      f"Distance to goal={info['distance_to_goal']:.2f}")
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if done:
            successes += 1
            logger.info(f"Episode {episode}: Success! Reward={total_reward:.2f}, Steps={steps}")
        else:
            logger.info(f"Episode {episode}: Failed! Reward={total_reward:.2f}, Steps={steps}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = successes / num_episodes
    avg_length = np.mean(episode_lengths)
    
    results = {
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'success_rate': success_rate,
        'avg_episode_length': float(avg_length),
        'total_episodes': num_episodes,
        'successful_episodes': successes
    }
    
    logger.info(f"Evaluation Results over {num_episodes} episodes:")
    logger.info(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Average Episode Length: {avg_length:.2f}")
    
    # Save results
    if save_results:
        results_dir = Path("results/val")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = np.datetime_as_string(np.datetime64('now'), unit='s')
        timestamp = timestamp.replace(':', '-')
        
        results_file = results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent for navigation planning')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config/advanced_ppo_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--env-type', type=str, default='simple',
                        choices=['simple', 'complex'],
                        help='Type of environment for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save evaluation results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    results = evaluate_model(
        model_path=args.model_path,
        config_path=args.config,
        num_episodes=args.episodes,
        env_type=args.env_type,
        render=args.render,
        save_results=not args.no_save
    )
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()