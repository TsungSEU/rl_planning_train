#!/usr/bin/env python3
"""
Enhanced evaluation script for trained PPO navigation planner
Provides detailed performance analysis and comparison
"""

import numpy as np
import torch
import yaml
import argparse
import logging
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PathPlanningEnvironment, SimplePathPlanningEnv
from planner_rl_train import ActorCritic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model_detailed(model_path: str, config_path: str, num_episodes: int = 100, 
                           env_type: str = 'simple', render: bool = False) -> dict:
    """
    Evaluate a trained model with detailed performance analysis.
    
    Args:
        model_path: Path to the trained model weights
        config_path: Path to the training configuration file
        num_episodes: Number of evaluation episodes
        env_type: Type of environment ('simple' or 'complex')
        render: Whether to render the environment
        
    Returns:
        Dictionary containing detailed evaluation metrics
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
    distances_to_goal = []
    
    # Action distribution tracking
    action_counts = np.zeros(config['action_dim'])
    
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
                
            # Track action distribution
            action_counts[action] += 1
                
            # Execute action
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Store final distance to goal
            if done or steps >= config.get('max_steps', 200) - 1:
                distances_to_goal.append(info['distance_to_goal'])
            
            # Render if requested (only for first few episodes)
            if render and episode < 3:
                print(f"Episode {episode}, Step {steps}: Position={state[:2]}, "  # Only show position
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
    std_length = np.std(episode_lengths)
    avg_distance = np.mean(distances_to_goal)
    std_distance = np.std(distances_to_goal)
    
    # Action distribution
    action_distribution = action_counts / np.sum(action_counts)
    
    results = {
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'success_rate': success_rate,
        'avg_episode_length': float(avg_length),
        'std_episode_length': float(std_length),
        'avg_final_distance': float(avg_distance),
        'std_final_distance': float(std_distance),
        'total_episodes': num_episodes,
        'successful_episodes': successes,
        'action_distribution': action_distribution.tolist()
    }
    
    logger.info(f"Detailed Evaluation Results over {num_episodes} episodes:")
    logger.info(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Average Episode Length: {avg_length:.2f} ± {std_length:.2f}")
    logger.info(f"  Average Final Distance to Goal: {avg_distance:.2f} ± {std_distance:.2f}")
    logger.info(f"  Action Distribution: {action_distribution}")
    
    return results


def compare_models(models_configs: dict, num_episodes: int = 100, 
                   env_type: str = 'simple') -> dict:
    """
    Compare multiple models and generate comparative analysis.
    
    Args:
        models_configs: Dictionary mapping model names to (model_path, config_path) tuples
        num_episodes: Number of episodes for evaluation
        env_type: Type of environment
        
    Returns:
        Dictionary containing comparison results
    """
    comparison_results = {}
    
    for name, (model_path, config_path) in models_configs.items():
        logger.info(f"Evaluating model: {name}")
        results = evaluate_model_detailed(model_path, config_path, num_episodes, env_type)
        comparison_results[name] = results
        
    return comparison_results


def plot_comparison(comparison_results: dict, save_path: str = None):
    """
    Plot comparison results.
    
    Args:
        comparison_results: Results from compare_models function
        save_path: Path to save the plot
    """
    models = list(comparison_results.keys())
    
    # Extract metrics
    success_rates = [comparison_results[model]['success_rate'] for model in models]
    avg_rewards = [comparison_results[model]['avg_reward'] for model in models]
    avg_lengths = [comparison_results[model]['avg_episode_length'] for model in models]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison')
    
    # Success rate
    axes[0].bar(models, success_rates, color='skyblue')
    axes[0].set_title('Success Rate')
    axes[0].set_ylabel('Rate')
    axes[0].set_ylim(0, 1)
    
    # Average reward
    axes[1].bar(models, avg_rewards, color='lightcoral')
    axes[1].set_title('Average Reward')
    axes[1].set_ylabel('Reward')
    
    # Average episode length
    axes[2].bar(models, avg_lengths, color='lightgreen')
    axes[2].set_title('Average Episode Length')
    axes[2].set_ylabel('Steps')
    
    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Enhanced evaluation of PPO agent for navigation planning')
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
    parser.add_argument('--compare', type=str, 
                        help='Path to JSON file with models to compare')
    parser.add_argument('--output', type=str,
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    # Run evaluation or comparison
    if args.compare and os.path.exists(args.compare):
        # Load comparison configuration
        with open(args.compare, 'r') as f:
            models_configs = json.load(f)
        
        logger.info("Starting model comparison...")
        results = compare_models(models_configs, args.episodes, args.env_type)
        
        # Plot comparison
        plot_comparison(results, 'model_comparison.png')
    else:
        # Run single model evaluation
        logger.info("Starting detailed model evaluation...")
        results = evaluate_model_detailed(
            model_path=args.model_path,
            config_path=args.config,
            num_episodes=args.episodes,
            env_type=args.env_type,
            render=args.render
        )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()