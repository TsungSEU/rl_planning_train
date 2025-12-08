#!/usr/bin/env python3
"""
Benchmark script for comparing different planners
Measures performance metrics of trained models vs baselines
"""

import numpy as np
import torch
import yaml
import argparse
import logging
import time
from pathlib import Path
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SimplePathPlanningEnv
from planner_rl_train import ActorCritic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_random_policy(episodes=100):
    """
    Benchmark random policy baseline.
    
    Args:
        episodes: Number of episodes to run
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Benchmarking random policy...")
    
    # Initialize environment
    env = SimplePathPlanningEnv(width=20, height=20)
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    successes = 0
    inference_times = []
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            # Time the action selection
            start_time = time.time()
            
            # Random action
            action = np.random.choice(env.action_space)
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if done:
            successes += 1
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = successes / episodes
    avg_length = np.mean(episode_lengths)
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds
    
    results = {
        'policy_type': 'random',
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'success_rate': success_rate,
        'avg_episode_length': float(avg_length),
        'avg_inference_time_ms': float(avg_inference_time),
        'total_episodes': episodes
    }
    
    logger.info(f"Random Policy Results:")
    logger.info(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Average Episode Length: {avg_length:.2f}")
    logger.info(f"  Average Inference Time: {avg_inference_time:.4f} ms")
    
    return results


def benchmark_trained_model(model_path, config_path, episodes=100):
    """
    Benchmark trained model.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        episodes: Number of episodes to run
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking trained model from {model_path}...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    env = SimplePathPlanningEnv(width=20, height=20)
    
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
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    successes = 0
    inference_times = []
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            # Time the action selection
            start_time = time.time()
            
            # Get action from trained model
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                logits, _ = model(state_tensor)
                # Use logits directly for action selection (argmax)
                action = torch.argmax(logits).item()
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if done:
            successes += 1
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = successes / episodes
    avg_length = np.mean(episode_lengths)
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds
    
    results = {
        'policy_type': 'trained_model',
        'model_path': model_path,
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'success_rate': success_rate,
        'avg_episode_length': float(avg_length),
        'avg_inference_time_ms': float(avg_inference_time),
        'total_episodes': episodes
    }
    
    logger.info(f"Trained Model Results:")
    logger.info(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Average Episode Length: {avg_length:.2f}")
    logger.info(f"  Average Inference Time: {avg_inference_time:.4f} ms")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark path planning policies')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config/advanced_ppo_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for benchmarking')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Path to save benchmark results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    # Run benchmarks
    logger.info("Starting benchmarks...")
    
    # Benchmark random policy
    random_results = benchmark_random_policy(args.episodes)
    
    # Benchmark trained model
    model_results = benchmark_trained_model(args.model_path, args.config, args.episodes)
    
    # Combine results
    benchmark_results = {
        'random_policy': random_results,
        'trained_model': model_results
    }
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmarking completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()