#!/usr/bin/env python3
"""
Final evaluation script for trained models
Tests models on specific scenarios to demonstrate learning
"""

import torch
import yaml
import argparse
import logging
import json
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.environment import SimplePathPlanningEnv
from planner_train import ActorCritic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_on_specific_tasks(model_path: str, config_path: str) -> dict:
    """
    Evaluate model on specific predetermined tasks to demonstrate learning.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        
    Returns:
        Evaluation results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # Define specific test cases with denser and wider distribution
    test_cases = [
        # Very short distances (1-3 units)
        {"start": (0, 0), "goal": (1, 1)},
        {"start": (5, 5), "goal": (6, 5)},
        {"start": (10, 10), "goal": (10, 12)},
        {"start": (3, 7), "goal": (5, 7)},
        {"start": (15, 2), "goal": (15, 4)},
        
        # Short distances (4-7 units)
        {"start": (0, 0), "goal": (4, 4)},
        {"start": (2, 3), "goal": (6, 3)},
        {"start": (12, 8), "goal": (12, 13)},
        {"start": (7, 1), "goal": (11, 5)},
        {"start": (16, 16), "goal": (19, 19)},
        
        # Medium distances (8-15 units)
        {"start": (0, 0), "goal": (9, 9)},
        {"start": (1, 1), "goal": (10, 15)},
        {"start": (5, 5), "goal": (15, 10)},
        {"start": (3, 17), "goal": (18, 3)},
        {"start": (19, 0), "goal": (0, 19)},
        {"start": (10, 10), "goal": (0, 19)},
        {"start": (8, 2), "goal": (2, 18)},
        
        # Long distances (16+ units) on standard map
        {"start": (0, 0), "goal": (19, 19)},
        {"start": (19, 0), "goal": (0, 19)},
        {"start": (0, 19), "goal": (19, 0)},
        {"start": (9, 0), "goal": (9, 19)},
        {"start": (0, 9), "goal": (19, 9)},
        
        # Very long distances on large map
        {"start": (0, 0), "goal": (50, 50), "width": 51, "height": 51},
        {"start": (0, 0), "goal": (100, 100), "width": 101, "height": 101},
        {"start": (25, 25), "goal": (75, 75), "width": 101, "height": 101},
        {"start": (0, 50), "goal": (100, 50), "width": 101, "height": 101},
        {"start": (50, 0), "goal": (50, 100), "width": 101, "height": 101},
        
        # Edge cases
        {"start": (0, 0), "goal": (19, 0)},
        {"start": (0, 0), "goal": (0, 19)},
        {"start": (19, 19), "goal": (0, 19)},
        {"start": (19, 19), "goal": (19, 0)},
    ]
    
    results = {}
    
    for i, case in enumerate(test_cases):
        logger.info(f"Testing case {i+1}: Start {case['start']} -> Goal {case['goal']}")
        
        # Initialize environment with specific start/goal
        # Check if this is a large map case
        if 'width' in case and 'height' in case:
            env = SimplePathPlanningEnv(width=case['width'], height=case['height'])
        else:
            env = SimplePathPlanningEnv(width=20, height=20)
            
        state = env.reset(start_pos=case['start'], goal_pos=case['goal'])
        
        total_reward = 0
        steps = 0
        max_steps = 300  # Increased limit for longer paths
        done = False
        
        # Track path for visualization
        path = [state[:2].copy()]  # Only track actual positions
        
        # Execute episode
        while not done and steps < max_steps:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                logits, _ = model(state_tensor, use_softmax=False)
                # Use logits directly for action selection (argmax)
                action = torch.argmax(logits).item()  # Greedy action selection
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            path.append(state[:2].copy())  # Only track actual positions
            
        # Record results
        case_key = f"case_{i+1}_{case['start']}_to_{case['goal']}"
        results[case_key] = {
            "start": case['start'],
            "goal": case['goal'],
            "reward": float(total_reward),
            "steps": steps,
            "success": bool(done),
            "final_distance": float(info['distance_to_goal']),
            "path": [pos.tolist() for pos in path]
        }
        
        status = "SUCCESS" if done else "FAILED"
        logger.info(f"  Result: {status} | Reward: {total_reward:.2f} | Steps: {steps}")
    
    # Calculate overall statistics
    successful_cases = sum(1 for r in results.values() if r['success'])
    total_cases = len(results)
    success_rate = successful_cases / total_cases if total_cases > 0 else 0
    
    results['summary'] = {
        'successful_cases': successful_cases,
        'total_cases': total_cases,
        'success_rate': success_rate
    }
    
    logger.info(f"Overall Success Rate: {success_rate:.2%} ({successful_cases}/{total_cases})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Final evaluation of trained models')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config/ppo_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='final_evaluation.json',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    logger.info("Starting final evaluation...")
    results = evaluate_on_specific_tasks(args.model_path, args.config)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {args.output}")
    

if __name__ == "__main__":
    main()