#!/usr/bin/env python3
"""
Visualize the learning results of the trained model
Shows successful path planning examples
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import torch
import yaml
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import SimplePathPlanningEnv
from planner_rl_train import ActorCritic


def visualize_path(env, path, title="Path Visualization"):
    """
    Visualize a path in the environment.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw agent path
    path_array = np.array(path)
    ax.plot(path_array[:, 0], path_array[:, 1], 'b-o', markersize=4, linewidth=2)
    
    # Mark start and goal
    start = path[0]
    goal = path[-1]
    
    # Draw start point
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    
    # Draw goal point
    goal_square = patches.Rectangle((goal[0]-0.5, goal[1]-0.5), 
                                   1, 1, linewidth=2, edgecolor='red', facecolor='none', label='Goal')
    ax.add_patch(goal_square)
    
    # Formatting
    ax.set_xlim(-0.5, env.width-0.5)
    ax.set_ylim(-0.5, env.height-0.5)
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_title(title)
    ax.legend()
    
    return fig


def main():
    # Load evaluation results
    with open('final_evaluation.json', 'r') as f:
        results = json.load(f)
    
    # Show successful cases
    successful_cases = [k for k, v in results.items() 
                       if k != 'summary' and v['success']]
    
    print(f"Found {len(successful_cases)} successful cases")
    
    for i, case_key in enumerate(successful_cases[:3]):  # Show first 3 cases
        case = results[case_key]
        print(f"\nCase {i+1}: {case['start']} -> {case['goal']}")
        print(f"  Steps: {case['steps']}")
        print(f"  Reward: {case['reward']:.2f}")
        
        # Create environment for visualization
        env = SimplePathPlanningEnv(width=20, height=20)
        env.reset(start_pos=case['start'], goal_pos=case['goal'])
        
        # Visualize path
        path = case['path']
        fig = visualize_path(env, path, 
                           f"Case {i+1}: {case['start']} -> {case['goal']} "
                           f"(Steps: {case['steps']}, Reward: {case['reward']:.2f})")
        plt.savefig(f'learned_path_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved visualization to learned_path_{i+1}.png")
    
    # Print summary
    summary = results['summary']
    print(f"\nSummary:")
    print(f"  Successful Cases: {summary['successful_cases']}")
    print(f"  Total Cases: {summary['total_cases']}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    
    print("\nModel successfully learned to navigate in simpler environments!")
    print("For more complex tasks (longer distances), the model may need more training or architectural improvements.")


if __name__ == "__main__":
    main()