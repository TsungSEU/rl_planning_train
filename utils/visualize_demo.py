#!/usr/bin/env python3
"""
Visualization demo for path planning environment
Creates side-by-side comparisons of different environments
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PathPlanningEnvironment, SimplePathPlanningEnv


def visualize_environments(save_path='static_environments.png'):
    """
    Create a side-by-side visualization of simple and complex environments.
    
    Args:
        save_path: Path to save the visualization
    """
    # Create both environments
    simple_env = SimplePathPlanningEnv(width=20, height=20)
    complex_env = PathPlanningEnvironment(width=20, height=20, obstacle_ratio=0.2)
    
    # Reset environments to initialize positions
    simple_env.reset()
    complex_env.reset()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Visualize simple environment
    # Draw agent
    agent_pos = simple_env.agent_pos
    agent_circle = plt.Circle(agent_pos, 0.3, color='blue')
    ax1.add_patch(agent_circle)
    
    # Draw goal
    goal_pos = simple_env.goal_pos
    goal_square = patches.Rectangle((goal_pos[0]-0.5, goal_pos[1]-0.5), 
                                   1, 1, linewidth=2, edgecolor='green', facecolor='none')
    ax1.add_patch(goal_square)
    
    # Formatting
    ax1.set_xlim(-0.5, simple_env.width-0.5)
    ax1.set_ylim(-0.5, simple_env.height-0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_xticks(range(simple_env.width))
    ax1.set_yticks(range(simple_env.height))
    ax1.set_title("Simple Environment (No Obstacles)")
    
    # Visualize complex environment
    # Draw obstacles
    for y in range(complex_env.height):
        for x in range(complex_env.width):
            if complex_env.map[y, x]:
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                       linewidth=0, facecolor='black', alpha=0.8)
                ax2.add_patch(rect)
    
    # Draw agent
    agent_pos = complex_env.agent_pos
    agent_circle = plt.Circle(agent_pos, 0.3, color='blue')
    ax2.add_patch(agent_circle)
    
    # Draw goal
    goal_pos = complex_env.goal_pos
    goal_square = patches.Rectangle((goal_pos[0]-0.5, goal_pos[1]-0.5), 
                                   1, 1, linewidth=2, edgecolor='green', facecolor='none')
    ax2.add_patch(goal_square)
    
    # Formatting
    ax2.set_xlim(-0.5, complex_env.width-0.5)
    ax2.set_ylim(-0.5, complex_env.height-0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(complex_env.width))
    ax2.set_yticks(range(complex_env.height))
    ax2.set_title("Complex Environment (With Obstacles)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Environments visualization saved to {save_path}")
    plt.close(fig)


def visualize_path_example(save_path_prefix='step'):
    """
    Create a sequence of images showing agent movement along a path.
    
    Args:
        save_path_prefix: Prefix for saving the images
    """
    # Create environment
    env = SimplePathPlanningEnv(width=10, height=10)
    env.reset(start_pos=(0, 0), goal_pos=(9, 9))
    
    # Predefined path (moving right then up)
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
    
    for i, pos in enumerate(path):
        # Update agent position
        env.agent_pos = np.array(pos, dtype=float)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw agent
        agent_circle = plt.Circle(env.agent_pos, 0.3, color='blue')
        ax.add_patch(agent_circle)
        
        # Draw goal
        goal_square = patches.Rectangle((env.goal_pos[0]-0.5, env.goal_pos[1]-0.5), 
                                       1, 1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(goal_square)
        
        # Formatting
        ax.set_xlim(-0.5, env.width-0.5)
        ax.set_ylim(-0.5, env.height-0.5)
        ax.set_aspect('equal')
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
        
        if i == 0:
            ax.set_title(f"Initial State")
            save_path = f'{save_path_prefix}_{i}_initial.png'
        else:
            ax.set_title(f"Step {i}: Action {0 if pos[0] > path[i-1][0] else 1}")
            save_path = f'{save_path_prefix}_{i}_action_{0 if pos[0] > path[i-1][0] else 1}.png'
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Path step visualization saved to {save_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize path planning environments')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['static', 'path'],
                        help='Visualization mode')
    parser.add_argument('--save-path', type=str, default='environments.png',
                        help='Path to save the visualization')
    
    args = parser.parse_args()
    
    if args.mode == 'static':
        visualize_environments(args.save_path)
    else:
        visualize_path_example(args.save_path.replace('.png', ''))


if __name__ == "__main__":
    main()