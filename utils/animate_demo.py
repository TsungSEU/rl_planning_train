#!/usr/bin/env python3
"""
Animation demo for path planning environment
Creates animated GIFs showing agent movement in the environment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PathPlanningEnvironment, SimplePathPlanningEnv
from planner_rl_train import ActorCritic


def create_policy_animation(env, model=None, save_path='agent_movement.gif', max_steps=50):
    """
    Create an animation of agent moving in the environment.
    
    Args:
        env: Environment instance
        model: Trained model (optional, for intelligent actions)
        save_path: Path to save the animation
        max_steps: Maximum steps for the animation
    """
    # Reset environment
    state = env.reset()
    
    # Setup figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Initialize visualization elements
    # Extract actual position from state (first two elements)
    agent_pos = state[:2] * np.array([env.width-1, env.height-1])  # Denormalize
    agent_circle = plt.Circle(agent_pos, 0.3, color='blue')
    
    goal_pos = env.goal_pos
    goal_square = patches.Rectangle((goal_pos[0]-0.5, goal_pos[1]-0.5), 
                                   1, 1, linewidth=2, edgecolor='green', facecolor='none')
    
    # For complex environment, draw obstacles
    obstacles = []
    if hasattr(env, 'map'):
        for y in range(env.height):
            for x in range(env.width):
                if env.map[y, x]:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           linewidth=0, facecolor='black', alpha=0.8)
                    obstacles.append(rect)
                    ax.add_patch(rect)
    
    ax.add_patch(agent_circle)
    ax.add_patch(goal_square)
    
    # Formatting
    ax.set_xlim(-0.5, env.width-0.5)
    ax.set_ylim(-0.5, env.height-0.5)
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    
    # Title
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center')
    
    # Store positions for animation
    positions = [agent_pos.copy()]
    rewards = [0]
    done_flags = [False]
    
    # Simulate agent movement
    for step in range(max_steps):
        if done_flags[-1]:  # If already done, stop
            positions.append(positions[-1].copy())
            rewards.append(0)
            done_flags.append(True)
            continue
            
        # Select action
        if model is not None:
            # Use trained model to select action
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                logits, _ = model(state_tensor)
                # Use logits directly for action selection (argmax)
                action = torch.argmax(logits).item()
        else:
            # Random action
            action = np.random.choice(env.action_space)
        
        # Execute action
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
        # Extract actual position from state
        current_pos = state[:2] * np.array([env.width-1, env.height-1])  # Denormalize
        positions.append(current_pos.copy())
        rewards.append(reward)
        done_flags.append(done)
        
        if done:
            break
    
    # Animation update function
    def update(frame):
        # Update agent position
        agent_circle.center = positions[frame]
        
        # Update title
        status = "Goal Reached!" if done_flags[frame] else "Moving..."
        title.set_text(f'Step {frame} | Reward: {rewards[frame]:.2f} | Status: {status}')
        
        return agent_circle, title
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(positions), interval=500, blit=True, repeat=True)
    
    # Save animation
    anim.save(save_path, writer=PillowWriter(fps=2))
    print(f"Animation saved to {save_path}")
    
    plt.close(fig)
    return anim


def load_trained_model(model_path, config_path):
    """
    Load a trained model from file.
    """
    import yaml
    
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
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Create animation of agent in environment')
    parser.add_argument('--model-path', type=str, 
                        help='Path to trained model weights (optional, for intelligent actions)')
    parser.add_argument('--config-path', type=str, default='config/advanced_ppo_config.yaml',
                        help='Path to model configuration')
    parser.add_argument('--save-path', type=str, default='agent_movement.gif',
                        help='Path to save the animation')
    parser.add_argument('--env-type', type=str, default='simple',
                        choices=['simple', 'complex'],
                        help='Type of environment for animation')
    parser.add_argument('--obstacle-ratio', type=float, default=0.2,
                        help='Obstacle ratio for complex environment')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Maximum steps for animation')
    
    args = parser.parse_args()
    
    # Initialize environment
    if args.env_type == 'simple':
        env = SimplePathPlanningEnv(width=20, height=20)
    else:
        env = PathPlanningEnvironment(width=20, height=20, obstacle_ratio=args.obstacle_ratio)
    
    # Load model if provided
    model = None
    if args.model_path and Path(args.model_path).exists():
        try:
            model, config = load_trained_model(args.model_path, args.config_path)
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using random actions instead.")
    
    # Create animation
    print("Creating animation...")
    anim = create_policy_animation(
        env=env,
        model=model,
        save_path=args.save_path,
        max_steps=args.max_steps
    )
    
    print("Animation creation complete!")


if __name__ == "__main__":
    main()