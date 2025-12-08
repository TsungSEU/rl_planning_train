#!/usr/bin/env python3
"""
Curriculum learning training script for PPO path planning
Starts with simple tasks and gradually increases difficulty
"""

import numpy as np
import torch
import yaml
import argparse
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SimplePathPlanningEnv
from planner_rl_train import PPOTrainer, load_training_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_curriculum(config, stages=5):
    """
    Train using curriculum learning approach.
    
    Args:
        config: Training configuration
        stages: Number of curriculum stages
    """
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    trainer = None
    
    for stage in range(stages):
        logger.info(f"Starting curriculum stage {stage + 1}/{stages}")
        
        # Adjust environment difficulty based on stage
        if stage == 0:
            # Stage 1: Very close goal (2x2 grid)
            env_width, env_height = 2, 2
            start_pos = (0, 0)
            goal_pos = (1, 1)
        elif stage == 1:
            # Stage 2: Close goal (5x5 grid)
            env_width, env_height = 5, 5
            start_pos = (0, 0)
            goal_pos = (4, 4)
        elif stage == 2:
            # Stage 3: Medium distance (10x10 grid)
            env_width, env_height = 10, 10
            start_pos = (0, 0)
            goal_pos = (9, 9)
        elif stage == 3:
            # Stage 4: Far goal (15x15 grid)
            env_width, env_height = 15, 15
            start_pos = (0, 0)
            goal_pos = (14, 14)
        else:
            # Stage 5: Full environment (20x20 grid)
            env_width, env_height = 20, 20
            start_pos = None  # Random start
            goal_pos = None   # Random goal
            
        # Adjust episodes based on stage
        stage_episodes = config['episodes'] // stages
        
        # Initialize environment for this stage
        env = SimplePathPlanningEnv(width=env_width, height=env_height)
        
        # Initialize or reuse trainer
        if trainer is None:
            trainer = PPOTrainer(config)
        # Note: We keep the same trainer to preserve learned weights across stages
        
        # Training metrics for this stage
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        logger.info(f"Stage {stage + 1} environment: {env_width}x{env_height} grid")
        
        # Training loop for this stage
        for episode in range(stage_episodes):
            # Reset environment with specific start/goal positions for early stages
            if stage < 4:
                state = env.reset(start_pos=start_pos, goal_pos=goal_pos)
            else:
                state = env.reset()
                
            episode_reward = 0
            episode_length = 0
            
            # Collect episode trajectory
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            
            for step in range(config.get('max_steps', 200)):
                # Convert to tensor
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32)
                
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
                episode_length += 1
                
                if done:
                    success_count += 1
                    break
            
            # Update agent
            if len(states) > 0:
                trainer.update(states, actions, rewards, log_probs, values, dones)
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Periodic logging
            if episode % config.get('log_interval', 100) == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                success_rate = success_count / max(episode, 1)
                logger.info(f"Stage {stage + 1} - Episode {episode}/{stage_episodes}, "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Avg Length: {avg_length:.2f}, "
                           f"Success Rate: {success_rate:.2%}")
        
        # Save stage checkpoint
        stage_model_path = models_dir / f"curriculum_stage_{stage + 1}_weights.pth"
        torch.save(trainer.actor_critic.state_dict(), stage_model_path)
        logger.info(f"Stage {stage + 1} completed. Model saved to {stage_model_path}")
    
    # Save final model
    final_model_path = models_dir / "curriculum_final_weights.pth"
    torch.save(trainer.actor_critic.state_dict(), final_model_path)
    logger.info(f"Curriculum learning completed. Final model saved to {final_model_path}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Curriculum learning for PPO path planning')
    parser.add_argument('--config', type=str, default='config/advanced_ppo_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--stages', type=int, default=5,
                        help='Number of curriculum stages')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_training_config(args.config)
    
    logger.info("Starting curriculum learning...")
    trainer = train_curriculum(config, args.stages)
    
    logger.info("Curriculum learning finished!")


if __name__ == "__main__":
    main()