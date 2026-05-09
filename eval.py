#!/usr/bin/env python3
"""
Evaluation script for nav_data (hierarchical navigation + data collection).

Supports deterministic and stochastic evaluation modes.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from tqdm import tqdm

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

from core.config import load_config, PPOConfig
from core.factory import ScenarioFactory

logger = logging.getLogger(__name__)


def print_results(results: Dict[str, Any]):
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*50}\n")


class EvaluationRunner:
    def __init__(self, config_path: str, model_path: str, device: str = None):
        self.config = load_config(config_path)
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        self.env = ScenarioFactory.create_environment(self.config)
        self.agent = ScenarioFactory.create_agent(self.config, self.device)
        self.agent.load_weights(model_path)

        logger.info(f"Evaluation on {self.device}, model: {model_path}")

    def evaluate(self, num_episodes: int = 100, save_results: bool = True, output_dir: str = "runs") -> Dict[str, Any]:
        self.agent.set_eval_mode()

        episode_rewards = []
        episode_lengths = []
        successes = 0

        for _ in tqdm(range(num_episodes), desc='Evaluating'):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done and episode_length < self.config.max_steps:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if isinstance(info, dict) and info.get('all_collected', False):
                successes += 1

        self.agent.set_train_mode()

        results = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': successes / num_episodes,
            'successes': successes,
        }

        if save_results:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "nav_data_eval_results.npy"
            np.save(results_file, results)
            logger.info(f"Results saved to {results_file}")

        return results

    def evaluate_deterministic(self, num_episodes: int = 10) -> Dict[str, Any]:
        self.agent.set_eval_mode()

        episode_rewards = []
        episode_lengths = []
        successes = 0

        for _ in tqdm(range(num_episodes), desc='Deterministic eval'):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done and episode_length < self.config.max_steps:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if isinstance(info, dict) and info.get('all_collected', False):
                successes += 1

        self.agent.set_train_mode()

        return {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': successes / num_episodes,
            'successes': successes,
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate nav_data RL agent')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--model-path', '-m', type=str, required=True, help='Model weights path (.pt)')
    parser.add_argument('--episodes', '-n', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic action selection')
    parser.add_argument('--output-dir', type=str, default='runs', help='Directory to save results')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return 1

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1

    runner = EvaluationRunner(args.config, args.model_path, device=None if args.device == 'auto' else args.device)

    if args.deterministic:
        results = runner.evaluate_deterministic(args.episodes)
    else:
        results = runner.evaluate(
            num_episodes=args.episodes,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )

    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())