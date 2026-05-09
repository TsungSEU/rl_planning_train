#!/usr/bin/env python3
"""
Training script for nav_data (hierarchical navigation + data collection).

Supports two modes:
- Serial (--num-envs 1): Single environment, episode-by-episode training
- Vectorized (--num-envs >1): N parallel environments, batch GPU inference

Output goes to runs/train/expNNN/ (auto-incremented).
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

from core.config import load_config, PPOConfig
from core.factory import ScenarioFactory
from core.vectorized_env import VectorizedEnvRunner
from core.base_agent import ContinuousPPOAgent
logger = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that routes output through tqdm.write to avoid breaking the progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class TrainingRunner:
    """
    Serial training runner (num_envs=1).
    """

    def __init__(self, config: PPOConfig, device: str = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        self.env = ScenarioFactory.create_environment(config)
        self.agent = ScenarioFactory.create_agent(config, self.device)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_count: int = 0

        logger.info(f"Serial training on {self.device}, state_dim={config.state_dim}, action_dim={config.action_dim}")

    def train(self, episodes: int = None, save_interval: int = None, update_interval: int = 4):
        num_episodes = episodes or self.config.episodes
        save_interval = save_interval or self.config.save_interval

        accumulated_states = []
        accumulated_actions = []
        accumulated_rewards = []
        accumulated_log_probs = []
        accumulated_values = []
        accumulated_dones = []

        pbar = tqdm(range(num_episodes), desc='Training', ncols=120, leave=True)

        for episode in pbar:
            result = self._run_episode()

            if result['states']:
                accumulated_states.extend(result['states'])
                accumulated_actions.extend(result['actions'])
                accumulated_rewards.extend(result['rewards'])
                accumulated_log_probs.extend(result['log_probs'])
                accumulated_values.extend(result['values'])
                accumulated_dones.extend(result['dones'])

            self.episode_rewards.append(result['reward'])
            self.episode_lengths.append(result['length'])
            if result['success']:
                self.success_count += 1

            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0
            success_rate = self.success_count / max(1, episode + 1)

            pbar.set_postfix(avg_r=f'{avg_reward:.1f}', avg_l=f'{avg_length:.0f}', succ=f'{success_rate:.0%}')

            if (episode + 1) % update_interval == 0 or episode == num_episodes - 1:
                if accumulated_states:
                    self.agent.update(
                        accumulated_states, accumulated_actions, accumulated_rewards,
                        accumulated_log_probs, accumulated_values, accumulated_dones
                    )
                    self.agent.decay_entropy()
                    accumulated_states.clear()
                    accumulated_actions.clear()
                    accumulated_rewards.clear()
                    accumulated_log_probs.clear()
                    accumulated_values.clear()
                    accumulated_dones.clear()

            if save_interval > 0 and (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)

        pbar.close()
        self._save_final()

    def _run_episode(self) -> Dict[str, Any]:
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        for step in range(self.config.max_steps):
            action, log_prob, value = self.agent.select_action(state)

            states.append(state.copy())
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

            state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            episode_length += 1

            if done:
                break

        success = info.get('all_collected', False) if isinstance(info, dict) else False

        return {
            'states': states, 'actions': actions, 'rewards': rewards,
            'log_probs': log_probs, 'values': values, 'dones': dones,
            'reward': episode_reward, 'length': episode_length,
            'success': success, 'info': info,
        }

    def _save_checkpoint(self, episode: int):
        path = self.exp_dir / f"nav_data_weights_iter_{episode}.pt"
        self.agent.save_weights(str(path))
        logger.info(f"Checkpoint saved: {path}")

    def _save_final(self):
        path = self.exp_dir / "nav_data_weights.pt"
        self.agent.save_weights(str(path))
        self._save_metrics()
        logger.info(f"Training completed. Model saved to {path}")

    def _save_metrics(self):
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_count': self.success_count,
        }
        np.save(self.exp_dir / "nav_data_training_metrics.npy", metrics)

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        self.agent.set_eval_mode()

        eval_rewards, eval_lengths, eval_successes = [], [], 0

        for _ in tqdm(range(num_episodes), desc='Evaluating'):
            state = self.env.reset()
            episode_reward, episode_length = 0.0, 0
            done = False

            while not done and episode_length < self.config.max_steps:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            if info.get('all_collected', False):
                eval_successes += 1

        self.agent.set_train_mode()

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': eval_successes / num_episodes,
        }


class VectorizedTrainingRunner:
    """
    Vectorized training runner (num_envs > 1).
    N parallel envs, batch GPU inference, steps_per_env per iteration.
    """

    def __init__(self, config: PPOConfig, num_envs: int, device: str = None):
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        env_factory = lambda: ScenarioFactory.create_environment(config)
        self.env_runner = VectorizedEnvRunner(env_factory, num_envs)
        self.agent = ScenarioFactory.create_agent(config, self.device)

        self.iteration = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_count: int = 0
        self.total_episodes: int = 0
        self.total_targets_collected: int = 0
        self.total_targets_available: int = 0

        # Curriculum learning
        self._recent_successes: List[bool] = []
        self._curriculum_stage = 0
        training_dict = getattr(config, 'planner_config', None)
        if training_dict is not None:
            training_dict = getattr(training_dict, 'training', None)
        self._curriculum_stages = []
        self._curriculum_threshold = 0.4
        self._curriculum_window = 100
        if training_dict is not None:
            self._curriculum_stages = getattr(training_dict, 'curriculum_stages', [])
            self._curriculum_threshold = getattr(training_dict, 'curriculum_success_threshold', 0.4)
            self._curriculum_window = getattr(training_dict, 'curriculum_window', 100)

    def train(self, max_iterations: int = None, save_interval: int = None):
        max_iterations = max_iterations or self.config.max_iterations
        save_interval = save_interval or self.config.save_interval

        steps_per_env = self.config.steps_per_env
        max_steps_per_episode = self.config.max_steps
        total_steps_per_iter = self.num_envs * steps_per_env
        estimated_eps_per_iter = total_steps_per_iter / max_steps_per_episode

        pbar = tqdm(range(max_iterations), desc=f'VecTrain nav_data', ncols=150, leave=True, mininterval=0.5)

        # Reset all envs ONCE before training; step_batch handles per-env resets on done
        states = self.env_runner.reset_all()

        iter_start = time.time()
        for iteration in pbar:
            self.iteration = iteration

            accumulated_states = []
            accumulated_actions = []
            accumulated_rewards = []
            accumulated_log_probs = []
            accumulated_values = []
            accumulated_dones = []

            for step in range(steps_per_env):
                actions, log_probs, values = self.agent.select_actions_batch(states)

                next_states, rewards, dones, infos = self.env_runner.step_batch(actions)

                # Track episode completions and successes (only on done)
                for i, info in enumerate(infos):
                    if dones[i] and info is not None:
                        self.total_targets_collected += info.get('collected_targets', 0)
                        self.total_targets_available += info.get('total_targets', 0)
                        is_success = info.get('all_collected', False)
                        if is_success:
                            self.success_count += 1
                        self._recent_successes.append(is_success)
                self.total_episodes += int(np.sum(dones))

                accumulated_states.append(states)
                accumulated_actions.append(actions)
                accumulated_rewards.append(rewards)
                accumulated_log_probs.append(log_probs)
                accumulated_values.append(values)
                accumulated_dones.append(dones)

                states = next_states

            # Stack to (steps_per_env, num_envs, ...) then transpose to env-major
            # for correct per-env GAE computation.
            rewards_2d = np.stack(accumulated_rewards, axis=0)       # (steps, envs)
            values_2d = np.stack(accumulated_values, axis=0)         # (steps, envs)
            dones_2d = np.stack(accumulated_dones, axis=0)           # (steps, envs)

            # Compute bootstrap values (V(s) of state AFTER last collected step)
            bootstrap_values = self.agent.compute_bootstrap_values(states)  # (envs,)

            # Compute GAE per-env with proper bootstrapping
            advantages_2d = ContinuousPPOAgent._compute_gae_per_env(
                rewards_2d, values_2d, dones_2d, bootstrap_values,
                self.config.gamma, self.config.lam,
            )  # (steps, envs)

            # Flatten to env-major order
            _s = np.stack(accumulated_states, axis=0)       # (steps, envs, state_dim)
            accumulated_states = _s.transpose(1, 0, 2).reshape(-1, _s.shape[2])
            _a = np.stack(accumulated_actions, axis=0)      # (steps, envs, action_dim)
            accumulated_actions = _a.transpose(1, 0, 2).reshape(-1, _a.shape[2])
            accumulated_rewards = rewards_2d.T.ravel()
            advantages_flat = advantages_2d.T.ravel()
            accumulated_log_probs = np.stack(accumulated_log_probs, axis=0).T.ravel()
            values_flat = values_2d.T.ravel()
            accumulated_dones = dones_2d.T.ravel()

            # Update agent with pre-computed advantages
            self.agent.update_from_arrays_with_advantages(
                accumulated_states, accumulated_actions, accumulated_rewards,
                accumulated_log_probs, values_flat, accumulated_dones,
                advantages_flat,
            )
            self.agent.decay_entropy()

            # Progress bar update
            mean_reward = np.mean(accumulated_rewards)
            elapsed = time.time() - iter_start
            iter_speed = elapsed / (iteration + 1) if iteration >= 0 else 0
            eps_completed = self.total_episodes

            # steps per second: total steps collected so far / elapsed time
            total_steps_done = (iteration + 1) * total_steps_per_iter
            steps_s = total_steps_done / elapsed if elapsed > 0 else 0

            # Success rate and target collection rate
            success_rate = self.success_count / max(1, eps_completed)
            target_rate = self.total_targets_collected / max(1, self.total_targets_available)

            # Curriculum advancement
            cur_stage_name = ""
            if self._curriculum_stages and self._curriculum_stage < len(self._curriculum_stages):
                cur_stage_name = f"[{self._curriculum_stages[self._curriculum_stage].get('name', '')}]"
                recent_rate = (sum(self._recent_successes[-self._curriculum_window:])
                               / max(1, len(self._recent_successes[-self._curriculum_window:])))
                if (recent_rate >= self._curriculum_threshold
                        and len(self._recent_successes) >= self._curriculum_window
                        and self._curriculum_stage < len(self._curriculum_stages) - 1):
                    self._curriculum_stage += 1
                    stage = self._curriculum_stages[self._curriculum_stage]
                    logger.info(f"Curriculum: advancing to stage {stage['name']} "
                                f"(recent rate={recent_rate:.0%})")
                    self.env_runner.set_difficulty_for_all(stage)
                    self._recent_successes.clear()

            pbar.set_postfix(
                avg_r=f'{mean_reward:.1f}',
                eps=eps_completed,
                tgt=f'{target_rate:.0%}',
                succ=f'{success_rate:.0%}',
                cur=cur_stage_name,
            )

            # Save checkpoint
            if save_interval > 0 and (iteration + 1) % save_interval == 0:
                self._save_checkpoint(iteration + 1)

        pbar.close()
        self._save_final()

    def _save_checkpoint(self, iteration: int):
        path = self.exp_dir / f"nav_data_weights_iter_{iteration}.pt"
        self.agent.save_weights(str(path))

    def _save_final(self):
        path = self.exp_dir / "nav_data_weights.pt"
        self.agent.save_weights(str(path))
        self._save_metrics()
        logger.info(f"Training completed. Model saved to {path}")

    def _save_metrics(self):
        metrics = {
            'iteration': self.iteration,
            'total_episodes': self.total_episodes,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(1, self.total_episodes),
        }
        np.save(self.exp_dir / "nav_data_training_metrics.npy", metrics)
        logger.info(f"Metrics saved to {self.exp_dir / 'nav_data_training_metrics.npy'}")


def _setup_exp_dir(base: str = "runs/train") -> Path:
    """Create auto-incremented experiment directory."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("exp")]
    nums = [int(d.name[3:]) for d in existing if d.name[3:].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    exp_dir = base_path / f"exp{next_num:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description='Train nav_data RL agent')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--num-envs', type=int, default=None, help='Parallel envs (1=serial, >1=vectorized)')
    parser.add_argument('--episodes', type=int, default=None, help='Training episodes (serial mode)')
    parser.add_argument('--max-iterations', type=int, default=None, help='Training iterations (vectorized mode)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--save-interval', type=int, default=None)
    parser.add_argument('--update-interval', type=int, default=4, help='Episodes per update (serial only)')
    parser.add_argument('--model-path', type=str, default=None, help='Checkpoint to continue from')
    parser.add_argument('--exp-dir', type=str, default=None, help='Experiment output directory')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=100)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = load_config(args.config)

    if args.episodes is not None:
        config.episodes = args.episodes

    num_envs = args.num_envs if args.num_envs is not None else config.num_envs

    # Replace default handler with tqdm-safe one for vectorized mode
    if num_envs > 1:
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        _handler = _TqdmLoggingHandler()
        _handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(_handler)
        root_logger.setLevel(logging.INFO)

    device_str = None if args.device == 'auto' else args.device

    # Setup experiment directory
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {exp_dir}")
    else:
        exp_dir = _setup_exp_dir()

    if num_envs == 1:
        runner = TrainingRunner(config, device_str)
    else:
        max_iters = args.max_iterations or config.max_iterations
        logger.info(f"Vectorized mode on {device_str or torch.device('cuda' if torch.cuda.is_available() else 'cpu')}, {num_envs} envs, state_dim={config.state_dim}, action_dim={config.action_dim}")
        logger.info(f"Output directory: {exp_dir}")
        logger.info(f"Starting vectorized training: {max_iters} iterations × {num_envs} envs × {config.steps_per_env} steps/iter (max_ep={config.max_steps}) = ~{num_envs * max_iters * config.steps_per_env / config.max_steps:.0f} episodes")
        runner = VectorizedTrainingRunner(config, num_envs, device_str)

    runner.exp_dir = exp_dir

    if args.model_path:
        logger.info(f"Loading checkpoint from {args.model_path}")
        runner.agent.load_weights(args.model_path)

    if num_envs == 1:
        runner.train(episodes=args.episodes, save_interval=args.save_interval, update_interval=args.update_interval)
    else:
        runner.train(max_iterations=args.max_iterations, save_interval=args.save_interval)

    if args.evaluate:
        if num_envs == 1:
            results = runner.evaluate(args.eval_episodes)
        else:
            # Vectorized eval not implemented yet; fall back to serial eval
            serial_runner = TrainingRunner(config, device_str)
            serial_runner.agent.load_weights(str(exp_dir / "nav_data_weights.pt"))
            serial_runner.exp_dir = exp_dir
            results = serial_runner.evaluate(args.eval_episodes)

        print(f"\n{'='*50}")
        print(f"Evaluation Results")
        print(f"{'='*50}")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*50}\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()