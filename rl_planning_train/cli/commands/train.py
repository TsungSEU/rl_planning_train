#!/usr/bin/env python3
"""
RL Planning Train CLI - Training command for nav_data scenario.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import click
import logging
import torch
from pathlib import Path

from core.config import load_config
from train import TrainingRunner, VectorizedTrainingRunner, _setup_exp_dir

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="配置文件路径 (YAML)")
@click.option("--num-envs", type=int, default=None, help="并行环境数 (1=串行, >1=向量化)")
@click.option("--episodes", type=int, default=None, help="训练回合数 (串行模式)")
@click.option("--max-iterations", type=int, default=None, help="训练迭代数 (向量化模式)")
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"], case_sensitive=False), default="auto")
@click.option("--save-interval", type=int, default=None, help="保存间隔")
@click.option("--update-interval", type=int, default=4, help="累积N回合后更新 (串行模式)")
@click.option("--model-path", type=click.Path(exists=True), default=None, help="预训练模型路径")
@click.option("--exp-dir", type=str, default=None, help="实验输出目录")
@click.option("--evaluate", is_flag=True, default=False, help="训练后运行评估")
@click.option("--eval-episodes", type=int, default=100, help="评估回合数")
def train(config, num_envs, episodes, max_iterations, device, save_interval,
          update_interval, model_path, exp_dir, evaluate, eval_episodes):
    """
    训练 Aurora 高层导航策略 (nav_data 场景)

    示例:
        aurora train --config config/nav_data_training.yaml
        aurora train --config config/nav_data_training.yaml --num-envs 4096
        aurora train --config config/nav_data_training.yaml --num-envs 1 --episodes 5000
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    click.echo(f"Training nav_data scenario...")
    click.echo(f"Config: {config}")

    cfg = load_config(config)

    if episodes is not None:
        cfg.episodes = episodes

    if max_iterations is not None:
        cfg.max_iterations = max_iterations

    num_envs_actual = num_envs if num_envs is not None else cfg.num_envs
    device_str = None if device == "auto" else device

    # Setup experiment directory
    if exp_dir:
        exp_path = Path(exp_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {exp_path}")
    else:
        exp_path = _setup_exp_dir()

    # Create runner
    if num_envs_actual == 1:
        runner = TrainingRunner(cfg, device_str)
    else:
        max_iters = max_iterations or cfg.max_iterations
        actual_device = device_str or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Vectorized mode on {actual_device}, {num_envs_actual} envs, state_dim={cfg.state_dim}")
        logger.info(f"Output directory: {exp_path}")
        logger.info(f"Starting vectorized training: {max_iters} iterations × {num_envs_actual} envs × {cfg.steps_per_env} steps/iter (max_ep={cfg.max_steps}) = ~{num_envs_actual * max_iters * cfg.steps_per_env / cfg.max_steps:.0f} episodes")
        runner = VectorizedTrainingRunner(cfg, num_envs_actual, device_str)

    runner.exp_dir = exp_path

    if model_path:
        click.echo(f"Loading checkpoint from {model_path}")
        runner.agent.load_weights(model_path)

    # Train
    if num_envs_actual == 1:
        runner.train(episodes=episodes, save_interval=save_interval, update_interval=update_interval)
    else:
        runner.train(max_iterations=max_iterations, save_interval=save_interval)

    # Evaluate
    if evaluate:
        click.echo(f"\nEvaluating for {eval_episodes} episodes...")
        if num_envs_actual == 1:
            results = runner.evaluate(eval_episodes)
        else:
            serial_runner = TrainingRunner(cfg, device_str)
            serial_runner.agent.load_weights(str(exp_path / "nav_data_weights.pt"))
            serial_runner.exp_dir = exp_path
            results = serial_runner.evaluate(eval_episodes)

        click.echo(f"\n{'='*50}")
        click.echo("Evaluation Results")
        click.echo(f"{'='*50}")
        for key, value in results.items():
            click.echo(f"  {key}: {value:.4f}")
        click.echo(f"{'='*50}\n")

    click.echo("Training completed!")