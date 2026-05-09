#!/usr/bin/env python3
"""
RL Planning Train CLI - Evaluation commands

Wraps the existing EvaluationRunner from eval.py.
"""

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import click
import logging

from eval import EvaluationRunner, print_results

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="训练好的模型权重文件路径 (.pt)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="配置文件路径 (YAML格式)",
)
@click.option(
    "--episodes",
    "-n",
    type=int,
    default=100,
    help="评估回合数 (默认: 100)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "auto"], case_sensitive=False),
    default="auto",
    help="评估设备",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="runs",
    help="结果输出目录",
)
@click.option(
    "--render",
    is_flag=True,
    help="渲染环境可视化",
)
@click.option(
    "--deterministic",
    is_flag=True,
    help="使用确定性动作选择",
)
@click.option(
    "--no-save",
    is_flag=True,
    help="不保存评估结果",
)
def evaluate(model_path, config, episodes, device, output_dir,
             render, deterministic, no_save):
    """
    评估训练好的RL模型

    \b
    示例:
        aurora eval --model-path models/nav_data_weights.pt --config config/nav_data_training.yaml --episodes 50
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    click.echo("Starting NAV_DATA evaluation...")
    click.echo(f"Model: {model_path}")
    click.echo(f"Episodes: {episodes}")

    # Create evaluation runner
    device_str = None if device == "auto" else device
    runner = EvaluationRunner(
        config_path=config,
        model_path=model_path,
        device=device_str
    )

    # Run evaluation
    if deterministic:
        click.echo("Using deterministic action selection")
        results = runner.evaluate_deterministic(episodes)
    else:
        results = runner.evaluate(
            num_episodes=episodes,
            save_results=not no_save,
            output_dir=output_dir
        )

    if render:
        click.echo("Note: render flag received but rendering is not yet supported.")

    # Print results
    print_results(results)

    click.echo("Evaluation completed!")
