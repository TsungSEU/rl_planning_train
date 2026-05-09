#!/usr/bin/env python3
"""
RL Planning Train CLI - Main entry point

A unified command-line interface for training, evaluation, and deployment
of RL navigation + data collection models.
"""

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import click

from .commands.train import train
from .commands.eval import evaluate
from .commands.export import export_model


@click.group()
@click.version_option(version="0.8.0", prog_name="rl_train")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """
    RL Planning Train - RL导航与数据采集训练平台

    统一命令行工具，支持导航+数据采集场景的强化学习训练。

    \b
    示例:
        rl_train train --config config/nav_data_training.yaml --episodes 5000
        rl_train eval --model-path models/nav_data_weights.pt --config config/nav_data_training.yaml
        rl_train export --model-path models/nav_data_weights.pt --config config/nav_data_training.yaml --output models/nav_data.onnx
    """
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Register subcommands
cli.add_command(train)
cli.add_command(evaluate, name="eval")
cli.add_command(export_model, name="export")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
