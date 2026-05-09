#!/usr/bin/env python3
"""
RL Planning Train CLI - Export commands

Wraps the existing export_to_onnx from utils/export_onnx.py.
"""

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import click
import logging

from core.config import load_config
from utils.export_onnx import export_to_onnx

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
    help="模型配置文件路径 (YAML格式)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="输出ONNX模型路径",
)
def export_model(model_path, config, output):
    """
    导出模型为ONNX格式

    将训练好的PyTorch模型导出为ONNX格式，用于机器人部署。

    \b
    注意:
        模型输出action_mean，需配合log_std使用

    \b
    示例:
        aurora export --model-path models/nav_data_weights.pt --config config/nav_data_training.yaml --output models/nav_data.onnx
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    click.echo("Exporting NAV_DATA model to ONNX...")
    click.echo(f"Model: {model_path}")
    click.echo(f"Output: {output}")

    config_obj = load_config(config)

    try:
        export_to_onnx(model_path, config_obj, output)
        click.echo(f"\nModel exported successfully to {output}")
    except Exception as e:
        click.echo(f"\nExport failed: {e}", err=True)
        raise click.Abort()
