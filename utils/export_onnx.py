#!/usr/bin/env python3
"""
Export trained model to ONNX format for deployment.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PPOConfig, load_config
from core.base_agent import ActorCriticNet

logger = logging.getLogger(__name__)


class ONNXExportWrapper(nn.Module):
    """Wrapper for continuous action space ONNX export - avoids dynamic control flow."""

    def __init__(self, actor_critic: ActorCriticNet):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, state: torch.Tensor):
        """Forward pass - returns action mean for continuous actions."""
        shared_features = self.actor_critic.shared_layers(state)
        action_mean = self.actor_critic.actor_head(shared_features)
        value = self.actor_critic.critic_head(shared_features)
        return action_mean, value


def export_to_onnx(
    model_path: str,
    config: PPOConfig,
    output_path: str,
    sample_input: Optional[torch.Tensor] = None
):
    """
    Export robot model to ONNX format.

    Args:
        model_path: Path to trained model weights
        config: PPO configuration
        output_path: Path to save exported ONNX model
        sample_input: Optional sample input for export
    """
    # Initialize model with custom architecture support
    model = ActorCriticNet(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.network_layers,
        dropout_rate=config.dropout_rate,
        action_space='continuous',
        hidden_dims=getattr(config, 'hidden_dims', None),
        activation=getattr(config, 'activation', 'ReLU'),
    )

    # Load trained weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract log_std for separate export
    log_std = state_dict.pop('log_std', torch.zeros(config.action_dim))

    # Load network weights
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Loaded log_std: {log_std}")

    # Create wrapper for ONNX export
    wrapped_model = ONNXExportWrapper(model)
    wrapped_model.eval()

    # Create dummy input
    if sample_input is None:
        sample_input = torch.randn(1, config.state_dim)

    # Export actor-critic to ONNX
    torch.onnx.export(
        wrapped_model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_mean', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action_mean': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

    # Export log_std separately
    log_std_path = output_path.replace('.onnx', '_log_std.pt')
    torch.save(log_std, log_std_path)

    logger.info(f"Model exported to {output_path}")
    logger.info(f"Log std saved to {log_std_path}")
    logger.info("Note: On deployment side, combine action_mean with exp(log_std) for Gaussian policy")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export trained model to ONNX format for deployment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model configuration file (YAML)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save exported ONNX model'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return 1

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1

    # Load configuration
    config = load_config(args.config)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export model
    try:
        export_to_onnx(args.model_path, config, args.output)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify the exported model
    logger.info("Verifying exported model...")
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        logger.info("Exported model is valid ONNX")
    except ImportError:
        logger.warning("onnx package not installed - skipping verification")
    except Exception as e:
        logger.warning(f"Model verification failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
