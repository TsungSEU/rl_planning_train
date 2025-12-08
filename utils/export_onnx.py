#!/usr/bin/env python3
"""
Export trained model to ONNX format for deployment
Converts PyTorch model to ONNX for use with ONNX Runtime on vehicle side
"""

import torch
import yaml
import argparse
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planner_rl_train import ActorCritic


def export_to_onnx(model_path, config_path, output_path):
    """
    Export trained PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to trained model weights
        config_path (str): Path to model configuration file
        output_path (str): Path to save exported ONNX model
    """
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
    
    print(f"Loaded model from {model_path}")

    # Create dummy input tensor with the same shape as expected input
    dummy_input = torch.randn(1, config['state_dim'])
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output_policy', 'output_value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output_policy': {0: 'batch_size'},
            'output_value': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully exported to {output_path}")
    print("Model is ready for deployment with ONNX Runtime on vehicle side")


def main():
    parser = argparse.ArgumentParser(description='Export trained model to ONNX format')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--config', type=str, default='config/advanced_ppo_config.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--output', type=str, default='models/planner_model.onnx',
                        help='Path to save exported ONNX model')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return 1
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export model
    export_to_onnx(args.model_path, args.config, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())