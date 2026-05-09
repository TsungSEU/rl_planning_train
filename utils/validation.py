#!/usr/bin/env python3
"""
Configuration validation utilities for training.

EarlyStopping and TrainingMetrics have been moved to utils/training_utils.py.
ReachabilityTracker has been moved to utils/reachability.py.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from utils.constants import PlannerMode

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates training configuration parameters.
    Ensures all required parameters are present and within valid ranges.
    """

    REQUIRED_KEYS = [
        'state_dim', 'action_dim', 'hidden_dim',
        'learning_rate', 'gamma', 'lam', 'epsilon',
        'epochs', 'batch_size', 'episodes',
    ]

    PARAM_RANGES = {
        'state_dim': (1, 512),
        'action_dim': (1, 64),
        'hidden_dim': (8, 1024),
        'learning_rate': (1e-6, 1e-1),
        'gamma': (0.9, 0.9999),
        'lam': (0.9, 0.9999),
        'epsilon': (0.01, 0.5),
        'epochs': (1, 100),
        'batch_size': (1, 4096),
        'episodes': (1, 1000000),
        'network_layers': (1, 10),
        'dropout_rate': (0.0, 0.9),
        'entropy_coef': (0.0, 1.0),
        'gradient_clip_value': (0.1, 10.0),
        'lr_decay_rate': (0.8, 0.9999),
        'reachability_decay': (0.5, 0.999),
        'reachability_min_samples': (1, 100),
        'reachability_position_tolerance': (0.1, 2.0),
        'goal_reward': (0.0, 2000.0),
        'collision_penalty': (-200.0, -0.1),
    }

    @classmethod
    def validate(cls, config: Dict[str, Any], raise_on_error: bool = True,
                 mode: Optional[PlannerMode] = None) -> tuple:
        errors = []

        if mode is None:
            mode_str = config.get('planner_mode', 'nav_data')
            try:
                mode = PlannerMode(mode_str)
            except ValueError:
                errors.append(f"Invalid planner_mode: {mode_str}")
                mode = PlannerMode.NAV_DATA

        for key in cls.REQUIRED_KEYS:
            if key not in config:
                errors.append(f"Missing required parameter: {key}")

        for key, (min_val, max_val) in cls.PARAM_RANGES.items():
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{key}' must be a number, got {type(value).__name__}")
                elif value < min_val or value > max_val:
                    errors.append(f"Parameter '{key}'={value} out of range [{min_val}, {max_val}]")

        if 'gamma' in config and config['gamma'] >= 1.0:
            errors.append("Parameter 'gamma' must be < 1.0 for proper discounting")
        if 'lam' in config and config['lam'] >= 1.0:
            errors.append("Parameter 'lam' must be < 1.0 for proper GAE")

        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} error(s):")
            for error in errors:
                logger.error(f"  - {error}")
        else:
            logger.info(f"Configuration validation passed for mode: {mode.value}")

        if raise_on_error and errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return len(errors) == 0, errors

    @classmethod
    def validate_config_file(cls, config_path: Union[str, Path],
                             raise_on_error: bool = True) -> tuple:
        try:
            from utils.config_loader import ConfigLoader
            config = ConfigLoader.load(config_path)
            config_dict = {
                'planner_mode': config.planner_mode.value,
                'state_dim': config.get_state_dim(),
                'action_dim': config.get_action_dim(),
            }
            return cls.validate(config_dict, raise_on_error, config.planner_mode)
        except Exception as e:
            errors = [f"Failed to load configuration file: {str(e)}"]
            if raise_on_error:
                raise ValueError("\n".join(errors))
            return False, errors
