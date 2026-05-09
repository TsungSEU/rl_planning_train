"""
RL Planning Train - Reinforcement Learning Path Planning Training Platform

A unified platform for training RL agents for humanoid robot navigation and data collection.
"""

__version__ = "0.8.0"
__author__ = "Aurora Planning Team"

# Import key classes for easy access
from core.config import PPOConfig, load_config
from utils.config_loader import PlannerConfig, ConfigLoader

__all__ = [
    "__version__",
    "PPOConfig",
    "load_config",
    "PlannerConfig",
    "ConfigLoader",
]
