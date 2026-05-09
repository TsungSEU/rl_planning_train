"""
Scenario factory — creates environments and agents from a PPOConfig.
"""

from typing import Optional

import torch

from .base_agent import ContinuousPPOAgent
from .base_environment import BaseEnvironment
from .config import PPOConfig

from humanoid.nav_data_environment import NavDataEnvironment
from humanoid.nav_data_reward import NavDataRewardConfig


class ScenarioFactory:
    """Create environments and agents for navigation + data collection."""

    @staticmethod
    def create_environment(config: PPOConfig) -> BaseEnvironment:
        """
        Create a NavDataEnvironment.

        Args:
            config: PPO configuration

        Returns:
            NavDataEnvironment instance
        """
        grid_res = config.grid_resolution
        reward_config = NavDataRewardConfig(**(config.reward or {}))
        return NavDataEnvironment(
            width=int(config.env_width / grid_res),
            height=int(config.env_height / grid_res),
            grid_resolution=grid_res,
            max_steps=config.max_steps,
            num_targets=config.num_targets,
            num_obstacles=config.num_obstacles,
            reward_config=reward_config,
        )

    @staticmethod
    def create_agent(
        config: PPOConfig,
        device: Optional[torch.device] = None,
    ) -> ContinuousPPOAgent:
        """
        Create a ContinuousPPOAgent.

        Args:
            config: PPO configuration
            device: Torch device

        Returns:
            ContinuousPPOAgent instance
        """
        return ContinuousPPOAgent(config, device)
