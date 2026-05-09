"""
Base environment class for RL training.
Provides Gym-compatible interface.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    """
    Abstract base class for RL environments.
    Provides Gym-compatible interface with reset() and step() methods.
    """

    def __init__(self, width: int = 50, height: int = 50, max_steps: int = 200):
        """
        Initialize the base environment.

        Args:
            width: Environment width
            height: Environment height
            max_steps: Maximum steps per episode
        """
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.step_count = 0

        # To be defined by subclasses
        self._state_dim = None
        self._action_dim = None
        self._action_space_type = None  # 'discrete' or 'continuous'

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """State space dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action space dimension."""
        pass

    @property
    @abstractmethod
    def action_space_type(self) -> str:
        """Action space type: 'discrete' or 'continuous'."""
        pass

    @abstractmethod
    def reset(self, **kwargs) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get current state observation.

        Returns:
            Current state vector
        """
        pass

    @abstractmethod
    def compute_reward(self, **kwargs) -> float:
        """
        Compute reward for current state.

        Returns:
            Reward value
        """
        pass

    def is_done(self) -> bool:
        """
        Check if episode is done.

        Returns:
            True if episode should end
        """
        return self.step_count >= self.max_steps

    def get_action_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get action bounds for continuous action spaces.

        Returns:
            Tuple of (action_low, action_high) or None for discrete actions
        """
        if self.action_space_type == 'continuous':
            return np.ones(self.action_dim) * -1.0, np.ones(self.action_dim)
        return None

    def render(self, mode: str = 'human', ax: Optional[plt.Axes] = None, save_path: Optional[str] = None):
        """
        Render the environment.

        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            ax: Matplotlib axis to render on
            save_path: Path to save figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            self._render_on_axis(ax)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
        else:
            self._render_on_axis(ax)

    @abstractmethod
    def _render_on_axis(self, ax: plt.Axes):
        """
        Render environment on given matplotlib axis.

        Args:
            ax: Matplotlib axis
        """
        pass

    def close(self):
        """Clean up environment resources."""
        pass

    def seed(self, seed: int):
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed
        """
        np.random.seed(seed)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"action_space={self.action_space_type})")
