"""
Base reward calculator for RL training.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardState:
    """State information for reward calculation."""

    # Agent state
    position: np.ndarray
    goal_position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    orientation: Optional[float] = None

    # Environment state
    step_count: int = 0
    max_steps: int = 200
    is_collision: bool = False
    is_goal_reached: bool = False
    is_fallen: bool = False

    # History
    trajectory: Optional[list] = None
    visited_cells: Optional[np.ndarray] = None
    previous_position: Optional[np.ndarray] = None

    # Data collection
    data_scarcity: float = 0.0
    coverage_ratio: float = 0.0
    data_quality: float = 1.0
    scene_diversity: float = 0.0

    # Energy and efficiency
    energy_consumed: float = 0.0
    path_length: float = 0.0
    direct_distance: float = 0.0


@dataclass
class RewardBreakdown:
    """Breakdown of reward components."""

    total: float
    distance: float = 0.0
    time_penalty: float = 0.0
    goal: float = 0.0
    collision: float = 0.0
    data: float = 0.0
    coverage: float = 0.0
    efficiency: float = 0.0
    repetition: float = 0.0
    stability: float = 0.0
    safety: float = 0.0
    energy: float = 0.0
    smoothness: float = 0.0
    fall: float = 0.0
    navigation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total': self.total,
            'distance': self.distance,
            'time_penalty': self.time_penalty,
            'goal': self.goal,
            'collision': self.collision,
            'data': self.data,
            'coverage': self.coverage,
            'efficiency': self.efficiency,
            'repetition': self.repetition,
            'stability': self.stability,
            'safety': self.safety,
            'energy': self.energy,
            'smoothness': self.smoothness,
            'fall': self.fall,
            'navigation': self.navigation,
        }


class BaseRewardCalculator(ABC):
    """
    Abstract base class for reward calculators.
    Supports multi-component reward functions with configurable weights.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reward calculator.

        Args:
            config: Configuration dictionary with reward weights
        """
        self.config = config or {}

    @abstractmethod
    def compute(self, state: RewardState) -> RewardBreakdown:
        """
        Compute reward breakdown for given state.

        Args:
            state: Current reward state

        Returns:
            RewardBreakdown with all components
        """
        pass

    def compute_total(self, state: RewardState) -> float:
        """
        Compute total reward for given state.

        Args:
            state: Current reward state

        Returns:
            Total reward value
        """
        breakdown = self.compute(state)
        return breakdown.total

    def distance_reward(self, state: RewardState) -> float:
        """
        Compute distance-based reward (progress toward goal).

        Args:
            state: Current reward state

        Returns:
            Distance reward
        """
        if state.goal_position is None:
            return 0.0

        if state.previous_position is None:
            return 0.0

        old_distance = np.linalg.norm(state.previous_position - state.goal_position)
        new_distance = np.linalg.norm(state.position - state.goal_position)

        # Reward for getting closer to goal
        weight = self.config.get('w_distance', 2.0)
        return (old_distance - new_distance) * weight

    def time_penalty(self, state: RewardState) -> float:
        """
        Compute time penalty to encourage efficiency.

        Args:
            state: Current reward state

        Returns:
            Time penalty (negative)
        """
        weight = self.config.get('w_time_penalty', 0.1)
        return -weight

    def goal_reward(self, state: RewardState) -> float:
        """
        Compute reward for reaching goal.

        Args:
            state: Current reward state

        Returns:
            Goal reward
        """
        if not state.is_goal_reached:
            return 0.0

        return self.config.get('goal_reward', 10.0)

    def collision_penalty(self, state: RewardState) -> float:
        """
        Compute penalty for collision.

        Args:
            state: Current reward state

        Returns:
            Collision penalty (negative)
        """
        if not state.is_collision:
            return 0.0

        weight = self.config.get('w_collision', 5.0)
        return -weight

    def data_reward(self, state: RewardState) -> float:
        """
        Compute data collection reward.

        Args:
            state: Current reward state

        Returns:
            Data collection reward
        """
        weight = self.config.get('w_data', 1.0)
        return state.data_scarcity * weight

    def coverage_reward(self, state: RewardState) -> float:
        """
        Compute coverage reward based on exploration.

        Args:
            state: Current reward state

        Returns:
            Coverage reward
        """
        weight = self.config.get('w_coverage', 0.1)
        return state.coverage_ratio * weight

    def efficiency_penalty(self, state: RewardState) -> float:
        """
        Compute path efficiency penalty.

        Args:
            state: Current reward state

        Returns:
            Efficiency penalty (negative)
        """
        if state.direct_distance <= 0:
            return 0.0

        efficiency_ratio = state.path_length / (state.direct_distance + 1)
        weight = self.config.get('w_efficiency', 0.01)
        return -weight * efficiency_ratio

    def repetition_penalty(self, state: RewardState) -> float:
        """
        Compute penalty for revisiting locations.

        Args:
            state: Current reward state

        Returns:
            Repetition penalty (negative)
        """
        if state.visited_cells is None:
            return 0.0

        # Check if current position was visited before
        x, y = int(state.position[0]), int(state.position[1])
        if 0 <= y < state.visited_cells.shape[0] and 0 <= x < state.visited_cells.shape[1]:
            if state.visited_cells[y, x]:
                weight = self.config.get('w_repetition', 0.05)
                return -weight

        return 0.0
