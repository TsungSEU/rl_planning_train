"""
Navigation + Data Collection reward calculator (10 components).

Designed for hierarchical control where LivelyBot handles locomotion
stability/gait/energy, and Aurora focuses on:
  - Navigating to high-value data areas
  - Maximizing data collection value
  - Efficient path planning
  - Safety (collision avoidance)
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from core.base_reward import BaseRewardCalculator, RewardBreakdown


@dataclass
class NavDataRewardConfig:
    """Configuration for nav_data reward calculation."""

    # Main reward weights (aligned with architecture documentation)
    w_approach: float = 15.0      # Target approach reward (stronger gradient)
    w_data_value: float = 3.0     # Data value collection (reduced to avoid nav conflict)
    w_value_guide: float = 0.5    # High-value area guidance (reduced to avoid target conflict)
    w_coverage: float = 0.5       # Coverage reward
    w_goal: float = 30.0          # Goal completion reward (reduced from 100 to not dominate GAE)
    w_efficiency: float = 0.5     # Path efficiency penalty
    w_repeat: float = 1.0         # Repeat visit penalty
    w_collision: float = 10.0     # Collision/danger penalty
    w_speed: float = 1.0          # Speed tracking (increased to encourage movement)
    w_time: float = 0.05          # Time penalty (increased for urgency)
    w_proximity: float = 8.0      # Proximity shaping reward (increased for strong dense signal)

    # Reward values
    first_visit_bonus: float = 0.3
    collision_penalty: float = -2.0
    near_obstacle_scale: float = 5.0
    safe_distance: float = 1.0      # meters
    stall_penalty: float = -0.5
    min_speed_threshold: float = 0.05  # m/s
    max_repeat_penalty: float = 0.03   # Cap on per-step repeat penalty


@dataclass
class NavDataRewardState:
    """State for nav_data reward calculation."""

    # Position and motion
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    previous_position: Optional[np.ndarray] = None
    heading: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Goal
    goal_position: Optional[np.ndarray] = None
    goal_distance: float = 0.0
    is_goal_reached: bool = False

    # Data value
    current_data_value: float = 0.0
    current_rarity: float = 0.0
    data_value_sectors: np.ndarray = field(default_factory=lambda: np.zeros(8))
    is_first_visit: bool = False

    # Collection status
    collected_value_ratio: float = 0.0
    new_cells_visited: int = 0
    visit_count_at_pos: int = 0

    # Path tracking
    path_length: float = 0.0
    direct_distance: float = 0.0

    # Safety
    is_collision: bool = False
    min_obstacle_distance: float = 10.0

    # Episode
    step_count: int = 0
    max_steps: int = 600

    # Action
    action: np.ndarray = field(default_factory=lambda: np.zeros(3))


class NavDataRewardCalculator(BaseRewardCalculator):
    """
    Reward calculator for high-level navigation + data collection.

    10 components:
    1. Target approach (w=3.0)
    2. Data value collection (w=5.0) — core reward
    3. High-value area guidance (w=2.0)
    4. Coverage reward (w=1.0)
    5. Goal completion (w=100.0)
    6. Path efficiency (w=0.5)
    7. Repeat visit penalty (w=1.0)
    8. Collision/danger (w=10.0)
    9. Speed tracking (w=0.5)
    10. Time penalty (w=0.1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        if isinstance(config, NavDataRewardConfig):
            self.cfg = config
        else:
            self.cfg = NavDataRewardConfig(**(config or {}))

    def compute(self, state: NavDataRewardState) -> RewardBreakdown:
        """Compute all reward components."""
        r_approach = self._reward_approach(state)
        r_proximity = self._reward_proximity(state)
        r_data = self._reward_data_value(state)
        r_guide = self._reward_value_guide(state)
        r_coverage = self._reward_coverage(state)
        r_goal = self._reward_goal(state)
        r_efficiency = self._reward_efficiency(state)
        r_repeat = self._reward_repeat(state)
        r_collision = self._reward_collision(state)
        r_speed = self._reward_speed(state)
        r_time = self._reward_time(state)

        total = (r_approach + r_proximity + r_data + r_guide + r_coverage + r_goal +
                 r_efficiency + r_repeat + r_collision + r_speed + r_time)

        return RewardBreakdown(
            total=total,
            distance=r_approach,
            data=r_data,
            navigation=r_guide,
            coverage=r_coverage,
            goal=r_goal,
            efficiency=r_efficiency,
            repetition=r_repeat,
            collision=r_collision,
            time_penalty=r_time,
        )

    def _reward_approach(self, state: NavDataRewardState) -> float:
        """
        Reward for making progress toward current target.

        Formula: (old_dist - new_dist) × w_approach + proximity_bonus
        """
        if state.goal_position is None or state.previous_position is None:
            return 0.0

        old_dist = np.linalg.norm(state.previous_position - state.goal_position)
        new_dist = np.linalg.norm(state.position - state.goal_position)

        progress_reward = (old_dist - new_dist) * self.cfg.w_approach

        # Proximity bonus: increasing reward as robot gets closer to goal
        proximity_bonus = 0.0
        if new_dist < 5.0:
            proximity_bonus = (5.0 - new_dist) / 5.0 * 0.5

        return progress_reward + proximity_bonus

    def _reward_proximity(self, state: NavDataRewardState) -> float:
        """
        Dense shaping reward for being close to the target.

        Uses quadratic shaping: reward increases steeply near the target,
        creating a "funnel" gradient that guides the agent in.
        Far from target → weak signal; close → very strong signal.

        Formula: (1 - dist/max_range)^2 × w  when within max_range
        """
        if state.goal_position is None:
            return 0.0

        dist = state.goal_distance
        max_range = 15.0
        if dist < max_range:
            normalized = 1.0 - dist / max_range
            return normalized ** 2 * self.cfg.w_proximity
        return 0.0

    def _reward_data_value(self, state: NavDataRewardState) -> float:
        """
        Core reward: data value collection at current position.

        Formula: value × rarity × w + first_visit_bonus
        """
        value = state.current_data_value * state.current_rarity
        reward = value * self.cfg.w_data_value

        if state.is_first_visit:
            reward += self.cfg.first_visit_bonus

        return reward

    def _reward_value_guide(self, state: NavDataRewardState) -> float:
        """
        Guide robot toward highest-value sector.

        Formula: cos(heading_error) × forward_vel × w
        """
        if np.max(state.data_value_sectors) < 0.01:
            return 0.0

        best_sector = int(np.argmax(state.data_value_sectors))
        target_angle = best_sector * (2 * np.pi / 8)
        heading_error = (target_angle - state.heading + np.pi) % (2 * np.pi) - np.pi

        forward_vel = state.velocity[0] if len(state.velocity) > 0 else 0.0
        return np.cos(heading_error) * max(0, forward_vel) * self.cfg.w_value_guide

    def _reward_coverage(self, state: NavDataRewardState) -> float:
        """
        Reward for visiting new cells.

        Formula: new_cells × w_coverage
        """
        return state.new_cells_visited * self.cfg.w_coverage

    def _reward_goal(self, state: NavDataRewardState) -> float:
        """
        Large reward for reaching target (high-value area center).
        """
        if not state.is_goal_reached:
            return 0.0
        return self.cfg.w_goal

    def _reward_efficiency(self, state: NavDataRewardState) -> float:
        """
        Penalize inefficient paths (path >> direct distance).

        Formula: -(1 - min(1, direct/path)) × w
        Clipped so it only penalizes wandering, never rewards short paths.
        """
        if state.path_length <= 0 or state.direct_distance <= 0:
            return 0.0

        efficiency = min(1.0, state.direct_distance / state.path_length)
        return -(1.0 - efficiency) * self.cfg.w_efficiency

    def _reward_repeat(self, state: NavDataRewardState) -> float:
        """
        Penalize revisiting already-collected areas.

        Formula: -min(visit_count-1, cap) × w
        """
        if state.visit_count_at_pos <= 1:
            return 0.0
        penalty = (state.visit_count_at_pos - 1) * self.cfg.w_repeat
        return -min(penalty, self.cfg.max_repeat_penalty)

    def _reward_collision(self, state: NavDataRewardState) -> float:
        """
        Penalize collisions and proximity to obstacles.

        Collision: fixed penalty
        Near obstacle: -(safe_dist - obs_dist) × scale
        """
        if state.is_collision:
            return self.cfg.collision_penalty

        if state.min_obstacle_distance < self.cfg.safe_distance:
            return -(self.cfg.safe_distance - state.min_obstacle_distance) * self.cfg.near_obstacle_scale

        return 0.0

    def _reward_speed(self, state: NavDataRewardState) -> float:
        """
        Penalize stalling when there are remaining targets.

        Below min_speed_threshold: -stall_penalty
        """
        if state.is_goal_reached:
            return 0.0

        speed = np.linalg.norm(state.velocity[:2]) if len(state.velocity) >= 2 else 0.0
        if speed < self.cfg.min_speed_threshold:
            return self.cfg.stall_penalty * self.cfg.w_speed

        return 0.0

    def _reward_time(self, state: NavDataRewardState) -> float:
        """Fixed time penalty per step."""
        return -self.cfg.w_time

    def get_config(self) -> NavDataRewardConfig:
        """Get reward configuration."""
        return self.cfg
