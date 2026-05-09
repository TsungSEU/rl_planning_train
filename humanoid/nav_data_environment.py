"""
Navigation + Data Collection environment (2D prototype).

2D grid prototype for validating Aurora's high-level navigation policy.
Uses simplified dynamics to approximate LivelyBot's velocity response,
allowing rapid iteration on state/reward design before physics integration.

State space: 42 dimensions (NavDataState)
Action space: 3 continuous velocity commands [vx, vy, wz]
Control: 10 Hz Aurora decision, simulates 0.1s per step
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import logging

from core.base_environment import BaseEnvironment
from .nav_data_state import (
    NavDataState, NAV_DATA_STATE_DIM,
    VALUE_SECTOR_COUNT, OBSTACLE_SECTOR_COUNT,
)
from .velocity_action import VelocityAction, VelocityActionBounds
from .nav_data_reward import NavDataRewardCalculator, NavDataRewardConfig, NavDataRewardState

logger = logging.getLogger(__name__)


class NavDataEnvironment(BaseEnvironment):
    """
    2D prototype environment for navigation + data collection.

    Simulates a grid world with:
    - Data value map (areas with different data collection value)
    - Obstacle map (navigation hazards)
    - Multiple target waypoints (high-value data collection points)
    - Simplified dynamics matching LivelyBot velocity response

    The agent learns to navigate to high-value areas efficiently while
    maximizing data collection value.
    """

    def __init__(
        self,
        width: int = 120,
        height: int = 120,
        grid_resolution: float = 0.5,
        max_steps: int = 600,
        num_targets: int = 5,
        num_obstacles: int = 40,
        reward_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize nav_data environment.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            grid_resolution: Meters per grid cell
            max_steps: Maximum steps per episode (600 = 60s at 10Hz)
            num_targets: Number of high-value target waypoints
            num_obstacles: Number of obstacle cells
            reward_config: Reward configuration
        """
        super().__init__(width, height, max_steps)

        self.grid_resolution = grid_resolution
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles

        # Dimensions
        self._state_dim = NAV_DATA_STATE_DIM
        self._action_dim = VelocityAction.ACTION_DIM
        self._action_space_type = 'continuous'

        # Maps
        self.obstacle_map = np.zeros((height, width), dtype=bool)
        self.data_value_map = np.zeros((height, width), dtype=float)
        self.visited_cells = np.zeros((height, width), dtype=int)
        self.visit_times = np.zeros((height, width), dtype=float)

        # Tracking
        self._visited_count = 0
        self.trajectory: List[np.ndarray] = []
        self.start_pos: Optional[np.ndarray] = None

        # Targets (high-value collection points)
        self.targets: List[np.ndarray] = []
        self.collected_targets: List[bool] = []
        self.current_target_idx: int = 0

        # State
        self.nav_state = NavDataState(
            width=width * grid_resolution,
            height=height * grid_resolution,
            max_steps=max_steps,
        )
        self.heading: float = 0.0
        self.position: np.ndarray = np.zeros(2)
        self.velocity: np.ndarray = np.zeros(3)

        # Reward
        self.reward_calculator = NavDataRewardCalculator(reward_config)
        self.action_bounds = VelocityActionBounds()

        # Episode
        self.current_time = 0.0
        self.path_length = 0.0
        self.direct_distance = 0.0
        self.total_value_collected = 0.0
        self.total_value_available = 0.0

        # Precomputed
        self._goal_threshold = 3.0  # meters
        self._max_range = 10.0  # meters, for observation normalization

        # Initialize
        self._generate_obstacles()
        self._generate_data_value_map()

    def set_difficulty(self, params: Dict[str, Any]):
        """Reconfigure environment difficulty for curriculum learning."""
        if 'num_targets' in params:
            self.num_targets = params['num_targets']
        if 'num_obstacles' in params:
            self.num_obstacles = params['num_obstacles']
        if 'env_size' in params:
            size = params['env_size']
            grid_res = self.grid_resolution
            self.width = int(size / grid_res)
            self.height = int(size / grid_res)
            self.obstacle_map = np.zeros((self.height, self.width), dtype=bool)
            self.data_value_map = np.zeros((self.height, self.width), dtype=float)
            self.visited_cells = np.zeros((self.height, self.width), dtype=int)
            self.visit_times = np.zeros((self.height, self.width), dtype=float)
        if 'goal_threshold' in params:
            self._goal_threshold = params['goal_threshold']
        if 'steps' in params:
            self.max_steps = params['steps']
        # Regenerate maps with new parameters
        self._generate_obstacles()
        self._generate_data_value_map()

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_space_type(self) -> str:
        return self._action_space_type

    def _generate_obstacles(self):
        """Generate obstacle map with clustered obstacles."""
        self.obstacle_map.fill(False)

        margin = min(10, self.width // 4, self.height // 4)
        inner_start = margin
        inner_end_w = max(inner_start + 1, self.width - margin)
        inner_end_h = max(inner_start + 1, self.height - margin)

        for _ in range(self.num_obstacles):
            # Cluster obstacles in groups
            cx = np.random.randint(margin, inner_end_w)
            cy = np.random.randint(margin, inner_end_h)
            size = np.random.randint(1, 4)

            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.obstacle_map[y, x] = np.random.random() < 0.6

    def _generate_data_value_map(self):
        """
        Generate data value map with high-value regions.
        Value range: [0, 1], where higher means more valuable data.
        """
        self.data_value_map.fill(0.1)  # baseline low value

        # Create high-value hotspots (Gaussian blobs)
        margin = min(15, self.width // 4, self.height // 4)
        num_hotspots = self.num_targets + 3
        for _ in range(num_hotspots):
            cx = np.random.randint(margin, max(margin + 1, self.width - margin))
            cy = np.random.randint(margin, max(margin + 1, self.height - margin))
            radius = np.random.randint(3, max(4, min(15, self.width // 4)))
            peak_value = np.random.uniform(0.5, 1.0)

            for y in range(max(0, cy - radius), min(self.height, cy + radius)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius)):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        value = peak_value * np.exp(-0.5 * (dist / (radius * 0.5))**2)
                        self.data_value_map[y, x] = max(self.data_value_map[y, x], value)

    def _bfs_path_length(self, start_cell: np.ndarray, end_cell: np.ndarray) -> Optional[int]:
        """BFS shortest path on obstacle map. Returns path length in cells or None if unreachable."""
        if np.array_equal(start_cell, end_cell):
            return 0
        visited = set()
        visited.add((start_cell[0], start_cell[1]))
        queue = deque([(start_cell[0], start_cell[1], 0)])
        while queue:
            y, x, dist = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if ny == end_cell[0] and nx == end_cell[1]:
                    return dist + 1
                if (0 <= ny < self.height and 0 <= nx < self.width
                        and not self.obstacle_map[ny, nx]
                        and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx, dist + 1))
        return None

    def _select_targets(self):
        """Select target waypoints from high-value regions with reachability validation."""
        self.targets = []
        self.collected_targets = []

        # Find high-value cells
        high_value_threshold = 0.6
        high_value_cells = np.argwhere(self.data_value_map > high_value_threshold)

        if len(high_value_cells) < self.num_targets:
            high_value_cells = np.argwhere(self.data_value_map > 0.3)

        if len(high_value_cells) == 0:
            for _ in range(self.num_targets):
                tx = np.random.randint(5, max(6, self.width - 5))
                ty = np.random.randint(5, max(6, self.height - 5))
                self.targets.append(np.array([tx, ty], dtype=float) * self.grid_resolution)
                self.collected_targets.append(False)
        else:
            np.random.shuffle(high_value_cells)
            selected = []
            min_dist_cells = 8
            max_first_target_m = 8.0
            max_path_m = self.max_steps * 0.06 * 0.5  # effective speed * 50% margin

            start_cell = (self.position / self.grid_resolution).astype(int)
            start_cell = np.clip(start_cell[::-1], 0, [self.height - 1, self.width - 1])  # (row, col)

            # First target: near start + reachable
            dists_to_start = np.array([
                np.linalg.norm(np.array([c[1], c[0]]) * self.grid_resolution - self.position * self.grid_resolution)
                for c in high_value_cells
            ])
            near_mask = dists_to_start < max_first_target_m
            near_cells = high_value_cells[near_mask]

            first_selected = False
            if len(near_cells) > 0:
                sorted_idx = np.argsort(dists_to_start[near_mask])
                for idx in sorted_idx:
                    cell = near_cells[idx]
                    path_len = self._bfs_path_length(start_cell, cell)
                    if path_len is not None and path_len * self.grid_resolution < max_path_m:
                        selected.append(cell)
                        self.targets.append(np.array([cell[1], cell[0]], dtype=float) * self.grid_resolution)
                        self.collected_targets.append(False)
                        first_selected = True
                        break

            # Subsequent targets: reachable from previous target
            for cell in high_value_cells:
                if len(selected) >= self.num_targets:
                    break
                too_close = any(np.linalg.norm(cell - s) < min_dist_cells for s in selected)
                if too_close:
                    continue
                if len(selected) > 0:
                    prev = selected[-1]
                    path_len = self._bfs_path_length(prev, cell)
                    if path_len is None or path_len * self.grid_resolution >= max_path_m:
                        continue
                selected.append(cell)
                self.targets.append(
                    np.array([cell[1], cell[0]], dtype=float) * self.grid_resolution
                )
                self.collected_targets.append(False)

            # Fallback: if BFS filtered everything out, use simple distance-based selection
            if len(selected) < self.num_targets:
                for cell in high_value_cells:
                    if len(selected) >= self.num_targets:
                        break
                    if any(np.array_equal(cell, s) for s in selected):
                        continue
                    too_close = any(np.linalg.norm(cell - s) < min_dist_cells for s in selected)
                    if too_close:
                        continue
                    selected.append(cell)
                    self.targets.append(
                        np.array([cell[1], cell[0]], dtype=float) * self.grid_resolution
                    )
                    self.collected_targets.append(False)

        self.total_value_available = float(np.sum(self.data_value_map))
        self.current_target_idx = 0

    def reset(
        self,
        start_pos: Optional[Tuple[float, float]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Reset environment.

        Args:
            start_pos: Optional (x, y) start position in meters
            goal_pos: Ignored (targets auto-selected from data value map)

        Returns:
            Initial 42-dim state observation
        """
        self.step_count = 0
        self.current_time = 0.0
        self.trajectory = []
        self._visited_count = 0
        self.path_length = 0.0
        self.total_value_collected = 0.0

        # Reset maps
        self.visited_cells.fill(0)
        self.visit_times.fill(0)

        # Select targets
        self._select_targets()

        # Reset position
        env_w = self.width * self.grid_resolution
        env_h = self.height * self.grid_resolution

        if start_pos is None:
            margin = 5  # meters from edge
            env_w_m = self.width * self.grid_resolution
            env_h_m = self.height * self.grid_resolution
            sx = np.random.uniform(margin, env_w_m - margin)
            sy = np.random.uniform(margin, env_h_m - margin)
            self.position = np.array([sx, sy]) / self.grid_resolution
        else:
            self.position = np.array(start_pos, dtype=float)

        self.start_pos = self.position.copy()
        self.heading = np.random.uniform(-0.3, 0.3)
        self.velocity = np.zeros(3)

        # Reset nav state
        self.nav_state = NavDataState(
            width=env_w,
            height=env_h,
            max_steps=self.max_steps,
            max_range=self._max_range,
        )
        self.nav_state.position = self.position.copy() * self.grid_resolution
        self.nav_state.heading = self.heading
        self.nav_state.action_history = np.zeros(8)

        # Set initial goal (nearest uncollected target)
        self._update_current_target()

        # Record trajectory
        self.trajectory.append(self.position.copy())
        self.direct_distance = self._compute_direct_distance()

        return self.get_state()

    def _update_current_target(self):
        """Set current target to nearest uncollected target."""
        best_dist = float('inf')
        best_idx = 0
        pos_m = self.position * self.grid_resolution  # convert to meters

        for i, collected in enumerate(self.collected_targets):
            if not collected:
                dist = np.linalg.norm(self.targets[i] - pos_m)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

        self.current_target_idx = best_idx
        if self.targets:
            self.nav_state.goal_position = self.targets[best_idx]
            # Update direct_distance when target changes
            self.direct_distance = self._compute_direct_distance()

    def _compute_direct_distance(self) -> float:
        """Compute direct distance from start to current target (in meters)."""
        if self.nav_state.goal_position is None:
            return 0.0
        start_m = self.start_pos * self.grid_resolution
        return float(np.linalg.norm(self.nav_state.goal_position - start_m))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one Aurora step (0.1s simulated time).

        Args:
            action: 3-dim velocity command [forward_vel, lateral_vel, angular_vel]

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        self.current_time += 0.1

        old_pos = self.position.copy()
        old_velocity = self.velocity.copy()

        # Clip action to LivelyBot ranges
        lower, upper = self.action_bounds.get_bounds()
        action = np.clip(action, lower, upper)

        forward_vel, lateral_vel, angular_vel = action[0], action[1], action[2]

        # --- Simplified dynamics (approximates LivelyBot velocity response) ---
        # Heading update
        self.heading += angular_vel * 0.1

        # World-frame velocity from body-frame commands
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        world_vx = (forward_vel * cos_h - lateral_vel * sin_h)
        world_vy = (forward_vel * sin_h + lateral_vel * cos_h)

        # Smooth velocity (simulate LivelyBot response lag)
        alpha = 0.8  # response smoothing (0.8 → effective speed ~0.48 m/s)
        self.velocity[0] = self.velocity[0] * (1 - alpha) + world_vx * alpha
        self.velocity[1] = self.velocity[1] * (1 - alpha) + world_vy * alpha
        self.velocity[2] = angular_vel

        # Position update (grid_resolution converts cells to meters)
        new_pos = self.position + np.array([
            self.velocity[0] * 0.1 / self.grid_resolution,
            self.velocity[1] * 0.1 / self.grid_resolution,
        ])

        # Collision check
        is_collision = False
        gx = int(new_pos[0])
        gy = int(new_pos[1])

        if not (0 <= gx < self.width and 0 <= gy < self.height):
            # Out of bounds — clamp
            new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
        elif self.obstacle_map[gy, gx]:
            is_collision = True
            # Decay velocity instead of zeroing — enables faster recovery
            self.velocity[0] *= 0.3
            self.velocity[1] *= 0.3
        else:
            self.position = new_pos

        # Update path length
        self.path_length += np.linalg.norm(self.position - old_pos)

        # --- Update visit tracking ---
        cx = int(self.position[0])
        cy = int(self.position[1])
        cx = np.clip(cx, 0, self.width - 1)
        cy = np.clip(cy, 0, self.height - 1)

        old_visit_count = self.visited_cells[cy, cx]
        self.visited_cells[cy, cx] += 1
        self.visit_times[cy, cx] = self.current_time
        if old_visit_count == 0:
            self._visited_count += 1

        new_cells = 1 if old_visit_count == 0 else 0

        # Data value at current position
        current_value = self.data_value_map[cy, cx]
        total_value = float(np.sum(self.data_value_map))
        rarity = 1.0 / (1.0 + self.visited_cells[cy, cx] * 0.1)

        if old_visit_count == 0:
            self.total_value_collected += current_value

        # --- Check target collection ---
        is_goal_reached = False
        min_obstacle_dist = self._get_min_obstacle_distance()

        # Use current position (not stale nav_state.position from previous step)
        pos_m = self.position * self.grid_resolution
        dist_to_target = float('inf')
        if self.nav_state.goal_position is not None:
            dist_to_target = np.linalg.norm(
                pos_m - self.nav_state.goal_position
            )
            if dist_to_target < self._goal_threshold:
                target_idx = self.current_target_idx
                if target_idx < len(self.collected_targets) and not self.collected_targets[target_idx]:
                    self.collected_targets[target_idx] = True
                    is_goal_reached = True
                    self._update_current_target()
                    # Recompute distance after target switch
                    if self.nav_state.goal_position is not None:
                        dist_to_target = np.linalg.norm(pos_m - self.nav_state.goal_position)

        # --- Update NavDataState ---
        self.nav_state.base_lin_vel = np.array([
            self.velocity[0], self.velocity[1], 0.0
        ])
        self.nav_state.base_ang_vel = np.array([0.0, 0.0, self.velocity[2]])
        self.nav_state.position = self.position.copy() * self.grid_resolution
        self.nav_state.heading = self.heading
        self.nav_state.current_data_value = current_value
        self.nav_state.current_rarity = rarity
        self.nav_state.collected_value_ratio = (
            self.total_value_collected / max(1, total_value)
        )
        self.nav_state.coverage_ratio = self._visited_count / (self.width * self.height)
        self.nav_state.step_count = self.step_count
        self.nav_state.action_history = np.roll(self.nav_state.action_history, -1)
        action_mag = np.linalg.norm(action) / np.sqrt(0.6**2 + 0.3**2 + 0.3**2)
        self.nav_state.action_history[-1] = np.clip(action_mag, 0, 1)

        # Gait phase (simulated 0.4s cycle = 2.5Hz stepping)
        self.nav_state.gait_phase = (self.current_time / 0.4) % 1.0

        # Update sector scans
        self.nav_state.update_value_sectors(
            self.data_value_map, self.grid_resolution
        )
        self.nav_state.update_obstacle_sectors(
            self.obstacle_map.astype(np.float64), self.grid_resolution
        )

        # Obstacle density
        search_r = 5
        x_min = max(0, cx - search_r)
        x_max = min(self.width, cx + search_r + 1)
        y_min = max(0, cy - search_r)
        y_max = min(self.height, cy + search_r + 1)
        local_area = self.obstacle_map[y_min:y_max, x_min:x_max]
        self.nav_state.obstacle_density = float(np.mean(local_area))

        # Record trajectory
        self.trajectory.append(self.position.copy())

        # --- Build reward state ---
        reward_state = NavDataRewardState(
            position=self.position.copy() * self.grid_resolution,
            previous_position=old_pos.copy() * self.grid_resolution,
            heading=self.heading,
            velocity=self.velocity.copy(),
            goal_position=self.nav_state.goal_position,
            goal_distance=dist_to_target,
            is_goal_reached=is_goal_reached,
            current_data_value=current_value,
            current_rarity=rarity,
            data_value_sectors=self.nav_state.data_value_sectors.copy(),
            is_first_visit=(old_visit_count == 0),
            collected_value_ratio=self.nav_state.collected_value_ratio,
            new_cells_visited=new_cells,
            visit_count_at_pos=self.visited_cells[cy, cx],
            path_length=self.path_length * self.grid_resolution,
            direct_distance=self.direct_distance,
            is_collision=is_collision,
            min_obstacle_distance=min_obstacle_dist * self.grid_resolution,
            step_count=self.step_count,
            max_steps=self.max_steps,
            action=action.copy(),
        )

        reward_breakdown = self.reward_calculator.compute(reward_state)

        # --- Done check ---
        all_collected = all(self.collected_targets) if self.collected_targets else False
        done = all_collected or self.is_done()

        # --- Info ---
        info = {
            'position': self.position.copy(),
            'heading': self.heading,
            'speed': np.linalg.norm(self.velocity[:2]),
            'current_value': current_value,
            'is_goal_reached': is_goal_reached,
            'collected_targets': sum(self.collected_targets),
            'total_targets': len(self.targets),
            'all_collected': all_collected,
            'coverage': self.nav_state.coverage_ratio,
            'path_efficiency': (
                self.direct_distance / max(0.1, self.path_length * self.grid_resolution)
            ),
            'reward_breakdown': reward_breakdown.to_dict(),
        }

        return self.get_state(), reward_breakdown.total, done, info

    def get_state(self) -> np.ndarray:
        """Get 42-dimensional state observation."""
        return self.nav_state.get_state_vector()

    def _get_min_obstacle_distance(self) -> float:
        """Get distance to nearest obstacle in grid cells."""
        cx = int(self.position[0])
        cy = int(self.position[1])

        search_r = 10
        x_min = max(0, cx - search_r)
        x_max = min(self.width, cx + search_r + 1)
        y_min = max(0, cy - search_r)
        y_max = min(self.height, cy + search_r + 1)

        region = self.obstacle_map[y_min:y_max, x_min:x_max]
        if not region.any():
            return float(search_r)

        obs_ys, obs_xs = np.nonzero(region)
        dx = obs_xs + x_min - cx
        dy = obs_ys + y_min - cy
        dists = np.sqrt(dx.astype(float)**2 + dy.astype(float)**2)
        return float(np.min(dists))

    def compute_reward(self, **kwargs) -> float:
        """Compute reward for current state."""
        return self.reward_calculator.compute_total(**kwargs)

    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action bounds."""
        return self.action_bounds.get_bounds()

    def _render_on_axis(self, ax: plt.Axes):
        """Render environment on matplotlib axis."""
        # Data value map (background heatmap)
        extent = [0, self.width, 0, self.height]
        ax.imshow(self.data_value_map.T, origin='lower', cmap='YlOrRd',
                  alpha=0.6, extent=extent, vmin=0, vmax=1)

        # Obstacles
        obs_y, obs_x = np.where(self.obstacle_map)
        ax.scatter(obs_x, obs_y, c='black', s=3, alpha=0.8, marker='s')

        # Visited cells (green overlay)
        vis_y, vis_x = np.where(self.visited_cells > 0)
        if len(vis_x) > 0:
            intensities = np.minimum(0.5, self.visited_cells[vis_y, vis_x] * 0.05)
            ax.scatter(vis_x, vis_y, c='green', s=5, alpha=intensities, marker='s')

        # Targets
        for i, target in enumerate(self.targets):
            tx = target[0] / self.grid_resolution
            ty = target[1] / self.grid_resolution
            color = 'lime' if self.collected_targets[i] else 'red'
            marker = 'o' if self.collected_targets[i] else '*'
            ax.plot(tx, ty, marker=marker, color=color, markersize=12,
                    markeredgecolor='black', markeredgewidth=1)

        # Current target indicator
        if self.nav_state.goal_position is not None:
            gx = self.nav_state.goal_position[0] / self.grid_resolution
            gy = self.nav_state.goal_position[1] / self.grid_resolution
            circle = plt.Circle((gx, gy), self._goal_threshold / self.grid_resolution,
                                fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(circle)

        # Agent
        agent_gx = self.position[0]
        agent_gy = self.position[1]
        agent_circle = plt.Circle((agent_gx, agent_gy), 0.8, color='blue', zorder=10)
        ax.add_patch(agent_circle)

        # Heading arrow
        arrow_len = 1.5
        dx = np.cos(self.heading) * arrow_len
        dy = np.sin(self.heading) * arrow_len
        ax.arrow(agent_gx, agent_gy, dx, dy,
                 head_width=0.3, head_length=0.3, fc='red', ec='red', zorder=11)

        # Formatting
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')

        collected = sum(self.collected_targets)
        total = len(self.targets)
        ax.set_title(
            f"Nav+Data (Step: {self.step_count}/{self.max_steps}, "
            f"Targets: {collected}/{total}, "
            f"Coverage: {self.nav_state.coverage_ratio:.1%})"
        )
