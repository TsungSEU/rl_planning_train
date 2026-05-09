"""
Navigation + Data Collection state representation (43-dimensional).

Designed for hierarchical control: Aurora provides high-level navigation
at 10Hz, LivelyBot handles low-level locomotion at 50Hz.

State structure:
- Indices 0-2:   Base linear velocity [vx, vy, vz] (from LivelyBot)
- Indices 3-5:   Base angular velocity [wx, wy, wz] (from LivelyBot)
- Indices 6-7:   Normalized position [x/W, y/H]
- Indices 8-9:   Heading [sin(theta), cos(theta)]
- Indices 10-11: Goal direction [sin(delta_theta), cos(delta_theta)]
- Indices 12-14: Goal distance [dx, dy, norm] (normalized)
- Indices 15-22: Data value sectors (8 directions)
- Indices 23-26: Obstacle sectors (4 directions: front, back, left, right)
- Indices 27-28: Current position data [value, rarity]
- Indices 29-30: Collection status [collected_ratio, coverage_ratio]
- Index 31:      Terrain type (normalized)
- Index 32:      Local obstacle density
- Index 33:      Gait phase from LivelyBot
- Indices 34-41: Action history (8 recent action magnitudes)
- Index 42:      Remaining time budget
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


NAV_DATA_STATE_DIM = 43

# State index ranges for reference
BASE_LIN_VEL_IDX = (0, 3)
BASE_ANG_VEL_IDX = (3, 6)
POSITION_IDX = (6, 8)
HEADING_IDX = (8, 10)
GOAL_DIR_IDX = (10, 12)
GOAL_DIST_IDX = (12, 15)
VALUE_SECTOR_IDX = (15, 23)
OBSTACLE_SECTOR_IDX = (23, 27)
CURRENT_VALUE_IDX = (27, 29)
COLLECTION_STATUS_IDX = (29, 31)
TERRAIN_IDX = (31, 32)
OBSTACLE_DENSITY_IDX = (32, 33)
GAIT_PHASE_IDX = (33, 34)
ACTION_HISTORY_IDX = (34, 42)
BUDGET_IDX = (42, 43)

# Sector definitions
VALUE_SECTOR_COUNT = 8
VALUE_SECTOR_ANGLE = 2 * np.pi / VALUE_SECTOR_COUNT
OBSTACLE_SECTOR_COUNT = 4


@dataclass
class NavDataState:
    """
    Navigation + Data Collection state for Aurora high-level policy.

    Excludes joint-level details (managed by LivelyBot).
    Focuses on navigation awareness and data value assessment.
    """

    STATE_DIM = NAV_DATA_STATE_DIM

    # --- Robot state (from LivelyBot feedback) ---
    base_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # --- Position & heading ---
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0  # radians

    # --- Goal information ---
    goal_position: Optional[np.ndarray] = None

    # --- Data value map observations ---
    data_value_sectors: np.ndarray = field(default_factory=lambda: np.zeros(VALUE_SECTOR_COUNT))
    current_data_value: float = 0.0
    current_rarity: float = 0.0

    # --- Obstacle observations ---
    obstacle_sectors: np.ndarray = field(default_factory=lambda: np.full(OBSTACLE_SECTOR_COUNT, 10.0))
    obstacle_density: float = 0.0

    # --- Collection tracking ---
    collected_value_ratio: float = 0.0
    coverage_ratio: float = 0.0

    # --- Environment ---
    terrain_type: int = 0
    gait_phase: float = 0.0  # from LivelyBot

    # --- History & budget ---
    action_history: np.ndarray = field(default_factory=lambda: np.zeros(8))

    # --- Episode state ---
    step_count: int = 0
    max_steps: int = 600  # 60s at 10Hz
    width: float = 60.0
    height: float = 60.0
    max_range: float = 10.0

    def get_state_vector(self) -> np.ndarray:
        """
        Build the 42-dimensional state vector.

        Returns:
            42-dimensional numpy array
        """
        # Base linear velocity (indices 0-2)
        # [vz] forced to 0 in 2D prototype
        lin_vel = np.array([self.base_lin_vel[0] * 2.0, self.base_lin_vel[1] * 2.0, 0.0])

        # Base angular velocity (indices 3-5)
        # [wx, wy] forced to 0 in 2D prototype (only yaw matters)
        ang_vel = np.array([0.0, 0.0, self.base_ang_vel[2] * 1.0])

        # Normalized position (indices 6-7)
        norm_pos = np.array([
            self.position[0] / max(1.0, self.width),
            self.position[1] / max(1.0, self.height),
        ])

        # Heading (indices 8-9)
        heading_vec = np.array([np.sin(self.heading), np.cos(self.heading)])

        # Goal direction and distance (indices 10-14)
        goal_dir, goal_dist = self._compute_goal_info()

        # Data value sectors (indices 15-22)
        value_sectors = np.clip(self.data_value_sectors, 0.0, 1.0)

        # Obstacle sectors (indices 23-26)
        obs_sectors = np.clip(self.obstacle_sectors / max(1.0, self.max_range), 0.0, 1.0)

        # Current data value and rarity (indices 27-28)
        current_value = np.array([
            np.clip(self.current_data_value, 0.0, 1.0),
            np.clip(self.current_rarity, 0.0, 1.0),
        ])

        # Collection status (indices 29-30)
        collection = np.array([
            np.clip(self.collected_value_ratio, 0.0, 1.0),
            np.clip(self.coverage_ratio, 0.0, 1.0),
        ])

        # Terrain type (index 31) — 0 in 2D prototype
        terrain = np.array([0.0])

        # Obstacle density (index 32)
        density = np.array([np.clip(self.obstacle_density, 0.0, 1.0)])

        # Gait phase (index 33) — simulated stepping cycle
        gait = np.array([np.sin(2 * np.pi * self.gait_phase)])

        # Action history (indices 34-41)
        action_hist = np.clip(self.action_history, 0.0, 1.0)

        # Remaining budget (index 42)
        budget = np.array([max(0.0, 1.0 - self.step_count / max(1, self.max_steps))])

        return np.concatenate([
            lin_vel,            # 3
            ang_vel,            # 3
            norm_pos,           # 2
            heading_vec,        # 2
            goal_dir,           # 2
            goal_dist,          # 3
            value_sectors,      # 8
            obs_sectors,        # 4
            current_value,      # 2
            collection,         # 2
            terrain,            # 1
            density,            # 1
            gait,               # 1
            action_hist,        # 8
            budget,             # 1
        ])

    def _compute_goal_info(self) -> tuple:
        """
        Compute goal direction and distance vectors.

        Returns:
            Tuple of (goal_dir: [sin, cos], goal_dist: [dx, dy, norm])
        """
        if self.goal_position is None:
            return np.zeros(2), np.zeros(3)

        delta = self.goal_position - self.position
        dist = np.linalg.norm(delta)

        # Direction to goal relative to current heading
        angle_to_goal = np.arctan2(delta[1], delta[0]) - self.heading
        goal_dir = np.array([np.sin(angle_to_goal), np.cos(angle_to_goal)])

        # Normalized distance — use max_range for better resolution in 0-10m
        goal_dist = np.array([
            delta[0] / self.max_range,
            delta[1] / self.max_range,
            dist / self.max_range,
        ])

        return goal_dir, goal_dist

    def update_action_history(self, action: np.ndarray):
        """
        Update action history with new velocity command.

        Args:
            action: 3-dimensional velocity command [vx, vy, wz]
        """
        self.action_history = np.roll(self.action_history, -1)
        magnitude = np.linalg.norm(action) / np.sqrt(0.6**2 + 0.3**2 + 0.3**2)
        self.action_history[-1] = np.clip(magnitude, 0.0, 1.0)

    def update_value_sectors(self, data_value_map: np.ndarray,
                             grid_resolution: float = 0.5):
        """
        Scan data value map in 8 directions to build sector observations.

        Args:
            data_value_map: 2D array of data values
            grid_resolution: meters per grid cell
        """
        if data_value_map is None:
            return

        for i in range(VALUE_SECTOR_COUNT):
            angle = self.heading + i * VALUE_SECTOR_ANGLE
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Sample along ray with distance-weighted averaging
            sector_value = 0.0
            weight_sum = 0.0
            for r in np.linspace(0, self.max_range, 40):
                gx = int((self.position[0] + r * dx) / grid_resolution)
                gy = int((self.position[1] + r * dy) / grid_resolution)
                if (0 <= gx < data_value_map.shape[1] and
                        0 <= gy < data_value_map.shape[0]):
                    # Closer samples get higher weight
                    w = np.exp(-0.3 * r)
                    sector_value += data_value_map[gy, gx] * w
                    weight_sum += w

            self.data_value_sectors[i] = sector_value / max(0.01, weight_sum)

    def update_obstacle_sectors(self, obstacle_map: np.ndarray,
                                grid_resolution: float = 0.5):
        """
        Scan for obstacles in 4 directions (front, back, left, right).

        Args:
            obstacle_map: 2D boolean array of obstacles
            grid_resolution: meters per grid cell
        """
        if obstacle_map is None:
            return

        directions = [0, np.pi, np.pi / 2, -np.pi / 2]  # front, back, left, right

        for i, offset in enumerate(directions):
            angle = self.heading + offset
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Ray march for closest obstacle
            for r in np.linspace(0, self.max_range, 50):
                gx = int((self.position[0] + r * dx) / grid_resolution)
                gy = int((self.position[1] + r * dy) / grid_resolution)
                if (0 <= gx < obstacle_map.shape[1] and
                        0 <= gy < obstacle_map.shape[0]):
                    if obstacle_map[gy, gx]:
                        self.obstacle_sectors[i] = r
                        break
            else:
                self.obstacle_sectors[i] = self.max_range

    @classmethod
    def from_vector(cls, vector: np.ndarray,
                    width: float = 60.0, height: float = 60.0) -> 'NavDataState':
        """
        Create NavDataState from state vector.

        Args:
            vector: 42-dimensional state vector
            width: Environment width
            height: Environment height

        Returns:
            NavDataState instance
        """
        if len(vector) != cls.STATE_DIM:
            raise ValueError(f"Expected {cls.STATE_DIM}-dim vector, got {len(vector)}")

        state = cls(width=width, height=height)

        state.base_lin_vel = vector[0:3] / 2.0
        state.base_ang_vel = vector[3:6]
        state.position = np.array([vector[6] * width, vector[7] * height])
        state.heading = np.arctan2(vector[8], vector[9])
        state.data_value_sectors = vector[15:23]
        state.obstacle_sectors = vector[23:27] * state.max_range
        state.current_data_value = vector[27]
        state.current_rarity = vector[28]
        state.collected_value_ratio = vector[29]
        state.coverage_ratio = vector[30]
        state.terrain_type = int(vector[31] * 6)
        state.obstacle_density = vector[32]
        state.action_history = vector[34:42]

        return state
