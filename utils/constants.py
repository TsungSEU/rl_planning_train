#!/usr/bin/env python3
"""
Centralized constants for the RL robot path planning training system.
Navigation + Data Collection mode (hierarchical with LivelyBot).
"""

from enum import Enum
from typing import Dict, Any


class PlannerMode(Enum):
    """Planner mode enumeration."""
    NAV_DATA = "nav_data"  # High-level navigation + data collection (hierarchical with LivelyBot)


class SceneType(Enum):
    """Types of scenes for navigation."""
    INDOOR_FLAT = 0
    INDOOR_STAIRS = 1
    INDOOR_SLOPE = 2
    OUTDOOR_FLAT = 3
    OUTDOOR_ROUGH = 4
    OUTDOOR_SLOPE = 5


class TerrainType(Enum):
    """Terrain types for locomotion."""
    FLAT = 0
    STAIRS_UP = 1
    STAIRS_DOWN = 2
    SLOPE_UP = 3
    SLOPE_DOWN = 4
    ROUGH = 5
    UNEVEN = 6


# ============================================================================
# Planner Mode Configuration
# ============================================================================

# Default planner mode
DEFAULT_PLANNER_MODE = PlannerMode.NAV_DATA

# ============================================================================
# Navigation + Data Collection Mode Constants (Hierarchical with LivelyBot)
# ============================================================================

# Planner mode: Aurora provides high-level navigation at 10Hz,
# LivelyBot handles low-level locomotion at 50Hz.

# State space dimensions (43 dimensions total)
NAV_DATA_STATE_DIM = 43
NAV_DATA_VELOCITY_DIM = 3       # [vx, vy, vz] base linear velocity
NAV_DATA_ANGULAR_VEL_DIM = 3    # [wx, wy, wz] base angular velocity
NAV_DATA_POSITION_DIM = 2       # [norm_x, norm_y]
NAV_DATA_HEADING_DIM = 2        # [sin_theta, cos_theta]
NAV_DATA_GOAL_DIR_DIM = 2       # [sin_delta_theta, cos_delta_theta] to target
NAV_DATA_GOAL_DIST_DIM = 3      # [dx, dy, norm] relative to target
NAV_DATA_VALUE_SECTOR_DIM = 8   # Data value in 8 directions
NAV_DATA_OBSTACLE_SECTOR_DIM = 4  # Obstacle distance in 4 directions
NAV_DATA_CURRENT_VALUE_DIM = 2  # [data_value, rarity] at current position
NAV_DATA_COLLECTION_STATUS_DIM = 2  # [collected_ratio, coverage_ratio]
NAV_DATA_TERRAIN_DIM = 1        # Terrain type normalized
NAV_DATA_OBSTACLE_DENSITY_DIM = 1  # Local obstacle density
NAV_DATA_GAIT_PHASE_DIM = 1     # Gait phase from LivelyBot
NAV_DATA_ACTION_HISTORY_DIM = 8  # Last 8 action magnitudes
NAV_DATA_BUDGET_DIM = 1         # Remaining time budget

# Action space (3 continuous velocity commands)
NAV_DATA_ACTION_DIM = 3
NAV_DATA_ACTION_SPACE_TYPE = 'continuous'

# Velocity command ranges (aligned with LivelyBot Pi training ranges)
NAV_DATA_FORWARD_VEL_RANGE = (-0.3, 0.6)   # m/s
NAV_DATA_LATERAL_VEL_RANGE = (-0.3, 0.3)   # m/s
NAV_DATA_ANGULAR_VEL_RANGE = (-0.3, 0.3)   # rad/s

# Control frequencies
NAV_DATA_POLICY_FREQ = 10      # Hz (Aurora decision frequency)
LIVELYBOT_CONTROL_FREQ = 50    # Hz (LivelyBot execution frequency)
PHYSICS_SIM_FREQ = 1000        # Hz (Isaac Gym / MuJoCo physics)
NAV_DATA_DECIMATION = PHYSICS_SIM_FREQ // NAV_DATA_POLICY_FREQ  # 100 steps per Aurora decision
LIVELYBOT_DECIMATION = PHYSICS_SIM_FREQ // LIVELYBOT_CONTROL_FREQ  # 20 steps per LivelyBot action
COMMAND_HOLD_STEPS = LIVELYBOT_CONTROL_FREQ // NAV_DATA_POLICY_FREQ  # 5 LivelyBot steps per Aurora command

# Observation scaling
NAV_DATA_LIN_VEL_SCALE = 2.0
NAV_DATA_ANG_VEL_SCALE = 1.0
NAV_DATA_DIST_SCALE = 1.0      # Normalized by max_range

# Reward weights
NAV_DATA_W_APPROACH = 3.0           # Target approach reward
NAV_DATA_W_DATA_VALUE = 5.0         # Data value collection (core reward)
NAV_DATA_W_VALUE_GUIDE = 2.0        # High-value area guidance
NAV_DATA_W_COVERAGE = 1.0           # Coverage reward
NAV_DATA_W_GOAL = 100.0             # Goal completion reward
NAV_DATA_W_EFFICIENCY = 0.5         # Path efficiency penalty
NAV_DATA_W_REPEAT = 1.0             # Repeat visit penalty
NAV_DATA_W_COLLISION = 10.0         # Collision/danger penalty
NAV_DATA_W_SPEED = 0.5              # Speed tracking
NAV_DATA_W_TIME = 0.1               # Time penalty

# Reward values
NAV_DATA_FIRST_VISIT_BONUS = 2.0    # Bonus for first visit to a cell
NAV_DATA_COLLISION_PENALTY = -10.0
NAV_DATA_NEAR_OBSTACLE_PENALTY_SCALE = 5.0
NAV_DATA_SAFE_DISTANCE = 1.0        # meters
NAV_DATA_STALL_PENALTY = -0.5
NAV_DATA_MIN_SPEED_THRESHOLD = 0.05  # m/s, below this is "stalling"

# Data value map parameters
NAV_DATA_VALUE_SECTOR_COUNT = 8     # Number of directional sectors
NAV_DATA_VALUE_SECTOR_ANGLE = 2 * 3.14159265 / 8  # 45 degrees per sector
NAV_DATA_OBSTACLE_SECTOR_COUNT = 4  # Front, back, left, right
NAV_DATA_MAX_RANGE = 10.0           # meters, for distance normalization

# Training hyperparameters
NAV_DATA_LEARNING_RATE = 3e-4
NAV_DATA_GAMMA = 0.99
NAV_DATA_LAM = 0.95
NAV_DATA_EPSILON = 0.2
NAV_DATA_EPOCHS = 4
NAV_DATA_MINI_BATCHES = 4
NAV_DATA_ENTROPY_COEF = 0.01
NAV_DATA_MAX_GRAD_NORM = 1.0
NAV_DATA_INIT_NOISE_STD = 0.5

# Network architecture
NAV_DATA_ACTOR_HIDDEN = [256, 128, 64]
NAV_DATA_CRITIC_HIDDEN = [256, 128, 64]
NAV_DATA_ACTIVATION = "ELU"

# Environment parameters
NAV_DATA_NUM_ENVS = 4096
NAV_DATA_EPISODE_LENGTH_S = 60     # 60 seconds per episode
NAV_DATA_STEPS_PER_ENV = 24        # Aurora steps per update (= 2.4s real time)
NAV_DATA_MAX_ITERATIONS = 5000
NAV_DATA_SAVE_INTERVAL = 100

# Domain randomization (for data value map, not physics)
NAV_DATA_VALUE_MAP_NOISE = 0.1     # ±10% Gaussian noise on data values

# ============================================================================
# Common Parameters
# ============================================================================

# Data closed-loop parameters
COMMON_SPARSE_THRESHOLD = 0.15
COMMON_EXPLORATION_BONUS = 10.0
COMMON_REDUNDANCY_PENALTY = 5.0
COMMON_GRID_RESOLUTION = 1.0  # meters

# Data value model weights
COMMON_W_SPATIAL_RARITY = 0.3
COMMON_W_TEMPORAL_FRESHNESS = 0.15
COMMON_W_SCENE_DIVERSITY = 0.2
COMMON_W_QUALITY = 0.15
COMMON_W_COVERAGE = 0.2

# Scene rarity mapping
SCENE_RARITY = {
    'indoor_flat': 0.3,
    'indoor_stair': 0.7,
    'indoor_ramp': 0.6,
    'outdoor_flat': 0.4,
    'outdoor_rough': 0.8,
    'outdoor_slope': 0.7,
    'mixed': 0.9
}

# ============================================================================
# Environment Defaults
# ============================================================================

DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 20
DEFAULT_MAX_STEPS = 200
DEFAULT_OBSTACLE_RATIO = 0.2

# Goal distance thresholds
GOAL_DISTANCE_THRESHOLD_COMPLEX = 1.0  # For complex environment with obstacles
GOAL_DISTANCE_THRESHOLD_SIMPLE = 0.5  # For simple environment

# ============================================================================
# PPO Training Constants (Default Values)
# ============================================================================

# PPO hyperparameters
DEFAULT_GAMMA = 0.999  # Discount factor
DEFAULT_LAM = 0.95  # GAE lambda
DEFAULT_EPSILON = 0.2  # PPO clipping parameter
DEFAULT_EPOCHS = 10  # Number of optimization epochs per update
DEFAULT_BATCH_SIZE = 64  # Mini-batch size
DEFAULT_ENTROPY_COEF = 0.01  # Entropy coefficient for exploration

# Network architecture
DEFAULT_HIDDEN_DIM = 128  # Hidden layer size
DEFAULT_NETWORK_LAYERS = 3  # Number of hidden layers
DEFAULT_DROPOUT_RATE = 0.0  # Dropout rate

# Optimization
DEFAULT_LEARNING_RATE = 3e-4  # Adam learning rate
DEFAULT_GRADIENT_CLIP_VALUE = 0.5  # Gradient clipping threshold
DEFAULT_LR_DECAY_RATE = 0.99  # Learning rate decay rate

# Numerical stability
LOGIT_MIN = -10.0
LOGIT_MAX = 10.0
LOG_PROB_CLAMP_MIN = -10.0
LOG_PROB_CLAMP_MAX = 10.0

# ============================================================================
# Training Settings
# ============================================================================

# Episode settings
DEFAULT_TRAINING_EPISODES = 1000
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_LOG_INTERVAL = 100

# Progress bar settings
EPISODES_PER_PROGRESS_BAR = 200

# Moving average for smoothing
SMOOTHING_WINDOW_SIZE = 9

# ============================================================================
# Evaluation Constants
# ============================================================================

DEFAULT_EVAL_EPISODES = 100
SUCCESS_RATE_THRESHOLD = 0.8  # Consider 80%+ success as good performance

# ============================================================================
# State Representation Constants
# ============================================================================

# Normalization bounds
COORDINATE_MIN = 0.0
COORDINATE_MAX = 1.0

# Action history
ACTION_HISTORY_BUFFER_SIZE = 10

# ============================================================================
# Helper Functions
# ============================================================================

def get_mode_config(mode: PlannerMode = PlannerMode.NAV_DATA) -> Dict[str, Any]:
    """
    Get configuration dictionary for the planner mode.

    Returns:
        Dictionary with mode-specific configuration
    """
    return {
        'state_dim': NAV_DATA_STATE_DIM,
        'action_dim': NAV_DATA_ACTION_DIM,
        'action_space_type': NAV_DATA_ACTION_SPACE_TYPE,
        'control_freq': NAV_DATA_POLICY_FREQ,
        'low_level_freq': LIVELYBOT_CONTROL_FREQ,
        'reward': {
            'w_approach': NAV_DATA_W_APPROACH,
            'w_data_value': NAV_DATA_W_DATA_VALUE,
            'w_value_guide': NAV_DATA_W_VALUE_GUIDE,
            'w_coverage': NAV_DATA_W_COVERAGE,
            'w_goal': NAV_DATA_W_GOAL,
            'w_efficiency': NAV_DATA_W_EFFICIENCY,
            'w_repeat': NAV_DATA_W_REPEAT,
            'w_collision': NAV_DATA_W_COLLISION,
            'w_speed': NAV_DATA_W_SPEED,
            'w_time': NAV_DATA_W_TIME,
        },
        'velocity_ranges': {
            'forward': NAV_DATA_FORWARD_VEL_RANGE,
            'lateral': NAV_DATA_LATERAL_VEL_RANGE,
            'angular': NAV_DATA_ANGULAR_VEL_RANGE,
        },
    }


def get_action_names(mode: PlannerMode = PlannerMode.NAV_DATA) -> Dict[int, str]:
    """
    Get action name mapping.

    Returns:
        Dictionary mapping action indices to names
    """
    return {0: "forward_vel", 1: "lateral_vel", 2: "angular_vel"}
