#!/usr/bin/env python3
"""
Unified configuration loader for the Aurora planning system.
Navigation + Data Collection mode (hierarchical with LivelyBot).
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

from utils.constants import PlannerMode, NAV_DATA_STATE_DIM

logger = logging.getLogger(__name__)


@dataclass
class CommonConfig:
    """Common configuration."""
    sparse_threshold: float = 0.15
    exploration_bonus: float = 10.0
    redundancy_penalty: float = 5.0
    grid_resolution: float = 1.0

    # Reachability parameters
    reachability_decay: float = 0.95
    reachability_min_samples: int = 3
    reachability_position_tolerance: float = 0.5

    # Data value model weights
    w_spatial_rarity: float = 0.3
    w_temporal_freshness: float = 0.15
    w_scene_diversity: float = 0.2
    w_quality: float = 0.15
    w_coverage: float = 0.2

    # Scene rarity mapping
    scene_rarity: Dict[str, float] = field(default_factory=lambda: {
        'indoor_flat': 0.3,
        'indoor_stair': 0.7,
        'indoor_ramp': 0.6,
        'outdoor_flat': 0.4,
        'outdoor_rough': 0.8,
        'outdoor_slope': 0.7,
        'mixed': 0.9
    })


@dataclass
class ArchitectureConfig:
    """Model architecture configuration."""
    hidden_dim: int = 128
    network_layers: int = 3
    dropout_rate: float = 0.0


@dataclass
class StateConfig:
    """State space configuration."""
    state_dim: int = 43
    enable_reachability: bool = True


@dataclass
class ActionConfig:
    """Action space configuration."""
    action_dim: int = 3
    action_scale: float = 1.0
    clip_observations: float = 18.0
    clip_actions: float = 18.0


@dataclass
class ReachabilityConfig:
    """Reachability configuration."""
    decay: float = 0.95
    min_samples: int = 3
    position_tolerance: float = 0.5
    effective_cost_threshold: float = 2.0


@dataclass
class ScenarioConfig:
    """Scenario configuration (state, action, reachability, architecture)."""
    state: StateConfig = field(default_factory=StateConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    architecture: ArchitectureConfig = field(default_factory=lambda: ArchitectureConfig(hidden_dim=256, network_layers=4))
    reward: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """PPO training configuration."""
    # Network architecture
    hidden_dim: int = 128
    network_layers: int = 3
    dropout_rate: float = 0.0
    actor_hidden_dims: Optional[list] = None
    critic_hidden_dims: Optional[list] = None
    activation: str = 'ReLU'
    init_noise_std: float = 1.0

    # PPO hyperparameters
    learning_rate: float = 0.0003
    gamma: float = 0.999
    lam: float = 0.95
    epsilon: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    entropy_coef: float = 0.01

    # Entropy decay
    entropy_decay: float = 1.0
    min_entropy_coef: float = 0.001

    # Continuous action parameters
    init_log_std: float = 0.0
    min_log_std: float = -20.0
    max_log_std: float = 2.0

    # Optimization settings
    use_gradient_clipping: bool = True
    gradient_clip_value: float = 0.5
    use_lr_scheduler: bool = False
    lr_decay_rate: float = 0.99
    use_amp: bool = False

    # Training settings
    episodes: int = 1000
    max_steps: int = 200
    steps_per_env: int = 24
    max_iterations: int = 0           # vectorized mode iterations (0 = derive from episodes)
    save_interval: int = 1000
    num_envs: int = 1

    # Environment settings
    env_width: int = 20
    env_height: int = 20
    obstacle_ratio: float = 0.2
    data_sparse_factor: float = 0.1
    coverage_factor: float = 0.1
    grid_resolution: float = 1.0
    num_targets: int = 2
    num_obstacles: int = 20

    # Curriculum learning
    curriculum: bool = False
    curriculum_success_threshold: float = 0.4
    curriculum_window: int = 100
    curriculum_stages: list = field(default_factory=list)


@dataclass
class PlannerConfig:
    """Unified planner configuration."""
    planner_mode: PlannerMode = PlannerMode.NAV_DATA
    common: CommonConfig = field(default_factory=CommonConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def get_state_dim(self) -> int:
        """Get the state dimension."""
        return NAV_DATA_STATE_DIM

    def get_action_dim(self) -> int:
        """Get the action dimension."""
        return 3

    def get_action_space_type(self) -> str:
        """Get the action space type."""
        return 'continuous'


class ConfigLoader:
    """
    Loads and parses the planner configuration YAML.
    """

    @staticmethod
    def load(config_path: Union[str, Path]) -> PlannerConfig:
        """Load configuration from a YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        return ConfigLoader.from_dict(yaml_config)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> PlannerConfig:
        """Create PlannerConfig from a dictionary."""
        # Parse planner mode
        mode_str = config_dict.get('planner_mode', 'nav_data')
        planner_mode = PlannerMode(mode_str)

        # Parse common config
        common_dict = config_dict.get('common', {})
        common_config = CommonConfig(
            sparse_threshold=common_dict.get('sparse_threshold', 0.15),
            exploration_bonus=common_dict.get('exploration_bonus', 10.0),
            redundancy_penalty=common_dict.get('redundancy_penalty', 5.0),
            grid_resolution=common_dict.get('grid_resolution', 1.0),
            reachability_decay=common_dict.get('reachability_decay', 0.95),
            reachability_min_samples=common_dict.get('reachability_min_samples', 3),
            reachability_position_tolerance=common_dict.get('reachability_position_tolerance', 0.5),
            w_spatial_rarity=common_dict.get('w_spatial_rarity', 0.3),
            w_temporal_freshness=common_dict.get('w_temporal_freshness', 0.15),
            w_scene_diversity=common_dict.get('w_scene_diversity', 0.2),
            w_quality=common_dict.get('w_quality', 0.15),
            w_coverage=common_dict.get('w_coverage', 0.2),
            scene_rarity=common_dict.get('scene_rarity', {})
        )

        # Parse scenario config (from nav_data: section in YAML)
        nav_data_dict = config_dict.get('nav_data', config_dict.get('humanoid', {}))
        scenario_config = ConfigLoader._parse_scenario_config(nav_data_dict)

        # Parse training config
        training_dict = config_dict.get('training', {})
        training_config = TrainingConfig(
            hidden_dim=training_dict.get('hidden_dim', 128),
            network_layers=training_dict.get('network_layers', 3),
            dropout_rate=training_dict.get('dropout_rate', 0.0),
            actor_hidden_dims=training_dict.get('actor_hidden_dims'),
            critic_hidden_dims=training_dict.get('critic_hidden_dims'),
            activation=training_dict.get('activation', 'ReLU'),
            init_noise_std=training_dict.get('init_noise_std', 1.0),
            learning_rate=training_dict.get('learning_rate', 0.0003),
            gamma=training_dict.get('gamma', 0.999),
            lam=training_dict.get('lam', 0.95),
            epsilon=training_dict.get('epsilon', 0.2),
            epochs=training_dict.get('epochs', 10),
            batch_size=training_dict.get('batch_size', 64),
            entropy_coef=training_dict.get('entropy_coef', 0.01),
            entropy_decay=training_dict.get('entropy_decay', 1.0),
            min_entropy_coef=training_dict.get('min_entropy_coef', 0.001),
            init_log_std=training_dict.get('init_log_std', 0.0),
            min_log_std=training_dict.get('min_log_std', -20.0),
            max_log_std=training_dict.get('max_log_std', 2.0),
            use_gradient_clipping=training_dict.get('use_gradient_clipping', True),
            gradient_clip_value=training_dict.get('gradient_clip_value', 0.5),
            use_lr_scheduler=training_dict.get('use_lr_scheduler', False),
            lr_decay_rate=training_dict.get('lr_decay_rate', 0.99),
            use_amp=training_dict.get('use_amp', False),
            episodes=training_dict.get('episodes', 1000),
            max_steps=training_dict.get('max_steps',
                int(training_dict.get('episode_length_s', 20) * 10)),
            steps_per_env=training_dict.get('steps_per_env', 24),
            max_iterations=training_dict.get('max_iterations', 0),
            save_interval=training_dict.get('save_interval', 1000),
            num_envs=training_dict.get('num_envs', 1),
            env_width=training_dict.get('env_width', 20),
            env_height=training_dict.get('env_height', 20),
            obstacle_ratio=training_dict.get('obstacle_ratio', 0.2),
            data_sparse_factor=training_dict.get('data_sparse_factor', 0.1),
            coverage_factor=training_dict.get('coverage_factor', 0.1),
            grid_resolution=training_dict.get('grid_resolution', 1.0),
            num_targets=training_dict.get('num_targets', 2),
            num_obstacles=training_dict.get('num_obstacles', 20),
            curriculum=training_dict.get('curriculum', False),
            curriculum_success_threshold=training_dict.get('curriculum_success_threshold', 0.4),
            curriculum_window=training_dict.get('curriculum_window', 100),
            curriculum_stages=training_dict.get('curriculum_stages', []),
        )

        return PlannerConfig(
            planner_mode=planner_mode,
            common=common_config,
            scenario=scenario_config,
            training=training_config
        )

    @staticmethod
    def _parse_scenario_config(section_dict: Dict[str, Any]) -> ScenarioConfig:
        """Parse scenario configuration from YAML section."""
        # State config
        state_dict = section_dict.get('state', {})
        state_config = StateConfig(
            state_dim=state_dict.get('state_dim', 43),
            enable_reachability=state_dict.get('enable_reachability', True),
        )

        # Action config
        action_dict = section_dict.get('action', {})
        action_config = ActionConfig(
            action_dim=action_dict.get('action_dim', 3),
            action_scale=action_dict.get('action_scale', 1.0),
            clip_observations=action_dict.get('clip_observations', 18.0),
            clip_actions=action_dict.get('clip_actions', 18.0),
        )

        # Reachability config
        reach_dict = section_dict.get('reachability', {})
        reachability_config = ReachabilityConfig(
            decay=reach_dict.get('decay', 0.95),
            min_samples=reach_dict.get('min_samples', 3),
            position_tolerance=reach_dict.get('position_tolerance', 0.5),
            effective_cost_threshold=reach_dict.get('effective_cost_threshold', 2.0),
        )

        # Architecture config
        arch_dict = section_dict.get('architecture', {})
        architecture_config = ArchitectureConfig(
            hidden_dim=arch_dict.get('hidden_dim', 256),
            network_layers=arch_dict.get('network_layers', 4),
            dropout_rate=arch_dict.get('dropout_rate', 0.0),
        )

        # Reward config (flat dict from nav_data.reward section)
        reward_config = section_dict.get('reward', {})

        return ScenarioConfig(
            state=state_config,
            action=action_config,
            reachability=reachability_config,
            architecture=architecture_config,
            reward=reward_config,
        )

    @staticmethod
    def save(config: PlannerConfig, config_path: Union[str, Path]):
        """Save configuration to a YAML file."""
        config_dict = ConfigLoader.to_dict(config)
        config_path = Path(config_path)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {config_path}")

    @staticmethod
    def to_dict(config: PlannerConfig) -> Dict[str, Any]:
        """Convert PlannerConfig to dictionary."""
        s = config.scenario
        return {
            'planner_mode': config.planner_mode.value,
            'common': {
                'sparse_threshold': config.common.sparse_threshold,
                'exploration_bonus': config.common.exploration_bonus,
                'redundancy_penalty': config.common.redundancy_penalty,
                'grid_resolution': config.common.grid_resolution,
                'reachability_decay': config.common.reachability_decay,
                'reachability_min_samples': config.common.reachability_min_samples,
                'reachability_position_tolerance': config.common.reachability_position_tolerance,
                'w_spatial_rarity': config.common.w_spatial_rarity,
                'w_temporal_freshness': config.common.w_temporal_freshness,
                'w_scene_diversity': config.common.w_scene_diversity,
                'w_quality': config.common.w_quality,
                'w_coverage': config.common.w_coverage,
                'scene_rarity': config.common.scene_rarity,
            },
            'nav_data': {
                'state': {
                    'state_dim': s.state.state_dim,
                    'enable_reachability': s.state.enable_reachability,
                },
                'action': {
                    'action_dim': s.action.action_dim,
                    'action_scale': s.action.action_scale,
                    'clip_observations': s.action.clip_observations,
                    'clip_actions': s.action.clip_actions,
                },
                'reachability': {
                    'decay': s.reachability.decay,
                    'min_samples': s.reachability.min_samples,
                    'position_tolerance': s.reachability.position_tolerance,
                    'effective_cost_threshold': s.reachability.effective_cost_threshold,
                },
            },
        }


def load_config(config_path: Union[str, Path]) -> PlannerConfig:
    """Convenience function to load configuration."""
    return ConfigLoader.load(config_path)


def merge_with_training_config(planner_config: PlannerConfig,
                                training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge planner configuration with training configuration."""
    merged = training_config.copy()

    merged['planner_mode'] = planner_config.planner_mode.value
    merged['state_dim'] = planner_config.get_state_dim()
    merged['action_dim'] = planner_config.get_action_dim()
    merged['action_space_type'] = planner_config.get_action_space_type()

    # Add common parameters
    merged['sparse_threshold'] = planner_config.common.sparse_threshold
    merged['exploration_bonus'] = planner_config.common.exploration_bonus
    merged['redundancy_penalty'] = planner_config.common.redundancy_penalty
    merged['grid_resolution'] = planner_config.common.grid_resolution

    return merged
