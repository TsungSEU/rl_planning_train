"""
Configuration classes for PPO training.

Single entry point: load_config() reads YAML -> PlannerConfig -> PPOConfig.
Reward config is handled by NavDataRewardCalculator internally.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import yaml
from pathlib import Path

from utils.config_loader import ConfigLoader, PlannerConfig
from utils.constants import PlannerMode


@dataclass
class PPOConfig:
    """PPO training configuration — derived from PlannerConfig."""

    # ── State and action dimensions ──────────────────────────────────
    state_dim: int
    action_dim: int
    action_space: str = 'continuous'

    # ── Network architecture ─────────────────────────────────────────
    hidden_dim: int = 128
    network_layers: int = 3
    dropout_rate: float = 0.0
    hidden_dims: Optional[list] = None  # e.g. [256, 128, 64]
    activation: str = 'ReLU'

    # ── PPO hyper-parameters ─────────────────────────────────────────
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

    # ── Training loop ────────────────────────────────────────────────
    episodes: int = 1000
    max_steps: int = 200
    steps_per_env: int = 24
    max_iterations: int = 0           # vectorized mode iterations (0 = derive from episodes)
    save_interval: int = 1000

    # ── Optimisation ─────────────────────────────────────────────────
    use_gradient_clipping: bool = True
    gradient_clip_value: float = 0.5
    use_lr_scheduler: bool = False
    lr_decay_rate: float = 0.99
    use_amp: bool = False

    # ── Continuous action noise ──────────────────────────────────────
    init_log_std: float = 0.0
    min_log_std: float = -20.0
    max_log_std: float = 2.0

    # ── Environment geometry ─────────────────────────────────────────
    env_width: int = 20
    env_height: int = 20
    obstacle_ratio: float = 0.2
    data_sparse_factor: float = 0.1
    coverage_factor: float = 0.1
    grid_resolution: float = 1.0

    # ── Reachability ─────────────────────────────────────────────────
    enable_reachability: bool = True
    reachability_decay: float = 0.95
    reachability_min_samples: int = 3
    reachability_position_tolerance: float = 0.5

    # ── Scenario tag ─────────────────────────────────────────────────
    scenario: str = 'nav_data'

    # ── Reward weights (flat dict, consumed by reward calculators) ───
    reward: Dict[str, Any] = field(default_factory=dict)

    # ── Nav_Data specific ────────────────────────────────────────────
    num_targets: int = 2
    num_obstacles: int = 20

    # ── Value model toggle ───────────────────────────────────────────
    enable_value_model: bool = True

    # ── Vectorized training ──────────────────────────────────────────
    num_envs: int = 1

    # ── Keep a reference to the full planner config ──────────────────
    planner_config: Optional[Any] = field(default=None, repr=False)

    # ─────────────────────────────────────────────────────────────────
    # Factory: PlannerConfig -> PPOConfig
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def from_planner_config(cls, planner_config: PlannerConfig) -> 'PPOConfig':
        """Derive a PPOConfig from the unified PlannerConfig."""
        state_dim = planner_config.get_state_dim()
        action_dim = planner_config.get_action_dim()
        action_space = planner_config.get_action_space_type()

        training = planner_config.training
        arch = planner_config.scenario.architecture

        hidden_dims = training.actor_hidden_dims
        activation = training.activation

        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=arch.hidden_dim,
            network_layers=arch.network_layers,
            dropout_rate=arch.dropout_rate,
            hidden_dims=hidden_dims,
            activation=training.activation,
            learning_rate=training.learning_rate,
            gamma=training.gamma,
            lam=training.lam,
            epsilon=training.epsilon,
            epochs=training.epochs,
            batch_size=training.batch_size,
            entropy_coef=training.entropy_coef,
            entropy_decay=training.entropy_decay,
            min_entropy_coef=training.min_entropy_coef,
            init_log_std=training.init_log_std,
            min_log_std=training.min_log_std,
            max_log_std=training.max_log_std,
            use_gradient_clipping=training.use_gradient_clipping,
            gradient_clip_value=training.gradient_clip_value,
            use_lr_scheduler=training.use_lr_scheduler,
            lr_decay_rate=training.lr_decay_rate,
            use_amp=training.use_amp,
            episodes=training.episodes,
            max_steps=training.max_steps,
            steps_per_env=training.steps_per_env,
            max_iterations=training.max_iterations,
            save_interval=training.save_interval,
            num_envs=training.num_envs,
            env_width=training.env_width,
            env_height=training.env_height,
            obstacle_ratio=training.obstacle_ratio,
            data_sparse_factor=training.data_sparse_factor,
            coverage_factor=training.coverage_factor,
            grid_resolution=planner_config.common.grid_resolution,
            num_targets=training.num_targets if hasattr(training, 'num_targets') else 2,
            num_obstacles=training.num_obstacles if hasattr(training, 'num_obstacles') else 20,
            enable_reachability=planner_config.scenario.state.enable_reachability,
            reachability_decay=planner_config.scenario.reachability.decay,
            reachability_min_samples=planner_config.scenario.reachability.min_samples,
            reachability_position_tolerance=planner_config.scenario.reachability.position_tolerance,
            reward=planner_config.scenario.reward,
            planner_config=planner_config,
        )

    # ─────────────────────────────────────────────────────────────────
    # Legacy YAML loader (for configs without planner_mode key)
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PPOConfig':
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Unified planner config
        if 'planner_mode' in config_dict:
            planner_config = ConfigLoader.load(config_path)
            return cls.from_planner_config(planner_config)

        # Legacy fallback — minimal extraction
        return cls._from_legacy_dict(config_dict)

    @classmethod
    def _from_legacy_dict(cls, d: Dict[str, Any]) -> 'PPOConfig':
        d.setdefault('scenario', 'nav_data')
        if 'planner_mode' in d:
            d.pop('planner_mode')
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if k != 'planner_config'}


def load_config(config_path: str) -> PPOConfig:
    """Load configuration from YAML file."""
    return PPOConfig.from_yaml(config_path)
