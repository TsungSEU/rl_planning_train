# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **RL Planning Train** — a Reinforcement Learning training module for humanoid robot navigation and data collection. It trains a high-level navigation policy using Proximal Policy Optimization (PPO).

The system implements a hierarchical architecture with LivelyBot Pi:
- **Aurora** (this repo): High-level navigation + data value assessment (43-dim state → 3 velocity commands)
- **LivelyBot** (separate repo): Low-level bipedal locomotion (705-dim state → 12 joint targets)

## Hierarchical Architecture

```
Aurora Policy (10 Hz)                    LivelyBot Policy (50 Hz)
┌─────────────────────┐                  ┌─────────────────────┐
│  State: 43 维        │                  │  State: 705 维       │
│  - 位置/朝向/速度     │                  │  - 关节/步态/IMU     │
│  - 数据价值地图       │                  │                     │
│  - 目标/覆盖率       │                  │                     │
│  Action: 3 维        │──vx,vy,ωz───→   │  Action: 12 维      │
│  [vx, vy, ωz]       │                  │  [joint offsets]    │
│  Reward: 导航+数据    │←──robot state── │  Reward: 步态+稳定  │
└─────────────────────┘                  └─────────────────────┘
        ↓ 0.1s                                   ↓ 0.02s
  Isaac Gym / MuJoCo 物理仿真 (1000 Hz)
```

## Development Commands

### Training
```bash
python train.py --config config/nav_data_training.yaml
```

### Evaluation
```bash
python eval.py --model-path runs/train/exp001/nav_data_weights.pt --config config/nav_data_training.yaml --episodes 100
```

### Deployment (ONNX Export)
```bash
python -m utils.export_onnx --model-path runs/train/exp001/nav_data_weights.pt --config config/nav_data_training.yaml --output models/nav_data.onnx
```

### Testing
```bash
pytest tests/ -v
```

## Architecture

### State Space (43 dimensions)
- **[0-2]** Base linear velocity [vx, vy, vz] (×2.0, from LivelyBot)
- **[3-5]** Base angular velocity [wx, wy, wz] (×1.0, from LivelyBot)
- **[6-7]** Normalized position [x/W, y/H]
- **[8-9]** Heading [sinθ, cosθ]
- **[10-11]** Goal direction [sinΔθ, cosΔθ]
- **[12-14]** Goal distance [Δx, Δy, ‖Δ‖] (÷max_range)
- **[15-22]** Data value sectors (8 directions)
- **[23-26]** Obstacle sectors (4 directions: front, back, left, right)
- **[27-28]** Current position data [value, rarity]
- **[29-30]** Collection status [collected_ratio, coverage_ratio]
- **[31]** Terrain type (normalized ÷6)
- **[32]** Local obstacle density
- **[33]** Gait phase (from LivelyBot, sin encoding)
- **[34-41]** Action history (8 steps)
- **[42]** Remaining time budget

### Action Space (3 continuous)
- 0: forward_vel [-0.3, 0.6] m/s
- 1: lateral_vel [-0.3, 0.3] m/s
- 2: angular_vel [-0.3, 0.3] rad/s

### Reward Function (10 components)
1. **Target approach** (w=3.0): Δdist progress toward goal
2. **Data value collection** (w=5.0): value × rarity (core reward)
3. **High-value guidance** (w=2.0): cos(Δθ) × forward_vel toward best sector
4. **Coverage** (w=1.0): reward for new cells visited
5. **Goal completion** (w=100.0): sparse reward for reaching target
6. **Path efficiency** (w=0.5): penalize inefficient routes
7. **Repeat visit** (w=1.0): penalize revisiting collected areas
8. **Collision/danger** (w=10.0): collision and proximity penalty
9. **Speed tracking** (w=0.5): penalize stalling
10. **Time** (w=0.1): fixed step penalty

### Model Architecture
- **ActorCriticNet**: Custom hidden_dims architecture [256, 128, 64] with ELU activation
- Continuous action spaces with Gaussian policy

## Module Organization

### Core
- `core/base_agent.py`: ActorCriticNet, ContinuousPPOAgent (includes take_action, get_action_mean_std, evaluate_batch, select_actions_batch, update_from_arrays)
- `core/base_environment.py`: BaseEnvironment abstract class
- `core/base_reward.py`: BaseRewardCalculator, RewardState, RewardBreakdown
- `core/config.py`: PPOConfig, load_config (YAML → PlannerConfig → PPOConfig pipeline)
- `core/factory.py`: ScenarioFactory — creates NavDataEnvironment and ContinuousPPOAgent
- `core/vectorized_env.py`: VectorizedEnvRunner — manages N parallel env instances

### Scenario (humanoid/)
- `humanoid/nav_data_state.py`: NavDataState (43-dim state)
- `humanoid/nav_data_reward.py`: NavDataRewardCalculator (10 reward components)
- `humanoid/nav_data_environment.py`: NavDataEnvironment (with value_model + reachability integration)
- `humanoid/velocity_action.py`: VelocityAction (3-DOF, aligned with LivelyBot)

### Value Model
- `value_model/base_value.py`: BaseDataValueModel, DataValueConfig, DataValueResult
- `value_model/humanoid_value.py`: HumanoidDataValueModel (4D value: spatial rarity, temporal freshness, scene diversity, quality)

### Training
- `train.py`: TrainingRunner (sequential) and VectorizedTrainingRunner (parallel envs)
- `eval.py`: EvaluationRunner

### Utilities
- `utils/constants.py`: PlannerMode enum, SceneType, TerrainType, dimension constants
- `utils/config_loader.py`: YAML config loader (PlannerConfig hierarchy)
- `utils/export_onnx.py`: ONNX model export
- `utils/validation.py`: ConfigValidator
- `utils/reachability.py`: ReachabilityTracker (position accessibility scoring)
- `utils/training_utils.py`: EarlyStopping, TrainingMetrics
- `utils/experience_loader.py`: Edge device experience data loader

### CLI
- `rl_planning_train/cli/main.py`: CLI entry point (`rl_train` command)
- `rl_planning_train/cli/commands/train.py`: Training command
- `rl_planning_train/cli/commands/eval.py`: Evaluation command
- `rl_planning_train/cli/commands/export.py`: ONNX export command

## Configuration

All hyperparameters are managed through YAML files in `config/`:
- `nav_data_training.yaml`: Training configuration

Configuration loading: `YAML file → ConfigLoader → PlannerConfig → PPOConfig.from_planner_config() → PPOConfig`

## Integrated Systems

### Value Model
NavDataEnvironment integrates `HumanoidDataValueModel`:
- Tracks spatial rarity, temporal freshness, scene diversity, and quality per cell
- `update_visit()` called on each step to maintain 4D value assessment
- Results available for reward calculation and target selection

### Reachability Tracker
NavDataEnvironment includes `ReachabilityTracker`:
- Records visit success/failure per grid cell
- Computes reachability scores [0, 1] for each position
- Decays over time to encourage exploration

## LivelyBot Integration

The system interfaces with LivelyBot Pi through velocity commands:
1. Aurora outputs `[vx, vy, ωz]` at 10 Hz
2. Commands held for 5 LivelyBot steps (50Hz)
3. LivelyBot translates to 12 joint targets via PD controller
4. Physical simulation at 1000 Hz

LivelyBot repository: `/home/xucong/caicAD/01datainfra/robot/livelybot_pi_rl_baseline`

## Directory Structure

- `config/`: Training configuration YAML files
- `models/`: Trained model weights (.pt) and ONNX exports (.onnx)
- `runs/`: Training results and logs
- `core/`: Base classes + factory (agent, environment, reward, config)
- `humanoid/`: Navigation + data collection modules (state, reward, environment, velocity action)
- `value_model/`: Data value assessment models
- `deploy/`: Deployment scripts (Aurora + LivelyBot simulation)
- `utils/`: Export, config, reachability, training utilities
- `rl_planning_train/`: CLI package
- `tests/`: Unit tests (26 tests)
