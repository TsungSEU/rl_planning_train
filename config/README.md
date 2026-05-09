# Planner Configuration Files

This directory contains unified configuration files for the dual-scenario RL platform, supporting both autonomous driving (`auto`) and humanoid robot (`humanoid`) modes.

## Configuration File Structure

All configuration files follow a unified structure with `planner_mode` as the top-level selector:

```yaml
planner_mode: auto  # or "humanoid"

# Common parameters shared across modes
common:
  sparse_threshold: 0.15
  exploration_bonus: 10.0
  grid_resolution: 1.0
  # ...

# Auto mode specific configuration
auto:
  state:
    state_dim: 24
    enable_reachability: false
  action:
    action_dim: 4
    action_space: discrete
  reward:
    goal_reward: 50.0
    collision_penalty: -50.0
  # ...

# Humanoid mode specific configuration
humanoid:
  state:
    state_dim: 75
    enable_joint_state: true
    enable_gait_phase: true
    enable_system_monitoring: true
    enable_reachability: true
  action:
    action_dim: 18
    action_space: continuous
  reward:
    goal_reward: 1000.0
    fall_penalty: -20.0
  # ...

# Training parameters (for training configs)
training:
  hidden_dim: 128
  learning_rate: 0.0003
  episodes: 15000
  # ...
```

## Configuration Files

### Training Configurations

- **`auto_training.yaml`** - Auto mode training configuration
  - State dim: 25 (position + drivable network + actions + budget + density + reachability)
  - Action dim: 4 (discrete: forward, left, right, U-turn)
  - Episodes: 15000
  - Hidden dim: 128
  - Network layers: 3

- **`humanoid_training.yaml`** - Humanoid mode training configuration
  - State dim: 75 (position + joints + gait + monitoring + budget + reachability)
  - Action dim: 18 (continuous: joint offsets)
  - Episodes: 3000
  - Hidden dim: 256
  - Network layers: 4

### Evaluation Configurations

- **`auto_evaluation.yaml`** - Auto mode evaluation configuration
- **`humanoid_evaluation.yaml`** - Humanoid mode evaluation configuration

### Export Configurations

- **`auto_export.yaml`** - Auto mode ONNX export configuration
- **`humanoid_export.yaml`** - Humanoid mode ONNX export configuration

### Reference Configuration

- **`planner_weights_v2.yaml`** - Complete reference configuration with both modes

## Usage

### Training

```bash
# Train auto mode
python train.py --scenario auto --config config/auto_training.yaml

# Train humanoid mode
python train.py --scenario humanoid --config config/humanoid_training.yaml

# Override episodes
python train.py --scenario auto --config config/auto_training.yaml --episodes 5000

# Train with CUDA
python train.py --scenario humanoid --config config/humanoid_training.yaml --device cuda
```

### Evaluation

```bash
# Evaluate auto mode
python eval.py --scenario auto --config config/auto_evaluation.yaml

# Evaluate humanoid mode
python eval.py --scenario humanoid --config config/humanoid_evaluation.yaml
```

### Export to ONNX

```bash
# Export auto mode to ONNX
python utils/export_onnx.py --scenario auto --config config/auto_export.yaml

# Export humanoid mode to ONNX
python utils/export_onnx.py --scenario humanoid --config config/humanoid_export.yaml
```

## Configuration Parameters Reference

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sparse_threshold` | float | 0.15 | Sparse area threshold for exploration |
| `exploration_bonus` | float | 10.0 | Exploration reward coefficient |
| `redundancy_penalty` | float | 5.0 | Redundancy penalty coefficient |
| `grid_resolution` | float | 1.0 | Grid resolution in meters |
| `reachability_decay` | float | 0.95 | Historical data decay coefficient |
| `reachability_min_samples` | int | 3 | Minimum samples for reliability |
| `reachability_position_tolerance` | float | 0.5 | Position tolerance in meters |

### Auto Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state.state_dim` | int | 24 | State space dimension |
| `action.action_dim` | int | 4 | Number of discrete actions |
| `reward.goal_reward` | float | 50.0 | Reward for reaching goal |
| `reward.collision_penalty` | float | -50.0 | Penalty for collision |
| `reward.new_sparse_reward` | float | 10.0 | Reward for exploring sparse areas |

### Humanoid Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state.state_dim` | int | 75 | State space dimension |
| `action.action_dim` | int | 18 | Number of continuous actions |
| `reward.goal_reward` | float | 1000.0 | Reward for reaching goal |
| `reward.fall_penalty` | float | -20.0 | Penalty for falling |
| `reward.new_sparse_reward` | float | 1.0 | Reward for exploring sparse areas |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training.hidden_dim` | int | 128 | Hidden layer dimension |
| `training.network_layers` | int | 3 | Number of network layers |
| `training.learning_rate` | float | 0.0003 | Learning rate |
| `training.gamma` | float | 0.999 | Discount factor |
| `training.episodes` | int | 1000 | Number of training episodes |
| `training.max_steps` | int | 200 | Maximum steps per episode |
| `training.batch_size` | int | 64 | Mini-batch size |
| `training.epochs` | int | 10 | PPO optimization epochs |
