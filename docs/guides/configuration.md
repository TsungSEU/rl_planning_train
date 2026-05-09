# 配置说明

## 配置文件结构

所有配置文件使用YAML格式，位于 `config/` 目录。

### 统一配置格式

```yaml
planner_mode: nav_data

# 通用参数
common:
  sparse_threshold: 0.15
  exploration_bonus: 10.0
  # ...

# 导航+数据采集参数
nav_data:
  state:
    state_dim: 43
  action:
    action_dim: 3
    action_space: continuous
  # ...

# 训练参数
training:
  hidden_dim: 128
  episodes: 15000
  # ...
```

---

## Nav_Data 配置 (nav_data_training.yaml)

### 场景参数

```yaml
planner_mode: nav_data

nav_data:
  state:
    state_dim: 43             # 状态维度
  action:
    action_dim: 3             # 3个连续速度命令
    action_space: continuous
```

### 控制频率

```yaml
nav_data:
  control:
    policy_freq: 10           # Aurora 决策频率 (Hz)
    low_level_freq: 50        # LivelyBot 执行频率 (Hz)
    physics_freq: 1000        # 物理仿真频率 (Hz)
    command_hold_steps: 5     # LivelyBot 每次执行 Aurora 命令的步数
```

### 网络架构

```yaml
training:
  actor_hidden_dims: [256, 128, 64]  # Actor 隐藏层
  critic_hidden_dims: [256, 128, 64]  # Critic 隐藏层
  activation: ELU                     # 激活函数
  init_noise_std: 0.5                 # 初始噪声标准差
```

### PPO超参数

```yaml
training:
  learning_rate: 0.0003       # 学习率
  gamma: 0.95                 # 折扣因子
  lam: 0.95                   # GAE lambda
  epsilon: 0.2                # PPO裁剪参数
  epochs: 8                   # 每次更新轮数
  mini_batches: 4             # mini-batch 数
  entropy_coef: 0.1           # 熵系数
  max_grad_norm: 1.0          # 梯度裁剪
  use_amp: true               # 混合精度训练
```

### 连续动作参数

```yaml
training:
  init_log_std: 0.0           # 初始log标准差
  min_log_std: -20.0          # log标准差下限
  max_log_std: 2.0            # log标准差上限
```

### 向量化训练设置

```yaml
training:
  num_envs: 4096              # 并行环境数
  steps_per_env: 24           # 每次 update 的 Aurora 步数 (= 2.4s)
  episode_length_s: 60        # 每回合 60 秒
  max_iterations: 5000        # 最大迭代数
  save_interval: 100          # 保存间隔
```

### 环境设置

```yaml
training:
  env_width: 40               # 环境宽度 (米)
  env_height: 40              # 环境高度 (米)
  grid_resolution: 0.5        # 网格分辨率 (米)
  max_range: 10.0             # 观测最大范围 (米)
```

---

## 奖励配置 (nav_data_training.yaml)

```yaml
nav_data:
  reward:
    # 导航奖励
    w_approach: 3.0            # 目标接近奖励
    w_goal: 100.0              # 目标完成奖励

    # 数据采集奖励 (核心)
    w_data_value: 5.0          # 数据价值采集
    w_value_guide: 2.0         # 高价值区域引导
    w_coverage: 1.0            # 覆盖率奖励

    # 效率/安全惩罚
    w_efficiency: 0.5          # 路径效率
    w_repeat: 1.0              # 重复访问惩罚
    w_collision: 10.0          # 碰撞危险惩罚
    w_speed: 0.5               # 速度跟踪
    w_time: 0.1                # 时间惩罚
```

---

## 混合精度训练

```yaml
training:
  use_amp: true               # 启用混合精度训练（需GPU）
```

---

## 配置文件列表

| 文件名 | 用途 |
|--------|------|
| `nav_data_training.yaml` | 导航+数据采集训练配置 |

---

## 导出配置说明

导出时使用的网络架构参数由训练配置决定，无需单独的导出配置文件：

```bash
python -m utils.export_onnx \
  --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml \
  --output models/nav_data.onnx
```
