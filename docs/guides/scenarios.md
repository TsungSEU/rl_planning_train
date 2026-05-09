# 场景详解

Aurora Planning Engine 实现单一导航+数据采集场景，采用分层控制架构。

## Nav_Data (分层导航+数据采集) - 43维状态 / 3连续动作

**分层控制架构**: Aurora 高层导航 (10Hz) → LivelyBot 低层行走 (50Hz) → MuJoCo 物理仿真 (1000Hz)

### 状态空间 (43 维)

| 索引 | 组件 | 维度 | 说明 |
|------|------|------|------|
| 0-2 | 基座线速度 | 3 | [vx, vy, vz] ×2.0 (来自 LivelyBot) |
| 3-5 | 基座角速度 | 3 | [wx, wy, wz] ×1.0 (来自 LivelyBot) |
| 6-7 | 归一化位置 | 2 | [x/W, y/H] |
| 8-9 | 朝向 | 2 | [sinθ, cosθ] |
| 10-11 | 目标方向 | 2 | [sinΔθ, cosΔθ] |
| 12-14 | 目标距离 | 3 | [Δx, Δy, ‖Δ‖] ÷ max_range |
| 15-22 | 数据价值扇区 | 8 | 8 方向的数据价值评分 |
| 23-26 | 障碍物扇区 | 4 | [前, 后, 左, 右] 障碍距离 |
| 27-28 | 当前位置数据 | 2 | [value, rarity] |
| 29-30 | 收集状态 | 2 | [collected_ratio, coverage_ratio] |
| 31 | 地形类型 | 1 | 地形类型 ÷6 |
| 32 | 局部障碍密度 | 1 | 局部障碍物密度 |
| 33 | 步态相位 | 1 | 来自 LivelyBot，sin 编码 |
| 34-41 | 动作历史 | 8 | 最近 8 步的速度命令 |
| 42 | 剩余时间预算 | 1 | 归一化剩余时间 |

### 动作空间 (3 维连续 - 速度命令)

| 索引 | 名称 | 范围 | 单位 |
|------|------|------|------|
| 0 | forward_vel | [-0.3, 0.6] | m/s |
| 1 | lateral_vel | [-0.3, 0.3] | m/s |
| 2 | angular_vel | [-0.3, 0.3] | rad/s |

### 奖励函数 (10 组件)

| 组件 | 权重 | 说明 |
|------|------|------|
| 目标接近 | 3.0 | 向目标移动的进度 |
| 数据价值收集 | 5.0 | value × rarity (核心奖励) |
| 高价值引导 | 2.0 | cos(Δθ) × forward_vel |
| 覆盖率 | 1.0 | 访问新网格奖励 |
| 目标完成 | 100.0 | 到达目标的稀疏奖励 |
| 路径效率 | 0.5 | 惩罚低效路线 |
| 重复访问 | 1.0 | 惩罚重访已收集区域 |
| 碰撞/危险 | 10.0 | 碰撞和接近惩罚 |
| 速度跟踪 | 0.5 | 惩罚停滞 |
| 时间 | 0.1 | 固定步数惩罚 |

---

## 数据价值模型 (Value Model)

Aurora 集成了 `HumanoidDataValueModel`，实现四维数据价值评估：

### 价值维度

| 维度 | 说明 | 计算方式 |
|------|------|----------|
| **空间稀缺度** | 该区域被访问的频率 | 访问次数越少，价值越高 |
| **时间新鲜度** | 距离上次采集的时间 | 时间越久，价值越高 |
| **场景多样性** | 该区域场景的丰富程度 | 基于场景类型分布 |
| **数据质量** | 采集数据的质量评分 | 基于采集条件 |

### 价值模型集成

```python
# 在 NavDataEnvironment 中集成
from value_model.humanoid_value import HumanoidDataValueModel

value_model = HumanoidDataValueModel(
    spatial_decay=0.95,    # 空间衰减因子
    temporal_decay=0.9,    # 时间衰减因子
    diversity_weight=0.3,  # 多样性权重
    quality_weight=0.2     # 质量权重
)

# 每步更新价值评估
value_result = value_model.evaluate(
    position=(x, y),
    scene_type=SceneType.INDOOR,
    timestamp=current_time
)
```

### 价值结果结构

```python
DataValueResult(
    spatial_rarity=0.85,   # 空间稀缺度 [0, 1]
    temporal_freshness=0.72, # 时间新鲜度 [0, 1]
    scene_diversity=0.68,   # 场景多样性 [0, 1]
    quality_score=0.90,     # 数据质量 [0, 1]
    total_value=0.79        # 综合价值 [0, 1]
)
```

---

## 可达性追踪 (Reachability Tracker)

Aurora 实现执行感知的可达性追踪系统，记录规划位置与实际到达位置的差异。

### 可达性分数

| 分数范围 | 含义 |
|----------|------|
| 0.0-0.3 | 低可达性 - 规划路径难以执行 |
| 0.3-0.7 | 中等可达性 - 部分规划可执行 |
| 0.7-1.0 | 高可达性 - 规划路径可可靠执行 |

### 可达性追踪机制

```python
# 在 NavDataEnvironment 中集成
from utils.reachability import ReachabilityTracker

reachability = ReachabilityTracker(
    grid_width=80,          # 网格宽度
    grid_height=80,         # 网格高度
    decay_rate=0.95,        # 时间衰减因子
    min_samples=3           # 最小样本数
)

# 记录执行尝试
reachability.record_attempt(grid_x, grid_y)

# 记录执行成功
reachability.record_success(grid_x, grid_y)

# 获取可达性分数
score = reachability.get_score(grid_x, grid_y)
```

### 可达性在状态空间中的使用

可达性信息通过以下方式融入决策：

1. **数据价值扇区 (状态 [15-22])**: 结合可达性调整价值评分
   ```python
   effective_value = base_value * reachability_score
   ```

2. **奖励函数**: 可达性影响数据采集奖励
   ```python
   reward = data_value * reachability_bonus
   ```

3. **路径规划**: 避免指向低可达性区域

---

## 模型架构

### 推荐配置

```yaml
# Nav_Data 推荐配置
actor_hidden_dims: [256, 128, 64]
critic_hidden_dims: [256, 128, 64]
activation: ELU
init_noise_std: 0.5
```

### 网络结构

```
Input: State (43-dim)
    ↓
Actor Network: [256 → 128 → 64]
    ↓
Output: Action Mean (3-dim) + Log Std (3-dim)
    ↓
Action Sample: Gaussian Policy
```

```
Input: State (43-dim)
    ↓
Critic Network: [256 → 128 → 64]
    ↓
Output: State Value (1-dim)
```

---

## 向量化训练

### 配置参数

```yaml
training:
  num_envs: 4096              # 并行环境数
  steps_per_env: 24           # 每次 update 的 Aurora 步数
  episode_length_s: 60        # 每回合 60 秒
```

### 训练流程

```
1. 初始化 4096 个并行环境
2. 每个环境运行 24 步 (Aurora 决策)
3. 收集所有 transitions (4096 × 24 = 98,304)
4. 计算 advantages (GAE)
5. 更新 PPO 策略 (8 epochs, 4 mini-batches)
6. 重复步骤 2-5
```

### 性能优化

| 优化项 | 加速比 |
|--------|--------|
| GPU 向量化推理 | 10-50x |
| 混合精度训练 (AMP) | ~2x |
| 批量经验收集 | ~3x |

---

## 联合仿真

### Aurora + LivelyBot MuJoCo 联合仿真

```bash
# 使用启动脚本 (推荐)
bash deploy/run_sim.sh              # 默认 10 秒
bash deploy/run_sim.sh 30           # 指定时长

# 或直接运行 Python
python deploy/aurora_livelybot_sim.py \
    --aurora_model runs/train/exp001/nav_data_weights.pt \
    --livelybot_model /path/to/livelybot/policy.pt \
    --duration 30
```

### 可视化

- **MuJoCo 3D 窗口**: 实时机器人姿态
- **Matplotlib 曲线**: 速度跟踪、位置轨迹、目标距离

### 分层控制频率

| 层级 | 频率 | 周期 | 功能 |
|------|------|------|------|
| Aurora 高层 | 10 Hz | 100 ms | 导航决策 |
| LivelyBot 低层 | 50 Hz | 20 ms | 步态控制 |
| 物理仿真 | 1000 Hz | 1 ms | 力学计算 |

---

## 配置文件

训练时使用 `config/nav_data_training.yaml`。

详细配置参数请参阅 [配置说明](configuration.md)。

---

## 相关文档

- [系统架构](architecture.md) — 分层控制架构详解
- [配置说明](configuration.md) — 训练配置参数
