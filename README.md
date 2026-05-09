#RL Planning Training

基于 PPO 的分层强化学习训练平台，为双足人形机器人提供高层导航与数据采集策略。

## 系统总览

Aurora 是面向人形机器人的**自主数据采集系统**，采用云-边-端三层架构：

| 组件 | 定位 | 核心职责 | 运行环境 |
|------|------|----------|----------|
| **rl_planning_train** (本仓库) | 云端训练平台 | 训练高层导航策略，导出 ONNX 模型 | GPU 服务器 (Isaac Gym/MuJoCo) |
| **aurora-edge-runtime** | 边缘推理引擎 | 加载 ONNX 模型实时推理，数据采集与上传 | 机器人端 (ARM/x86 + ROS2) |
| **livelybot_pi_rl_baseline** | 底层运动控制 | 双足行走策略，速度指令 → 关节动作 | 机器人端 (Isaac Gym/MuJoCo/实机) |

```
┌─────────────────────────────────────────────────────────────┐
│                     云端 (Cloud)                             │
│                                                             │
│   rl_planning_train                                            │
│   输入: 仿真环境 + 边缘端经验数据                               │
│   算法: PPO 强化学习                                          │
│   状态: 43维 → 动作: 3维速度指令 [vx, vy, ωz]                  │
│   输出: ONNX 模型 (nav_data.onnx)                            │
│                         │                                   │
│                         │ ONNX 模型 (部署到边缘端)             │
└─────────────────────────┼────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     边缘端 (Edge)                            │
│                                                             │
│   aurora-edge-runtime                                       │
│   · ONNX Runtime 10Hz 实时推理 → 速度指令                     │
│   · 安全约束层 (速度限制 + 碰撞检测)                            │
│   · 数据采集 (Rosbag2, 15s前+5s后环形缓冲)                     │
│   · S3 数据上传 → 回传云端 (闭环迭代)                           │
│                                                             │
│          │ cmd_vel (10Hz)        │ 经验数据                  │
│          ↓                       ↓                          │
│   ┌──────────────────┐    数据回传至云端                      │
│   │  LivelyBot (ROS2) │                                     │
│   └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     机器人端 (Robot)                         │
│                                                             │
│   livelybot_pi_rl_baseline                                  │
│   · 50Hz 低层运动控制                                         │
│   · 输入: 3维速度指令 + 705维本体感知                           │
│   · 输出: 12维关节目标 → PD控制器 → 关节力矩                     │
│   · 自主处理: 步态生成、平衡控制、地面接触                        │
└─────────────────────────────────────────────────────────────┘
```

### 分层控制架构

```
Aurora Policy (10 Hz)                    LivelyBot Policy (50 Hz)
┌─────────────────────┐                  ┌─────────────────────┐
│  State: 43 维       │                  │  State: 705 维       │
│  - 位置/朝向/速度     │                  │  - 关节/步态/IMU     │
│  - 数据价值地图       │                  │                     │
│  - 目标/覆盖率        │                  │                     │
│  Action: 3 维       │──vx,vy,ωz───→    │  Action: 12 维       │
│  [vx, vy, ωz]       │                  │  [joint offsets]    │
│  Reward: 导航+数据   │←──robot state──  │  Reward: 步态+稳定    │
└─────────────────────┘                  └─────────────────────┘
        ↓ 0.1s                                   ↓ 0.02s
  Isaac Gym / MuJoCo 物理仿真 (1000 Hz)
```

## 特性

- **云-边-端三层架构**: 训练、推理、执行分离部署
- **分层控制**: Aurora 高层导航 (10Hz) → LivelyBot 低层行走 (50Hz) → 物理仿真 (1000Hz)
- **数据价值驱动**: 4 维价值评估 (空间稀缺度、时间新鲜度、场景多样性、数据质量)
- **向量化训练**: 4096 并行环境，GPU 批量推理，混合精度加速
- **可达性追踪**: 执行感知的路径可达性评估
- **联合仿真**: Aurora + LivelyBot MuJoCo 联合仿真，实时可视化
- **ONNX 导出**: 模型导出用于边缘端部署 (ONNX Runtime)

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd rl_planning_train

# 安装依赖
pip install -e .

# 验证安装
aurora --version
pytest tests/ -v
```

详细安装说明请参阅 [安装文档](docs/getting-started/installation.md)。

### 配置文件

训练配置位于 `config/nav_data_training.yaml`，主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 4096 | 并行环境数 (1=串行, >1=向量化) |
| `max_iterations` | 5000 | 向量化训练迭代数 |
| `steps_per_env` | 24 | 每次迭代每个环境的 rollout 步数 (= 2.4s) |
| `episode_length_s` | 60 | 单个 episode 最大时长 (秒), 超出截断 |
| `learning_rate` | 0.0003 | PPO 学习率 |
| `gamma` | 0.95 | 折扣因子 |
| `epsilon` | 0.2 | PPO 裁剪参数 |
| `epochs` | 8 | 每次更新优化轮数 |
| `entropy_coef` | 0.1 | 熵系数 (控制探索) |
| `use_amp` | true | 混合精度训练 |
| `env_width/height` | 40 | 环境尺寸 (米) |

详细配置说明请参阅 [配置文档](docs/guides/configuration.md)。

### 训练命令

#### 使用 Aurora CLI (推荐)

```bash
# 标准训练 (向量化, 4096 并行环境)
aurora train --config config/nav_data_training.yaml

# 指定 GPU
aurora train --config config/nav_data_training.yaml --device cuda

# 从检查点继续训练
aurora train --config config/nav_data_training.yaml \
  --model-path runs/train/exp001/nav_data_weights_iter_100.pt
```

#### 使用 Python 脚本

```bash
# 向量化训练
python train.py --config config/nav_data_training.yaml

# 串行训练 (单环境, 用于调试)
python train.py --config config/nav_data_training.yaml --num-envs 1

# 训练并评估
python train.py --config config/nav_data_training.yaml --evaluate --eval-episodes 100
```

#### Python 脚本参数

**通用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 (必需) | - |
| `--device` | 设备 (cpu/cuda/auto) | cuda |
| `--num-envs` | 并行环境数。1=串行模式, >1=向量化模式 | 配置文件值 |
| `--save-interval` | 保存间隔 | 配置文件值 |
| `--evaluate` | 训练后运行评估 | false |
| `--eval-episodes` | 评估回合数 | 1000 |
| `--model-path` | 预训练模型路径 (继续训练) | - |

**串行模式参数** (`--num-envs 1`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--episodes` | 训练回合数 | 配置文件值 |
| `--update-interval` | 累积 N 个 episode 后更新 | 4 |

**向量化模式参数** (`--num-envs >1`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-iterations` | 训练迭代数 (覆盖配置) | 配置文件值 (5000) |
| `--exp-dir` | 实验输出目录 | 自动递增 `runs/train/expNNN` |

### 训练方法

训练采用 **On-Policy PPO**，根据 `num_envs` 自动选择模式：

| 模式 | num_envs | 适用场景 |
|------|----------|----------|
| **串行** | 1 | 调试，逐 episode 收集经验后更新 |
| **向量化** | >1 | 大规模训练，N 个并行环境批量推理 |

#### 向量化训练流程 (默认)

```
每次迭代 (iteration):
  1. Rollout: 对 N 个环境各跑 steps_per_env 步
     → 收集 N × steps_per_env 个 transitions
     → GPU 批量推理 (单次前向传播处理全部环境)
     → 超过 max_steps 的 episode 强制截断
  2. Update: 用收集的 transitions 做 PPO 策略更新
     → 计算 GAE advantage
     → epochs 轮 mini-batch 梯度更新
     → 熵系数衰减

重复 max_iterations 次
```

**默认配置下的数据量**：

| 指标 | 值 |
|------|------|
| Transitions / 迭代 | 4096 × 24 = 98,304 |
| Episode 最大步数 | 600 (= 60s × 10Hz) |
| 总迭代数 | 5000 |
| 预估总 episodes | ~819,200 |

### 训练监控

训练过程中进度条显示：
- `avg_r`: 最近 100 个 episode 的平均奖励
- `eps`: 已完成的 episode 总数
- `succ`: 成功率
- `steps_s`: 每秒环境步数

### 评估模型

```bash
# Aurora CLI
aurora eval --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml --episodes 100

# Python 脚本
python eval.py --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml --episodes 100

# 确定性策略评估
python eval.py --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml --deterministic
```

### 导出 ONNX 模型

```bash
# Aurora CLI
aurora export --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml --output models/nav_data.onnx

# Python 脚本
python -m utils.export_onnx --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml --output models/nav_data.onnx
```

### 输出文件

每次训练自动创建独立的实验目录，输出文件保存在 `runs/train/expNNN/`：

```
runs/train/
├── exp001/
│   ├── nav_data_weights.pt              # 最终模型
│   ├── nav_data_weights_iter_100.pt     # checkpoint (向量化模式)
│   ├── nav_data_weights_ep_500.pt       # checkpoint (串行模式)
│   └── nav_data_training_metrics.npy    # 训练指标
├── exp002/
│   └── ...
└── exp003/
    └── ...
```

| 文件 | 说明 |
|------|------|
| `nav_data_weights.pt` | 最终模型权重 |
| `nav_data_weights_iter_N.pt` | 定期保存的检查点 (向量化模式) |
| `nav_data_weights_ep_N.pt` | 定期保存的检查点 (串行模式) |
| `nav_data_training_metrics.npy` | 训练指标 (奖励、长度、成功率) |

实验目录自动递增 (exp001, exp002, ...)，也可通过 `--exp-dir` 手动指定。

### GPU 显存与 num_envs 建议

| GPU 显存 | 建议 num_envs |
|----------|---------------|
| 8 GB | 512-1024 |
| 16 GB | 2048-4096 |
| 24 GB+ | 4096-8192 |

遇到 OOM 时减小 `num_envs` 或关闭 `use_amp`。

## 状态空间 (43 维)

| 索引 | 组件 | 维度 | 说明 |
|------|------|------|------|
| 0-2 | 基座线速度 | 3 | [vx, vy, vz] ×2.0 |
| 3-5 | 基座角速度 | 3 | [wx, wy, wz] ×1.0 |
| 6-7 | 归一化位置 | 2 | [x/W, y/H] |
| 8-9 | 朝向 | 2 | [sinθ, cosθ] |
| 10-11 | 目标方向 | 2 | [sinΔθ, cosΔθ] |
| 12-14 | 目标距离 | 3 | [Δx, Δy, ‖Δ‖] ÷max_range |
| 15-22 | 数据价值扇区 | 8 | 8 方向价值评分 |
| 23-26 | 障碍物扇区 | 4 | [前, 后, 左, 右] |
| 27-28 | 当前位置数据 | 2 | [value, rarity] |
| 29-30 | 收集状态 | 2 | [collected_ratio, coverage_ratio] |
| 31 | 地形类型 | 1 | 归一化 ÷6 |
| 32 | 局部障碍密度 | 1 | [0, 1] |
| 33 | 步态相位 | 1 | sin 编码 |
| 34-41 | 动作历史 | 8 | 最近 8 步速度命令 |
| 42 | 剩余时间预算 | 1 | [0, 1] |

## 奖励函数 (10 组件)

| 组件 | 权重 | 说明 |
|------|------|------|
| 目标接近 | 3.0 | 向目标移动进度 |
| 数据价值收集 | 5.0 | value × rarity (核心) |
| 高价值引导 | 2.0 | 朝高价值区域移动 |
| 覆盖率 | 1.0 | 访问新网格奖励 |
| 目标完成 | 100.0 | 到达目标稀疏奖励 |
| 路径效率 | 0.5 | 惩罚低效路线 |
| 重复访问 | 1.0 | 惩罚重访已收集区域 |
| 碰撞/危险 | 10.0 | 碰撞和接近惩罚 |
| 速度跟踪 | 0.5 | 惩罚停滞 |
| 时间 | 0.1 | 固定步数惩罚 |

## 目录结构

```
rl_planning_train/
├── rl_planning_train/      # CLI 包
├── core/                   # 核心模块 (PPO, 环境基类, 配置, 工厂)
├── humanoid/               # Nav_Data 场景 (状态, 奖励, 环境, 动作)
├── value_model/            # 4 维数据价值模型
├── utils/                  # 工具 (导出, 配置, 可达性, 训练工具)
├── deploy/                 # 联合仿真部署脚本
├── config/                 # YAML 配置文件
├── models/                 # 训练权重输出
├── tests/                  # 单元测试
├── docs/                   # 文档
├── train.py                # 训练入口
└── eval.py                 # 评估入口
```

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/getting-started/quick-start.md) | 5 分钟上手训练 |
| [安装说明](docs/getting-started/installation.md) | 依赖安装和环境配置 |
| [系统架构](docs/guides/architecture.md) | 分层控制架构详解 |
| [场景详解](docs/guides/scenarios.md) | 状态/动作/奖励空间详解 |
| [配置说明](docs/guides/configuration.md) | YAML 配置参数 |
| [性能优化](docs/guides/performance.md) | 训练加速指南 |
| [命令参考](docs/reference/cli-reference.md) | CLI 完整参数 |
| [故障排除](docs/troubleshooting.md) | 常见问题解决方案 |

## 版本历史

详细变更记录请查看 [CHANGELOG](CHANGELOG.md)

| 版本 | 日期 | 变更类型 | 主要变更 |
|------|------|----------|----------|
| v0.7.0 | 2026-04-28 | MINOR | **向量化训练关键 bug 修复**: GAE 数据顺序、环境重置、奖励平衡修正；移除自动驾驶场景，专注 Nav_Data |
| v0.6.0 | 2026-04-16 | MINOR | Aurora + LivelyBot 联合仿真：分层控制架构、Matplotlib 实时可视化、Nav_Data 模式、ONNX 导出 |
| v0.5.0 | 2026-04-09 | MINOR | CLI命令行工具：统一 aurora 命令、支持 pip 安装、训练/评估/导出子命令 |
| v0.4.0 | 2026-04-08 | MINOR | 重大性能优化：环境热路径向量化（10-50x加速）、GAE O(n²)→O(n)、批量经验收集、混合精度训练支持、PPO epochs 优化 |
| v0.3.0 | 2026-03-10 | MINOR | 统一配置文件格式、自动驾驶状态维度扩展至25维、人形机器人状态维度扩展至75维（均含可达性特征） |
| v0.2.0 | 2026-03-06 | MINOR | ONNX模型导出功能 (Opset 14/17)、log_std参数保存/加载、部署端推理示例 |
| v0.1.0 | 2025-12-24 | MINOR | 双场景强化学习训练平台初始版本 |
| v0.0.1 | 2025-12-20 | PATCH | 项目初始结构、基础PPO实现 |

## 许可证

MIT
