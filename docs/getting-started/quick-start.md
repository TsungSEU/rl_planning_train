# 快速开始

本指南将在 5 分钟内让你开始训练第一个模型。

## 安装 Aurora CLI

```bash
# 进入项目目录
cd rl_planning_train

# 开发模式安装（包含 aurora 命令）
pip install -e .

# 验证安装
aurora --version
```

## 训练你的第一个模型

### 使用 Aurora CLI (推荐)

```bash
# 训练 Aurora 高层导航策略
aurora train --config config/nav_data_training.yaml

# 指定 GPU
aurora train --config config/nav_data_training.yaml --device cuda

# 从检查点继续训练
aurora train --config config/nav_data_training.yaml \
# 从检查点继续训练
aurora train --config config/nav_data_training.yaml \
  --model-path runs/train/exp001/nav_data_weights_iter_100.pt
```

### 使用 Python 脚本

```bash
# 向量化训练 (默认, 4096 并行环境)
python train.py --config config/nav_data_training.yaml

# 指定迭代数
python train.py --config config/nav_data_training.yaml --max-iterations 1000

# 切换到串行模式 (调试用)
python train.py --config config/nav_data_training.yaml --num-envs 1 --episodes 5000

# 训练并评估
python train.py --config config/nav_data_training.yaml --evaluate --eval-episodes 100
```

### Nav_Data 场景特点

- **43 维状态空间**: 位置、速度、目标、数据价值、障碍物...
- **3 维速度命令**: [vx, vy, ωz] 输出到 LivelyBot
- **分层控制**: Aurora (10Hz) → LivelyBot (50Hz) → MuJoCo (1000Hz)
- **数据价值驱动**: 智能导航采集高价值数据
- **向量化训练**: 4096 并行环境，GPU 加速

## 训练参数说明

### Aurora CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 (必需) | - |
| `--episodes` | 训练回合数 (串行模式) | 配置文件指定 |
| `--max-iterations` | 训练迭代数 (向量化模式) | 配置文件指定 |
| `--device` | 训练设备 (cpu/cuda/auto) | auto |
| `--num-envs` | 并行环境数。1=串行, >1=向量化 | 配置文件指定 |
| `--model-path` | 预训练模型路径 | - |
| `--exp-dir` | 实验输出目录 | 自动 runs/train/expNNN |

### Python 脚本参数

**通用：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 (必需) | - |
| `--device` | 训练设备 (cpu/cuda/auto) | cuda |
| `--num-envs` | 并行环境数。1=串行, >1=向量化 | 配置文件指定 |
| `--save-interval` | 保存间隔 | 配置文件指定 |
| `--evaluate` | 训练后运行评估 | false |
| `--eval-episodes` | 评估回合数 | 1000 |
| `--model-path` | 预训练模型路径 | - |
| `--exp-dir` | 实验输出目录 | 自动 runs/train/expNNN | (`--num-envs 1`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--episodes` | 训练回合数 | 配置文件指定 |
| `--update-interval` | 累积 N 个 episode 后更新 | 4 |

**向量化模式** (`--num-envs >1`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-iterations` | 训练迭代数 | 配置文件指定 (5000) |

## 训练模式

根据 `num_envs` 自动选择：

| 模式 | num_envs | 特点 |
|------|----------|------|
| **串行** | 1 | 单环境逐回合训练，适合调试 |
| **向量化** | >1 | N 个并行环境，GPU 批量推理 |

### 向量化训练流程 (默认)

每次迭代：对 N 个环境各跑 `steps_per_env` (24) 步 → 收集 N×24 个 transitions → PPO 更新。超过 `max_steps` (600) 的 episode 强制截断。重复 `max_iterations` (5000) 次。

## 输出文件

每次训练自动创建独立实验目录 `runs/train/expNNN/`：

```
runs/train/
├── exp001/
│   ├── nav_data_weights.pt
│   ├── nav_data_weights_iter_100.pt
│   └── nav_data_training_metrics.npy
├── exp002/
│   └── ...
└── exp003/
```

实验目录自动递增。可通过 `--exp-dir` 手动指定输出目录。

## 评估模型

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

## 导出 ONNX 模型

ONNX 模型用于部署到边缘端 (aurora-edge-runtime)：

```bash
# Aurora CLI
aurora export \
  --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml \
  --output models/nav_data.onnx

# Python 脚本
python -m utils.export_onnx \
  --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml \
  --output models/nav_data.onnx
```

## 联合仿真验证

### Aurora + LivelyBot MuJoCo 联合仿真

```bash
python deploy/aurora_livelybot_sim.py \
    --aurora_model runs/train/exp001/nav_data_weights.pt \
    --livelybot_model /path/to/livelybot/policy_torch.pt \
    --duration 30
```

### 可视化

- **MuJoCo 3D 窗口**: 实时机器人姿态
- **Matplotlib 曲线**: 速度跟踪、位置轨迹、目标距离

## 完整训练-部署流程

```
Step 1: 训练 LivelyBot (独立)
  cd livelybot_pi_rl_baseline
  python humanoid/scripts/train.py --task=pai_ppo --num_envs 4096
  python humanoid/scripts/play.py --task=pai_ppo --run_name v1

Step 2: 训练 Aurora (本仓库)
  python train.py --config config/nav_data_training.yaml
  # 输出到 runs/train/exp001/

Step 3: 导出 ONNX
  python -m utils.export_onnx \
    --model-path runs/train/exp001/nav_data_weights.pt \
    --config config/nav_data_training.yaml --output models/nav_data.onnx

Step 4: 联合仿真验证
  python deploy/aurora_livelybot_sim.py \
    --aurora_model runs/train/exp001/nav_data_weights.pt ...

Step 5: 部署到边缘端
  cd aurora-edge-runtime
  source setup.bash && colcon build
  ros2 run aurora_edge_runtime dcp
```

## 从检查点继续训练

```bash
# Aurora CLI
aurora train --config config/nav_data_training.yaml \
  --model-path runs/train/exp001/nav_data_weights_iter_100.pt

# Python 脚本
python train.py --config config/nav_data_training.yaml \
  --model-path runs/train/exp001/nav_data_weights_iter_100.pt
```

## 性能优化建议

### GPU 显存与 num_envs

| GPU 显存 | 建议 num_envs |
|----------|---------------|
| 8 GB | 512-1024 |
| 16 GB | 2048-4096 |
| 24 GB+ | 4096-8192 |

### 混合精度训练

在配置文件中启用：

```yaml
training:
  use_amp: true  # ~2x 速度, ~50% 显存
```

## 验证安装

```bash
# 运行测试
pytest tests/ -v

# 检查 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 下一步

- 阅读 [系统架构](../guides/architecture.md) 了解云-边-端三层架构
- 阅读 [场景详解](../guides/scenarios.md) 了解状态/动作空间
- 阅读 [配置说明](../guides/configuration.md) 调整训练参数
- 阅读 [命令参考](../reference/cli-reference.md) 了解完整命令列表
