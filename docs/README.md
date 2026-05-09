# 文档

Aurora — 面向人形机器人的自主数据采集系统，采用云-边-端三层架构（planning-engine → edge-runtime → livelybot）。

## 快速导航

| 文档 | 说明 |
|------|------|
| [快速开始](getting-started/quick-start.md) | 5 分钟上手训练 |
| [安装说明](getting-started/installation.md) | 依赖安装和环境配置 |
| [系统架构](guides/architecture.md) | 云-边-端三层架构 + 分层控制详解 |
| [场景详解](guides/scenarios.md) | 导航+数据采集的状态/动作空间 |
| [配置说明](guides/configuration.md) | YAML 配置文件结构和参数 |
| [性能优化](guides/performance.md) | 训练性能调优指南 |
| [命令参考](reference/cli-reference.md) | 训练、评估、导出命令完整参数 |
| [故障排除](troubleshooting.md) | 常见问题和解决方案 |

## 核心概念

### 云-边-端三层架构

| 组件 | 定位 | 核心职责 |
|------|------|----------|
| rl_planning_train | 云端训练平台 | 训练高层导航策略，导出 ONNX 模型 |
| aurora-edge-runtime | 边缘推理引擎 | ONNX 实时推理，数据采集与上传 |
| livelybot_pi_rl_baseline | 底层运动控制 | 双足行走，速度指令 → 关节动作 |

完整系统架构请参阅 [Aurora系统架构与组件协作详解](../Aurora系统架构与组件协作详解.md)。

### 分层强化学习

1. **Aurora 高层** (10 Hz): 导航决策，输出速度命令
2. **LivelyBot 低层** (50 Hz): 双足行走，输出关节目标
3. **物理仿真** (1000 Hz): MuJoCo 刚体动力学

### 闭环迭代

训练 (云端) → 部署 (边缘端) → 采集 (机器人) → 回传经验数据 → 再训练

## 训练流程

```bash
# 1. 训练 Aurora (输出到 runs/train/expNNN/)
python train.py --config config/nav_data_training.yaml

# 2. 评估模型
python eval.py --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml

# 3. 联合仿真验证
python deploy/aurora_livelybot_sim.py \
    --aurora_model runs/train/exp001/nav_data_weights.pt \
    --livelybot_model /path/to/livelybot/policy_torch.pt \
    --duration 30

# 4. 导出 ONNX 模型 (部署到边缘端)
python -m utils.export_onnx --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml \
  --output models/nav_data.onnx

# 5. 部署到边缘端
cd ../aurora-edge-runtime && source setup.bash
ros2 run aurora_edge_runtime dcp
```
