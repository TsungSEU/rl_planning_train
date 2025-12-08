# 强化学习路径规划训练模块使用说明

## 概述

本目录包含了用于训练强化学习路径规划模型的所有代码和工具。通过这些脚本，您可以训练、评估和导出模型，用于自动驾驶车辆的路径规划任务。

## 目录结构

```
training/
├── config/                    # 训练配置文件
│   ├── advanced_ppo_config.yaml
│   ├── example_ppo_config.yaml
│   └── optimized_ppo_config.yaml
├── models/                    # 训练好的模型权重（训练后生成）
├── results/                   # 评估结果（评估后生成）
├── utils/                     # 工具脚本
│   ├── animate_demo.py
│   ├── hyperparameter_tuning.py
│   └── visualize_demo.py
├── benchmark/                 # 性能基准测试
│   └── benchmark_planner.py
├── environment.py             # 训练环境实现
├── planner_rl_train.py        # PPO训练主脚本
├── curriculum_train.py        # 课程学习训练脚本
├── planner_rl_eval.py         # 模型评估脚本
├── enhanced_eval.py           # 增强评估脚本
├── final_evaluation.py        # 最终评估脚本
├── show_learning_results.py   # 结果可视化脚本
├── hyperparameter_tuning.py   # 超参数调优脚本
├── export_onnx.py             # ONNX模型导出脚本
└── settings.py                # 训练设置
```

## 环境准备

### 依赖安装

确保您的环境中安装了以下依赖：

```bash
pip install torch numpy matplotlib pyyaml pillow
```

### 环境变量设置

训练脚本会自动创建所需的目录，无需额外设置环境变量。

## 训练模型

### 基本训练

使用PPO算法训练路径规划模型：

```bash
python planner_rl_train.py --config config/advanced_ppo_config.yaml --episodes 5000
```

常用参数：
- `--config`: 指定配置文件路径
- `--episodes`: 训练回合数
- `--env-type`: 环境类型 (`simple` 或 `complex`)
- `--save-interval`: 模型保存间隔

### 课程学习训练

使用课程学习方法逐步增加训练难度：

```bash
python curriculum_train.py --config config/advanced_ppo_config.yaml --stages 5
```

这种方法从简单的2x2网格开始，逐步过渡到20x20的复杂环境。

### 配置文件说明

配置文件包含训练所需的所有超参数：

- `state_dim`: 状态维度 (24)
- `action_dim`: 动作空间维度 (4)
- `hidden_dim`: 隐藏层维度
- `learning_rate`: 学习率
- `gamma`: 折扣因子
- `episodes`: 训练回合数
- 更多参数请参考配置文件中的注释

## 评估模型

### 基本评估

评估训练好的模型性能：

```bash
python planner_rl_eval.py --model-path models/ppo_weights.pth --config config/advanced_ppo_config.yaml --episodes 100
```

### 详细评估

进行更全面的性能分析：

```bash
python enhanced_eval.py --model-path models/ppo_weights.pth --config config/advanced_ppo_config.yaml --episodes 100
```

### 特定任务评估

在预定义的任务上评估模型：

```bash
python final_evaluation.py --model-path models/ppo_weights.pth --config config/advanced_ppo_config.yaml
```

## 可视化

### 生成路径动画

创建GIF动画可视化智能体行为：

```bash
python utils/animate_demo.py --model-path models/ppo_weights.pth --config config/advanced_ppo_config.yaml --save-path agent_demo.gif
```

### 显示学习结果

可视化学习到的路径：

```bash
python show_learning_results.py
```

## 超参数调优

搜索最优超参数组合：

```bash
python hyperparameter_tuning.py --output tuning_results.json
```

## 模型导出

将训练好的PyTorch模型导出为ONNX格式以便在车端部署：

```bash
python export_onnx.py --model-path models/ppo_weights.pth --config config/advanced_ppo_config.yaml --output models/planner_model.onnx
```

## 性能基准测试

比较不同模型的性能：

```bash
python benchmark/benchmark_planner.py --model-path models/ppo_weights.pth --config config/advanced_ppo_config.yaml --episodes 100
```

## 注意事项

1. 训练时间取决于训练回合数和环境复杂度，可能需要几分钟到几小时
2. 模型保存在 `models/` 目录下
3. 评估结果保存在 `results/` 目录下
4. 确保在训练和推理时使用相同的状态表示
5. 车端推理必须使用ONNX Runtime加载模型
6. 模型输出logits而非概率，车端需要执行softmax操作

## 故障排除

### 训练效果不佳

1. 增加训练回合数
2. 调整超参数（学习率、折扣因子等）
3. 使用课程学习方法
4. 检查奖励函数设计

### 内存不足

1. 减少批次大小 ([batch_size](file:///workspaces/ad_data_closed_loop/src/codec/gpujpeg/gpujpeg_encoder.h#L72-L72))
2. 减少网络层数或隐藏单元数
3. 使用更简单的环境进行训练

### 模型收敛慢

1. 调整学习率
2. 增加训练回合数
3. 使用更复杂的网络架构

### 导入模块错误

如果遇到 `ModuleNotFoundError` 错误，比如：

```
ModuleNotFoundError: No module named 'environment'
```

这是因为Python无法找到模块。可以通过以下方式解决：

1. 确保在 `training` 目录下运行脚本
2. 如果在子目录中运行脚本，可以添加父目录到Python路径：

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

所有脚本都已经修复了这个问题，可以直接运行。