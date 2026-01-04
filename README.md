# 强化学习路径规划训练模块使用说明

## 概述

本目录包含了用于训练强化学习路径规划模型的所有代码和工具。通过这些脚本，您可以训练、评估和导出模型，用于自动驾驶车辆的路径规划任务。

## 环境准备

### 依赖安装

确保您的环境中安装了以下依赖：

```bash
pip install torch numpy matplotlib pyyaml pillow
```

### 环境变量设置

训练脚本会自动创建所需的目录，无需额外设置环境变量。

## 环境定义

### 状态空间 (State Space)

状态向量是一个24维的特征向量，结构如下：
- **位置坐标 (indices 0-1)**: 代理的归一化坐标 [norm_lat, norm_lon]
- **可行驶路网 (indices 2-17, 16个值)**: 局部可行驶区域信息，表示周围网格的可行驶性
- **最近动作 (indices 18-21, 4个值)**: 最近4个动作的独热编码
- **剩余预算归一化值 (index 22)**: 剩余步数的归一化值 [0,1]
- **局部交通密度 (index 23)**: 代理周围区域的障碍物密度

### 动作空间 (Action Space)

动作空间包含4个道路相关的动作：
- **0**: 继续前行 (Go Forward) - 沿当前方向移动
- **1**: 左转 (Turn Left) - 逆时针改变方向
- **2**: 右转 (Turn Right) - 顺时针改变方向
- **3**: 掉头 (U-turn) - 180度转向

### 奖励函数 (Reward Function)

奖励函数由多个组件构成，以鼓励智能体学习有效的路径规划策略：

1. **距离奖励 (Distance Reward)**: 基于代理向目标靠近的距离变化
   - `distance_reward = (old_distance - new_distance) * 2.0`

2. **时间惩罚 (Time Penalty)**: 每步施加 `-0.1` 的小惩罚，鼓励高效路径

3. **目标奖励 (Goal Reward)**: 到达目标时给予 `10.0` 的大奖励

4. **碰撞惩罚 (Collision Penalty)**: 当代理进入障碍物区域时施加 `-5.0` 惩罚

5. **数据稀缺度奖励 (Data Scarcity Reward)**: 访问数据稀缺度高的区域时获得奖励
   - `data_scarcity_reward = current_cell_data_scarcity * data_sparse_factor`

6. **覆盖率奖励 (Coverage Reward)**: 基于环境探索程度的奖励
   - `coverage_reward = visited_cells_ratio * coverage_factor`

7. **路径效率惩罚 (Path Efficiency Penalty)**: 对低效路径施加惩罚
   - `path_efficiency_penalty = -0.01 * path_length / direct_distance`

8. **重复路径惩罚 (Repeated Path Penalty)**: 重复访问已访问区域时施加 `-0.05` 惩罚

## 目录结构

```
├── benchmark                       # 性能基准测试
│   └── benchmark_planner.py
├── config                          # 训练配置文件
│   ├── advanced_ppo_config.yaml
│   └── ppo_config.yaml
├── curriculum_train.py             # 课程学习训练脚本
├── final_evaluation.py             # 最终评估脚本
├── models                          # 训练好的模型权重
│   ├── planner_model.onnx
│   ├── ppo_weights.pt
│   └── training_metrics.npy
├── planner_rl_eval.py              # 模型评估脚本
├── planner_rl_train.py             # PPO训练主脚本
├── runs                            # 运行结果
├── settings.py                     # 训练设置
└── utils                           # 工具脚本
    ├── animate_demo.py             # 生成路径动画
    ├── hyperparameter_tuning.py    # 超参数调优
    ├── show_learning_results.py    # 绘制学习结果
    ├── export_onnx.py              # 导出ONNX模型
    ├── environment.py              # 训练环境实现
    └── visualize_demo.py           # 可视化
```

## 训练模型

### 基本训练
在云端训练时，模型支持同时输出原始logits和softmax概率：
1. **Logits模式** (`use_softmax=False`)：
   - 用于训练过程
   - 输出原始logits值
   - 用于Categorical分布采样

2. **Softmax模式** (`use_softmax=True`)：
   - 用于指标计算
   - 输出概率分布
   - 用于监控模型性能

这种设计允许在训练过程中保持数值稳定性，同时在云环境中计算各种性能指标。

```python
# 加载模型
model = ActorCritic(state_dim=24, action_dim=4)
model.load_state_dict(torch.load('model.pt'))

# 训练时使用logits
logits, value = model(state, use_softmax=False)

# 计算指标时使用softmax
probs, value = model(state, use_softmax=True)
```

**关键改进**:
- .pt模型文件只保存模型参数，不包含softmax或logits的特定输出
- 训练时使用logits (`use_softmax=False`) 以保持数值稳定性和性能
- 计算指标时使用softmax (`use_softmax=True`)，但不改变模型保存格式
- 保持向后兼容性，现有代码无需修改
```bash
python planner_rl_train.py --config config/ppo_config.yaml --episodes 5000 --device cuda
```

常用参数：
- `--config`: 指定配置文件路径
- `--episodes`: 训练回合数
- `--device`: 使用的计算设备 (`cpu` 或 `cuda`)
- `--weights-path`: 指定初始权重文件路径
- `--env-type`: 环境类型 (`simple` 或 `complex`)
- `--save-interval`: 模型保存间隔
- `--plot-results`: 是否绘制结果

### 课程学习训练
使用课程学习方法逐步增加训练难度：

```bash
python curriculum_train.py --config config/advanced_ppo_config.yaml --stages 5
```

这种方法从简单的2x2网格开始，逐步过渡到20x20的复杂环境。

### PPO配置文件说明
配置文件包含训练所需的所有超参数：

- `state_dim`: 状态维度 (24)
- `action_dim`: 动作空间维度 (4)
- `hidden_dim`: 隐藏层维度
- `learning_rate`: 学习率
- `gamma`: 折扣因子
- `episodes`: 训练回合数
- 更多参数请参考配置文件中的注释

## 评估模型

### 评估
评估训练好的模型性能：
```bash
python planner_eval.py --model-path models/ppo_weights.pt --config config/advanced_ppo_config.yaml --episodes 100
```

### 特定任务评估
在预定义的任务上评估模型：

```bash
python final_evaluation.py --model-path models/ppo_weights.pt --config config/advanced_ppo_config.yaml
```

## 可视化

### 生成路径动画
创建GIF动画可视化智能体行为：

```bash
python utils/animate_demo.py --model-path models/ppo_weights.pt --config config/advanced_ppo_config.yaml --save-path agent_demo.gif
```

### 显示学习结果
可视化学习到的路径：

```bash
python utils/show_learning_results.py
```

## 超参数调优
搜索最优超参数组合：

```bash
python utils/hyperparameter_tuning.py --output tuning_results.json
```

## 模型导出
将训练好的PyTorch模型导出为ONNX格式以便在车端部署：

```bash
python utils/export_onnx.py --model-path models/ppo_weights.pt --config config/ppo_config.yaml --output models/planner_model.onnx
```

## 性能基准测试
比较不同模型的性能：

```bash
python benchmark/benchmark_planner.py --model-path models/ppo_weights.pt --config config/ppo_config.yaml --episodes 100
```

## 注意事项
1. 训练时间取决于训练回合数和环境复杂度，可能需要几分钟到几小时
2. 模型保存在 `models/` 目录下
3. 评估结果保存在 `runs/` 目录下
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
1. 减少批次大小 (batch_size)
2. 减少网络层数或隐藏单元数
3. 使用更简单的环境进行训练

### 模型收敛慢
1. 调整学习率
2. 增加训练回合数
3. 使用更复杂的网络架构