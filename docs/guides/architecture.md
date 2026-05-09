# 系统架构详解

Aurora 采用云-边-端三层架构，通过分层强化学习（Hierarchical RL）实现人形机器人智能导航与数据采集。

## 三层架构

| 层级 | 组件 | 解决的问题 | 接口 |
|------|------|-----------|------|
| 高层认知 | rl_planning_train → aurora-edge-runtime | **去哪里**采集数据最有价值 | 3 维速度指令 |
| 低层运动 | livelybot_pi_rl_baseline | **怎么走**过去（稳定双足行走） | 12 维关节力矩 |

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        云端 (Cloud)                                  │
│                                                                     │
│   rl_planning_train                                            │
│   · PPO 强化学习训练                                                 │
│   · 43维状态 → 3维速度指令                                           │
│   · 导出 ONNX 模型                                                  │
│   · 接收边缘端回传的经验数据，持续改进                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ ONNX 模型部署
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        边缘端 (Edge)                                 │
│                                                                     │
│   aurora-edge-runtime                                               │
│   · ONNX Runtime 10Hz 实时推理                                      │
│   · 安全约束层 (速度限制 + 碰撞检测)                                  │
│   · 数据采集 (Rosbag2, 环形缓冲)                                     │
│   · S3 数据上传 → 回传云端                                          │
│   · 状态机 (8状态事件驱动架构)                                       │
│              │ cmd_vel (10Hz)        │ 经验数据                      │
│              ↓                       ↓                              │
│   ┌──────────────────┐    数据回传至云端                             │
│   │  LivelyBot (ROS2) │                                              │
│   └──────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        机器人端 (Robot)                               │
│                                                                     │
│   livelybot_pi_rl_baseline                                          │
│   · 50Hz 低层运动控制                                                │
│   · 输入: 3维速度指令 + 705维本体感知                                 │
│   · 输出: 12维关节目标 → PD控制器 → 关节力矩                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 闭环反馈

```
┌──────────────┐  ONNX模型  ┌──────────────┐
│   训练平台    │ ────────→ │   边缘运行时   │
│   (Cloud)    │           │   (Edge)     │
│  PPO 训练    │ ←──────── │  数据采集     │
│  经验回放    │  经验数据   │  S3上传      │
└──────────────┘           └──────┬───────┘
                                  │ cmd_vel
                                  ↓
                           ┌──────────────┐
                           │  LivelyBot    │
                           │  (Robot)     │
                           │  状态反馈     │
                           └──────────────┘

循环: 训练 → 部署 → 采集 → 回传 → 再训练 → 迭代改进
```

## Aurora 高层规划 (10 Hz)

### 职责

1. **导航决策**: 决定机器人应该向哪个方向移动
2. **数据价值评估**: 评估不同区域的数据采集价值
3. **路径规划**: 避开障碍物，规划最优路径
4. **目标追踪**: 向任务目标点移动

### 输入状态 (43 维)

| 组件 | 维度 | 来源 |
|------|------|------|
| 基座速度 (线/角) | 6 | LivelyBot 里程计 + IMU |
| 位置与朝向 | 4 | LivelyBot 里程计 |
| 目标信息 | 5 | 任务配置 |
| 数据价值扇区 | 8 | CostMap (Value Model) |
| 障碍物扇区 | 4 | 射线检测 |
| 采集状态 | 2 | 内部统计 |
| 步态相位 | 1 | LivelyBot 关节状态 |
| 动作历史 | 8 | 内部缓存 |
| 其他 | 5 | 环境/配置 |

### 输出动作 (3 维)

```python
action = [forward_vel, lateral_vel, angular_vel]
# forward_vel: [-0.3, 0.6] m/s
# lateral_vel: [-0.3, 0.3] m/s
# angular_vel: [-0.3, 0.3] rad/s
```

### 训练参数

```yaml
training:
  policy_freq: 10           # 决策频率 10 Hz
  command_hold_steps: 5     # 每个命令保持 5 个 LivelyBot 步长
  episode_length_s: 60      # 每回合 60 秒
```

## LivelyBot 低层控制 (50 Hz)

### 职责

1. **双足行走**: 将速度命令转换为稳定的双足步态
2. **平衡控制**: 维持机器人平衡，防止跌倒
3. **步态生成**: 生成摆动相和站立相的足端轨迹
4. **逆运动学**: 计算关节目标角度

### 输入状态 (705 维)

| 组件 | 维度 | 说明 |
|------|------|------|
| 步态相位 | 2 | [sin(φ), cos(φ)] |
| 速度指令 | 3 | 来自 Aurora |
| 关节位置 | 12 | 12 个关节的当前角度 |
| 关节速度 | 12 | 12 个关节的角速度 |
| 历史动作 | 12 | 上一步的关节目标 |
| IMU 数据 | 6 | 角速度 + 欧拉角 |
| 历史堆叠 | 47 × 14 | 15 帧历史 (含当前帧) |

47 维单帧 × 15 帧历史堆叠 = **705 维**。

### 输出动作 (12 维)

```python
action = [
    left_hip_pitch, left_hip_roll, left_thigh, left_calf,
    left_ankle_pitch, left_ankle_roll,
    right_hip_pitch, right_hip_roll, right_thigh, right_calf,
    right_ankle_pitch, right_ankle_roll
]
# 每个关节的偏移量 (action_scale=0.25)
```

PD 控制器：`τ = Kp × (target_q - q) + Kd × (0 - dq)`

## 频率协调

### 时间轴

```
Aurora 决策 (10 Hz):
  t=0.0s    t=0.1s    t=0.2s    t=0.3s
    ↓         ↓         ↓         ↓
  [cmd1]    [cmd2]    [cmd3]    [cmd4]
    │         │         │         │
    └─────────┴─────────┴─────────┘
              │ 持续 100ms
              ↓
LivelyBot 执行 (50 Hz):
  每 20ms 读取一次 cmd_vel，更新内部状态
  执行步态控制，输出关节目标

物理仿真 (1000 Hz):
  每 1ms 执行一次物理步进
  计算 PD 控制器输出，更新刚体状态
```

### 命令保持机制

Aurora 每 100ms 输出一个新命令，LivelyBot 在 100ms 内执行 5 个控制周期 (50 Hz × 0.1s = 5)，每个控制周期使用相同的 Aurora 命令。

## 模块通信

### Aurora → LivelyBot

```python
# Aurora 输出 → ROS2 Topic: /robot/cmd_vel
cmd_vel = geometry_msgs.Twist()
cmd_vel.linear.x = forward_vel
cmd_vel.linear.y = lateral_vel
cmd_vel.angular.z = angular_vel
```

### LivelyBot → Aurora

```python
# LivelyBot 输出 → ROS2 Topics
/robot/odom          # nav_msgs/Odometry (50Hz) — 位姿、线速度、角速度
/robot/joint_states  # sensor_msgs/JointState (50Hz) — 12个关节
/robot/imu           # sensor_msgs/Imu (50Hz) — IMU数据
```

## Edge-Runtime 内部处理流程

```
1. 状态采集与构造
   ├── 订阅 /robot/odom        → 位置、线速度、角速度 [0-5]
   ├── 订阅 /robot/imu         → 姿态信息
   ├── 订阅 /robot/joint_states → 步态相位 [33]
   ├── CostMap 更新            → 数据价值扇区 [15-22]
   ├── 射线检测                 → 障碍物扇区 [23-26]
   ├── 任务目标                 → 目标方向/距离 [10-14]
   ├── 采集状态                 → 覆盖率/采集比 [29-30]
   └── 动作历史                 → 前8步动作 [34-41]

2. 状态向量组装 (43维)
   HumanoidStateInfo → std::vector<float>(43)

3. ONNX 模型推理 (10Hz)
   输入: state[43]  →  输出: action_mean[3], value[1]

4. 安全约束层
   ├── 速度限幅 (匹配动作空间范围)
   ├── 碰撞预检测
   └── 紧急停止条件判断

5. 指令发布
   安全动作 → geometry_msgs::Twist → /robot/cmd_vel

6. 数据采集 (并行)
   ├── 触发器检测 (步态触发 / 规则触发)
   ├── 环形缓冲区 (15s预缓冲 + 5s后缓冲)
   ├── Rosbag2 录制 (.db3 格式)
   ├── LZ4 压缩 (.tar.lz4)
   └── S3 分片上传 (caic-dataset/robot_dev/)
```

## 训练策略

### 独立训练

1. **LivelyBot 训练** (独立，不依赖 Aurora)
   ```bash
   cd livelybot_pi_rl_baseline
   python humanoid/scripts/train.py --task=pai_ppo --num_envs 4096
   ```
   - 随机下发速度命令，学习稳定双足行走
   - 冻结后用于 Aurora 训练

2. **Aurora 训练** (依赖冻结的 LivelyBot)
   ```bash
   cd rl_planning_train
   python train.py --config config/nav_data_training.yaml
   ```
   - LivelyBot 作为底层控制器
   - Aurora 学习导航决策
   - 联合仿真验证

### 联合仿真

```bash
python deploy/aurora_livelybot_sim.py \
    --aurora_model runs/train/exp001/nav_data_weights.pt \
    --livelybot_model /path/to/livelybot/policy_torch.pt \
    --duration 30
```

## 模型部署路径

```
Planning-Engine                    Edge-Runtime
─────────────                      ─────────────

训练产出:                           运行时加载:
nav_data_weights.pt
      │
      ↓ export_onnx.py
nav_data.onnx  ──── 文件系统共享 ──→  ONNX Runtime 加载推理
```

环境变量配置：

```bash
# setup.bash
export AER_MODEL_PATH=/home/xucong/caicAD/01datainfra/Aurora/rl_planning_train/models/nav_data.onnx
```

## 优势分析

### 1. 任务分解

- **Aurora**: 专注高层认知 (去哪里采集数据最有价值)
- **LivelyBot**: 专注低层运动 (怎么走过去保持稳定)

### 2. 独立训练

- 两层策略可并行开发，互不阻塞
- 更换机器人只需重训 LivelyBot，Aurora 策略可复用

### 3. 简洁接口

- 仅通过 3 维速度指令解耦
- 降低系统复杂度，便于调试和维护

### 4. 闭环迭代

- 边缘端采集的数据回传云端，持续改进模型
- 经验数据可用于 fine-tune 导航策略

## 相关文档

- [场景详解](scenarios.md) — 状态/动作/奖励空间详细说明
- [配置说明](configuration.md) — 训练配置参数
- [完整系统架构文档](../../docs/Aurora系统架构与组件协作详解.md) — 各组件文件索引与部署细节
