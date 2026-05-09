# Changelog

All notable changes to this project will be documented in this file.

## 格式说明

- 遵循 https://keepachangelog.com/ 标准
- 遵循 https://semver.org/ 规范
- MAJOR.MINOR.PATCH:
    - MAJOR: 不兼容的API变更
    - MINOR: 向后兼容的新功能
    - PATCH: 向后兼容的bug修复
    - 预发布标识: -alpha, -rc.1, -rc.2 等

---

## [0.8.0] - 2026-05-06

### Added
- **课程学习框架**: 4 阶段渐进训练（easy→medium→hard→full），成功率达标自动推进
- **BFS 目标可达性验证**: `_select_targets()` 使用 BFS 验证路径可达，排除不可能完成的 episode
- **奖励归一化**: PPO update 中使用 running mean/var 归一化 reward，稳定训练
- **Value function clipping**: 防止 critic 大幅更新，提升 GAE 估计稳定性

### Changed
- `steps_per_env: 24→128`（rollout 覆盖 ~21% episode，修复 GAE bootstrap 问题）
- `num_envs: 4096→1024`（配合更长 rollout，保持总样本量 ~131k）
- `gamma: 0.995→0.99`（有效视野从 200 步降至 100 步，重视近期密集奖励）
- 速度平滑 `alpha: 0.6→0.8`（有效速度从 0.36 提升至 0.48 m/s）
- 奖励权重重平衡: `w_goal 100→30`, `w_approach 10→15`, `w_proximity 新增 8.0`, `w_value_guide 2.0→0.5`
- Proximity 奖励改为二次函数 `(1-d/r)²`，靠近目标梯度陡增
- `VectorizedEnvRunner` 新增 `set_difficulty_for_all()` 方法
- `NavDataEnvironment` 新增 `set_difficulty()` 方法支持动态难度调整
- `TrainingConfig` 新增 curriculum 相关字段（curriculum_stages, curriculum_success_threshold, curriculum_window）

---

## [0.7.0] - 2026-04-28

### Added
- **向量化训练环境** `core/vectorized_env.py`: N 并行环境批量推理
- **场景工厂** `core/factory.py`: 统一创建 NavData 环境和 PPO 智能体
- **可达性追踪器** `utils/reachability.py`: 位置可达性评分系统
- **训练工具集** `utils/training_utils.py`: 早停、训练指标记录
- **架构文档** `docs/guides/architecture.md`: 云-边-端三层架构详细说明
- **Nav_Data 专用配置** `config/nav_data_training.yaml`: 独立训练配置文件
- 进度条添加 `tgt=` 指标追踪部分目标收集率

### Changed
- 移除自动驾驶场景，专注 Nav_Data 分层导航和人形机器人场景
- `eval.py` 简化为 Nav_Data 专用评估脚本
- 配置加载重构: `utils/config_loader.py` 精简，奖励配置从 YAML 正确加载
- 训练指标保存: `VectorizedTrainingRunner` 现在保存训练指标

### Fixed
- **关键: GAE 数据顺序错误** — 向量化训练数据从 step-major(交错) 改为 env-major(连续)，修正 PPO 优势估计
- **关键: 每次迭代重置环境** — `reset_all()` 移到循环外，保证 episode 跨迭代连续
- **关键: 奖励平衡** — 重复访问惩罚 `max_repeat_penalty` 从 1.0 降至 0.1，修正奖励信号
- `_compute_gae_array` 硬编码 gamma/lam 改为参数传入
- `evaluate_deterministic` 循环条件从 `episode_reward` 改为 `episode_length`
- CLI `eval.py` 移除不支持的 `render=` 参数
- CLI `train.py` 添加缺失的 `Path` 导入
- 目标距离检测使用当前帧位置（非上一帧）
- `gait_phase` 在状态向量中使用实际计算值（原为固定 0）
- `action_history` 保留完整 8 步历史（原前 4 步被清零）
- 日志重复: 移除 `VectorizedTrainingRunner` 重复的 logging handler

---

## [0.6.0] - 2026-04-16

### Added
- **Aurora + LivelyBot 联合仿真**: 集成式分层控制架构，Aurora 高层导航 (10Hz) + LivelyBot 低层行走 (50Hz)
- **联合仿真可视化**: Matplotlib 实时绘图，显示速度跟踪曲线 (Aurora 命令 vs LivelyBot 实际) 和位置轨迹
- **Nav_Data 分层导航模式**: 43 维状态空间、3 维速度命令 (vx, vy, ωz)、10 组件奖励函数
- **ONNX 导出支持**: Nav_Data 模型导出为 ONNX 格式用于端侧部署
- **联合仿真启动脚本**: `deploy/run_sim.sh` 一键启动 Aurora + LivelyBot 联合仿真

### Changed
- Aurora 输出速度命令范围: forward_vel [-0.3, 0.6], lateral_vel [-0.3, 0.3], angular_vel [-0.3, 0.3]
- LivelyBot 观测空间集成 Aurora 速度命令替代键盘输入
- 仿真环境从 Isaac Gym 改为 MuJoCo (联合仿真模式)

### Configuration Structure
```yaml
# Nav_Data 新配置
humanoid_nav_data_training.yaml:
  state_dim: 43
  action_dim: 3
  actor_hidden_dims: [256, 128, 64]
  activation: ELU
```

### Fixed
- 修复联合仿真中 LivelyBot 模型路径配置
- 修复 nav_data 训练日志格式化错误

---

## [0.5.0] - 2026-04-09

### Added
- 统一命令行接口 (CLI) 模块 `aurora/`，提供 `aurora` 命令支持训练、评估、导出操作
- 支持 pip 安装方式 (`pip install -e .`)
- CLI 子命令: `aurora train`, `aurora eval`, `aurora export`, `aurora train:curriculum`
- 场景参数支持: `--scenario auto` 或 `--scenario humanoid`

---

## [0.4.0] - 2026-04-08

### Added
- 混合精度训练 (AMP) 支持，通过 `use_amp` 配置启用
- 批量经验收集功能，通过 `--update-interval` 参数控制累积 N 个 episode 后更新（默认 4）
- 默认自动检测 CUDA 设备（`--device auto`）

### Changed
- **重大性能优化** - 人形机器人环境热路径向量化：
  - `_update_spatial_rarity()` 从 O(W×H) Python 循环改为 numpy 向量化操作（约 100x 加速）
  - `_get_obstacle_distance()` 从 O(radius²) 双层循环改为 numpy 切片+向量化计算（约 20x 加速）
  - `coverage` 计算从每步全网格扫描改为增量追踪 `_visited_count`
  - `goal_threshold` 预计算避免每步重复 `np.sqrt(width² + height²)`
- **重大性能优化** - 自动驾驶环境热路径向量化：
  - `_calculate_local_density()` 从 O(radius²) 双层循环改为 numpy 切片操作（约 5x 加速）
- **重大性能优化** - GAE 计算从 O(n²) 优化为 O(n)：
  - 将 `advantages.insert(0, gae)` 改为 `append` + `reverse()`（约 2x 加速）
- 人形机器人 PPO `epochs` 从 20 降至 10，提升训练效率 2x
- 训练脚本 `train.py` 默认设备从 `cpu` 改为 `auto`（自动检测 CUDA）

### Performance Improvements
- 预计整体训练速度提升 **10-50x**（取决于场景和硬件）
- 人形机器人场景每步计算开销从 ~10,000 操作降至 ~100 操作
- 自动驾驶场景每步计算开销从 ~50 操作降至 ~10 操作

### Configuration Structure
```yaml
training:
  use_amp: false  # 启用混合精度训练（需 GPU）
  # ...
```

---

## [0.3.0] - 2026-03-10

### Added
- 统一配置文件格式，通过 `planner_mode` 字段选择场景模式
- 配置文件命名规范：使用 `auto_*` 和 `humanoid_*` 前缀区分不同场景
- 自动驾驶可达性特征 (Reachability Score)，状态维度从 24 扩展到 25
- 人形机器人可达性特征 (Reachability Score)，状态维度从 74 扩展到 75
- 配置文件说明文档 (`config/README.md`)

### Changed
- 重构配置加载逻辑，支持统一格式的 YAML 配置文件
- 自动驾驶状态空间更新为 25 维（添加位置可达性评分特征）
- 人形机器人状态空间更新为 75 维（添加位置可达性评分特征）
- 配置文件结构包含 `common`、`auto`/`humanoid` 和 `training` 三个主要部分

### Fixed
- 修复配置加载时 `env_width`/`env_height` 被 `grid_resolution` 错误覆盖的问题
- 修复人形机器人环境在小尺寸网格下的初始化错误

### Removed
- 清理旧版配置文件 (`auto_config.yaml`, `humanoid_v1_config.yaml`, `humanoid_v2_config.yaml`, `humanoid_export_config.yaml`)

### Configuration Structure
```yaml
planner_mode: auto  # or "humanoid"

common:
  sparse_threshold: 0.15
  exploration_bonus: 10.0
  # ...

auto:  # or humanoid:
  state:
    state_dim: 25  # or 75 for humanoid
  # ...

training:
  hidden_dim: 128
  episodes: 15000
  # ...
```

---

## [0.2.0] - 2026-03-06

### Added
- ONNX导出功能支持，用于模型部署
  - 车辆模型导出 (Opset 14)
  - 人形机器人模型导出 (Opset 17)
  - 自动保存/加载log_std参数（连续动作空间）
- ONNX推理示例脚本 (`utils/onnx_inference_example.py`)
- 部署端模型运行器类 (`VehicleONNXRunner`, `HumanoidONNXRunner`)

### Fixed
- 修复ONNX导出时的TracerWarning警告 - 使用wrapper类避免动态控制流
- 修复ContinuousPPOAgent的save_weights/load_weights方法 - 现在正确保存log_std参数
- 修复ONNX推理示例中value的类型处理

### Changed
- 改进导出脚本，支持新旧checkpoint格式兼容
- 人形机器人ONNX模型现在输出action_mean和value

### Technical Details
- **ONNX Export Wrappers**:
  - `ONNXExportWrapperDiscrete`: 离散动作空间导出包装器
  - `ONNXExportWrapperContinuous`: 连续动作空间导出包装器
- **Opset Versions**:
  - Vehicle: Opset 14
  - Humanoid: Opset 17

---

## [0.1.0] - 2025-12-24

### Added
- 双场景强化学习训练平台
  - 自动驾驶车辆场景 (24维状态, 4离散动作)
  - 人形机器人场景 (74维状态, 18连续动作)
- PPO (Proximal Policy Optimization) 算法实现
- 统一训练接口 (`train.py`)
- 统一评估接口 (`eval.py`)
- 模块化架构设计
- 单元测试框架
- 多维数据价值模型
- 课程学习支持 (`curriculum_train.py`)

### Features
- 离散和连续动作空间支持
- Actor-Critic网络架构
- GAE (Generalized Advantage Estimation)
- 熵衰减机制
- 梯度裁剪支持
- 学习率调度器
- 多组件奖励系统

---

## [0.0.1] - 2025-12-20

### Added
- Initial project structure
- Basic PPO implementation
- Vehicle navigation environment
