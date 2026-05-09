# 安装说明

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA (可选，用于GPU加速)
- Click 8.0+ (用于 aurora CLI)

## 安装依赖

### 基础安装

```bash
pip install torch numpy onnx onnxruntime pytest tqdm matplotlib click
```

### CUDA 支持（推荐）

如果你有 NVIDIA GPU，安装支持 CUDA 的 PyTorch：

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 安装 Aurora CLI

### 开发模式安装

```bash
# 进入项目目录
cd rl_planning_train

# 开发模式安装（包含 aurora 命令）
pip install -e .

# 验证安装
aurora --version
# 输出: aurora, version 0.7.0

aurora --help
# 输出: Aurora Planning Engine - RL导航与数据采集训练平台
```

### 使用 Aurora CLI

```bash
# 训练模型
aurora train --config config/nav_data_training.yaml

# 评估模型
aurora eval --model-path runs/train/exp001/nav_data_weights.pt --config config/nav_data_training.yaml

# 导出 ONNX 模型
aurora export --model-path runs/train/exp001/nav_data_weights.pt --config config/nav_data_training.yaml --output models/nav_data.onnx
```

## 可选依赖

### 可视化训练曲线

```bash
pip install matplotlib tensorboard
```

### 代码质量工具

```bash
pip install pytest pytest-cov black flake8 mypy
```

### LivelyBot 集成（用于联合仿真）

```bash
# LivelyBot 仓库位于
# /home/xucong/caicAD/01datainfra/robot/livelybot_pi_rl_baseline

# 安装 LivelyBot 依赖
cd /home/xucong/caicAD/01datainfra/robot/livelybot_pi_rl_baseline
pip install -r requirements.txt
```

## 开发环境设置

### 克隆仓库

```bash
git clone <repository-url>
cd rl_planning_train
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_core.py -v          # 核心模块
pytest tests/test_value_model.py -v   # 价值模型

# 生成覆盖率报告
pytest tests/ --cov=core --cov-report=html
```

## 项目结构

```
rl_planning_train/
├── aurora/                 # CLI 包
│   └── cli/               # 命令行接口
│       ├── commands/      # train, eval, export 命令
│       └── main.py        # CLI 入口
├── core/                  # 核心模块
│   ├── base_agent.py      # Actor-Critic, PPO
│   ├── base_environment.py # 环境基类
│   ├── base_reward.py     # 奖励基类
│   └── config.py          # 配置管理
├── humanoid/              # 人形机器人模块
│   ├── nav_data_state.py  # 43 维状态
│   ├── nav_data_reward.py # 奖励函数
│   ├── nav_data_environment.py # 环境
│   └── velocity_action.py # 3 维速度动作
├── value_model/           # 数据价值模型
│   └── humanoid_value.py  # 4D 价值评估
├── utils/                 # 工具函数
│   ├── export_onnx.py     # ONNX 导出
│   ├── reachability.py    # 可达性追踪
│   └── training_utils.py  # 训练工具
├── config/                # 配置文件
│   └── nav_data_training.yaml
├── deploy/                # 部署脚本
│   └── aurora_livelybot_sim.py # 联合仿真
├── docs/                  # 文档
├── tests/                 # 测试
├── train.py               # 训练脚本
├── eval.py                # 评估脚本
├── setup.py               # 安装脚本
└── CLAUDE.md              # 项目指南
```

## 故障排除

### ImportError: No module named 'torch'

确保已安装 PyTorch：
```bash
pip install torch
```

### CUDA not available

即使安装了 CUDA 版本的 PyTorch，也可能遇到此问题。检查：
1. NVIDIA 驱动是否正确安装
2. CUDA toolkit 版本是否与 PyTorch 兼容
3. 运行 `python -c "import torch; print(torch.cuda.is_available())"` 诊断

### ONNX Runtime 导入错误

```bash
pip install --upgrade onnxruntime
```

对于 GPU 推理：
```bash
pip install onnxruntime-gpu
```

### aurora 命令不可用

确保已安装 Aurora CLI：
```bash
# 重新安装
pip install -e .

# 验证安装
aurora --version
```

### 依赖冲突

使用虚拟环境隔离依赖：
```bash
# 创建虚拟环境
python -m venv aurora_env
source aurora_env/bin/activate  # Linux/Mac
# 或
aurora_env\Scripts\activate     # Windows

# 安装依赖
pip install -e .
```

### LivelyBot 路径问题

联合仿真需要正确设置 LivelyBot 模型路径：
```bash
python deploy/aurora_livelybot_sim.py \
    --livelybot_model /home/xucong/caicAD/01datainfra/robot/livelybot_pi_rl_baseline/logs/Pai_ppo/exported/policies/policy_torch.pt
```

## 下一步

安装完成后，请阅读 [快速开始](quick-start.md) 开始训练你的第一个模型。
