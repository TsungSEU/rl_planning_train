# 命令行参考

Aurora Planning Engine 提供两种命令行接口：
1. **Python 脚本** - 直接运行 `train.py`, `eval.py` 等
2. **Aurora CLI** - 统一的 `aurora` 命令（推荐）

---

## Aurora CLI (推荐)

Aurora CLI 是统一的命令行工具，提供更简洁的接口。

### 安装 CLI

```bash
# 开发模式安装（包含 aurora 命令）
pip install -e .
```

### 全局命令

```bash
aurora [OPTIONS] COMMAND [ARGS]
```

| 选项 | 说明 |
|------|------|
| `--verbose`, `-v` | 启用详细日志 |
| `--version` | 显示版本信息 (v0.7.0) |
| `--help` | 显示帮助信息 |

---

### aurora train - 训练模型

```bash
aurora train [OPTIONS]
```

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--config` | 配置文件路径 (YAML) | - | **是** |
| `--episodes` | 训练回合数 | 配置文件值 | 否 |
| `--device` | 设备 (cpu/cuda/auto) | auto | 否 |
| `--num-envs` | 并行环境数 | 配置文件值 | 否 |
| `--model-path` | 预训练模型路径 | - | 否 |

**使用示例**:
```bash
# 基础训练
aurora train --config config/nav_data_training.yaml

# 指定训练回合数
aurora train --config config/nav_data_training.yaml --episodes 5000

# 使用 GPU 加速
aurora train --config config/nav_data_training.yaml --device cuda

# 从检查点继续训练
aurora train --config config/nav_data_training.yaml --model-path runs/train/exp001/nav_data_weights_iter_100.pt
```

---

### aurora eval - 评估模型

```bash
aurora eval [OPTIONS]
```

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--model-path` | 模型路径 | - | **是** |
| `--config` | 配置文件路径 | - | **是** |
| `--episodes` | 评估回合数 | 100 | 否 |
| `--deterministic` | 使用确定性策略 | false | 否 |
| `--device` | 设备 | auto | 否 |
| `--output-dir` | 结果输出目录 | runs | 否 |

**使用示例**:
```bash
# 基础评估
aurora eval --model-path runs/train/exp001/nav_data_weights.pt --config config/nav_data_training.yaml

# 确定性策略评估
aurora eval --model-path runs/train/exp001/nav_data_weights.pt --config config/nav_data_training.yaml --deterministic
```

---

### aurora export - 导出 ONNX 模型

```bash
aurora export [OPTIONS]
```

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--model-path` | 模型权重路径 (.pt) | - | **是** |
| `--config` | 配置文件路径 | - | **是** |
| `--output` | 输出 ONNX 模型路径 | - | **是** |

**使用示例**:
```bash
# 导出 ONNX 模型
aurora export \
  --model-path runs/train/exp001/nav_data_weights.pt \
  --config config/nav_data_training.yaml \
  --output models/nav_data.onnx
```

---

## Python 脚本命令

### train.py - 训练

```bash
python train.py [OPTIONS]
```

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--config` | 配置文件路径 (YAML) | - | **是** |
| `--num-envs` | 并行环境数。1=串行, >1=向量化 | 配置文件值 | 否 |
| `--episodes` | 训练回合数 (串行模式) | 配置文件值 | 否 |
| `--max-iterations` | 训练迭代数 (向量化模式) | 配置文件值 | 否 |
| `--exp-dir` | 实验输出目录 | 自动 runs/train/expNNN | 否 |
| `--update-interval` | 累积N个episode后更新 (串行模式) | 4 | 否 |
| `--device` | 设备 (cpu/cuda/auto) | cuda | 否 |
| `--save-interval` | 保存模型间隔 | 配置文件值 | 否 |
| `--evaluate` | 训练后运行评估 | false | 否 |
| `--eval-episodes` | 评估回合数 | 1000 | 否 |
| `--model-path` | 预训练模型路径 | - | 否 |

### eval.py - 评估

```bash
python eval.py [OPTIONS]
```

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--model-path` | 模型路径 | - | **是** |
| `--config` | 配置文件路径 (YAML) | - | **是** |
| `--episodes` | 评估回合数 | 100 | 否 |
| `--deterministic` | 使用确定性策略 | false | 否 |
| `--render` | 渲染环境 | false | 否 |
| `--device` | 设备 (cpu/cuda/auto) | auto | 否 |
| `--output-dir` | 结果输出目录 | runs | 否 |
| `--no-save` | 不保存结果 | false | 否 |

### export_onnx.py - ONNX 导出

```bash
python -m utils.export_onnx [OPTIONS]
```

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--model-path` | 模型权重路径 (.pt文件) | - | **是** |
| `--config` | 配置文件路径 (YAML) | - | **是** |
| `--output` | 输出ONNX模型路径 | - | **是** |

### 输出文件

| 文件 | 说明 |
|------|------|
| `nav_data.onnx` | ONNX模型文件 (Opset 17) |
| `nav_data_log_std.pt` | log_std参数 |

---

## 测试命令

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_core.py -v          # 核心模块
pytest tests/test_value_model.py -v   # 价值模型

# 生成覆盖率报告
pytest tests/ --cov=core --cov-report=html
```

---

## CLI vs Python 脚本对比

| 特性 | Aurora CLI | Python 脚本 |
|------|-----------|------------|
| 易用性 | 更简洁，推荐使用 | 灵活性高 |
| 安装 | 需要 pip install | 无需安装 |
| 参数 | 简化版 | 完整参数 |
| 未来支持 | 主要开发方向 | 保持兼容 |

**建议**: 新项目使用 `aurora` CLI，已有项目可继续使用 Python 脚本。
