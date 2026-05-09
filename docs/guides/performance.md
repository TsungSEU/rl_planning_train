# 性能优化指南

本文档介绍如何优化训练性能，充分利用硬件资源。

## 优化手段

| 优化项 | 加速比 | 说明 |
|--------|--------|------|
| 环境向量化 (num_envs) | 10-50x/step | N个并行环境，单次 GPU 推理 |
| GAE 计算 O(n²)→O(n) | ~2x | 列表插入改为 append+reverse |
| 混合精度训练 (AMP) | ~2x | GPU Tensor Core 加速 |
| 批量经验收集 | ~3x | 累积大量 transition 后更新 |

## 硬件优化

### 使用 GPU

训练脚本默认自动检测 CUDA：

```bash
python train.py --config config/nav_data_training.yaml
```

手动指定设备：

```bash
# 强制使用 GPU
python train.py --config config/nav_data_training.yaml --device cuda

# 强制使用 CPU
python train.py --config config/nav_data_training.yaml --device cpu
```

### 启用混合精度训练

在配置文件中设置：

```yaml
training:
  use_amp: true  # 启用混合精度（需 GPU）
```

混合精度训练可带来：
- **~2x 训练速度**
- **~50% 显存占用**
- 支持更大的 batch size

## 向量化训练优化

### 调整 num_envs

控制并行环境数，越大 GPU 利用率越高：

```yaml
training:
  num_envs: 4096  # 根据 GPU 内存调整
```

**建议**：
- GPU 显存 8GB: num_envs 512-1024
- GPU 显存 16GB: num_envs 2048-4096
- GPU 显存 24GB+: num_envs 4096-8192

### 调整 batch_size

```yaml
training:
  batch_size: 64   # mini-batch 大小
  mini_batches: 4   # mini-batch 数
```

### 调整 epochs

```yaml
training:
  epochs: 8  # PPO 优化轮数
```

**建议**：
- 大多数情况下 8-10 epochs 足够
- 更多的 epochs 不一定带来更好的性能，但会增加训练时间

## 环境参数优化

### 调整 max_steps

```yaml
training:
  max_steps: 600  # 每次迭代的 rollout 步数 (= 60s × 10Hz)
```

### 调整环境尺寸

```yaml
training:
  env_width: 40   # 环境宽度 (米)
  env_height: 40  # 环境高度 (米)
```

**建议**：
- 较小的环境 (20×20) 训练更快，适合调试
- 较大的环境 (40×40 或 60×60) 提供更丰富的训练场景

## 诊断性能问题

### 检查 GPU 利用率

```bash
# NVIDIA GPU
nvidia-smi -l 1

# 训练时观察 GPU 利用率
```

### 检查瓶颈

如果 GPU 利用率低，可能是：
1. **环境计算瓶颈** — 4096 个 Python 环境串行 step，GPU 在等待
2. **batch_size 太小** — GPU 未充分利用
3. **num_envs 太小** — 单次推理 batch 不够大

**解决方案**：
- 增大 num_envs（主要手段）
- 增大 batch_size
- 确保启用了 AMP

### 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减小 num_envs
2. 减小 batch_size
3. 减小 hidden_dims
4. 使用 CPU 训练

## 更多资源

- [配置说明](configuration.md) — 详细配置参数
- [故障排除](../troubleshooting.md) — 常见问题
