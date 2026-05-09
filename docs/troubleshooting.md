# 故障排除

## 常见问题

### CUDA设备错误

**错误信息**:
```
RuntimeError: Expected all tensors to be on the same device
```

**解决方案**:
- 确保所有张量在同一设备上
- 已在最新版本中修复，确保使用最新代码
- 如仍有问题，尝试 `--device cpu`

---

### 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少 `num_envs` (如从 4096 降至 1024)
- 减少 `batch_size` (如从 256 降至 128)
- 减少 `hidden_dims` (如从 [256, 128, 64] 降至 [128, 64])
- 使用 `--device cpu`

**配置调整示例**:
```yaml
training:
  num_envs: 1024  # 降低
  batch_size: 128  # 降低
```

---

### 机器人不收敛

**症状**: 训练过程中奖励不增长或波动很大

**解决方案**:
1. **增加探索**
   ```yaml
   entropy_coef: 0.1  # 增加
   ```

2. **调整学习率**
   ```yaml
   learning_rate: 0.0001  # 降低
   ```

3. **检查动作边界设置**
   - 确认 `init_log_std`, `min_log_std`, `max_log_std` 合理

4. **增加训练量**
   ```yaml
   max_iterations: 10000  # 增加迭代数
   ```

---

### ONNX导出失败

**错误信息**:
```
TracerWarning: Converting a tensor to a Python boolean...
```

**解决方案**:
- 确保使用最新版本的导出脚本
- 该警告已在v0.2.0中修复

**错误信息**:
```
RuntimeError: model state_dict missing keys
```

**解决方案**:
- 确认配置文件的网络架构与训练时一致
- 检查 `state_dim`, `action_dim`, `hidden_dims`

---

### 导出模型推理结果不一致

**症状**: ONNX模型输出与PyTorch模型不一致

**可能原因**:
1. **log_std未加载**
   - 确认加载了 `nav_data_log_std.pt`
   - 旧 checkpoint 不包含 log_std，使用默认值 0

2. **Opset版本不兼容**
   - 使用 Opset 17

---

### 测试失败

**错误信息**:
```
FAILED tests/test_xxx.py
```

**解决方案**:
```bash
# 更新pytest
pip install -U pytest

# 清理缓存
pytest --cache-clear

# 单独运行失败的测试获取详细输出
pytest tests/test_xxx.py::TestName::test_method -vvs
```

---

### 训练中断恢复

**症状**: 训练意外中断

**解决方案**:
使用定期保存的检查点继续训练：
```bash
python train.py \
  --config config/nav_data_training.yaml \
  --model-path runs/train/exp001/nav_data_weights_iter_100.pt \
  --max-iterations 5000
```

---

## 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 检查模型输出

```python
# 在训练脚本中添加
print(f"State: {state}")
print(f"Action: {action}, Log prob: {log_prob}, Value: {value}")
print(f"Reward: {reward}, Done: {done}")
```

### 可视化训练曲线

```bash
# 手动绘制training_metrics.npy
python -c "
import numpy as np
import matplotlib.pyplot as plt
metrics = np.load('runs/train/exp001/nav_data_training_metrics.npy', allow_pickle=True).item()
plt.plot(metrics['episode_rewards'])
plt.show()
"
```

---

## 获取帮助

如果以上方法无法解决问题：

1. 查看 CHANGELOG 确认版本
2. 检查配置文件与当前代码版本是否一致
3. 提供以下信息寻求帮助：
   - 完整错误信息
   - 配置文件内容
   - 命令行参数
   - 系统环境 (Python版本, PyTorch版本, CUDA版本)
