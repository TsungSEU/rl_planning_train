#!/usr/bin/env bash
# Aurora + LivelyBot 联合仿真启动脚本
#
# 用法:
#   bash deploy/run_sim.sh              # 默认 10 秒
#   bash deploy/run_sim.sh 30           # 指定时长
#   bash deploy/run_sim.sh 60 fast      # 加速模式（跳过 sleep）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 路径配置
AURORA_MODEL="${PROJECT_DIR}/models/nav_data_weights.pt"
LIVELYBOT_MODEL="/home/xucong/caicAD/01datainfra/robot/livelybot_pi_rl_baseline/logs/Pai_ppo/exported/policies/policy_torch.pt"
MJCF_PATH="/home/xucong/caicAD/01datainfra/robot/livelybot_pi_rl_baseline/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml"

DURATION="${1:-10}"

# 检查文件
for f in "$AURORA_MODEL" "$LIVELYBOT_MODEL" "$MJCF_PATH"; do
    if [ ! -f "$f" ]; then
        echo "文件不存在: $f"
        exit 1
    fi
done

echo "========================================"
echo " Aurora + LivelyBot 联合仿真"
echo "========================================"
echo " Aurora 模型:  $(basename $AURORA_MODEL)"
echo " LivelyBot:   $(basename $LIVELYBOT_MODEL)"
echo " 时长:         ${DURATION}s"
echo "========================================"

conda run -n rl_biped python "${SCRIPT_DIR}/aurora_livelybot_sim.py" \
    --aurora_model "$AURORA_MODEL" \
    --livelybot_model "$LIVELYBOT_MODEL" \
    --mjcf_path "$MJCF_PATH" \
    --duration "$DURATION"
