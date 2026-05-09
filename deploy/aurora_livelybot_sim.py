#!/usr/bin/env python3
"""
Aurora + LivelyBot Pi 联合仿真脚本

联合架构:
  Aurora (10Hz)  → 速度命令 [vx, vy, ωz]
  LivelyBot (50Hz) → 12 维关节目标
  MuJoCo (1000Hz)  → 物理仿真

用法:
  python deploy/aurora_livelybot_sim.py \
    --aurora_model models/nav_data_weights.pt \
    --livelybot_model /path/to/policy_1.pt \
    --duration 10
"""

import math
import argparse
import time
import numpy as np
import torch
from collections import deque
from scipy.spatial.transform import Rotation
import threading
import matplotlib.pyplot as plt

# MuJoCo
try:
    import mujoco
    import mujoco_viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    import glfw
    HAS_GLFW = True
except ImportError:
    HAS_GLFW = False

# Aurora imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoid.nav_data_state import NavDataState, NAV_DATA_STATE_DIM
from humanoid.velocity_action import VelocityActionBounds
from core.config import PPOConfig


# ============================================================================
# LivelyBot Pi 仿真配置
# ============================================================================

LIVELYBOT_NUM_JOINTS = 12
LIVELYBOT_SINGLE_OBS_DIM = 47
LIVELYBOT_FRAME_STACK = 15
LIVELYBOT_NUM_OBS = LIVELYBOT_SINGLE_OBS_DIM * LIVELYBOT_FRAME_STACK  # 705
LIVELYBOT_ACTION_SCALE = 0.25

# PD 增益
LIVELYBOT_KP = np.array([40, 20, 20, 40, 40, 20] * 2, dtype=np.float64)
LIVELYBOT_KD = np.array([1.8, 0.8, 0.8, 1.8, 1.8, 0.6] * 2, dtype=np.float64)
LIVELYBOT_TAU_LIMIT = 40.0

# 仿真频率
DT_PHYSICS = 0.001      # 1000 Hz
DECIMATION_LIVELYBOT = 20   # 50 Hz
DECIMATION_AURORA = 100     # 10 Hz

# LivelyBot 观测缩放
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 1.0
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
OBS_CLIP = 18.0
ACTION_CLIP = 18.0


class AuroraLivelyBotSim:
    """
    Aurora 高层导航 + LivelyBot 低层行走的联合仿真。

    频率关系:
        Aurora 每 100ms 决策一次 → vel_cmd 保持 5 个 LivelyBot 周期
        LivelyBot 每 20ms 推理一次 → joint targets 保持 20 个物理步
        MuJoCo 每 1ms 仿真一步
    """

    def __init__(self, aurora_model_path, livelybot_model_path,
                 mjcf_path=None, env_size=30.0, grid_resolution=0.5):
        if not HAS_MUJOCO:
            raise ImportError("需要 mujoco 和 mujoco_viewer。pip install mujoco mujoco_viewer")

        # --- Aurora 策略 ---
        self.aurora_agent = self._load_aurora_policy(aurora_model_path)
        self.nav_state = NavDataState(
            width=env_size, height=env_size,
            max_steps=60000,  # 60s at 10Hz
            max_range=10.0,
        )
        self.action_bounds = VelocityActionBounds()
        self.aurora_step_count = 0

        # --- LivelyBot 策略 ---
        self.livelybot_policy = torch.jit.load(livelybot_model_path)
        self.livelybot_policy.eval()
        self.prev_livelybot_action = np.zeros(LIVELYBOT_NUM_JOINTS, dtype=np.float64)
        self.target_q = np.zeros(LIVELYBOT_NUM_JOINTS, dtype=np.float64)

        # LivelyBot obs 历史 (15帧)
        self.hist_obs = deque(maxlen=LIVELYBOT_FRAME_STACK)
        for _ in range(LIVELYBOT_FRAME_STACK):
            self.hist_obs.append(np.zeros([1, LIVELYBOT_SINGLE_OBS_DIM], dtype=np.float64))

        # --- MuJoCo ---
        if mjcf_path is None:
            livelybot_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "robot", "livelybot_pi_rl_baseline"
            )
            mjcf_path = os.path.join(
                livelybot_root,
                "resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml"
            )
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(
                f"MuJoCo 模型未找到: {mjcf_path}\n"
                "请用 --mjcf_path 指定路径"
            )

        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = DT_PHYSICS
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_step(self.mj_model, self.mj_data)

        # --- Aurora 状态 ---
        self.env_size = env_size
        self.grid_resolution = grid_resolution
        self.vel_cmd = np.zeros(3, dtype=np.float64)

        # 设置目标 (MuJoCo 世界坐标，米)
        self.goal_position = np.array([env_size * 0.8, env_size * 0.5])
        self.nav_state.goal_position = self.goal_position

        # --- 跟踪 ---
        self.step_count = 0
        self.trajectory = []

        # --- Plot 数据 ---
        self.plot_stop_event = threading.Event()
        self.plot_lock = threading.Lock()
        self.plot_data = {
            'time': [],
            'cmd_vx': [],
            'cmd_vy': [],
            'cmd_wz': [],
            'actual_vx': [],
            'actual_vy': [],
            'actual_wz': [],
            'goal_dist': [],
            'pos_x': [],
            'pos_y': [],
        }
        self.actual_vel = np.zeros(3)  # [vx, vy, wz] body frame

    def _load_aurora_policy(self, model_path, config_path=None):
        """加载 Aurora 训练好的 nav_data 策略。"""
        if config_path is None:
            # 默认 nav_data 配置
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config/humanoid_nav_data_training.yaml"
            )

        from core.config import load_config
        config = load_config(config_path)

        from core.base_agent import ContinuousPPOAgent
        device = torch.device('cpu')
        agent = ContinuousPPOAgent(config, device)

        if os.path.exists(model_path):
            agent.load_weights(model_path)
            print(f"[Aurora] 已加载模型: {model_path}")
        else:
            print(f"[Aurora] 模型未找到: {model_path}，使用随机策略")

        agent.set_eval_mode()
        return agent

    def run(self, duration=10.0):
        """
        运行联合仿真。

        Args:
            duration: 仿真时长 (秒)
        """
        total_steps = int(duration / DT_PHYSICS)

        if not HAS_GLFW or not glfw.init():
            raise RuntimeError("无法初始化 GLFW")

        viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data)
        window = viewer.window

        print(f"\n{'='*60}")
        print(f"Aurora + LivelyBot 联合仿真")
        print(f"{'='*60}")
        print(f"  Aurora 频率: {1.0/DT_PHYSICS/DECIMATION_AURORA:.0f} Hz")
        print(f"  LivelyBot 频率: {1.0/DT_PHYSICS/DECIMATION_LIVELYBOT:.0f} Hz")
        print(f"  物理仿真: {1.0/DT_PHYSICS:.0f} Hz")
        print(f"  Plot 更新: 20 Hz")
        print(f"  时长: {duration}s ({total_steps} 步)")
        print(f"  目标位置: ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f})")
        print(f"{'='*60}\n")

        # 启动 plot 线程
        self._start_plot_thread()

        start_time = time.time()

        for step in range(total_steps):
            if glfw.window_should_close(viewer.window):
                break
            glfw.poll_events()

            # --- Aurora 决策 (10 Hz) ---
            if step % DECIMATION_AURORA == 0:
                self.vel_cmd = self._aurora_step()

            # --- LivelyBot 决策 (50 Hz) ---
            if step % DECIMATION_LIVELYBOT == 0:
                self._livelybot_step()

            # --- 物理仿真 (1000 Hz) ---
            self._physics_step()

            # 更新实际速度（body frame）
            quat = self._get_quat()
            r = Rotation.from_quat(quat)
            v_world = self.mj_data.qvel[:3]
            v_body = r.apply(v_world, inverse=True)
            omega = self.mj_data.sensor('angular-velocity').data
            self.actual_vel = np.array([v_body[0], v_body[1], omega[2]])

            # --- 渲染 (50 Hz) ---
            if step % DECIMATION_LIVELYBOT == 0:
                viewer.render()

            self.step_count += 1

            # 终端输出 + Plot 更新 (每 1 秒)
            if step % 1000 == 0 and step > 0:
                elapsed = time.time() - start_time
                pos = self._get_position()
                dist = np.linalg.norm(pos - self.goal_position)
                t = step * DT_PHYSICS

                print(
                    f"  t={t:.1f}s | "
                    f"pos=({pos[0]:.2f},{pos[1]:.2f}) | "
                    f"dist={dist:.2f}m | "
                    f"cmd=({self.vel_cmd[0]:.2f},{self.vel_cmd[1]:.2f},{self.vel_cmd[2]:.2f}) | "
                    f"actual=({self.actual_vel[0]:.2f},{self.actual_vel[1]:.2f},{self.actual_vel[2]:.2f}) | "
                    f"wall={elapsed:.1f}s"
                )

                # 更新 plot 数据
                self._update_plot_data(t, pos, dist)

        viewer.close()

        # 停止 plot 线程
        self.plot_stop_event.set()
        if hasattr(self, 'plot_thread'):
            self.plot_thread.join(timeout=2.0)

        glfw.terminate()

        elapsed = time.time() - start_time
        print(f"\n仿真完成。实际耗时: {elapsed:.1f}s")

    def _aurora_step(self) -> np.ndarray:
        """
        Aurora 高层决策: 43 维状态 → 3 维速度命令。
        """
        self._update_nav_state_from_mujoco()
        obs = self.nav_state.get_state_vector()

        # 推理 (确定性)
        action, _, _ = self.aurora_agent.select_action(obs, deterministic=True)

        # 裁剪到 LivelyBot 训练范围
        lower, upper = self.action_bounds.get_bounds()
        action = np.clip(action, lower, upper)

        self.nav_state.update_action_history(action)
        self.aurora_step_count += 1

        return action

    def _livelybot_step(self):
        """
        LivelyBot 低层决策: 705 维观测 → 12 维关节目标。
        """
        # 构建 47 维单帧观测
        obs = self._build_livelybot_obs(self.vel_cmd)
        self.hist_obs.append(obs)

        # 堆叠 15 帧 → 705 维
        stacked = np.zeros([1, LIVELYBOT_NUM_OBS], dtype=np.float32)
        for i in range(LIVELYBOT_FRAME_STACK):
            stacked[0, i * LIVELYBOT_SINGLE_OBS_DIM:(i + 1) * LIVELYBOT_SINGLE_OBS_DIM] = \
                self.hist_obs[i][0, :]

        # LivelyBot 推理
        action = self.livelybot_policy(torch.tensor(stacked))[0].detach().numpy()
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)

        self.target_q = action * LIVELYBOT_ACTION_SCALE
        self.prev_livelybot_action = action.copy()

    def _physics_step(self):
        """MuJoCo 物理步进 + PD 控制。"""
        q, dq = self._get_joint_state()

        # PD 控制器
        tau = (self.target_q - q) * LIVELYBOT_KP + (0 - dq) * LIVELYBOT_KD
        tau = np.clip(tau, -LIVELYBOT_TAU_LIMIT, LIVELYBOT_TAU_LIMIT)

        # 关节顺序交换 (左腿/右腿，与 sim2sim 一致)
        for i in range(6):
            tau[i], tau[i + 6] = tau[i + 6], tau[i]

        self.mj_data.ctrl = tau
        mujoco.mj_step(self.mj_model, self.mj_data)

    def _update_nav_state_from_mujoco(self):
        """从 MuJoCo 反馈更新 Aurora 的 NavDataState。"""
        # 位置 (MuJoCo 世界坐标 → Aurora 网格坐标)
        pos = self._get_position()
        self.nav_state.position = pos / self.grid_resolution

        # 朝向
        euler = self._get_euler()
        self.nav_state.heading = euler[2]  # yaw

        # 速度 (世界坐标 → body frame)
        quat = self._get_quat()
        r = Rotation.from_quat(quat)
        v_body = r.apply(self.mj_data.qvel[:3], inverse=True)
        self.nav_state.base_lin_vel = np.array([v_body[0], v_body[1], 0.0])

        # 角速度
        omega = self.mj_data.sensor('angular-velocity').data
        self.nav_state.base_ang_vel = np.array([omega[0], omega[1], omega[2]])

        # 步态相位 (0.4s 周期，与 LivelyBot 一致)
        t = self.step_count * DT_PHYSICS
        self.nav_state.gait_phase = (t / 0.4) % 1.0

        # 步数和预算
        self.nav_state.step_count = self.aurora_step_count

        # 覆盖率和可达性由 nav_state 内部维护（简化处理）
        # 实际部署中，这些由外部数据价值系统更新

    def _build_livelybot_obs(self, vel_cmd: np.ndarray) -> np.ndarray:
        """
        构建 LivelyBot 47 维单帧观测。
        vel_cmd 来自 Aurora 输出，替代 sim2sim 中的键盘输入。
        """
        obs = np.zeros([1, LIVELYBOT_SINGLE_OBS_DIM], dtype=np.float32)

        q, dq = self._get_joint_state()
        euler = self._get_euler()
        omega = self.mj_data.sensor('angular-velocity').data
        t = self.step_count * DT_PHYSICS

        # Gait phase (步态相位)
        obs[0, 0] = math.sin(2 * math.pi * t / 0.5)
        obs[0, 1] = math.cos(2 * math.pi * t / 0.5)

        # Velocity commands (Aurora 输出)
        obs[0, 2] = vel_cmd[0] * OBS_SCALE_LIN_VEL   # vx
        obs[0, 3] = vel_cmd[1] * OBS_SCALE_LIN_VEL   # vy
        obs[0, 4] = vel_cmd[2] * OBS_SCALE_ANG_VEL   # dyaw

        # Joint states
        obs[0, 5:17] = q * OBS_SCALE_DOF_POS
        obs[0, 17:29] = dq * OBS_SCALE_DOF_VEL

        # Previous action
        obs[0, 29:41] = self.prev_livelybot_action

        # IMU
        obs[0, 41:44] = omega
        obs[0, 44:47] = euler

        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    # --- MuJoCo 数据提取 ---

    def _get_position(self) -> np.ndarray:
        """获取基座 XY 位置 (世界坐标，米)。"""
        return np.array([self.mj_data.qpos[0], self.mj_data.qpos[1]])

    def _get_quat(self) -> np.ndarray:
        """获取四元数 [x, y, z, w]。"""
        return self.mj_data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.float64)

    def _get_euler(self) -> np.ndarray:
        """获取欧拉角 [roll, pitch, yaw]。"""
        quat = self._get_quat()
        r = Rotation.from_quat(quat)
        euler = r.as_euler('xyz')
        euler[euler > math.pi] -= 2 * math.pi
        return euler

    def _get_joint_state(self):
        """
        获取关节位置和速度。
        注意: 需要做左右腿顺序交换（与 sim2sim 一致）。
        """
        q = self.mj_data.qpos[-LIVELYBOT_NUM_JOINTS:].astype(np.float64).copy()
        dq = self.mj_data.qvel[-LIVELYBOT_NUM_JOINTS:].astype(np.float64).copy()

        # 左右腿交换 (与 LivelyBot sim2sim 一致)
        for i in range(6):
            q[i], q[i + 6] = q[i + 6], q[i]
            dq[i], dq[i + 6] = dq[i + 6], dq[i]

        return q, dq

    def _start_plot_thread(self):
        """启动 matplotlib 可视化线程。"""
        self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.plot_thread.start()

    def _plot_loop(self):
        """
        实时绘图线程：显示速度跟踪和位置信息。

        包含两个子图：
        1. 速度命令 vs 实际速度 (vx, vy, wz)
        2. 位置轨迹和目标距离
        """
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # --- 子图 1: 速度跟踪 ---
        # vx
        line_cmd_vx, = ax1.plot([], [], 'r-', linewidth=2, label='Cmd vx (Aurora)')
        line_actual_vx, = ax1.plot([], [], 'r--', linewidth=1.5, alpha=0.7, label='Actual vx')
        # vy
        line_cmd_vy, = ax1.plot([], [], 'g-', linewidth=2, label='Cmd vy (Aurora)')
        line_actual_vy, = ax1.plot([], [], 'g--', linewidth=1.5, alpha=0.7, label='Actual vy')
        # wz (angular vel)
        ax1_twin = ax1.twinx()
        line_cmd_wz, = ax1_twin.plot([], [], 'b-', linewidth=2, label='Cmd ωz (Aurora)')
        line_actual_wz, = ax1_twin.plot([], [], 'b--', linewidth=1.5, alpha=0.7, label='Actual ωz')

        ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Linear Velocity (m/s)', fontsize=11, fontweight='bold')
        ax1_twin.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
        ax1.set_title('Velocity Tracking: Aurora Command vs LivelyBot Actual',
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        ax1_twin.legend(loc='upper right', fontsize=9)

        # --- 子图 2: 位置和目标距离 ---
        line_pos_x, = ax2.plot([], [], 'k-', linewidth=1.5, alpha=0.5, label='Pos X')
        line_pos_y, = ax2.plot([], [], 'k-', linewidth=1.5, alpha=0.5, label='Pos Y')
        ax2_twin = ax2.twinx()
        line_goal_dist, = ax2_twin.plot([], [], 'm-', linewidth=2, label='Goal Distance')

        ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Position (m)', fontsize=11, fontweight='bold')
        ax2_twin.set_ylabel('Distance (m)', fontsize=11, fontweight='bold')
        ax2.set_title('Position Trajectory and Goal Distance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        ax2_twin.legend(loc='upper right', fontsize=9)

        plt.tight_layout()

        # 更新循环
        while not self.plot_stop_event.is_set():
            with self.plot_lock:
                if len(self.plot_data['time']) > 0:
                    t = self.plot_data['time']

                    # 更新速度曲线
                    line_cmd_vx.set_data(t, self.plot_data['cmd_vx'])
                    line_actual_vx.set_data(t, self.plot_data['actual_vx'])
                    line_cmd_vy.set_data(t, self.plot_data['cmd_vy'])
                    line_actual_vy.set_data(t, self.plot_data['actual_vy'])
                    line_cmd_wz.set_data(t, self.plot_data['cmd_wz'])
                    line_actual_wz.set_data(t, self.plot_data['actual_wz'])

                    # 更新位置曲线
                    line_pos_x.set_data(t, self.plot_data['pos_x'])
                    line_pos_y.set_data(t, self.plot_data['pos_y'])
                    line_goal_dist.set_data(t, self.plot_data['goal_dist'])

                    # 动态调整坐标轴
                    if len(t) > 1:
                        ax1.set_xlim(t[0], t[-1] + 0.5)
                        ax1.set_ylim(-0.8, 0.8)
                        ax1_twin.set_ylim(-0.5, 0.5)

                        ax2.set_xlim(t[0], t[-1] + 0.5)
                        ax2.set_ylim(min(self.plot_data['pos_x'] + self.plot_data['pos_y']) - 1,
                                    max(self.plot_data['pos_x'] + self.plot_data['pos_y']) + 1)
                        ax2_twin.set_ylim(0, max(self.plot_data['goal_dist']) + 1)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.05)  # 20 Hz 更新

        plt.ioff()
        plt.close(fig)

    def _update_plot_data(self, t, pos, goal_dist):
        """更新 plot 数据（线程安全）。"""
        with self.plot_lock:
            self.plot_data['time'].append(t)
            self.plot_data['cmd_vx'].append(self.vel_cmd[0])
            self.plot_data['cmd_vy'].append(self.vel_cmd[1])
            self.plot_data['cmd_wz'].append(self.vel_cmd[2])
            self.plot_data['actual_vx'].append(self.actual_vel[0])
            self.plot_data['actual_vy'].append(self.actual_vel[1])
            self.plot_data['actual_wz'].append(self.actual_vel[2])
            self.plot_data['goal_dist'].append(goal_dist)
            self.plot_data['pos_x'].append(pos[0])
            self.plot_data['pos_y'].append(pos[1])


def main():
    parser = argparse.ArgumentParser(description='Aurora + LivelyBot 联合仿真')
    parser.add_argument(
        '--aurora_model', type=str,
        default='models/nav_data_weights.pt',
        help='Aurora nav_data 模型路径 (.pt)'
    )
    parser.add_argument(
        '--livelybot_model', type=str,
        default=None,
        help='LivelyBot 行走策略路径 (.pt TorchScript)'
    )
    parser.add_argument(
        '--mjcf_path', type=str, default=None,
        help='LivelyBot MuJoCo 模型路径 (.xml)'
    )
    parser.add_argument(
        '--duration', type=float, default=10.0,
        help='仿真时长 (秒)'
    )
    parser.add_argument(
        '--env_size', type=float, default=30.0,
        help='Aurora 环境尺寸 (米)'
    )
    args = parser.parse_args()

    # 默认 LivelyBot 模型路径
    if args.livelybot_model is None:
        livelybot_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "robot", "livelybot_pi_rl_baseline"
        )
        args.livelybot_model = os.path.join(
            livelybot_root,
            "logs/Pai_ppo/exported/policies/policy_1.pt"
        )

    sim = AuroraLivelyBotSim(
        aurora_model_path=args.aurora_model,
        livelybot_model_path=args.livelybot_model,
        mjcf_path=args.mjcf_path,
        env_size=args.env_size,
    )
    sim.run(duration=args.duration)


if __name__ == '__main__':
    main()
