"""
Velocity command action space (3 continuous actions).

Simplified action representation for humanoid locomotion control.
Instead of directly controlling 12+ joint angles, this uses high-level velocity commands
that are translated to joint trajectories by the gait controller on the edge device.

Action structure:
- Index 0: forward_velocity [-1, 1] → [-0.5, 1.0] m/s
- Index 1: lateral_velocity [-1, 1] → [-1.0, 1.0] m/s
- Index 2: angular_velocity [-1, 1] → [-1.0, 1.0] rad/s
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VelocityActionBounds:
    """Bounds for 3-DOF velocity command actions (aligned with LivelyBot Pi)."""
    forward_vel_min: float = -0.3
    forward_vel_max: float = 0.6
    lateral_vel_min: float = -0.3
    lateral_vel_max: float = 0.3
    angular_vel_min: float = -0.3
    angular_vel_max: float = 0.3

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.forward_vel_min, self.lateral_vel_min, self.angular_vel_min])
        upper = np.array([self.forward_vel_max, self.lateral_vel_max, self.angular_vel_max])
        return lower, upper


class VelocityAction:
    """
    3-DOF velocity command action space for humanoid robot.

    This is the simplified action space that decouples high-level navigation
    from low-level gait control. The edge device's VelocityLocomotionController
    translates these velocity commands into actual joint trajectories.
    """

    ACTION_DIM = 3
    ACTION_SPACE = 'continuous'

    # Action indices
    FORWARD_IDX = 0
    LATERAL_IDX = 1
    ANGULAR_IDX = 2

    @staticmethod
    def get_default_bounds() -> VelocityActionBounds:
        return VelocityActionBounds()

    @staticmethod
    def clip_action(action: np.ndarray, bounds: Optional[VelocityActionBounds] = None) -> np.ndarray:
        if bounds is None:
            bounds = VelocityActionBounds()
        lower, upper = bounds.get_bounds()
        return np.clip(action, lower, upper)

    @staticmethod
    def zero_action() -> np.ndarray:
        return np.zeros(VelocityAction.ACTION_DIM)

    @staticmethod
    def random_action(bounds: Optional[VelocityActionBounds] = None) -> np.ndarray:
        if bounds is None:
            bounds = VelocityActionBounds()
        lower, upper = bounds.get_bounds()
        return np.random.uniform(lower, upper)

    @staticmethod
    def normalize_action(action: np.ndarray, bounds: Optional[VelocityActionBounds] = None) -> np.ndarray:
        if bounds is None:
            bounds = VelocityActionBounds()
        lower, upper = bounds.get_bounds()
        return 2 * (action - lower) / (upper - lower) - 1

    @staticmethod
    def denormalize_action(normalized: np.ndarray, bounds: Optional[VelocityActionBounds] = None) -> np.ndarray:
        if bounds is None:
            bounds = VelocityActionBounds()
        lower, upper = bounds.get_bounds()
        return (normalized + 1) / 2 * (upper - lower) + lower

    @staticmethod
    def get_action_description(action: np.ndarray) -> str:
        return (f"VelocityCmd: forward={action[0]:.2f} m/s, "
                f"lateral={action[1]:.2f} m/s, angular={action[2]:.2f} rad/s")

    @staticmethod
    def from_normalized(normalized: np.ndarray, bounds: Optional[VelocityActionBounds] = None) -> np.ndarray:
        """Convert normalized [-1,1] action to actual velocity range."""
        return VelocityAction.denormalize_action(normalized, bounds)

    @staticmethod
    def to_normalized(action: np.ndarray, bounds: Optional[VelocityActionBounds] = None) -> np.ndarray:
        """Convert actual velocity to normalized [-1,1] range."""
        return VelocityAction.normalize_action(action, bounds)
