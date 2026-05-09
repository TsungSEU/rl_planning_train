"""
Vectorized environment runner — manages N parallel env instances.

Enables batch inference by collecting (N, state_dim) arrays, feeding them
to the GPU in a single forward pass instead of N separate calls.

Note: env.step() is mostly pure-Python so ThreadPoolExecutor doesn't help
(more overhead than gain due to GIL).  The speedup comes from batch GPU
inference + pre-allocated numpy buffers + optimized PPO update.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VectorizedEnvRunner:
    """
    Manages N parallel environment instances.

    Environments step sequentially (Python GIL prevents true parallelism),
    but the batch inference and pre-allocated buffers provide significant
    GPU-side speedups.
    """

    def __init__(self, env_factory: Callable, num_envs: int):
        self.num_envs = num_envs
        self.envs = [env_factory() for _ in range(num_envs)]

        sample = self.envs[0]
        self._state_dim = sample.state_dim
        self._action_dim = sample.action_dim

        # logger.info(f"VectorizedEnvRunner: {num_envs} envs, "
        #             f"state_dim={self._state_dim}, action_dim={self._action_dim}")

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset_all(self) -> np.ndarray:
        """Reset all environments. Returns (num_envs, state_dim)."""
        states = np.empty((self.num_envs, self._state_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            states[i] = env.reset()
        return states

    def step_batch(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Advance all environments by one step.

        Args:
            actions: (num_envs, action_dim) array.

        Returns:
            states:   (num_envs, state_dim) — next states (fresh reset state if done)
            rewards:  (num_envs,)
            dones:    (num_envs,) bool
            infos:    list of num_envs info dicts
        """
        states = np.empty((self.num_envs, self._state_dim), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.empty(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = [None] * self.num_envs

        for i, env in enumerate(self.envs):
            next_state, reward, done, info = env.step(actions[i])

            rewards[i] = reward
            dones[i] = done
            infos[i] = info

            if done:
                next_state = env.reset()

            states[i] = next_state

        return states, rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()

    def set_difficulty_for_all(self, params: Dict[str, Any]):
        """Set difficulty parameters on all environments for curriculum learning."""
        for env in self.envs:
            env.set_difficulty(params)
