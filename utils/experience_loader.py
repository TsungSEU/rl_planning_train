"""
Experience data loader for edge device metadata.

Deserializes ExperienceMetadata uploaded from Aurora-Edge-Runtime
into PyTorch tensors suitable for PPO training.

Edge format: float16 (uint16) quantized vectors
  - state: [state_dim] uint16 values (quantized [-1,1] → [0, 65535])
  - next_state: [state_dim] uint16 values
  - action: [action_dim] uint16 values
  - reward: float32
  - env_change_type: uint8
  - env_change_severity: uint8
  - state_dim: uint16
  - action_dim: uint16
  - timestamp: uint64
"""

import numpy as np
import struct
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class EdgeExperience:
    """Single experience tuple from edge device."""
    state: np.ndarray          # [state_dim] float64
    action: np.ndarray         # [action_dim] float64
    reward: float
    next_state: np.ndarray     # [state_dim] float64
    env_change_type: int
    env_change_severity: float  # [0, 1]
    timestamp: int

    def to_tuple(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        return (self.state, self.action, self.reward, self.next_state)


def dequantize_vector(quantized: np.ndarray) -> np.ndarray:
    """Dequantize uint16 array to float64 in [-1, 1] range.

    Quantization formula (edge side): uint16 = (float + 1.0) * 32767.5
    Dequantization: float = uint16 / 32767.5 - 1.0
    """
    return quantized.astype(np.float64) / 32767.5 - 1.0


def load_experiences_from_json(
    filepath: str,
    expected_state_dim: int = 75,
    expected_action_dim: int = 3
) -> List[EdgeExperience]:
    """Load experiences from a JSON file (batch upload format).

    Args:
        filepath: Path to JSON file containing a list of experience metadata
        expected_state_dim: Expected state dimension (filters mismatched entries)
        expected_action_dim: Expected action dimension (filters mismatched entries)

    Returns:
        List of EdgeExperience objects
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Experience file not found: {filepath}")

    with open(path, 'r') as f:
        batch = json.load(f)

    experiences = []
    for entry in batch:
        state_q = np.array(entry.get('state', []), dtype=np.uint16)
        action_q = np.array(entry.get('action', []), dtype=np.uint16)
        next_state_q = np.array(entry.get('next_state', []), dtype=np.uint16)

        state = dequantize_vector(state_q) if len(state_q) > 0 else np.array([])
        action = dequantize_vector(action_q) if len(action_q) > 0 else np.array([])
        next_state = dequantize_vector(next_state_q) if len(next_state_q) > 0 else np.array([])

        if len(state) != expected_state_dim or len(action) != expected_action_dim:
            logger.warning(
                f"Skipping mismatched experience: state_dim={len(state)}, action_dim={len(action)} "
                f"(expected {expected_state_dim}, {expected_action_dim})")
            continue

        experiences.append(EdgeExperience(
            state=state,
            action=action,
            reward=float(entry.get('reward', 0.0)),
            next_state=next_state,
            env_change_type=int(entry.get('env_change_type', 0)),
            env_change_severity=float(entry.get('env_change_severity', 0)) / 100.0,
            timestamp=int(entry.get('timestamp', 0))
        ))

    logger.info(f"Loaded {len(experiences)} experiences from {filepath}")
    return experiences


def load_experiences_from_binary(
    filepath: str,
    expected_state_dim: int = 75,
    expected_action_dim: int = 3
) -> List[EdgeExperience]:
    """Load experiences from a binary file (compact format).

    Binary format per entry:
      uint16 state_dim (2 bytes)
      uint16 action_dim (2 bytes)
      uint16[state_dim] state (2*state_dim bytes)
      uint16[action_dim] action (2*action_dim bytes)
      uint16[state_dim] next_state (2*state_dim bytes)
      float32 reward (4 bytes)
      uint8 env_change_type (1 byte)
      uint8 env_change_severity (1 byte)
      uint64 timestamp (8 bytes)

    Args:
        filepath: Path to binary file
        expected_state_dim: Expected state dimension
        expected_action_dim: Expected action dimension

    Returns:
        List of EdgeExperience objects
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Experience file not found: {filepath}")

    experiences = []
    with open(path, 'rb') as f:
        data = f.read()

    offset = 0
    while offset < len(data):
        try:
            state_dim = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            action_dim = struct.unpack_from('<H', data, offset)[0]
            offset += 2

            if state_dim != expected_state_dim or action_dim != expected_action_dim:
                # Skip this entry
                skip = (2 * state_dim + 2 * action_dim + 2 * state_dim + 4 + 1 + 1 + 8)
                offset += skip
                continue

            state_bytes = 2 * state_dim
            state_q = np.frombuffer(data[offset:offset + state_bytes], dtype=np.uint16)
            offset += state_bytes

            action_bytes = 2 * action_dim
            action_q = np.frombuffer(data[offset:offset + action_bytes], dtype=np.uint16)
            offset += action_bytes

            next_state_q = np.frombuffer(data[offset:offset + state_bytes], dtype=np.uint16)
            offset += state_bytes

            reward = struct.unpack_from('<f', data, offset)[0]
            offset += 4

            env_change_type = struct.unpack_from('<B', data, offset)[0]
            offset += 1

            env_change_severity_raw = struct.unpack_from('<B', data, offset)[0]
            offset += 1

            timestamp = struct.unpack_from('<Q', data, offset)[0]
            offset += 8

            state = dequantize_vector(state_q)
            action = dequantize_vector(action_q)
            next_state = dequantize_vector(next_state_q)

            experiences.append(EdgeExperience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                env_change_type=env_change_type,
                env_change_severity=env_change_severity_raw / 100.0,
                timestamp=timestamp
            ))
        except struct.error:
            logger.warning(f"Truncated experience data at offset {offset}, stopping")
            break

    logger.info(f"Loaded {len(experiences)} experiences from {filepath}")
    return experiences


def experiences_to_tensors(
    experiences: List[EdgeExperience]
) -> Dict[str, torch.Tensor]:
    """Convert list of EdgeExperience to PyTorch tensors for PPO training.

    Returns:
        Dict with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'
    """
    states = torch.tensor(
        np.stack([e.state for e in experiences]), dtype=torch.float32)
    actions = torch.tensor(
        np.stack([e.action for e in experiences]), dtype=torch.float32)
    rewards = torch.tensor(
        np.array([e.reward for e in experiences]), dtype=torch.float32)
    next_states = torch.tensor(
        np.stack([e.next_state for e in experiences]), dtype=torch.float32)

    # Infer done flags from env_change_type (SENSOR_FAILURE or no next_state → done)
    dones = torch.zeros(len(experiences), dtype=torch.float32)
    for i, e in enumerate(experiences):
        if len(e.next_state) == 0 or e.env_change_type == 5:  # 5 = SENSOR_FAILURE
            dones[i] = 1.0

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones,
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.995,
    lam: float = 0.95
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: [T] reward at each step
        values: [T] value estimate at each step
        next_values: [T] value estimate at next step
        dones: [T] done flag at each step
        gamma: discount factor
        lam: GAE parameter

    Returns:
        advantages: [T] advantage estimates
    """
    advantages = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages
