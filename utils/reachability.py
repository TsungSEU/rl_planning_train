"""
Reachability tracker for position-based exploration rewards.

Tracks visit counts and success/failure statistics per grid cell,
computing reachability scores used in reward calculation.
"""

import numpy as np
from typing import Dict, Any


class ReachabilityTracker:
    """
    Tracks reachability statistics for positions in the environment.
    Used for data collection optimization and exploration rewards.
    """

    def __init__(self, decay: float = 0.95, min_samples: int = 3,
                 position_tolerance: float = 0.5, grid_resolution: float = 1.0):
        """
        Args:
            decay: History decay factor for reachability scores
            min_samples: Minimum samples before considering a position reliable
            position_tolerance: Distance tolerance for position matching (meters)
            grid_resolution: Grid resolution for discretizing space (meters)
        """
        self.decay = decay
        self.min_samples = min_samples
        self.position_tolerance = position_tolerance
        self.grid_resolution = grid_resolution

        self.visit_counts: Dict[tuple, int] = {}
        self.success_counts: Dict[tuple, int] = {}
        self.failure_counts: Dict[tuple, int] = {}
        self.reachability_scores: Dict[tuple, float] = {}
        self.last_visit_time: Dict[tuple, int] = {}

    def _discretize_position(self, position: np.ndarray) -> tuple:
        x = int(position[0] / self.grid_resolution)
        y = int(position[1] / self.grid_resolution)
        return (x, y)

    def record_visit(self, position: np.ndarray, success: bool = None,
                     episode: int = None) -> float:
        """
        Record a visit and return the updated reachability score.

        Args:
            position: Agent position [x, y] or [x, y, theta]
            success: Whether the visit was successful
            episode: Current episode number (for temporal decay)
        """
        grid_pos = self._discretize_position(position)

        self.visit_counts[grid_pos] = self.visit_counts.get(grid_pos, 0) + 1

        if success is not None:
            if success:
                self.success_counts[grid_pos] = self.success_counts.get(grid_pos, 0) + 1
            else:
                self.failure_counts[grid_pos] = self.failure_counts.get(grid_pos, 0) + 1

        if episode is not None:
            self.last_visit_time[grid_pos] = episode

        score = self._compute_reachability(grid_pos)
        self.reachability_scores[grid_pos] = score
        return score

    def _compute_reachability(self, grid_pos: tuple) -> float:
        visits = self.visit_counts.get(grid_pos, 0)
        successes = self.success_counts.get(grid_pos, 0)
        failures = self.failure_counts.get(grid_pos, 0)

        if visits < self.min_samples:
            return 0.5

        total_attempts = successes + failures
        if total_attempts == 0:
            return 0.5

        return float(np.clip(successes / total_attempts, 0.0, 1.0))

    def get_reachability(self, position: np.ndarray) -> float:
        grid_pos = self._discretize_position(position)
        return self.reachability_scores.get(grid_pos, 0.5)

    def get_confidence(self, position: np.ndarray) -> float:
        grid_pos = self._discretize_position(position)
        visits = self.visit_counts.get(grid_pos, 0)
        return min(visits / self.min_samples, 1.0)

    def decay_scores(self, current_episode: int, decay_factor: float = None):
        if decay_factor is None:
            decay_factor = self.decay

        for grid_pos, score in self.reachability_scores.items():
            last_visit = self.last_visit_time.get(grid_pos, current_episode)
            episodes_since = current_episode - last_visit
            self.reachability_scores[grid_pos] = score * (decay_factor ** episodes_since)

    def reset(self):
        self.visit_counts.clear()
        self.success_counts.clear()
        self.failure_counts.clear()
        self.reachability_scores.clear()
        self.last_visit_time.clear()

    def get_statistics(self) -> Dict[str, Any]:
        total_positions = len(self.visit_counts)
        if total_positions == 0:
            return {'total_positions': 0, 'avg_visits': 0, 'avg_reachability': 0.5}

        avg_visits = float(np.mean(list(self.visit_counts.values())))
        avg_reach = float(np.mean(list(self.reachability_scores.values()))) if self.reachability_scores else 0.5

        return {
            'total_positions': total_positions,
            'avg_visits': avg_visits,
            'avg_reachability': avg_reach,
            'high_reachability_count': sum(1 for s in self.reachability_scores.values() if s > 0.7),
            'low_reachability_count': sum(1 for s in self.reachability_scores.values() if s < 0.3),
        }
