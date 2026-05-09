"""
Training utilities: early stopping and metrics tracking.
"""

import re
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

from utils.constants import PlannerMode

logger = logging.getLogger(__name__)


def get_next_exp_dir(base_dir: str = "runs/train", prefix: str = "exp") -> Path:
    """
    Find the next available experiment directory under base_dir.

    Scans for existing {prefix}NNN directories and returns the next number.
    Creates the directory before returning.

    Returns:
        Path to the new experiment directory (e.g. runs/train/exp003)
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    max_num = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for d in base.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))

    exp_dir = base / f"{prefix}{max_num + 1:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


class EarlyStopping:
    """Monitors a metric and stops training when no improvement is observed."""

    def __init__(self, monitor: str = 'avg_reward', patience: int = 10,
                 min_delta: float = 0.01, mode: str = 'max', verbose: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_episode = 0
        self.stopped_episode = 0

        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
        else:
            self.is_better = lambda new, best: new < best - min_delta

    def __call__(self, current_score: float, episode: int) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self.best_episode = episode
            return False

        if self.is_better(current_score, self.best_score):
            if self.verbose:
                logger.info(f"EarlyStopping: {self.monitor} improved from "
                           f"{self.best_score:.4f} to {current_score:.4f} at episode {episode}")
            self.best_score = current_score
            self.best_episode = episode
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_episode = episode
                if self.verbose:
                    logger.info(f"EarlyStopping triggered! No improvement for {self.patience} episodes.")
                return True
            return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.best_episode = 0
        self.stopped_episode = 0


class TrainingMetrics:
    """Tracks and computes training metrics over a sliding window."""

    def __init__(self, window_size: int = 100, mode: PlannerMode = PlannerMode.NAV_DATA):
        self.window_size = window_size
        self.mode = mode
        self.metrics.update({
            'rewards': [],
            'lengths': [],
            'successes': [],
            'actor_losses': [],
            'critic_losses': [],
            'reachability_scores': [],
            'collision_events': [],
        })

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key] = self.metrics[key][-self.window_size:]

    def get_average(self, metric_name: str) -> float:
        values = self.metrics.get(metric_name, [])
        return float(np.mean(values)) if values else 0.0

    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f'avg_{key}'] = float(np.mean(values))
                summary[f'std_{key}'] = float(np.std(values))
                summary[f'last_{key}'] = values[-1]
        return summary

    def reset(self):
        for key in self.metrics:
            self.metrics[key] = []
