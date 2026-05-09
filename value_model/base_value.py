"""
Base data value model for assessing data collection value.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class DataValueConfig:
    """Configuration for data value assessment."""

    # Value component weights
    w_spatial_rarity: float = 0.3
    w_temporal_freshness: float = 0.15
    w_scene_diversity: float = 0.2
    w_quality: float = 0.15
    w_coverage: float = 0.2

    # Decay parameters
    temporal_decay_rate: float = 0.01
    spatial_decay_sigma: float = 2.0

    # Quality factors
    base_quality: float = 1.0

    # Resolution
    grid_resolution: float = 0.5


@dataclass
class DataValueResult:
    """Result of data value assessment."""

    total_value: float
    spatial_rarity: float = 0.0
    temporal_freshness: float = 0.0
    scene_diversity: float = 0.0
    quality: float = 0.0
    coverage: float = 0.0

    # Metadata
    position: Optional[Tuple[float, float]] = None
    scene_type: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_value': self.total_value,
            'spatial_rarity': self.spatial_rarity,
            'temporal_freshness': self.temporal_freshness,
            'scene_diversity': self.scene_diversity,
            'quality': self.quality,
            'coverage': self.coverage,
            'position': self.position,
            'scene_type': self.scene_type,
            'timestamp': self.timestamp,
        }


class BaseDataValueModel(ABC):
    """
    Abstract base class for data value models.
    Assesses the value of collected data based on multiple factors.
    """

    def __init__(self, config: Optional[DataValueConfig] = None):
        """
        Initialize the data value model.

        Args:
            config: Data value configuration
        """
        self.config = config or DataValueConfig()

        # Tracking maps
        self.visit_count: Optional[np.ndarray] = None
        self.visit_times: Optional[np.ndarray] = None
        self.scene_visits: Optional[Dict[int, int]] = None

        # Current time
        self.current_time: float = 0.0

    @abstractmethod
    def evaluate_location(
        self,
        x: float,
        y: float,
        **kwargs
    ) -> DataValueResult:
        """
        Evaluate data value at a specific location.

        Args:
            x: X coordinate
            y: Y coordinate
            **kwargs: Additional parameters (scene_type, etc.)

        Returns:
            DataValueResult with value components
        """
        pass

    @abstractmethod
    def evaluate_batch(
        self,
        positions: np.ndarray,
        **kwargs
    ) -> List[DataValueResult]:
        """
        Evaluate data value for multiple positions.

        Args:
            positions: Array of positions (N x 2)
            **kwargs: Additional parameters

        Returns:
            List of DataValueResult
        """
        pass

    def initialize_maps(self, width: int, height: int):
        """
        Initialize tracking maps for the given environment size.

        Args:
            width: Environment width
            height: Environment height
        """
        self.visit_count = np.zeros((height, width), dtype=int)
        self.visit_times = np.zeros((height, width), dtype=float)
        self.scene_visits = {}

    def update_visit(self, x: int, y: int, scene_type: int = 0):
        """
        Update visit tracking for a location.

        Args:
            x: X coordinate
            y: Y coordinate
            scene_type: Scene type at location
        """
        if self.visit_count is None:
            return

        # Check bounds
        if 0 <= y < self.visit_count.shape[0] and 0 <= x < self.visit_count.shape[1]:
            self.visit_count[y, x] += 1
            self.visit_times[y, x] = self.current_time

        # Update scene visits
        if scene_type not in self.scene_visits:
            self.scene_visits[scene_type] = 0
        self.scene_visits[scene_type] += 1

    def advance_time(self, dt: float = 0.1):
        """
        Advance the current time.

        Args:
            dt: Time increment
        """
        self.current_time += dt

    def reset_tracking(self):
        """Reset all tracking maps."""
        if self.visit_count is not None:
            self.visit_count.fill(0)
        if self.visit_times is not None:
            self.visit_times.fill(0)
        self.scene_visits = {}
        self.current_time = 0.0

    def _compute_spatial_rarity(
        self,
        x: float,
        y: float,
        visit_count: np.ndarray
    ) -> float:
        """
        Compute spatial rarity component.

        Args:
            x: X coordinate
            y: Y coordinate
            visit_count: Visit count map

        Returns:
            Spatial rarity score [0, 1]
        """
        xi, yi = int(x), int(y)

        # Check bounds
        if visit_count is None or not (0 <= yi < visit_count.shape[0] and 0 <= xi < visit_count.shape[1]):
            return 1.0  # Unknown areas are considered rare

        # Get local visit count
        count = visit_count[yi, xi]

        # Rarity is inversely proportional to visit count
        rarity = 1.0 / (1.0 + count * 0.1)

        return rarity

    def _compute_temporal_freshness(
        self,
        x: float,
        y: float,
        visit_times: np.ndarray
    ) -> float:
        """
        Compute temporal freshness component.

        Args:
            x: X coordinate
            y: Y coordinate
            visit_times: Last visit time map

        Returns:
            Temporal freshness score [0, 1]
        """
        xi, yi = int(x), int(y)

        # Check bounds
        if visit_times is None or not (0 <= yi < visit_times.shape[0] and 0 <= xi < visit_times.shape[1]):
            return 1.0  # Unvisited areas are fresh

        # Get last visit time
        last_visit = visit_times[yi, xi]

        if last_visit == 0:
            return 1.0  # Never visited

        # Freshness decays with time
        time_since_visit = self.current_time - last_visit
        freshness = np.exp(-self.config.temporal_decay_rate * time_since_visit)

        return freshness

    def _compute_scene_diversity(
        self,
        scene_type: int,
        scene_visits: Dict[int, int]
    ) -> float:
        """
        Compute scene diversity component.

        Args:
            scene_type: Current scene type
            scene_visits: Dictionary of scene visit counts

        Returns:
            Scene diversity score [0, 1]
        """
        if scene_visits is None or len(scene_visits) == 0:
            return 0.0

        # Diversity is inversely related to how often this scene has been visited
        total_visits = sum(scene_visits.values())
        scene_count = scene_visits.get(scene_type, 0)

        if total_visits == 0:
            return 0.0

        # Compute diversity as entropy-like measure
        proportion = scene_count / total_visits
        diversity = 1.0 - proportion

        return diversity

    def _compute_coverage(self, visit_count: np.ndarray) -> float:
        """
        Compute overall coverage component.

        Args:
            visit_count: Visit count map

        Returns:
            Coverage ratio [0, 1]
        """
        if visit_count is None:
            return 0.0

        total_cells = visit_count.size
        visited_cells = np.sum(visit_count > 0)

        return visited_cells / max(1, total_cells)

    def _compute_quality(self, **kwargs) -> float:
        """
        Compute data quality component.

        Args:
            **kwargs: Quality factors

        Returns:
            Quality score [0, 1]
        """
        # Base quality
        quality = self.config.base_quality

        # Apply modifiers if provided
        if 'lighting' in kwargs:
            quality *= kwargs['lighting']

        if 'noise_level' in kwargs:
            quality *= (1.0 - kwargs['noise_level'])

        return np.clip(quality, 0.0, 1.0)
