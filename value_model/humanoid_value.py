"""
Humanoid data value model for 4-dimensional value assessment.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from .base_value import BaseDataValueModel, DataValueResult, DataValueConfig
from utils.constants import SceneType, TerrainType


class DataQualityFactors(Enum):
    """Factors affecting data quality."""
    LIGHTING = 0
    MOTION_BLUR = 1
    OCCLUSION = 2
    SENSOR_NOISE = 3
    VIEW_ANGLE = 4


@dataclass
class HumanoidDataValueConfig(DataValueConfig):
    """Extended configuration for humanoid data value."""

    # Humanoid-specific weights
    w_posture_diversity: float = 0.1
    w_locomotion_mode: float = 0.1

    # Quality factor weights
    w_lighting_quality: float = 0.3
    w_motion_quality: float = 0.2
    w_occlusion_quality: float = 0.2
    w_noise_quality: float = 0.2
    w_view_quality: float = 0.1


class HumanoidDataValueModel(BaseDataValueModel):
    """
    4-dimensional data value model for humanoid robot.

    Dimensions:
    1. Spatial Rarity (30%): How rarely the location has been visited
    2. Temporal Freshness (15%): How recent the data is
    3. Scene Diversity (20%): Variety of scenes and environments
    4. Quality (15%): Data quality based on multiple factors
    5. Coverage (20%): Overall exploration percentage

    Additional considerations:
    - Posture diversity: variety of robot poses
    - Locomotion mode: walking, crawling, etc.
    """

    def __init__(
        self,
        config: Optional[HumanoidDataValueConfig] = None,
        width: int = 100,
        height: int = 100,
        grid_resolution: float = 0.5
    ):
        """
        Initialize humanoid data value model.

        Args:
            config: Data value configuration
            width: Environment width
            height: Environment height
            grid_resolution: Meters per grid cell
        """
        if config is None:
            config = HumanoidDataValueConfig()
        super().__init__(config)

        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution

        # Humanoid-specific tracking
        self.posture_visits: Optional[Dict[str, int]] = None
        self.locomotion_visits: Optional[Dict[str, int]] = None
        self.scene_coverage: Optional[Dict[SceneType, float]] = None

        # Quality maps
        self.lighting_map: Optional[np.ndarray] = None
        self.occlusion_map: Optional[np.ndarray] = None

        # Initialize maps
        self.initialize_maps(width, height)

    def initialize_maps(self, width: int, height: int):
        """Initialize tracking maps."""
        super().initialize_maps(width, height)
        self.width = width
        self.height = height

        # Posture and locomotion tracking
        self.posture_visits = {}
        self.locomotion_visits = {}

        # Scene coverage tracking
        self.scene_coverage = {scene: 0.0 for scene in SceneType}

        # Quality maps
        self.lighting_map = np.ones((height, width), dtype=float)
        self.occlusion_map = np.zeros((height, height), dtype=float)

    def evaluate_location(
        self,
        x: float,
        y: float,
        scene_type: SceneType = SceneType.INDOOR_FLAT,
        **kwargs
    ) -> DataValueResult:
        """
        Evaluate 4-dimensional data value at a specific location.

        Args:
            x: X coordinate
            y: Y coordinate
            scene_type: Type of scene
            **kwargs: Additional parameters (lighting, posture, locomotion, etc.)

        Returns:
            DataValueResult with 4D value components
        """
        # Compute main components
        spatial = self._compute_spatial_rarity(x, y, self.visit_count)
        temporal = self._compute_temporal_freshness(x, y, self.visit_times)
        scene = self._compute_scene_diversity_value(scene_type)
        quality = self._compute_humanoid_quality(x, y, **kwargs)
        coverage = self._compute_coverage(self.visit_count)

        # Compute total value
        total = (
            spatial * self.config.w_spatial_rarity +
            temporal * self.config.w_temporal_freshness +
            scene * self.config.w_scene_diversity +
            quality * self.config.w_quality +
            coverage * self.config.w_coverage
        )

        return DataValueResult(
            total_value=total,
            spatial_rarity=spatial,
            temporal_freshness=temporal,
            scene_diversity=scene,
            quality=quality,
            coverage=coverage,
            position=(x, y),
            scene_type=scene_type.name if isinstance(scene_type, SceneType) else str(scene_type),
            timestamp=self.current_time,
        )

    def evaluate_batch(
        self,
        positions: np.ndarray,
        scene_types: Optional[List[SceneType]] = None,
        **kwargs
    ) -> List[DataValueResult]:
        """
        Evaluate data value for multiple positions.

        Args:
            positions: Array of positions (N x 2)
            scene_types: List of scene types for each position
            **kwargs: Additional parameters

        Returns:
            List of DataValueResult
        """
        results = []

        for i, pos in enumerate(positions):
            scene = scene_types[i] if scene_types is not None else SceneType.INDOOR_FLAT
            result = self.evaluate_location(pos[0], pos[1], scene, **kwargs)
            results.append(result)

        return results

    def _compute_scene_diversity_value(self, scene_type: SceneType) -> float:
        """
        Compute scene diversity component for humanoid scenario.

        Args:
            scene_type: Type of scene

        Returns:
            Scene diversity score [0, 1]
        """
        if self.scene_coverage is None:
            return 0.0

        total_coverage = sum(self.scene_coverage.values())

        if total_coverage == 0:
            return 1.0  # First scene to explore

        # Current scene coverage
        scene_cov = self.scene_coverage.get(scene_type, 0.0)

        # Diversity is inversely related to how much of this scene has been explored
        diversity = 1.0 - (scene_cov / total_coverage)

        return diversity

    def _compute_humanoid_quality(self, x: float, y: float, **kwargs) -> float:
        """
        Compute data quality for humanoid scenario.

        Args:
            x: X coordinate
            y: Y coordinate
            **kwargs: Quality factors

        Returns:
            Quality score [0, 1]
        """
        quality = self.config.base_quality

        # Lighting quality
        if 'lighting' in kwargs:
            quality *= kwargs['lighting']

        # Motion blur (from velocity)
        if 'velocity' in kwargs:
            velocity = np.linalg.norm(kwargs['velocity'])
            motion_factor = 1.0 / (1.0 + velocity * 0.5)
            quality *= motion_factor

        # Occlusion
        if 'occlusion' in kwargs:
            quality *= (1.0 - kwargs['occlusion'] * 0.5)

        # Sensor noise
        if 'noise_level' in kwargs:
            quality *= (1.0 - kwargs['noise_level'])

        # View angle (trunk angle affects camera view)
        if 'trunk_angle' in kwargs:
            angle_penalty = abs(kwargs['trunk_angle']) / (np.pi / 2)
            quality *= (1.0 - angle_penalty * 0.3)

        return np.clip(quality, 0.0, 1.0)

    def update_visit(
        self,
        x: int,
        y: int,
        scene_type: SceneType = SceneType.INDOOR_FLAT,
        posture: Optional[str] = None,
        locomotion: Optional[str] = None
    ):
        """
        Update visit tracking for a location.

        Args:
            x: X coordinate
            y: Y coordinate
            scene_type: Type of scene
            posture: Robot posture (standing, crouching, etc.)
            locomotion: Locomotion mode (walking, crawling, etc.)
        """
        super().update_visit(x, y, scene_type.value)

        # Update scene coverage
        if self.scene_coverage is not None:
            self.scene_coverage[scene_type] = self.scene_coverage.get(scene_type, 0.0) + 1.0

        # Update posture tracking
        if posture is not None and self.posture_visits is not None:
            self.posture_visits[posture] = self.posture_visits.get(posture, 0) + 1

        # Update locomotion tracking
        if locomotion is not None and self.locomotion_visits is not None:
            self.locomotion_visits[locomotion] = self.locomotion_visits.get(locomotion, 0) + 1

    def reset_tracking(self):
        """Reset all tracking maps."""
        super().reset_tracking()
        self.posture_visits = {}
        self.locomotion_visits = {}
        if self.scene_coverage is not None:
            for scene in self.scene_coverage:
                self.scene_coverage[scene] = 0.0

    def get_value_heatmap(self) -> Optional[np.ndarray]:
        """
        Get heatmap of total data value across the environment.

        Returns:
            2D array of value scores or None if not initialized
        """
        if self.visit_count is None:
            return None

        heatmap = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                result = self.evaluate_location(x, y)
                heatmap[y, x] = result.total_value

        return heatmap

    def get_scene_coverage_report(self) -> Dict[str, float]:
        """
        Get report of scene coverage.

        Returns:
            Dictionary mapping scene names to coverage ratios
        """
        if self.scene_coverage is None:
            return {}

        total = sum(self.scene_coverage.values())

        if total == 0:
            return {scene.name: 0.0 for scene in SceneType}

        return {
            scene.name: self.scene_coverage[scene] / total
            for scene in SceneType
        }

    def get_high_value_targets(
        self,
        num_targets: int = 10,
        min_value: float = 0.6
    ) -> List[Tuple[float, float, float, str]]:
        """
        Get high-value target locations for data collection.

        Args:
            num_targets: Maximum number of targets to return
            min_value: Minimum value threshold

        Returns:
            List of (x, y, value, scene_type) tuples sorted by value
        """
        if self.visit_count is None:
            return []

        targets = []

        # Sample positions
        step = max(1, min(self.width, self.height) // 20)

        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                # Determine scene type
                scene = self._infer_scene_type(x, y)

                result = self.evaluate_location(x, y, scene)

                if result.total_value >= min_value:
                    targets.append((x, y, result.total_value, scene.name))

        # Sort by value (descending)
        targets.sort(key=lambda t: t[2], reverse=True)

        return targets[:num_targets]

    def _infer_scene_type(self, x: int, y: int) -> SceneType:
        """
        Infer scene type from position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Inferred scene type
        """
        # Simple heuristic: divide into quadrants
        hw, hh = self.width // 2, self.height // 2

        if x < hw:
            if y < hh:
                return SceneType.INDOOR_FLAT
            else:
                return SceneType.INDOOR_STAIRS
        else:
            if y < hh:
                return SceneType.OUTDOOR_FLAT
            else:
                return SceneType.OUTDOOR_ROUGH
