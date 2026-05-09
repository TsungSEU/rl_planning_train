#!/usr/bin/env python3
"""
Unit tests for value model module.
"""

import sys
import os
import unittest
import numpy as np
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from value_model import (
    BaseDataValueModel, DataValueResult, DataValueConfig,
    HumanoidDataValueModel
)
from utils.constants import SceneType


class TestDataValueConfig(unittest.TestCase):
    """Test DataValueConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataValueConfig()
        self.assertAlmostEqual(config.w_spatial_rarity, 0.3)
        self.assertAlmostEqual(config.w_temporal_freshness, 0.15)
        self.assertAlmostEqual(config.w_scene_diversity, 0.2)

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataValueConfig(
            w_spatial_rarity=0.5,
            w_temporal_freshness=0.2
        )
        self.assertEqual(config.w_spatial_rarity, 0.5)
        self.assertEqual(config.w_temporal_freshness, 0.2)


class TestDataValueResult(unittest.TestCase):
    """Test DataValueResult."""

    def test_result_creation(self):
        """Test creating a result."""
        result = DataValueResult(
            total_value=0.7,
            spatial_rarity=0.8,
            temporal_freshness=0.6,
            scene_diversity=0.5,
            quality=0.9,
            coverage=0.3
        )

        self.assertAlmostEqual(result.total_value, 0.7)

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = DataValueResult(
            total_value=0.7,
            position=(10.0, 20.0),
            scene_type="indoor"
        )

        d = result.to_dict()
        self.assertEqual(d['total_value'], 0.7)
        self.assertEqual(d['position'], (10.0, 20.0))
        self.assertEqual(d['scene_type'], 'indoor')


class TestHumanoidDataValueModel(unittest.TestCase):
    """Test HumanoidDataValueModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = HumanoidDataValueModel(
            width=100,
            height=100,
            grid_resolution=0.5
        )

    def test_initialize_maps(self):
        """Test map initialization."""
        self.assertIsNotNone(self.model.visit_count)
        self.assertEqual(self.model.visit_count.shape, (100, 100))
        self.assertIsNotNone(self.model.scene_coverage)

    def test_evaluate_location(self):
        """Test evaluating a location."""
        result = self.model.evaluate_location(
            x=50.0,
            y=50.0,
            scene_type=SceneType.INDOOR_FLAT
        )

        self.assertIsInstance(result, DataValueResult)
        self.assertGreaterEqual(result.total_value, 0.0)
        self.assertLessEqual(result.total_value, 1.0)

    def test_update_visit(self):
        """Test updating visit tracking."""
        self.model.update_visit(
            x=10,
            y=10,
            scene_type=SceneType.INDOOR_FLAT,
            posture="standing",
            locomotion="walking"
        )

        self.assertEqual(self.model.visit_count[10, 10], 1)
        self.assertEqual(self.model.posture_visits["standing"], 1)
        self.assertEqual(self.model.locomotion_visits["walking"], 1)

    def test_scene_diversity(self):
        """Test scene diversity calculation."""
        # First scene should have high diversity
        result1 = self.model.evaluate_location(
            10.0, 10.0,
            SceneType.INDOOR_FLAT
        )
        diversity1 = result1.scene_diversity

        # After visiting that scene many times, diversity should decrease
        for _ in range(10):
            self.model.update_visit(10, 10, SceneType.INDOOR_FLAT)

        result2 = self.model.evaluate_location(
            10.0, 10.0,
            SceneType.INDOOR_FLAT
        )
        diversity2 = result2.scene_diversity

        # The first visit should have higher or equal diversity
        self.assertGreaterEqual(diversity1, diversity2)

    def test_get_value_heatmap(self):
        """Test getting value heatmap."""
        heatmap = self.model.get_value_heatmap()

        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape, (100, 100))

    def test_get_scene_coverage_report(self):
        """Test scene coverage report."""
        self.model.update_visit(10, 10, SceneType.INDOOR_FLAT)
        self.model.update_visit(20, 20, SceneType.OUTDOOR_ROUGH)

        report = self.model.get_scene_coverage_report()

        self.assertIsInstance(report, dict)
        self.assertIn("INDOOR_FLAT", report)
        self.assertIn("OUTDOOR_ROUGH", report)

    def test_get_high_value_targets(self):
        """Test getting high value targets."""
        targets = self.model.get_high_value_targets(
            num_targets=5,
            min_value=0.0
        )

        self.assertLessEqual(len(targets), 5)
        for target in targets:
            self.assertEqual(len(target), 4)  # (x, y, value, scene_type)


if __name__ == '__main__':
    unittest.main()
