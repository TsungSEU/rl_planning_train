#!/usr/bin/env python3
"""
Settings and configuration constants for the training system
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Training directories
TRAINING_DIR = PROJECT_ROOT / "training"
CONFIG_DIR = TRAINING_DIR / "config"
MODELS_DIR = TRAINING_DIR / "models"
DATASETS_DIR = TRAINING_DIR / "datasets"
BENCHMARK_DIR = TRAINING_DIR / "benchmark"
UTILS_DIR = TRAINING_DIR / "utils"
RESULTS_DIR = TRAINING_DIR / "results"

# Create directories if they don't exist
for directory in [MODELS_DIR, DATASETS_DIR, BENCHMARK_DIR, UTILS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Default paths
DEFAULT_CONFIG_PATH = CONFIG_DIR / "ppo_train_config.yaml"
DEFAULT_MODEL_PATH = MODELS_DIR / "ppo_weights.pth"

# Environment settings
ENV_WIDTH = 20
ENV_HEIGHT = 20
OBSTACLE_RATIO = 0.2

# Training settings
DEFAULT_EPISODES = 10000
DEFAULT_MAX_STEPS = 200
DEFAULT_LOG_INTERVAL = 100

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import torch for DEVICE setting
import torch