#!/usr/bin/env python3
"""
Setup script for RL Planning Train.

Enables ``pip install -e .`` and the ``rl_train`` CLI command.
"""

from setuptools import setup, find_packages

setup(
    name="rl-planning-train",
    version="0.8.0",
    description="Reinforcement Learning Path Planning Training Platform for Humanoid Robots",
    author="Aurora Planning Team",
    python_requires=">=3.8",
    packages=find_packages(exclude=["archive", "tests", "benchmark", "runs", "models"]),
    install_requires=[
        "click>=7.0",
        "torch>=1.10",
        "numpy",
        "tqdm",
        "pyyaml",
        "matplotlib",
    ],
    extras_require={
        "export": ["onnx", "onnxruntime"],
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "rl_train=rl_planning_train.cli.main:main",
        ],
    },
    # Include non-Python data (config YAML files, etc.)
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)
