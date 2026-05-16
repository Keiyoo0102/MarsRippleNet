"""
Configuration package for MarsRippleNet.
"""

from .default_config import (
    TrainConfig,
    ModelConfig,
    LossConfig,
    DataConfig,
    CLASS_NAMES,
    CLASS_COLORS,
    EVAL_CLASSES,
    PROJECT_ROOT,
    DATA_ROOT,
    OUTPUT_DIR,
)

__all__ = [
    "TrainConfig",
    "ModelConfig",
    "LossConfig",
    "DataConfig",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "EVAL_CLASSES",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "OUTPUT_DIR",
]
