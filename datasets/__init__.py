"""
MarsRippleNet Datasets Package
================================
Data loading utilities for Mars ripple segmentation.

Exports:
    RippleDataset : Core dataset class for image-mask pairs
    get_dataloaders : Factory function for creating train/val/test loaders
    get_single_loader : Factory function for a single split loader
"""

from .ripple_dataset import RippleDataset, get_transforms
from .data_builder import get_dataloaders, get_single_loader

__all__ = [
    "RippleDataset",
    "get_transforms",
    "get_dataloaders",
    "get_single_loader",
]
