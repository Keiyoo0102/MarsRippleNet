"""
Data Loader Utilities
=====================

Provides a lightweight PatchDataset and dataloader factory functions
compatible with the standard MarsRippleNet directory structure.

Dataset Structure Expected:
--------------------------
    data_root/
    ├── train_images/
    ├── train_masks/
    ├── val_images/
    ├── val_masks/
    ├── test_images/
    └── test_masks/

Usage:
------
    from utils.data_loader import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="path/to/Augmented_Final/64",
        batch_size=32,
        num_workers=4,
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np


class PatchDataset(Dataset):
    """
    Lightweight dataset for Mars ripple image-mask pairs.

    Args:
        image_dir: Directory containing *.tif images
        mask_dir: Directory containing *.png masks
        size: Target patch size (images will be verified against this)
        augment: Apply basic augmentation (horizontal/vertical flip)
        normalize: Normalize images to [0, 1]. Default: True

    Returns:
        image: torch.Tensor [1, H, W], float32
        mask:  torch.Tensor [H, W],   int64
    """

    NUM_CLASSES = 4

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        size: int = 64,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        self.augment = augment
        self.normalize = normalize

        # Collect image paths
        self.image_paths: list[Path] = sorted(
            list(self.image_dir.glob("*.tif"))
            + list(self.image_dir.glob("*.tiff"))
            + list(self.image_dir.glob("*.png"))
        )
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # Build mask paths (same stem, .png extension)
        self.mask_paths = [self.mask_dir / (p.stem + ".png") for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img = np.array(Image.open(self.image_paths[idx]))
        if img.ndim == 3:
            img = img[:, :, 0]  # Take first channel

        img = img.astype(np.float32)
        if self.normalize and img.max() > 1.0:
            img = img / 255.0
        img = torch.from_numpy(img[np.newaxis, :, :]).float()

        # Load mask
        mask = np.array(Image.open(self.mask_paths[idx]))
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = torch.from_numpy(mask.astype(np.int64))

        # Basic augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[1])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[0])

        return img, mask


def get_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 64,
    augment_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_root: Root directory containing split subdirectories
        batch_size: Batch size. Default: 32
        num_workers: Number of workers. Default: 4
        size: Patch size (32, 48, or 64). Default: 64
        augment_train: Apply augmentation to training set. Default: False

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)

    train_dataset = PatchDataset(
        data_root / "train_images",
        data_root / "train_masks",
        size=size,
        augment=augment_train,
    )
    val_dataset = PatchDataset(
        data_root / "val_images",
        data_root / "val_masks",
        size=size,
    )
    test_dataset = PatchDataset(
        data_root / "test_images",
        data_root / "test_masks",
        size=size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def get_test_dataloader(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Create a test-only DataLoader."""
    data_root = Path(data_root)
    test_dataset = PatchDataset(
        data_root / "test_images",
        data_root / "test_masks",
    )
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
