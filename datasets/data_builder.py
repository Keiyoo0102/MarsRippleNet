"""
Data Builder: DataLoader Factory for Mars Ripple Segmentation
============================================================

Factory functions to create PyTorch DataLoaders for train, validation,
and test splits of the Mars Ripple dataset.

Directory Structure Expected:
-----------------------------
    data_root/
    ├── train_images/
    ├── train_masks/
    ├── val_images/
    ├── val_masks/
    ├── test_images/
    └── test_masks/

Usage:
------
    from datasets import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="path/to/Augmented_Final/64",
        batch_size=16,
        num_workers=4,
    )

    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images {images.shape}, masks {masks.shape}")
        break
    # Batch 0: images torch.Size([16, 1, 64, 64]), masks torch.Size([16, 64, 64])
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .ripple_dataset import RippleDataset


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
    size: Optional[int] = None,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    drop_last_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.

    Args:
        data_root: Root directory of the dataset.
            Can point to Augmented_Final/ (if size is specified)
            or Augmented_Final/64/ (direct path).
        batch_size: Number of samples per batch. Default: 8.
        num_workers: Number of worker processes. Default: 2.
            - 0: Main process only (debugging)
            - 2-4: Recommended for single GPU
        size: Image size. If provided and data_root is the parent dir,
            automatically constructs: data_root/{size}/
        pin_memory: Pin CPU memory for faster GPU transfer. Default: True.
        shuffle_train: Shuffle training data. Default: True.
        drop_last_train: Drop last incomplete batch in training. Default: False.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        FileNotFoundError: If required directories do not exist.
    """
    data_root = Path(data_root)

    # Auto-append size directory if it exists
    if size is not None:
        sized_root = data_root / str(size)
        if sized_root.exists():
            data_root = sized_root

    subdirs = {
        "train_images": "train_images",
        "train_masks":  "train_masks",
        "val_images":   "val_images",
        "val_masks":    "val_masks",
        "test_images":  "test_images",
        "test_masks":   "test_masks",
    }
    paths = {key: data_root / subdir for key, subdir in subdirs.items()}

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing directories:\n" +
            "\n".join([f"  - {name}: {paths[name]}" for name in missing])
        )

    print(f"\n{'='*60}")
    print(f"Initializing DataLoaders")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Batch size: {batch_size} | Num workers: {num_workers}")

    # Create datasets
    train_dataset = RippleDataset(paths["train_images"], paths["train_masks"])
    val_dataset   = RippleDataset(paths["val_images"],   paths["val_masks"])
    test_dataset  = RippleDataset(paths["test_images"],  paths["test_masks"])

    print(f"[Train] {len(train_dataset)} samples")
    print(f"[Val]   {len(val_dataset)} samples")
    print(f"[Test]  {len(test_dataset)} samples")

    # Print class distribution
    dist = train_dataset.get_class_distribution()
    print(f"\n[Train] Class distribution:")
    for cls, info in RippleDataset.LABEL_INFO.items():
        pct = dist["percentages"][cls]
        cnt = dist["counts"][cls]
        print(f"  {cls} ({info['name']:>8s}): {pct:6.2f}%  ({cnt:>10,} pixels)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    print(f"\nBatches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader


def get_single_loader(
    data_root: str | Path,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 2,
    size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for a single split (train/val/test).

    Args:
        data_root: Root directory of the dataset.
        split: One of "train", "val", "test".
        batch_size: Batch size. Default: 8.
        num_workers: Number of workers. Default: 2.
        size: Image size (auto-appended if directory exists).
        shuffle: Shuffle data. Default: True.

    Returns:
        DataLoader for the specified split.

    Raises:
        ValueError: If split is not "train", "val", or "test".
    """
    data_root = Path(data_root)

    if size is not None:
        sized_root = data_root / str(size)
        if sized_root.exists():
            data_root = sized_root

    split_map = {
        "train": ("train_images", "train_masks"),
        "val":   ("val_images",   "val_masks"),
        "test":  ("test_images",  "test_masks"),
    }
    if split not in split_map:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(split_map.keys())}")

    images_dir = data_root / split_map[split][0]
    masks_dir  = data_root / split_map[split][1]
    dataset = RippleDataset(images_dir, masks_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


__all__ = ["get_dataloaders", "get_single_loader"]
