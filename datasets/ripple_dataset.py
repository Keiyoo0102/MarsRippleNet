"""
RippleDataset: PyTorch Dataset for Mars Ripple Segmentation
============================================================

Loads Mars surface sand dune ripple images and their corresponding
segmentation masks as PyTorch tensors for model training/evaluation.

Dataset Structure:
------------------
    data_root/
        train_images/    - Training images (*.tif)
        train_masks/     - Training masks (*.png)
        val_images/      - Validation images
        val_masks/       - Validation masks
        test_images/     - Test images
        test_masks/      - Test masks

Mask Label Definitions:
------------------------
    0 = Masked  - Invalid/masked regions (excluded from metrics)
    1 = Straight - Straight, parallel ripple crests
    2 = Sinuous  - Sinuous, wave-like crestlines
    3 = Complex  - Irregular, multiply-oriented patterns

Example:
--------
    >>> dataset = RippleDataset(
    ...     images_dir="path/to/train_images",
    ...     masks_dir="path/to/train_masks"
    ... )
    >>> image, mask = dataset[0]
    >>> print(image.shape)   # torch.Size([1, H, W])
    >>> print(mask.shape)    # torch.Size([H, W])
    >>> print(torch.unique(mask))  # tensor([0, 1, 2, 3])
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple


class RippleDataset(Dataset):
    """
    PyTorch Dataset for Mars Ripple Segmentation.

    Loads single-channel grayscale images (*.tif) and corresponding
    segmentation masks (*.png) and returns them as PyTorch tensors.

    Args:
        images_dir: Directory containing input images (*.tif, *.tiff, *.png)
        masks_dir: Directory containing segmentation masks (*.png)
        transform: Optional transform to apply to image-mask pairs
        image_extensions: Valid image file extensions

    Attributes:
        NUM_CLASSES: Number of segmentation classes (4)
        LABEL_INFO: Class metadata (name, color for each class)
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files (aligned with images)
    """

    NUM_CLASSES = 4

    LABEL_INFO: Dict[int, Dict] = {
        0: {"name": "Masked",  "color": (128, 128, 128)},
        1: {"name": "Straight", "color": (255, 0,    0)  },
        2: {"name": "Sinuous",  "color": (0,   255, 0)  },
        3: {"name": "Complex",  "color": (0,   0,   255)},
    }

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = (".tif", ".tiff", ".png"),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Collect valid image paths
        self.image_paths: list[Path] = []
        for ext in image_extensions:
            self.image_paths.extend(list(self.images_dir.glob(f"*{ext}")))
        self.image_paths = sorted(self.image_paths)

        # Build aligned mask paths (assumes same filename with .png extension)
        self.mask_paths: list[Path] = []
        for img_path in self.image_paths:
            mask_name = img_path.stem + ".png"
            self.mask_paths.append(self.masks_dir / mask_name)

        self._validate()

    def _validate(self) -> None:
        """Validate that all image-mask pairs exist."""
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}. "
                f"Check the directory path and image extensions."
            )
        for img_path, mask_path in zip(self.image_paths[:3], self.mask_paths[:3]):
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load and preprocess image
        image = Image.open(self.image_paths[idx])
        image = np.array(image)

        if image.ndim == 2:
            # Grayscale: [H, W] -> [1, H, W]
            image = image[np.newaxis, :, :].astype(np.float32)
        elif image.ndim == 3:
            # Multi-channel: take first channel
            image = image[:, :, 0][np.newaxis, :, :].astype(np.float32)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Load and preprocess mask
        mask = Image.open(self.mask_paths[idx])
        mask = np.array(mask)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int64)

        # Apply optional transforms
        if self.transform is not None:
            transformed = self.transform(image=image.squeeze(), mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            if image.ndim == 2:
                image = image[np.newaxis, :, :]

        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def get_sample_info(self, idx: int) -> Dict:
        """Return metadata for a sample without loading pixel data."""
        mask = np.array(Image.open(self.mask_paths[idx]))
        return {
            "index": idx,
            "image_path": str(self.image_paths[idx]),
            "mask_path": str(self.mask_paths[idx]),
            "unique_labels": np.unique(mask).tolist(),
            "image_size": Image.open(self.image_paths[idx]).size,
        }

    def get_class_distribution(self) -> Dict:
        """Calculate pixel-level class distribution across all masks."""
        total_pixels = 0
        class_counts = {i: 0 for i in range(self.NUM_CLASSES)}

        for mask_path in self.mask_paths:
            mask = np.array(Image.open(mask_path))
            for cls in range(self.NUM_CLASSES):
                class_counts[cls] += np.sum(mask == cls)
            total_pixels += mask.size

        return {
            "counts": class_counts,
            "percentages": {
                cls: count / total_pixels * 100
                for cls, count in class_counts.items()
            },
            "total_pixels": total_pixels,
        }


def get_transforms() -> None:
    """
    Placeholder for default transforms.

    Currently returns None (data is pre-augmented offline).
    Extend to include online augmentation with albumentations:

    Example:
        from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=90, p=0.5),
        ])
    """
    return None
