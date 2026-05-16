"""
Dice Loss and Combined Loss for Mars Ripple Segmentation
========================================================

- DiceLoss: Directly optimizes the IoU/Dice metric for segmentation.
  Formula: L_dice = 1 - (2 * |X∩Y| + eps) / (|X| + |Y| + eps)

- CombinedLoss: Combines CrossEntropy + Dice for stable training.
  Leverages CE's easy gradient flow and Dice's metric-aligned optimization.

Usage:
------
    from loss_functions import DiceLoss, CombinedLoss

    # Dice only
    dice_loss = DiceLoss()

    # CE + Dice (recommended)
    combined_loss = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class semantic segmentation.

    Directly optimizes the Dice coefficient (F1-related), which is more
    aligned with the mIoU evaluation metric than cross-entropy.

    The loss is computed as:
        L_dice = 1 - mean_over_classes(2 * intersection / (pred + target))

    Args:
        smooth: Smoothing constant to prevent division by zero. Default: 1e-6
        ignore_index: Class to exclude from loss. Default: None (no exclusion)

    Note:
        Unlike CE-based losses, DiceLoss benefits from softmax-normalized
        probabilities and is scale-sensitive — good for handling class imbalance.
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  Model output, shape [B, num_classes, H, W]
            targets: Ground truth labels, shape [B, H, W], values are class indices

        Returns:
            Dice loss value (scalar, lower is better)
        """
        num_classes = logits.size(1)

        # Convert targets to one-hot: [B, H, W] -> [B, num_classes, H, W]
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Optionally mask ignored class
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # Dice per class: 2 * |pred ∩ target| / (|pred| + |target|)
        dims = (0, 2, 3)  # sum over batch, height, width dimensions
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality  = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Mean over all classes
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy + Dice Loss.

    Combines the benefits of both losses:
    - CrossEntropy: provides stable gradients, handles class imbalance via weights
    - DiceLoss: directly optimizes the segmentation metric (mIoU)

    Formula:
        L_total = ce_weight * CE(logits, targets) + dice_weight * Dice(logits, targets)

    Args:
        ce_weight: Weight for cross-entropy component. Default: 0.5
        dice_weight: Weight for dice component. Default: 0.5
        ignore_index: Class to ignore in CE. Default: 0 (Masked)
        class_weights: Per-class weights for CE, shape [num_classes]. Default: None
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = 0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  Model output, shape [B, num_classes, H, W]
            targets: Ground truth labels, shape [B, H, W]

        Returns:
            Combined loss value
        """
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_function(
    loss_type: str = "combined",
    ignore_index: int = 0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory function to create a loss function by name.

    Args:
        loss_type: "ce" | "dice" | "combined"
        ignore_index: Class to ignore. Default: 0
        class_weights: Per-class weights for CE. Default: None

    Returns:
        nn.Module loss function

    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type.lower() == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    elif loss_type.lower() == "dice":
        return DiceLoss()
    elif loss_type.lower() == "combined":
        return CombinedLoss(ignore_index=ignore_index, class_weights=class_weights)
    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. Use 'ce', 'dice', or 'combined'."
        )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size, num_classes, H, W = 4, 4, 64, 64
    logits  = torch.randn(batch_size, num_classes, H, W)
    targets = torch.randint(0, 4, (batch_size, H, W))

    print(f"\n[Test 1] CrossEntropyLoss:")
    ce = nn.CrossEntropyLoss(ignore_index=0)
    print(f"  Loss: {ce(logits, targets).item():.4f}")

    print(f"\n[Test 2] DiceLoss:")
    dice = DiceLoss()
    print(f"  Loss: {dice(logits, targets).item():.4f}")

    print(f"\n[Test 3] CombinedLoss (0.5/0.5):")
    combined = CombinedLoss(ignore_index=0)
    print(f"  Loss: {combined(logits, targets).item():.4f}")

    print(f"\n[Test 4] Factory function:")
    loss_fn = get_loss_function("combined")
    print(f"  Type: {type(loss_fn).__name__}")

    print("\n[OK] All loss tests passed!")
