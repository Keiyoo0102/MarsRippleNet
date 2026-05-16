"""
Focal Loss for Mars Ripple Segmentation
========================================

Focal Loss addresses class imbalance by down-weighting easy examples
and focusing on hard examples during training.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Formula:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
    p_t = p  if y=1 (correct class)
          1-p otherwise
    gamma (focusing parameter): higher = more focus on hard examples
    alpha_t: class weighting factor

For multi-class segmentation, applied per-pixel with ignore_index=0 (Masked).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class semantic segmentation.

    Reduces the relative loss for well-classified examples (p_t > 0.5)
    and focuses on hard, misclassified examples.

    Args:
        alpha: Per-class weights, shape [num_classes]. None = uniform.
        gamma: Focusing parameter. Default: 2.0.
            - gamma=0: equivalent to cross-entropy
            - gamma=2: standard focal loss
            - gamma>2: increasingly focuses on hard examples
        reduction: "mean" | "sum" | "none". Default: "mean"
        ignore_index: Class to ignore. Default: 0 (Masked region)

    Example:
        >>> loss_fn = FocalLoss(alpha=None, gamma=2.0, ignore_index=0)
        >>> logits = torch.randn(4, 4, 64, 64)   # [B, C, H, W]
        >>> targets = torch.randint(0, 4, (4, 64, 64))  # [B, H, W]
        >>> loss = loss_fn(logits, targets)
        >>> print(loss.item())
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = 0,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits:  Model predictions, shape [B, num_classes, H, W]
            targets: Ground truth labels, shape [B, H, W], values in [0, num_classes)

        Returns:
            Focal loss value
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        # Gather the log-probability of the correct class: [B, H, W]
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        probs = log_probs.exp()                    # [B, C, H, W]
        pt = (probs * targets_one_hot).sum(dim=1)  # [B, H, W], p_t

        # Create mask for ignored pixels
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(targets).float()

        # Apply class weights
        if self.alpha is not None:
            # alpha_t: weight for the target class
            alpha_t = (targets_one_hot * self.alpha.view(1, -1, 1, 1)).sum(dim=1)
        else:
            alpha_t = 1.0

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt).pow(self.gamma)

        # Cross-entropy: -log(p_t)
        ce_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        # Mask ignored pixels
        focal_loss = focal_loss * valid_mask

        if self.reduction == "mean":
            return focal_loss.sum() / (valid_mask.sum() + 1e-8)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing FocalLoss")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size, num_classes, H, W = 4, 4, 64, 64
    logits = torch.randn(batch_size, num_classes, H, W)
    targets = torch.randint(1, 4, (batch_size, H, W))  # No class 0

    loss_fn = FocalLoss(gamma=2.0, ignore_index=0)
    loss = loss_fn(logits, targets)
    print(f"Focal Loss (gamma=2.0): {loss.item():.4f}")

    loss_fn2 = FocalLoss(gamma=0.0, ignore_index=0)
    loss2 = loss_fn2(logits, targets)
    print(f"CE Loss (gamma=0.0):    {loss2.item():.4f}")
    print(f"Ratio (CE/Focal):       {loss2.item()/loss.item():.2f}x")
    print("[OK] FocalLoss test passed!")
