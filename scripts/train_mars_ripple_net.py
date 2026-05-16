"""
Training script for Mars-RippleNet (SAFM Dual-Branch Architecture)
=================================================================

Configuration:
    Model:     Mars-RippleNet (EfficientNet-B0 + SAFM)
    Epochs:    50
    Optimizer: AdamW (lr=1e-4, wd=1e-4)
    Scheduler: CosineAnnealingLR
    Loss:      CombinedLoss (CE=0.5 + Dice=0.5)
    Data:      64×64 patches, train/val/test = 57,600/6,400/400

Usage:
    python scripts/train_mars_ripple_net.py
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_dataloaders
from models.mars_ripple_net import get_mars_ripple_net_model
from loss_functions import CombinedLoss


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    # Paths
    DATA_ROOT = PROJECT_ROOT / "DATA" / "Label" / "Augmented_Final" / "64"
    OUTPUT_DIR = PROJECT_ROOT / "checkpoints"

    # Training
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Model
    ENCODER = "efficientnet-b0"
    NUM_CLASSES = 4
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # SAFM injection: Block 2 (16×16) and Block 3 (8×8)
    SAFM_BLOCK_INDICES = (2, 3)

    # DWR
    DWR_REDUCTION = 4
    DWR_DILATIONS = (1, 3, 5)

    # SPM
    SPM_REDUCTION = 16

    # SAFM fusion channels (None = auto)
    SAFM_FUSION_CHANNELS = None

    # Data
    PATCH_SIZE = 64
    NUM_WORKERS = 4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# IoU computation
# ==============================================================================

def compute_ious(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Compute per-class IoU for a batch."""
    ious = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls  = (targets == cls).float()
        inter = (pred_cls * tgt_cls).sum().item()
        union = pred_cls.sum() + tgt_cls.sum() - inter
        ious[cls] = inter / union if union > 0 else 1.0
    return ious


# ==============================================================================
# Training loop
# ==============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")

    return running_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> tuple[float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_ious = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device).long()

            outputs = model(images)
            loss    = criterion(outputs, masks)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_ious.append(compute_ious(preds, masks, num_classes))

    all_ious = np.array(all_ious)
    return running_loss / len(loader), all_ious.mean(axis=0)


# ==============================================================================
# Main
# ==============================================================================

def main():
    cfg = Config()

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.SEED)

    print("=" * 70)
    print("Mars-RippleNet Training  (SAFM Dual-Branch)")
    print("=" * 70)
    print(f"\nDevice: {cfg.DEVICE}")
    print(f"Data:   {cfg.DATA_ROOT}")
    print(f"Output: {cfg.OUTPUT_DIR}")
    print(f"Epochs: {cfg.EPOCHS} | Batch: {cfg.BATCH_SIZE} | LR: {cfg.LEARNING_RATE}")
    print(f"SAFM at decoder blocks: {cfg.SAFM_BLOCK_INDICES}")

    # Data
    print("\n[Data] Loading...")
    train_loader, val_loader, _ = get_dataloaders(
        data_root=cfg.DATA_ROOT,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        size=cfg.PATCH_SIZE,
    )
    print(f"[Data] Train: {len(train_loader.dataset):,} samples | "
          f"Val: {len(val_loader.dataset):,} samples")

    # Model
    print("\n[Model] Building Mars-RippleNet...")
    model = get_mars_ripple_net_model(
        encoder=cfg.ENCODER,
        encoder_weights=None,
        in_channels=1,
        num_classes=cfg.NUM_CLASSES,
        decoder_channels=cfg.DECODER_CHANNELS,
        safm_block_indices=cfg.SAFM_BLOCK_INDICES,
        dwr_reduction=cfg.DWR_REDUCTION,
        dwr_dilations=cfg.DWR_DILATIONS,
        spm_reduction=cfg.SPM_REDUCTION,
        safm_fusion_channels=cfg.SAFM_FUSION_CHANNELS,
    ).to(cfg.DEVICE)

    n_params = model.num_parameters()
    print(f"[Model] Parameters: {n_params:,} ({n_params / 1e6:.4f} M)")

    # Loss, optimizer, scheduler
    criterion  = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer  = AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                        weight_decay=cfg.WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # Training
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)

    best_val_miou = 0.0
    best_epoch = 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        t_epoch = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_ious = validate(model, val_loader, criterion, cfg.DEVICE, cfg.NUM_CLASSES)
        val_miou = val_ious[1:].mean()  # Exclude Mask (class 0)

        scheduler.step()
        epoch_time = time.time() - t_epoch

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "iou_mask": val_ious[0],
            "iou_straight": val_ious[1],
            "iou_sinuous": val_ious[2],
            "iou_complex": val_ious[3],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"\nEpoch {epoch:3d}/{cfg.EPOCHS} | {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val mIoU:   {val_miou:.4f}  "
              f"(Straight: {val_ious[1]:.4f} | "
              f"Sinuous: {val_ious[2]:.4f} | "
              f"Complex: {val_ious[3]:.4f})")

        # Save best
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            ckpt_path = cfg.OUTPUT_DIR / "mars_ripple_net_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "val_ious": val_ious.tolist(),
            }, ckpt_path)
            print(f"  ★ Best model saved (mIoU: {val_miou:.4f})")

    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Best epoch: {best_epoch} | Best val mIoU: {best_val_miou:.4f}")

    # Save history
    hist_path = cfg.OUTPUT_DIR / "training_history.csv"
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_miou",
                        "iou_mask", "iou_straight", "iou_sinuous", "iou_complex", "lr"]
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"\nHistory saved: {hist_path}")
    print("\nNext: python scripts/evaluate_final.py")


if __name__ == "__main__":
    main()
