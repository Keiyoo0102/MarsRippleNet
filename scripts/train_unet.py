"""
Training script for Baseline U-Net
==================================

Standard 4-level U-Net baseline for Mars ripple segmentation.
Reference: Ronneberger et al., MICCAI 2015.

Usage:
    python scripts/train_unet.py
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_dataloaders
from models import create_unet
from loss_functions import CombinedLoss


class Config:
    DATA_ROOT     = PROJECT_ROOT / "DATA" / "Label" / "Augmented_Final" / "64"
    OUTPUT_DIR    = PROJECT_ROOT / "checkpoints"
    EPOCHS        = 50
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    NUM_CLASSES   = 4
    BASE_FILTERS  = 64
    NUM_WORKERS   = 4
    SEED          = 42
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def compute_ious(preds, targets, num_classes):
    ious = np.zeros(num_classes)
    for cls in range(num_classes):
        inter = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        ious[cls] = inter / union if union > 0 else 1.0
    return ious


def main():
    cfg = Config()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.SEED)

    print("=" * 60)
    print("Baseline U-Net Training")
    print("=" * 60)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        cfg.DATA_ROOT, batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS, size=64,
    )
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    # Model
    model = create_unet(
        in_channels=1, num_classes=cfg.NUM_CLASSES,
        base_filters=cfg.BASE_FILTERS,
    ).to(cfg.DEVICE)
    print(f"Parameters: {model.num_parameters():,} ({model.num_parameters()/1e6:.2f}M)")

    criterion  = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer  = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_miou = 0.0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        # Train
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(cfg.DEVICE), masks.to(cfg.DEVICE).long()
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate
        model.eval()
        all_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                preds = model(images.to(cfg.DEVICE)).argmax(dim=1)
                all_ious.append(compute_ious(preds, masks, cfg.NUM_CLASSES))

        val_miou = np.array(all_ious).mean(axis=0)[1:].mean()
        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg.EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_miou": val_miou,
            }, cfg.OUTPUT_DIR / "unet_baseline_best.pth")

        history.append({"epoch": epoch, "train_loss": running_loss/len(train_loader),
                        "val_miou": val_miou, "lr": optimizer.param_groups[0]["lr"]})

    print(f"\nBest val mIoU: {best_miou:.4f} | Time: {(time.time()-t0)/60:.1f} min")

    with open(cfg.OUTPUT_DIR / "unet_history.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch","train_loss","val_miou","lr"]).writerows(history)


if __name__ == "__main__":
    main()
