"""
Evaluation script for Mars-RippleNet on the independent test set
================================================================

Evaluates the trained model and reports:
    - mIoU, mF1, Overall Accuracy
    - Per-class IoU, Precision, Recall, F1 (Straight, Sinuous, Complex)

Usage:
    python scripts/evaluate_final.py

The test set is from Scene 6 (independent from training Scenes 1-5).
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_dataloaders
from models.mars_ripple_net import get_mars_ripple_net_model


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    DATA_ROOT = PROJECT_ROOT / "DATA" / "Label" / "Augmented_Final" / "64"
    MODEL_DIR = PROJECT_ROOT / "checkpoints"

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_CLASSES = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model hyperparameters (must match training)
    ENCODER = "efficientnet-b0"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)
    SAFM_BLOCK_INDICES = (2, 3)
    DWR_REDUCTION = 4
    DWR_DILATIONS = (1, 3, 5)
    SPM_REDUCTION = 16


# ==============================================================================
# Metrics
# ==============================================================================

def compute_ious(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute per-class IoU from flat arrays."""
    ious = np.zeros(num_classes)
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum()
        fp = ((preds == cls) & (targets != cls)).sum()
        fn = ((preds != cls) & (targets == cls)).sum()
        denom = tp + fp + fn
        ious[cls] = tp / denom if denom > 0 else 0.0
    return ious


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict:
    """Compute confusion matrix and per-class precision/recall/F1."""
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            conf_matrix[true_cls, pred_cls] = (
                (targets == true_cls) & (preds == pred_cls)
            ).sum()

    precision = np.zeros(num_classes)
    recall    = np.zeros(num_classes)
    f1        = np.zeros(num_classes)

    for cls in range(num_classes):
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp

        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[cls]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[cls] + recall[cls]
        f1[cls] = 2 * precision[cls] * recall[cls] / denom if denom > 0 else 0.0

    return {
        "conf_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> dict:
    """Evaluate model on test set and return all metrics."""
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.numpy())

            if (batch_idx + 1) % 5 == 0:
                print(f"    Batch {batch_idx + 1}/{len(loader)}")

    all_preds   = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    ious  = compute_ious(all_preds, all_targets, num_classes)
    cm_metrics = compute_confusion_matrix(all_preds, all_targets, num_classes)

    # mIoU excluding class 0 (Masked)
    valid_ious = ious[1:]
    mIoU = valid_ious.mean()
    mF1  = cm_metrics["f1"][1:].mean()

    correct = (all_preds == all_targets).sum()
    OA = correct / all_targets.size

    return {
        "ious": ious,
        "mIoU": mIoU,
        "mF1": mF1,
        "OA": OA,
        "precision": cm_metrics["precision"],
        "recall":    cm_metrics["recall"],
        "f1":        cm_metrics["f1"],
        "conf_matrix": cm_metrics["conf_matrix"],
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    cfg = Config()

    print("=" * 70)
    print("Mars-RippleNet Evaluation  (Independent Test Set)")
    print("=" * 70)
    print(f"\nDevice:  {cfg.DEVICE}")
    print(f"Data:    {cfg.DATA_ROOT}")
    print(f"Model:   {cfg.MODEL_DIR}")

    # Load data
    print("\n[Data] Loading test set...")
    _, _, test_loader = get_dataloaders(
        data_root=cfg.DATA_ROOT,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        size=64,
    )
    print(f"[Data] Test: {len(test_loader.dataset):,} samples, {len(test_loader)} batches")

    # Build model
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
    ).to(cfg.DEVICE)

    # Load weights
    ckpt_path = cfg.MODEL_DIR / "mars_ripple_net_best.pth"
    if not ckpt_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {ckpt_path}")
        print("Run training first: python scripts/train_mars_ripple_net.py")
        return

    checkpoint = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Model] Loaded from epoch {checkpoint['epoch']} "
          f"(val mIoU: {checkpoint['val_miou']:.4f})")

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating...")
    print("=" * 70)

    results = evaluate_model(model, test_loader, cfg.DEVICE, cfg.NUM_CLASSES)

    # Print results
    class_names = ["Masked", "Straight", "Sinuous", "Complex"]

    print("\n" + "=" * 70)
    print("Test Set Results (Independent Scene 6)")
    print("=" * 70)

    print(f"\n{'Class':<10} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 46)
    for i in range(cfg.NUM_CLASSES):
        print(f"{class_names[i]:<10} {results['ious'][i]:>8.4f} "
              f"{results['precision'][i]:>10.4f} "
              f"{results['recall'][i]:>8.4f} "
              f"{results['f1'][i]:>8.4f}")

    print("-" * 46)
    print(f"{'mIoU':>10} {results['mIoU']:>8.4f}")
    print(f"{'mF1':>10} {results['mF1']:>8.4f}")
    print(f"{'OA':>10} {results['OA']:>8.4f}")

    # Save results
    out_path = cfg.MODEL_DIR / "test_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "IoU", "Precision", "Recall", "F1"])
        for i in range(cfg.NUM_CLASSES):
            writer.writerow([
                class_names[i],
                f"{results['ious'][i]:.4f}",
                f"{results['precision'][i]:.4f}",
                f"{results['recall'][i]:.4f}",
                f"{results['f1'][i]:.4f}",
            ])
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["mIoU", f"{results['mIoU']:.4f}"])
        writer.writerow(["mF1",  f"{results['mF1']:.4f}"])
        writer.writerow(["OA",   f"{results['OA']:.4f}"])

    print(f"\nResults saved: {out_path}")
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
