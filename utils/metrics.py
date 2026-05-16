"""
Metrics Module for Mars Ripple Segmentation
==========================================

Provides evaluation metrics for multi-class semantic segmentation:
- mIoU  (mean Intersection over Union)
- F1-Score (macro)
- Precision & Recall (macro)
- Overall Accuracy (OA)
- Per-class IoU, F1, Precision, Recall

Class 0 (Masked) is excluded from all metrics by default.

Usage:
------
    from utils.metrics import SegmentationMetrics

    tracker = SegmentationMetrics(num_classes=4, ignore_index=0)

    for images, masks in val_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        tracker.update(preds, masks)

    metrics = tracker.compute()
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"Straight IoU: {metrics['Straight_IoU']:.4f}")
"""

import numpy as np
import torch
from typing import Dict, Optional
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """
    Accumulated metrics tracker for semantic segmentation.

    Maintains a confusion matrix across all batches and computes
    mIoU, F1, Precision, Recall, and OA at evaluation time.

    Class 0 (Masked) is excluded from all averaged metrics.

    Args:
        num_classes: Total number of classes (including masked). Default: 4
        ignore_index: Class to ignore in metrics computation. Default: 0

    Attributes:
        CLASS_NAMES: Human-readable class names
        EVAL_CLASSES: List of classes included in averaged metrics [1, 2, 3]
    """

    CLASS_NAMES = {0: "Masked", 1: "Straight", 2: "Sinuous", 3: "Complex"}
    EVAL_CLASSES = [1, 2, 3]

    def __init__(self, num_classes: int = 4, ignore_index: int = 0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Clear accumulated confusion matrix."""
        self.confusion_matrix: Optional[np.ndarray] = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate predictions and targets into the confusion matrix.

        Args:
            preds:   Model predictions, shape [B, H, W], values are class indices
            targets: Ground truth labels, shape [B, H, W], values are class indices
        """
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()

        cm = confusion_matrix(
            targets, preds,
            labels=list(range(self.num_classes))
        )
        if self.confusion_matrix is None:
            self.confusion_matrix = cm
        else:
            self.confusion_matrix += cm

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary with macro metrics and per-class breakdowns:
            - mIoU, F1, Precision, Recall, OA (macro averages)
            - Straight_IoU, Straight_F1, Straight_Precision, Straight_Recall
            - Sinuous_IoU, Sinuous_F1, ...
            - Complex_IoU, Complex_F1, ...
        """
        if self.confusion_matrix is None:
            return self._empty_metrics()

        cm = self.confusion_matrix
        metrics: Dict[str, float] = {}

        # Per-class IoU, Precision, Recall, F1
        class_stats = {}
        for cls in self.EVAL_CLASSES:
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            class_stats[cls] = {"IoU": iou, "Precision": prec, "Recall": rec, "F1": f1}

        # Populate per-class entries
        for cls in self.EVAL_CLASSES:
            name = self.CLASS_NAMES[cls]
            for key in ["IoU", "Precision", "Recall", "F1"]:
                metrics[f"{name}_{key}"] = class_stats[cls][key]

        # Macro averages
        metrics["mIoU"]     = np.mean([class_stats[c]["IoU"]       for c in self.EVAL_CLASSES])
        metrics["F1"]       = np.mean([class_stats[c]["F1"]         for c in self.EVAL_CLASSES])
        metrics["Precision"]= np.mean([class_stats[c]["Precision"] for c in self.EVAL_CLASSES])
        metrics["Recall"]   = np.mean([class_stats[c]["Recall"]     for c in self.EVAL_CLASSES])

        # Overall Accuracy
        total_correct = sum(cm[i, i] for i in range(self.num_classes))
        total_pixels  = cm.sum()
        metrics["OA"] = total_correct / total_pixels if total_pixels > 0 else 0.0

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return zero metrics (when no data has been added)."""
        metrics = {"mIoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, "OA": 0.0}
        for cls in self.EVAL_CLASSES:
            name = self.CLASS_NAMES[cls]
            for key in ["IoU", "Precision", "Recall", "F1"]:
                metrics[f"{name}_{key}"] = 0.0
        return metrics

    def save_confusion_matrix(self, save_path: str) -> None:
        """Save the confusion matrix as a CSV file."""
        import pandas as pd
        if self.confusion_matrix is None:
            print("[WARNING] No confusion matrix to save.")
            return
        labels = [self.CLASS_NAMES[i] for i in range(self.num_classes)]
        df = pd.DataFrame(self.confusion_matrix, index=labels, columns=labels)
        df.to_csv(save_path)
        print(f"  [Confusion matrix saved] {save_path}")


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    ignore_index: int = 0,
) -> Dict[str, float]:
    """
    Convenience function: compute metrics for a single batch.

    Args:
        preds:   Predicted class indices, shape [B, H, W]
        targets: Ground truth class indices, shape [B, H, W]
        num_classes: Number of classes. Default: 4
        ignore_index: Class to ignore. Default: 0

    Returns:
        Metrics dictionary (same as SegmentationMetrics.compute())
    """
    tracker = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    tracker.update(preds, targets)
    return tracker.compute()
