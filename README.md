# MarsRippleNet

**Semantic Segmentation of Martian Aeolian Sand Ripple Patterns**

A PyTorch-based deep learning framework for 4-class semantic segmentation of aeolian sand ripples on Mars using HiRISE imagery. Features the novel **Mars-RippleNet** architecture with a **Spatial Adaptive Fusion Module (SAFM)** for dual-branch feature extraction.

---

## Overview

Mars surface images exhibit three distinctive aeolian ripple patterns that encode critical information about wind regimes, sediment transport dynamics, and surface evolution:

| Class | Name | Description |
|:-----:|:-----|:------------|
| 0 | Masked | Invalid/masked regions (excluded from metrics) |
| 1 | Straight | Straight, parallel ripple crests |
| 2 | Sinuous | Sinuous, wave-like crestlines |
| 3 | Complex | Irregular, multiply-oriented patterns |

---

## Architecture

### Mars-RippleNet

The proposed model extends an **EfficientNet-B0** encoder with a **SAFM Dual-Branch Decoder**:

```
Encoder (EfficientNet-B0)
  └── Block 0 (64×64, C=16)
  └── Block 1 (32×32, C=24)
  └── Block 2 (16×16, C=40)   ← SAFM injection
  └── Block 3 ( 8× 8, C=80)   ← SAFM injection
  └── Block 4 ( 4× 4, C=320)

Decoder (U-Net style with SAFM)
  └── Block 0: 256 ch → 128 ch
  └── Block 1: 128 ch → 64 ch
  └── Block 2: 64 ch → 32 ch   ← Dual-Branch SAFM (16×16)
  └── Block 3: 32 ch → 16 ch   ← Dual-Branch SAFM (8×8)
  └── Block 4: 16 ch → num_classes
```

### SAFM: Spatial Adaptive Fusion Module

SAFM resolves **feature interference** between branches designed for different ripple morphologies:

```
Input X
  ├──→ Branch A: Strip Pooling (SPM)    → F_A (straight/directional features)
  └──→ Branch B: Dilation-wise Residual (DWR) → F_B (curvilinear/texture features)
              │
              ↓ concat [F_A, F_B]
              ↓ Conv1×1 → Conv3×3 → Sigmoid
              ↓ W(h,w) ∈ [0,1]
F_out = W ⊙ F_A + (1-W) ⊙ F_B
  └──→ + residual X
Output Y
```

**Key insight**: Straight and sinuous patterns are spatially exclusive. Simple addition (Add) or concatenation (Concat) causes branch interference. SAFM learns a soft gate W(h,w) at each spatial location to weight branch contributions adaptively.

### Baselines

| Model | Description | Params |
|:------|:------------|-------:|
| **Mars-RippleNet (Ours)** | EfficientNet-B0 + SAFM Dual-Branch | 6.40 M |
| Baseline-UNet | Standard 4-level U-Net (base_filters=64) | 31.04 M |
| Baseline-Efficient-UNet | EfficientNet-B0 encoder + U-Net decoder | 6.25 M |
| Baseline-VM-UNet | Vision Mamba U-Net (pure PyTorch SSM) | 3.51 M |

---

## Quick Start

### Installation

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install timm albumentations
pip install scikit-learn pandas pillow
```

### Training

```bash
# Mars-RippleNet (SAFM)
python scripts/train_mars_ripple_net.py

# Baseline U-Net
python scripts/train_unet.py

# VM-UNet (pure PyTorch, no CUDA extensions)
python scripts/train_vmunet.py
```

### Evaluation

```bash
python scripts/evaluate_final.py --model mars_ripple_net
```

---

## Dataset Structure

```
DATA/
└── Label/
    └── Augmented_Final/
        ├── 32/           # 32×32 patches
        ├── 48/           # 48×48 patches
        └── 64/           # 64×64 patches
            ├── train_images/
            ├── train_masks/
            ├── val_images/
            ├── val_masks/
            ├── test_images/
            └── test_masks/
```

---

## Results (Independent Test Set, 64×64)

| Model | Params | mIoU | Straight | Sinuous | Complex |
|:------|-------:|-----:|---------:|--------:|--------:|
| Baseline-UNet | 31.04M | 0.5472 | 0.4449 | 0.3340 | 0.8628 |
| Baseline-Efficient-UNet | 6.25M | 0.6213 | 0.5439 | 0.4304 | 0.8896 |
| Baseline-VM-UNet | 3.51M | 0.5020 | 0.5025 | 0.3055 | 0.6980 |
| **Mars-RippleNet** | **6.40M** | **0.6382** | **0.5756** | **0.4155** | **0.9234** |

**+2.4% mIoU improvement** over the best baseline with only +2.4% additional parameters.

---

## Cross-Scale Robustness

| Model | 32×32 | 48×48 | 64×64 | Variance |
|:------|------:|------:|------:|--------:|
| Baseline-Efficient-UNet | 0.4646 | 0.6045 | 0.6213 | 0.004936 |
| **Mars-RippleNet** | **0.4979** | **0.6284** | **0.6382** | **0.004091** |

Mars-RippleNet is **16.8% more stable** across different patch sizes (lower variance = more robust).

---

## Project Structure

```
MarsRippleNet/
├── configs/           # Configuration files
│   └── default_config.py
├── datasets/          # Data loading
│   ├── ripple_dataset.py
│   └── data_builder.py
├── utils/            # Utilities
│   ├── data_loader.py
│   └── metrics.py
├── loss_functions/   # Loss functions
│   ├── focal_loss.py
│   └── dice_ce_loss.py
├── models/           # Model architectures
│   ├── unet.py              # Baseline U-Net
│   ├── vm_unet.py           # Vision Mamba U-Net
│   ├── safm_module.py       # SAFM core module
│   ├── spm_module.py        # Strip Pooling Module
│   ├── dwr_module.py        # Dilation-wise Residual Module
│   └── mars_ripple_net.py   # Mars-RippleNet (main model)
├── scripts/          # Training & evaluation
│   ├── train_mars_ripple_net.py
│   ├── train_unet.py
│   ├── train_vmunet.py
│   └── evaluate_final.py
└── checkpoints/      # Model weights (after training)
```

---

## License

MIT License
