"""
Strip Pooling Module (SPM)
==========================

Lightweight attention module for capturing long-range directional context
in Martian aeolian ripples.

Key idea:
    Standard pooling (2×2) misses elongated ripple patterns.
    Strip pooling uses elongated kernels:
        - Horizontal strip: (H, 1) — captures vertical stripe features
        - Vertical strip:   (1, W) — captures horizontal stripe features
    Combined attention gates the original features along ripple directions.

Reference:
    Hou et al., "Strip Pooling: Rethinking Pooling for Scene Parsing",
    CVPR 2020.

Injection point in Mars-RippleNet:
    Decoder Block 2 (16×16) and Block 3 (8×8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPoolingV2(nn.Module):
    """
    Strip Pooling attention module (v2, simplified).

    Captures elongated stripe patterns via horizontal and vertical
    strip pooling, then gates original features with learned attention.

    Process:
        1. Horizontal strip pool: AdaptiveAvgPool2d((H,1)) → Conv1d → expand → [B,C,H,W]
        2. Vertical strip pool:   AdaptiveAvgPool2d((1,W)) → Conv1d → expand → [B,C,H,W]
        3. Fusion: h_feat + v_feat → Conv2d → Sigmoid → attention
        4. Output: input * attention + input (residual connection)

    Args:
        in_channels: Input channel count
        reduction:    Channel compression ratio. Default: 16

    Example:
        >>> spm = StripPoolingV2(in_channels=64, reduction=16)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = spm(x)  # [2, 64, 32, 32]
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        inter_channels = max(in_channels // reduction, 4)

        # Horizontal: pool along H dimension → produce [B, C, H]
        self.h_conv = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # Vertical: pool along W dimension → produce [B, C, W]
        self.v_conv = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # Attention map generator
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Attended features [B, C, H, W] with residual connection
        """
        B, C, H, W = x.shape

        # Horizontal strip pooling: [B,C,H,W] → [B,C,H,1] → [B,C,H]
        h_pool = F.adaptive_avg_pool2d(x, (H, 1)).squeeze(-1)   # [B, C, H]
        h_feat = self.h_conv(h_pool)                            # [B, C', H]
        h_feat = h_feat.unsqueeze(-1).expand(-1, -1, -1, W)     # [B, C', H, W]

        # Vertical strip pooling: [B,C,H,W] → [B,C,1,W] → [B,C,W]
        v_pool = F.adaptive_avg_pool2d(x, (1, W)).squeeze(-2)  # [B, C, W]
        v_feat = self.v_conv(v_pool)                             # [B, C', W]
        v_feat = v_feat.unsqueeze(-2).expand(-1, -1, H, -1)     # [B, C', H, W]

        # Combine and generate attention map
        combined = h_feat + v_feat
        attn = self.fuse_conv(combined)

        # Gated output with residual
        return x * attn + x


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing StripPoolingV2 (SPM)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spm = StripPoolingV2(in_channels=64, reduction=16).to(device)

    x = torch.randn(2, 64, 32, 32).to(device)
    with torch.no_grad():
        y = spm(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == x.shape
    print("\n[OK] SPM test passed!")
