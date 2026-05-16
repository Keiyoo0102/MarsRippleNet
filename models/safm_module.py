"""
SAFM: Spatial Adaptive Fusion Module
===================================

Soft gating fusion for dual-branch feature extraction in Mars-RippleNet.

Key insight:
    Straight and sinuous ripple patterns are spatially exclusive.
    Simple addition (Add) or concatenation (Concat) causes "feature interference"
    — branches for different patterns dilute each other's gradients.

SAFM solution:
    At each spatial location (h, w), learns a soft gate W(h,w) ∈ [0, 1]
    to weight contributions from Branch A (SPM) and Branch B (DWR):

        F_out = W ⊙ F_A + (1 - W) ⊙ F_B

Architecture:
    [F_A, F_B] → Concat → Conv1×1 → Conv3×3 → Sigmoid → W
    F_out = W ⊙ F_A + (1-W) ⊙ F_B
"""

import torch
import torch.nn as nn


class SAFM(nn.Module):
    """
    Spatial Adaptive Fusion Module (SAFM).

    Soft-gating mechanism that adaptively weights two parallel branches
    at each spatial location, solving the feature interference problem.

    Architecture:
        Concat(F_A, F_B) → Conv1×1 → BatchNorm → ReLU
                         → Conv3×3 → BatchNorm → ReLU
                         → Conv1×1 → Sigmoid → W [B, 1, H, W]
        F_fused = W ⊙ F_A + (1-W) ⊙ F_B

    Args:
        in_channels:    Input/output channel count (same for both branches)
        fusion_channels: Middle layer channels. Default: max(in_channels // 4, 16)

    Example:
        >>> safm = SAFM(in_channels=64, fusion_channels=16)
        >>> F_A = torch.randn(2, 64, 32, 32)
        >>> F_B = torch.randn(2, 64, 32, 32)
        >>> F_out = safm(F_A, F_B)  # [2, 64, 32, 32]
    """

    def __init__(
        self,
        in_channels: int,
        fusion_channels: int = None,
    ):
        super().__init__()

        if fusion_channels is None:
            fusion_channels = max(in_channels // 4, 16)

        self.in_channels    = in_channels
        self.fusion_channels = fusion_channels

        # Spatial attention generator
        # Input: [B, 2C, H, W] — concatenated F_A and F_B
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial context extraction
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )

        # Attention mask: sigmoid output ∈ [0, 1]
        self.attention_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        F_A: torch.Tensor,
        F_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            F_A: Branch A output (SPM, straight patterns), shape [B, C, H, W]
            F_B: Branch B output (DWR, curvilinear patterns), shape [B, C, H, W]

        Returns:
            F_fused: Adaptively fused features, shape [B, C, H, W]
        """
        # Concatenate along channel dimension
        concat_feat = torch.cat([F_A, F_B], dim=1)  # [B, 2C, H, W]

        # Generate spatial attention mask W ∈ [0, 1]
        # W → 1: trust F_A (SPM, straight)
        # W → 0: trust F_B (DWR, curvilinear)
        attn = self.reduce_conv(concat_feat)        # [B, C/4, H, W]
        attn = self.spatial_conv(attn)               # [B, C/4, H, W]
        W    = self.attention_conv(attn)             # [B, 1, H, W]

        # Soft-gated fusion
        return W * F_A + (1 - W) * F_B


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing SAFM")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safm = SAFM(in_channels=64, fusion_channels=16).to(device)

    n_params = sum(p.numel() for p in safm.parameters())
    print(f"SAFM parameters: {n_params:,} ({n_params / 1e3:.2f}K)")

    F_A = torch.randn(2, 64, 32, 32).to(device)
    F_B = torch.randn(2, 64, 32, 32).to(device)

    with torch.no_grad():
        F_out = safm(F_A, F_B)

    print(f"F_A:    {F_A.shape}")
    print(f"F_B:    {F_B.shape}")
    print(f"F_out:  {F_out.shape}")
    assert F_out.shape == F_A.shape
    print("\n[OK] SAFM test passed!")
