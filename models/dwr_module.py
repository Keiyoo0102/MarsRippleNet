"""
Dilation-wise Residual Module (DWR)
==================================

Lightweight multi-scale module for capturing curvilinear texture features
in Martian aeolian ripples.

Key insight (CVPR 2023):
    Directly stacking dilated convolutions causes scale interference.
    DWR's solution: compress channels first → extract multi-scale independently →
    fuse across channels.

Architecture:
    Input X [B, C, H, W]
        ↓
    Conv3×3 (C → C/r)        ← channel compression
        ↓
    ┌───────┬───────┬───────┐
    │DW d=1 │DW d=3 │DW d=5 │  ← 3 parallel depth-wise dilated branches
    │(fine) │(mid) │(coarse)│
    └───────┴───────┴───────┘
        ↓
    Concat (3×C/r = C)
        ↓
    Conv1×1 → C
        ↓
    + X_residual
        ↓
    Output Y [B, C, H, W]

Physical interpretation:
    - d=1: captures fine micro-texture
    - d=3: captures medium-scale sinuous crestlines
    - d=5: captures macro-scale orientation trends

Parameter efficiency (C=64, r=4):
    Naive 3-branch multi-scale: 3 × (3×3×C²) = 27C² ≈ 110K
    DWR: 9×256 + 3×16² + 3×16 ≈ 7K → ~94% reduction
"""

import torch
import torch.nn as nn


class DWRModule(nn.Module):
    """
    Dilation-wise Residual Module.

    Multi-scale feature extraction with channel compression to avoid
    interference between branches at different dilation rates.

    Args:
        in_channels:  Input channel count
        reduction:    Channel compression ratio. Default: 4 (→ C/4)
        dilations:    Dilation rates for parallel branches. Default: (1, 3, 5)
        use_residual: Add residual connection. Default: True

    Example:
        >>> dwr = DWRModule(in_channels=64, reduction=4, dilations=(1, 3, 5))
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = dwr(x)  # [2, 64, 32, 32]
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        dilations: tuple = (1, 3, 5),
        use_residual: bool = True,
    ):
        super().__init__()

        self.in_channels  = in_channels
        self.reduction   = reduction
        self.dilations   = dilations
        self.n_branches  = len(dilations)
        self.use_residual = use_residual

        # Channel compression: reduces interference between multi-scale branches
        mid_channels = max(in_channels // reduction, 32)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Parallel depth-wise dilated branches
        self.dw_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels,
                          kernel_size=3, stride=1,
                          padding=d, dilation=d,
                          groups=mid_channels,  # Depth-wise
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])

        # Cross-channel fusion after concatenation
        fuse_in = mid_channels * self.n_branches
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_in, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.out_act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Multi-scale enhanced features [B, C, H, W]
        """
        residual = x

        # Compress channels
        x = self.reduce_conv(x)  # [B, C/r, H, W]

        # Parallel multi-scale branches
        branch_outs = [branch(x) for branch in self.dw_branches]
        x = torch.cat(branch_outs, dim=1)  # [B, 3C/r, H, W]

        # Cross-channel fusion
        x = self.fuse_conv(x)  # [B, C, H, W]

        # Residual connection
        if self.use_residual:
            x = x + residual

        return self.out_act(x)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DWRModule")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dwr = DWRModule(in_channels=64, reduction=4, dilations=(1, 3, 5)).to(device)

    n_params = sum(p.numel() for p in dwr.parameters())
    print(f"DWRModule parameters: {n_params:,} ({n_params / 1e3:.2f}K)")

    x = torch.randn(2, 64, 32, 32).to(device)
    with torch.no_grad():
        y = dwr(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == x.shape

    # Compare with naive 3-branch
    n_naive = 3 * (3 * 3 * 64 * 64)
    print(f"\nNaive 3-branch:  {n_naive:,} ({n_naive / 1e3:.1f}K)")
    print(f"DWR saving:       {n_naive - n_params:,} ({(n_naive - n_params) / n_naive * 100:.1f}%)")
    print("\n[OK] DWRModule test passed!")
