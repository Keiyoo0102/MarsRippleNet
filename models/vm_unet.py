"""
Vision Mamba U-Net (VM-UNet) Baseline
=====================================

Pure PyTorch implementation of a U-Net with Visual State Space (VSS) blocks.
No CUDA extensions required (mamba_ssm / causal_conv1d) — Windows compatible.

Core idea: Treat each spatial position as a token in a sequence and use
a selective state-space model (SSM) to capture global context.

Architecture:
    Encoder: ConvBNReLU → VSSBlock × 4 (with MaxPool between stages)
    Bottleneck: ConvBNReLU → VSSBlock
    Decoder: ConvTranspose → Concat → ConvBNReLU → VSSBlock × 4
    Output: 1×1 Conv

Channels: 32 → 64 → 128 → 256 → 256 → 128 → 64 → 32 → 4

Reference: Liu et al., "VM-UNet: Vision Mamba UNet for Medical Image Segmentation",
           arXiv:2402.02491.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Sequence


# =============================================================================
# 1. Selective SSM 1D (S6-Lite)
# =============================================================================

class SelectiveSSM1D(nn.Module):
    """
    Selective State Space Model in 1D (S6-Lite).

    The core SSM operation: computes a hidden state by scanning over a sequence.
    Uses Zero-Order Hold (ZOH) discretization for dt-projection.

    Key innovation vs. linear attention: the A, B, C matrices are
    INPUT-DEPENDENT (selected by x), hence "selective" SSM.

    Args:
        d_model:  Model dimension
        d_state:  SSM state dimension (higher = more expressive, slower). Default: 16
        dt_rank:  Rank of dt projection. Default: auto (d_model // 16)
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model

        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Selective projections (input-dependent selection mechanism)
        self.dt_proj   = nn.Linear(self.dt_rank, d_inner, bias=True)   # dt
        self.B_proj    = nn.Linear(d_model, d_state, bias=False)        # B
        self.C_proj    = nn.Linear(d_model, d_state, bias=False)        # C
        self.x_proj    = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # Log-A matrix (learnable, kept negative for stability)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
        self.A_log = nn.Parameter(A_log)

        # D residual
        self.D = nn.Parameter(torch.ones(d_model))

        nn.init.uniform_(self.dt_proj.bias, -4.6, -0.3)  # ~0.1 after softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] sequence

        Returns:
            y: [B, L, d_model]
        """
        B, L, d = x.shape

        # Compute dt, B, C from input (selective mechanism)
        xz = self.x_proj(x)
        x_dt, x_B, x_C = torch.split(xz, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt  = F.softplus(self.dt_proj(x_dt))              # [B, L, d]
        B_mat = self.B_proj(x_B)                           # [B, L, d_state]
        C_mat = self.C_proj(x_C)                           # [B, L, d_state]

        # Discretize A, B
        A = -torch.exp(self.A_log.float())                 # [d, d_state]
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [B, L, d, d_state]
        dA = torch.exp(dt_A)                                # [B, L, d, d_state]
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)         # [B, L, d, d_state]

        # Sequential scan: h_t = dA_t * h_{t-1} + dB_t * x_t
        h = torch.zeros(B, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class EfficientSSM1D(nn.Module):
    """
    Efficient parallel SSM using Gated Linear Recurrent Unit (GLRU).

    Approximates sequential scan with cumsum + exponential decay:
        h_t = alpha_t * h_{t-1} + (1 - alpha_t) * beta_t * x_t
        y_t = o_t * h_t

    Fully parallelizable (no sequential for-loop), no CUDA extensions needed.

    Args:
        d_model: Model dimension
        d_state: State dimension. Default: 16
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model

        self.alpha_proj = nn.Linear(d_model, d_model, bias=True)   # forget gate
        self.beta_proj  = nn.Linear(d_model, d_model, bias=True)  # input gate
        self.o_proj     = nn.Linear(d_model, d_model, bias=True)   # output gate
        nn.init.constant_(self.alpha_proj.bias, 1.0)                # ~0.9 initial forget

        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha_proj(x))
        beta  = torch.sigmoid(self.beta_proj(x))
        o     = torch.sigmoid(self.o_proj(x))

        log_alpha = torch.log(alpha + 1e-6)
        cumlog_alpha = torch.cumsum(log_alpha, dim=1)
        decay = torch.exp(cumlog_alpha)

        inp = (1 - alpha) * beta * x / (decay + 1e-6)
        h = decay * torch.cumsum(inp, dim=1)

        return o * h + x * self.D.unsqueeze(0).unsqueeze(0)


# =============================================================================
# 2. 2D VSS Block (4-directional scan)
# =============================================================================

class VSSBlock(nn.Module):
    """
    Visual State Space Block — 2D bidirectional scan.

    Scans along 4 directions: →, ←, ↓, ↑
    Outputs are summed and gated before projection.

    Structure:
        x → LayerNorm → expand → 4-dir SSM → merge → proj → + residual
          └──────────────────────────────────────────────────────→┘

    Args:
        dim:       Channel count
        d_state:   SSM state dimension. Default: 16
        expand:    Internal channel expansion. Default: 1
        drop_path: DropPath probability. Default: 0.0
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 1, drop_path: float = 0.0):
        super().__init__()
        d_inner = int(expand * dim)

        self.norm1 = nn.LayerNorm(dim)
        self.in_proj  = nn.Linear(dim, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.act = nn.SiLU()

        self.ssm = EfficientSSM1D(d_inner, d_state=d_state)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _scan(self, x: torch.Tensor, reverse: bool, transpose: bool) -> torch.Tensor:
        """Scan along one direction."""
        if transpose:
            x = x.permute(0, 2, 1, 3)   # [B, W, H, C] — scan along H

        B, H, W, C = x.shape
        seq = x.reshape(B, H * W, C)     # [B, L, C]

        if reverse:
            seq = seq.flip(1)

        y = self.ssm(seq)

        if reverse:
            y = y.flip(1)

        y = y.reshape(B, H, W, C)
        if transpose:
            y = y.permute(0, 2, 1, 3)   # back to [B, H, W, C]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x

        # Token mixer
        x_ln = x.permute(0, 2, 3, 1)          # [B, H, W, C] — channel-last
        x_ln = self.norm1(x_ln)
        xz = self.in_proj(x_ln)
        xi, gate = xz.chunk(2, dim=-1)
        xi   = self.act(xi)
        gate = self.act(gate)

        # 4-directional SSM scan
        y = (
            self._scan(xi, reverse=False, transpose=False)   # →
            + self._scan(xi, reverse=True,  transpose=False)   # ←
            + self._scan(xi, reverse=False, transpose=True)    # ↓
            + self._scan(xi, reverse=True,  transpose=True)    # ↑
        ) * gate

        y = self.out_proj(y).permute(0, 3, 1, 2)   # back to [B, C, H, W]
        x = shortcut + self.drop_path(y)

        # Channel FFN
        x_ffn = x.permute(0, 2, 3, 1)
        x_ffn = self.norm2(x_ffn)
        x_ffn = self.ffn(x_ffn).permute(0, 3, 1, 2)
        return x + self.drop_path(x_ffn)


class DropPath(nn.Module):
    """Stochastic depth (Drop Path) regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * mask / keep


# =============================================================================
# 3. Encoder / Decoder Blocks
# =============================================================================

class ConvBNReLU(nn.Sequential):
    """Conv → BatchNorm → ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class VSSEncoderBlock(nn.Module):
    """ConvBNReLU → VSSBlock (spatial resolution unchanged between stages)."""
    def __init__(self, in_ch: int, out_ch: int, d_state: int = 16):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch)
        self.vss  = VSSBlock(out_ch, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vss(self.conv(x))


class VSSDecoderBlock(nn.Module):
    """Transposed Conv → Concat → ConvBNReLU → VSSBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, d_state: int = 16):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)
        self.vss  = VSSBlock(out_ch, d_state=d_state)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.vss(self.conv(x))


# =============================================================================
# 4. VM-UNet Main Model
# =============================================================================

class VMUNet(nn.Module):
    """
    Vision Mamba U-Net.

    Architecture:
        Input  [B,1,H,W]
          ↓ E1: ConvBNReLU+VSS → [B,32,H,W]
          ↓ pool → [B,32,H/2,W/2]
          ↓ E2: → [B,64,H/2,W/2]
          ↓ pool → [B,64,H/4,W/4]
          ↓ E3: → [B,128,H/4,W/4]
          ↓ pool → [B,128,H/8,W/8]
          ↓ E4: → [B,256,H/8,W/8]
          ↓ pool → [B,256,H/16,W/16]
          ↓ BOT: → [B,256,H/16,W/16]
          ↑ D4: → [B,128,H/8,W/8]
          ↑ D3: → [B,64,H/4,W/4]
          ↑ D2: → [B,32,H/2,W/2]
          ↑ D1: → [B,32,H,W]
          ↓ HEAD: 1×1 → [B,4,H,W]

    Args:
        in_channels:   Input channels. Default: 1
        num_classes:   Output classes. Default: 4
        base_filters:  Filters in first stage. Default: 32
        d_state:       SSM state dimension. Default: 16
    """

    def __init__(
        self,
        in_channels:  int = 1,
        num_classes:  int = 4,
        base_filters: int = 32,
        d_state:      int = 16,
    ):
        super().__init__()
        c = base_filters

        # Encoder
        self.enc1 = VSSEncoderBlock(in_channels, c,     d_state)
        self.enc2 = VSSEncoderBlock(c,     c * 2,  d_state)
        self.enc3 = VSSEncoderBlock(c * 2, c * 4,  d_state)
        self.enc4 = VSSEncoderBlock(c * 4, c * 8,  d_state)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(c * 8, c * 8),
            VSSBlock(c * 8, d_state=d_state),
        )

        # Decoder
        self.dec4 = VSSDecoderBlock(c * 8, c * 8, c * 4, d_state)
        self.dec3 = VSSDecoderBlock(c * 4, c * 4, c * 2, d_state)
        self.dec2 = VSSDecoderBlock(c * 2, c * 2, c,     d_state)
        self.dec1 = VSSDecoderBlock(c,     c,     c,     d_state)

        # Output head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)
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
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                    # [B, 32, H,    W]
        e2 = self.enc2(self.pool(e1))       # [B, 64, H/2,  W/2]
        e3 = self.enc3(self.pool(e2))       # [B, 128, H/4, W/4]
        e4 = self.enc4(self.pool(e3))       # [B, 256, H/8, W/8]
        b  = self.bottleneck(self.pool(e4)) # [B, 256, H/16,W/16]

        d4 = self.dec4(b,  e4)              # [B, 128, H/8, W/8]
        d3 = self.dec3(d4, e3)              # [B,  64, H/4, W/4]
        d2 = self.dec2(d3, e2)              # [B,  32, H/2, W/2]
        d1 = self.dec1(d2, e1)              # [B,  32, H,   W]

        return self.head(d1)                  # [B, 4, H, W]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vmunet(
    in_channels:  int = 1,
    num_classes:  int = 4,
    base_filters: int = 32,
    d_state:      int = 16,
) -> VMUNet:
    """Factory function to create a VMUNet."""
    return VMUNet(in_channels, num_classes, base_filters, d_state)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing VM-UNet (Pure PyTorch, Windows-safe)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_vmunet(in_channels=1, num_classes=4, base_filters=32).to(device)

    n_params = model.num_parameters()
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    x = torch.randn(2, 1, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 4, 64, 64)

    for sz in [32, 48, 64]:
        xi = torch.randn(1, 1, sz, sz, device=device)
        with torch.no_grad():
            oi = model(xi)
        print(f"  {sz}×{sz} → {tuple(oi.shape)}")

    print("\n[OK] VM-UNet test passed!")
