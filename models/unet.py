"""
Standard U-Net Baseline for Mars Ripple Segmentation
====================================================

A 4-level U-Net baseline for aeolian ripple segmentation.
Reference: Ronneberger et al., "U-Net: Convolutional Networks for
          Biomedical Image Segmentation", MICCAI 2015.

Architecture:
    Encoder: DoubleConv x4 → MaxPool x4 (4 levels)
    Decoder: UpConv x4 → Concat x4 → DoubleConv x4
    Output:  1×1 Conv → num_classes

Input:  [B, 1, H, W]   (single-channel grayscale)
Output: [B, 4, H, W]   (4-class segmentation logits)
"""

import torch
import torch.nn as nn
from typing import Tuple


class DoubleConv(nn.Module):
    """
    Two consecutive (Conv → BatchNorm → ReLU) blocks.

    The basic building unit of U-Net encoder and decoder.

    Args:
        in_ch:  Input channels
        out_ch: Output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool → DoubleConv.

    Halves spatial resolution (H, W) while doubling channels.

    Args:
        in_ch:  Input channels
        out_ch: Output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """
    Upsampling block: Transposed Conv → Concat → DoubleConv.

    Doubles spatial resolution (H, W) and merges with skip features.

    Args:
        in_ch:      Channels from the lower decoder layer
        skip_ch:    Channels from the encoder skip connection
        out_ch:     Output channels
        bilinear:   Use bilinear upsampling instead of transposed conv
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()

        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            in_ch_after_up = in_ch
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            in_ch_after_up = in_ch // 2

        self.conv = DoubleConv(in_ch_after_up + skip_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: from lower layer (needs upsampling)
        # x2: from encoder skip connection
        x1 = self.up(x1)

        # Pad if H/W don't match (MaxPool truncates uneven dimensions)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = torch.nn.functional.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2,
                 diff_h // 2, diff_h - diff_h // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 4-level U-Net.

    Args:
        in_channels:   Number of input channels. Default: 1 (grayscale)
        num_classes:   Number of output classes. Default: 4
        base_filters:  Filters in the first layer. Default: 64
        bilinear:      Use bilinear upsampling in decoder. Default: False

    Example:
        >>> model = UNet(in_channels=1, num_classes=4, base_filters=64)
        >>> x = torch.randn(2, 1, 64, 64)
        >>> logits = model(x)  # [2, 4, 64, 64]
    """

    def __init__(
        self,
        in_channels: int  = 1,
        num_classes: int  = 4,
        base_filters: int  = 64,
        bilinear: bool     = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder path
        self.inc  = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters,      base_filters * 2)
        self.down2 = Down(base_filters * 2,  base_filters * 4)
        self.down3 = Down(base_filters * 4,  base_filters * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)

        # Decoder path
        self.up1 = Up(base_filters * 16, base_filters * 8,  base_filters * 8  // factor, bilinear)
        self.up2 = Up(base_filters * 8,  base_filters * 4,  base_filters * 4  // factor, bilinear)
        self.up3 = Up(base_filters * 4,  base_filters * 2,  base_filters * 2  // factor, bilinear)
        self.up4 = Up(base_filters * 2,  base_filters,      base_filters,        bilinear)

        # Output head
        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)        # [B,  64, H,    W]
        x2 = self.down1(x1)      # [B, 128, H/2,  W/2]
        x3 = self.down2(x2)      # [B, 256, H/4,  W/4]
        x4 = self.down3(x3)      # [B, 512, H/8,  W/8]
        x5 = self.down4(x4)      # [B, 512, H/16, W/16]

        # Decoder
        x = self.up1(x5, x4)    # [B, 256, H/8,  W/8]
        x = self.up2(x,  x3)    # [B, 128, H/4,  W/4]
        x = self.up3(x,  x2)    # [B,  64, H/2,  W/2]
        x = self.up4(x,  x1)    # [B,  64, H,    W]

        return self.outc(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(
    in_channels: int  = 1,
    num_classes: int  = 4,
    base_filters: int  = 64,
    bilinear: bool     = False,
) -> UNet:
    """Factory function to create a UNet."""
    return UNet(in_channels, num_classes, base_filters, bilinear)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("Testing UNet Baseline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_unet(in_channels=1, num_classes=4, base_filters=64).to(device)

    n_params = model.num_parameters()
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    x = torch.randn(2, 1, 64, 64).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 4, 64, 64)
    print("\n[OK] UNet test passed!")
