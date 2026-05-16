"""
Mars-RippleNet: Spatial Adaptive Dual-Branch Architecture
========================================================

MarsRippleNet = EfficientNet-B0 encoder + SAFM Dual-Branch Decoder

Key innovation: SAFM (Spatial Adaptive Fusion Module) resolves "feature
interference" between dual branches by learning a spatial gate W(h,w) at
each pixel location:
    - Branch A (SPM): responsible for straight/directional ripple features
    - Branch B (DWR): responsible for sinuous/curvilinear texture features
    - F_out = W ⊙ F_A + (1-W) ⊙ F_B

SAFM injection positions: Decoder Block 2 (16×16) and Block 3 (8×8)
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import modules as md

from .dwr_module import DWRModule
from .spm_module import StripPoolingV2
from .safm_module import SAFM


# =============================================================================
# 1. Dual-Branch Block: SPM || DWR → SAFM
# =============================================================================

class DualExpertSAFMModule(nn.Module):
    """
    Dual-Branch Block with SAFM adaptive fusion.

    Architecture:
        Input X
          ├──→ Branch A: StripPoolingV2 → F_A (straight features)
          └──→ Branch B: DWRModule       → F_B (curvilinear features)
                      ↓
                  SAFM(F_A, F_B) = W⊙F_A + (1-W)⊙F_B
                      ↓
                  + X_residual
                      ↓
        Output Y

    Args:
        in_channels:        Input channel count
        dwr_reduction:       DWR channel compression ratio. Default: 4
        dwr_dilations:      DWR dilation rates. Default: (1, 3, 5)
        spm_reduction:       SPM channel compression ratio. Default: 16
        safm_fusion_channels: SAFM middle channel count. Default: auto
        use_residual:        Add residual connection. Default: True
    """

    def __init__(
        self,
        in_channels: int,
        dwr_reduction: int = 4,
        dwr_dilations: tuple = (1, 3, 5),
        spm_reduction: int = 16,
        safm_fusion_channels: int = None,
        use_residual: bool = True,
    ):
        super().__init__()

        # Branch A: Strip Pooling — captures elongated straight patterns
        self.spm = StripPoolingV2(in_channels=in_channels, reduction=spm_reduction)

        # Branch B: Dilation-wise Residual — captures multi-scale curvilinear texture
        self.dwr = DWRModule(
            in_channels=in_channels,
            reduction=dwr_reduction,
            dilations=dwr_dilations,
            use_residual=False,  # Residual handled by SAFM
        )

        # SAFM: Spatial adaptive fusion
        self.safm = SAFM(
            in_channels=in_channels,
            fusion_channels=safm_fusion_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_A = self.spm(x)   # Branch A: straight/directional
        F_B = self.dwr(x)   # Branch B: curvilinear/texture

        F_fused = self.safm(F_A, F_B)
        return F_fused + x   # Residual connection


# =============================================================================
# 2. Decoder Block with SAFM injection
# =============================================================================

class UnetDecoderBlockWithSAFM(nn.Module):
    """
    U-Net decoder block that conditionally injects DualExpertSAFMModule.

    Forward flow: upsample → concat(skip) → attention →
                  [DualExpertSAFM] → conv1 → conv2 → out
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
        safm_module: Optional[nn.Module] = None,
        dwr_reduction: int = 4,
        dwr_dilations: tuple = (1, 3, 5),
        spm_reduction: int = 16,
        safm_fusion_channels: int = None,
    ):
        super().__init__()

        self.interpolation_mode = interpolation_mode

        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1, use_norm=use_norm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels, out_channels,
            kernel_size=3, padding=1, use_norm=use_norm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

        # SAFM injection
        self.safm_module = safm_module

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Upsample
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )

        # Concat with skip
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)

        # SAFM injection
        if self.safm_module is not None:
            feature_map = self.safm_module(feature_map)

        # Standard convolutions
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        return self.attention2(feature_map)


# =============================================================================
# 3. SAFM-Decoder
# =============================================================================

class UnetDecoderSAFMInject(nn.Module):
    """
    U-Net decoder with optional SAFM injection at specified block indices.

    Only Block 2 (16×16) and Block 3 (8×8) receive DualExpertSAFMModule.
    All other blocks use standard smp-style decoder blocks.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
        safm_block_indices: Sequence[int] = (2, 3),
        dwr_reduction: int = 4,
        dwr_dilations: tuple = (1, 3, 5),
        spm_reduction: int = 16,
        safm_fusion_channels: int = None,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth {n_blocks} != decoder_channels length {len(decoder_channels)}"
            )

        encoder_channels = encoder_channels[1:]  # Drop stem
        encoder_channels = encoder_channels[::-1]  # Reverse: bottom → top

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = md.UnetCenterBlock(head_channels, head_channels, use_norm=use_norm)
        else:
            self.center = nn.Identity()

        # Build decoder blocks
        self.blocks = nn.ModuleList()
        for block_idx, (block_in, block_skip, block_out) in enumerate(
            zip(in_channels, skip_channels, out_channels)
        ):
            safm_mod = None
            if block_idx in safm_block_indices:
                safm_mod = DualExpertSAFMModule(
                    in_channels=block_in + block_skip,
                    dwr_reduction=dwr_reduction,
                    dwr_dilations=dwr_dilations,
                    spm_reduction=spm_reduction,
                    safm_fusion_channels=safm_fusion_channels,
                    use_residual=True,
                )

            block = UnetDecoderBlockWithSAFM(
                block_in,
                block_skip,
                block_out,
                use_norm=use_norm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
                safm_module=safm_mod,
                dwr_reduction=dwr_reduction,
                dwr_dilations=dwr_dilations,
                spm_reduction=spm_reduction,
                safm_fusion_channels=safm_fusion_channels,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features[0] = stem, [1]..[4] = block outputs
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]   # Reverse

        features = features[1:][::-1]           # Remove stem, reverse

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            height, width = spatial_shapes[i + 1]
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip)

        return x


# =============================================================================
# 4. Mars-RippleNet (Main Model)
# =============================================================================

class MarsRippleNet(nn.Module):
    """
    Mars-RippleNet: EfficientNet-B0 encoder + SAFM Dual-Branch Decoder.

    Architecture:
        Encoder: EfficientNet-B0 (5 stages)
        Decoder: 5-block U-Net with SAFM at Block 2 & Block 3

    Key design choices:
        - Encoder frozen from ImageNet pretrained weights
        - SAFM injects DualExpertSAFMModule at 16×16 and 8×8 resolution
        - Dual branches: SPM (straight) || DWR (curvilinear)
        - SAFM resolves branch interference via spatial soft-gating

    Args:
        encoder_name:          Encoder backbone. Default: "efficientnet-b0"
        encoder_weights:       Pretrained weights. Default: None
        in_channels:            Input channels. Default: 1 (grayscale HiRISE)
        classes:               Output classes. Default: 4
        decoder_channels:       Decoder channel progression. Default: (256, 128, 64, 32, 16)
        safm_block_indices:     Where to inject SAFM. Default: (2, 3)
        dwr_reduction:          DWR compression ratio. Default: 4
        dwr_dilations:         DWR dilation rates. Default: (1, 3, 5)
        spm_reduction:          SPM compression ratio. Default: 16
        safm_fusion_channels:  SAFM middle channels. Default: auto

    Example:
        >>> model = MarsRippleNet(encoder_name="efficientnet-b0", in_channels=1, classes=4)
        >>> x = torch.randn(2, 1, 64, 64)
        >>> logits = model(x)  # [2, 4, 64, 64]
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_weights=None,
        in_channels: int = 1,
        classes: int = 4,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        safm_block_indices: Sequence[int] = (2, 3),
        dwr_reduction: int = 4,
        dwr_dilations: tuple = (1, 3, 5),
        spm_reduction: int = 16,
        safm_fusion_channels: int = None,
    ):
        super().__init__()

        # Encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        # SAFM-Decoder
        self.decoder = UnetDecoderSAFMInject(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            safm_block_indices=safm_block_indices,
            dwr_reduction=dwr_reduction,
            dwr_dilations=dwr_dilations,
            spm_reduction=spm_reduction,
            safm_fusion_channels=safm_fusion_channels,
        )

        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_mars_ripple_net_model(
    encoder: str = "efficientnet-b0",
    encoder_weights=None,
    in_channels: int = 1,
    num_classes: int = 4,
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    safm_block_indices: Sequence[int] = (2, 3),
    dwr_reduction: int = 4,
    dwr_dilations: tuple = (1, 3, 5),
    spm_reduction: int = 16,
    safm_fusion_channels: int = None,
) -> MarsRippleNet:
    """Factory function to create a MarsRippleNet."""
    return MarsRippleNet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=decoder_channels,
        safm_block_indices=safm_block_indices,
        dwr_reduction=dwr_reduction,
        dwr_dilations=dwr_dilations,
        spm_reduction=spm_reduction,
        safm_fusion_channels=safm_fusion_channels,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("=" * 70)
    print("Testing Mars-RippleNet (Dual-Branch + SAFM)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_mars_ripple_net_model(
        encoder="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        num_classes=4,
    ).to(device)

    n_params = model.num_parameters()
    print(f"\nParameters: {n_params:,} ({n_params / 1e6:.4f} M)")

    x = torch.randn(2, 1, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")

    # Compare with baseline
    base = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        classes=4,
    )
    n_base = sum(p.numel() for p in base.parameters())
    print(f"\nBaseNet (Efficient-UNet) params: {n_base:,} ({n_base / 1e6:.4f} M)")
    print(f"Overhead: +{(n_params - n_base) / 1e3:.1f}K (+{(n_params - n_base) / n_base * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("[PASS] Mars-RippleNet test passed!")
    print("=" * 70)
