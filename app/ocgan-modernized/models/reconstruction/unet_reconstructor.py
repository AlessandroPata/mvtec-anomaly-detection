"""
U-Net style decoder with skip connections from the encoder backbone.

Instead of reconstructing from a 128-dim latent vector alone, this decoder
receives multi-scale feature maps from the frozen ResNet backbone and fuses
them at each upsampling stage, preserving spatial detail that the bottleneck
would otherwise destroy.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(x + self.block(x))


class SkipFusionBlock(nn.Module):
    """Fuse decoder features with encoder skip connection via concatenation."""

    def __init__(self, decoder_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_ch + out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch),
        )

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        skip_proj = self.skip_proj(skip_feat)
        # Align spatial dims to skip (encoder) resolution
        if decoder_feat.shape[2:] != skip_proj.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, size=skip_proj.shape[2:],
                mode="bilinear", align_corners=False,
            )
        return self.fuse(torch.cat([decoder_feat, skip_proj], dim=1))


class UNetReconstructor(nn.Module):
    """
    Decoder that combines a latent vector with multi-scale encoder features.

    ResNet-50 on 256x256 input produces:
        layer4: 8x8,   2048 ch
        layer3: 16x16,  1024 ch
        layer2: 32x32,  512 ch
        layer1: 64x64,  256 ch

    Architecture:
        latent (128) -> FC -> 256x8x8
        fuse4: 8x8    + layer4 (2048) -> 256x8x8
        up -> 16x16,  fuse3 + layer3 (1024) -> 128x16x16
        up -> 32x32,  fuse2 + layer2 (512)  -> 64x32x32
        up -> 64x64,  fuse1 + layer1 (256)  -> 32x64x64
        up -> 128x128, conv -> 32x128x128
        up -> 256x256, final conv -> 3x256x256
    """

    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        base_channels: int = 256,
        output_size: int = 256,
        skip_channels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.output_size = output_size

        if skip_channels is None:
            skip_channels = {
                "layer4": 2048,
                "layer3": 1024,
                "layer2": 512,
                "layer1": 256,
            }

        ch = base_channels  # 256

        self.fc = nn.Linear(latent_dim, ch * 8 * 8)
        self.initial_block = ResidualBlock(ch)

        # At 8x8: fuse with layer4 (also 8x8)
        self.fuse4 = SkipFusionBlock(ch, skip_channels["layer4"], ch)  # -> 256x8x8

        # Upsample to 16x16, fuse with layer3
        self.fuse3 = SkipFusionBlock(ch, skip_channels["layer3"], ch // 2)  # -> 128x16x16

        # Upsample to 32x32, fuse with layer2
        self.fuse2 = SkipFusionBlock(ch // 2, skip_channels["layer2"], ch // 4)  # -> 64x32x32

        # Upsample to 64x64, fuse with layer1
        self.fuse1 = SkipFusionBlock(ch // 4, skip_channels["layer1"], ch // 8)  # -> 32x64x64

        # Upsample 64 -> 128 -> 256, final output
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 8, ch // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch // 8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Fallback decoder for inference without skip connections
        self.fallback_decoder = nn.Sequential(
            ResidualBlock(ch),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(ch // 2),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 2, ch // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch // 4),
            nn.ReLU(inplace=True),
            ResidualBlock(ch // 4),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 4, ch // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch // 8),
            nn.ReLU(inplace=True),
            ResidualBlock(ch // 8),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 8, ch // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch // 8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        print("[Reconstructor] type=unet_reconstructor (skip connections enabled)")

    def forward(
        self,
        z: torch.Tensor,
        skip_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.shape[0], -1, 8, 8)  # [B, 256, 8, 8]

        if skip_features is not None:
            x = self.initial_block(x)

            # Guard against AMP fp16 overflow in unfrozen backbone corrupting skip features.
            skip_features = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                             for k, v in skip_features.items()}

            # Fuse at 8x8 with layer4
            x = self.fuse4(x, skip_features["layer4"])  # 256x8x8

            # Upsample to 16x16, fuse with layer3
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.fuse3(x, skip_features["layer3"])  # 128x16x16

            # Upsample to 32x32, fuse with layer2
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.fuse2(x, skip_features["layer2"])  # 64x32x32

            # Upsample to 64x64, fuse with layer1
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.fuse1(x, skip_features["layer1"])  # 32x64x64

            # 64 -> 128 -> 256, final conv
            x = self.up_block(x)  # 3x256x256
        else:
            x = self.fallback_decoder(x)

        return x
