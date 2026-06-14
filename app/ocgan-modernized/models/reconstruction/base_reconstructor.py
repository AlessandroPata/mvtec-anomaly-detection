from __future__ import annotations

import torch
from torch import nn


class BaseReconstructor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        base_channels: int = 256,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 8)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels // 2, base_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels // 4, base_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels // 8, base_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels // 8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.shape[0], -1, 8, 8)
        x = self.decoder(x)
        return x
