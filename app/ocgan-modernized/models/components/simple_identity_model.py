from __future__ import annotations

import torch
from torch import nn


class SimpleIdentityModel(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
