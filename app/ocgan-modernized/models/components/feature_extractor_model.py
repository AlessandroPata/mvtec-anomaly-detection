from __future__ import annotations

import torch
from torch import nn

from models.backbones import build_backbone


class FeatureExtractorModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(x)
