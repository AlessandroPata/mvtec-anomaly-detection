from __future__ import annotations

import torch
from torch import nn

from models.backbones import build_backbone


class FeatureHeadModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg)

        global_dim = self.backbone.out_channels["global"]
        latent_dim = int(cfg.model.latent.dim)

        self.projection = nn.Sequential(
            nn.Linear(global_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        latent = self.projection(feats["global"])
        feats["latent"] = latent
        return feats
