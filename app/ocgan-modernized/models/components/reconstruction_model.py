from __future__ import annotations

import torch
from torch import nn

from models.backbones import build_backbone
from models.reconstruction import build_reconstructor


class ReconstructionModel(nn.Module):
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

        self.reconstructor = build_reconstructor(cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        latent = self.projection(feats["global"])

        if getattr(self.reconstructor, "uses_skip_connections", False):
            skip_features = {
                "layer1": feats["layer1"],
                "layer2": feats["layer2"],
                "layer3": feats["layer3"],
                "layer4": feats["layer4"],
            }
            reconstruction = self.reconstructor(latent, skip_features=skip_features)
        else:
            reconstruction = self.reconstructor(latent)

        recon_feats = self.backbone(reconstruction)

        feats["latent"] = latent
        feats["reconstruction"] = reconstruction

        feats["recon_layer1"] = recon_feats["layer1"]
        feats["recon_layer2"] = recon_feats["layer2"]
        feats["recon_layer3"] = recon_feats["layer3"]
        feats["recon_layer4"] = recon_feats["layer4"]
        feats["recon_global"] = recon_feats["global"]

        return feats
