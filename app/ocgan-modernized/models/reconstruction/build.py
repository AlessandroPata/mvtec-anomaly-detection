from __future__ import annotations

from models.reconstruction.base_reconstructor import BaseReconstructor
from models.reconstruction.unet_reconstructor import UNetReconstructor


def build_reconstructor(cfg):
    latent_dim = int(cfg.model.latent.dim)
    output_size = int(cfg.dataset.image_size)

    recon_cfg = cfg.model.reconstruction
    use_skip = bool(getattr(recon_cfg, "use_skip_connections", False))

    if use_skip:
        module = UNetReconstructor(
            latent_dim=latent_dim,
            out_channels=3,
            base_channels=256,
            output_size=output_size,
        )
        module.uses_skip_connections = True
        return module

    module = BaseReconstructor(
        latent_dim=latent_dim,
        out_channels=3,
        base_channels=256,
        output_size=output_size,
    )
    module.uses_skip_connections = False
    return module
