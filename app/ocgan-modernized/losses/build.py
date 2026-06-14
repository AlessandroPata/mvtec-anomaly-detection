from __future__ import annotations

from losses.reconstruction_losses import ReconstructionLoss


def build_reconstruction_loss(cfg):
    return ReconstructionLoss(
        use_l1=bool(cfg.losses.reconstruction.use_l1),
        use_mse=bool(cfg.losses.reconstruction.use_mse),
        use_ms_ssim=bool(cfg.losses.reconstruction.use_ms_ssim),
        l1_weight=float(cfg.losses.reconstruction.l1_weight),
        mse_weight=float(cfg.losses.reconstruction.mse_weight),
        ms_ssim_weight=float(cfg.losses.reconstruction.ms_ssim_weight),
        data_range=float(cfg.losses.reconstruction.data_range),
    )
