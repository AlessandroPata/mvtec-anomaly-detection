from __future__ import annotations

import torch
from torch import nn
from pytorch_msssim import ms_ssim


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        use_l1: bool = True,
        use_mse: bool = False,
        use_ms_ssim: bool = True,
        l1_weight: float = 1.0,
        mse_weight: float = 0.0,
        ms_ssim_weight: float = 1.0,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_l1 = use_l1
        self.use_mse = use_mse
        self.use_ms_ssim = use_ms_ssim

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.data_range = data_range

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        parts: dict[str, torch.Tensor] = {}

        if self.use_l1:
            l1_loss = self.l1(pred, target)
            total = total + self.l1_weight * l1_loss
            parts["l1_loss"] = l1_loss

        if self.use_mse:
            mse_loss = self.mse(pred, target)
            total = total + self.mse_weight * mse_loss
            parts["mse_loss"] = mse_loss

        if self.use_ms_ssim:
            ms_ssim_value = ms_ssim(pred, target, data_range=self.data_range, size_average=True)
            ms_ssim_loss = 1.0 - ms_ssim_value
            total = total + self.ms_ssim_weight * ms_ssim_loss
            parts["ms_ssim_loss"] = ms_ssim_loss

        parts["reconstruction_total"] = total
        return total, parts
