from __future__ import annotations

import torch
import torch.nn.functional as F


class LatentCenter:
    def __init__(
        self,
        dim: int,
        device: str = "cpu",
        momentum: float = 0.9,
        normalize_latent: bool = False,
    ) -> None:
        self.center = torch.zeros(dim, device=device)
        self.initialized = False
        self.momentum = momentum
        self.normalize_latent = normalize_latent

    def preprocess(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize_latent:
            z = F.normalize(z, dim=1)
        return z

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        z = self.preprocess(z.detach())
        batch_center = z.mean(dim=0)

        if not self.initialized:
            self.center.copy_(batch_center)
            self.initialized = True
            return

        self.center.mul_(self.momentum).add_(batch_center * (1.0 - self.momentum))

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        z = self.preprocess(z)
        center = self.center.unsqueeze(0)
        return torch.mean((z - center) ** 2)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        z = self.preprocess(z)
        center = self.center.unsqueeze(0)
        return torch.mean((z - center) ** 2, dim=1)
