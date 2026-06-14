from __future__ import annotations

import torch


def latent_anomaly_score(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    center = center.unsqueeze(0)
    return torch.mean((z - center) ** 2, dim=1)


def pgd_mine_latent(
    latent: torch.Tensor,
    center: torch.Tensor,
    steps: int = 3,
    step_size: float = 0.1,
    noise_std: float = 0.01,
    clamp_value: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z0 = latent.detach()

    if noise_std > 0:
        z = z0 + noise_std * torch.randn_like(z0)
    else:
        z = z0.clone()

    z = z.detach()

    initial_score = latent_anomaly_score(z0, center).detach()

    for _ in range(steps):
        z.requires_grad_(True)
        score = latent_anomaly_score(z, center).mean()
        grad = torch.autograd.grad(score, z, only_inputs=True)[0]

        z = z.detach() + step_size * torch.sign(grad.detach())
        z = torch.clamp(z, min=-clamp_value, max=clamp_value).detach()

    final_score = latent_anomaly_score(z, center).detach()
    return z0, z, initial_score, final_score
