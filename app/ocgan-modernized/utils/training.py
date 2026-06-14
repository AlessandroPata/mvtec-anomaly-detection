from __future__ import annotations

from typing import Iterable

import torch


def check_tensor_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contiene NaN o Inf")


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}
