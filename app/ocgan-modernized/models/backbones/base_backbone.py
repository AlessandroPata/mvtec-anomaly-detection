from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def out_channels(self) -> dict[str, int]:
        raise NotImplementedError
