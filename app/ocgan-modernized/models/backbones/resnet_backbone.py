from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, Wide_ResNet50_2_Weights, resnet50, wide_resnet50_2

from models.backbones.base_backbone import BaseBackbone


class ResNetBackbone(BaseBackbone):
    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        frozen: bool = True,
        unfreeze_from: str = "none",
    ) -> None:
        super().__init__()
        self.unfreeze_from = unfreeze_from

        if name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            channels = {
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
                "global": 2048,
            }
        elif name == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            backbone = wide_resnet50_2(weights=weights)
            channels = {
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
                "global": 2048,
            }
        else:
            raise ValueError(f"Backbone non supportato: {name}")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self._out_channels = channels

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        # Selective unfreeze: enable gradient for stages at and after unfreeze_from.
        # "layer3" unfreezes layer3 + layer4; "layer4" unfreezes only layer4.
        self._trainable_modules: list[nn.Module] = []
        if unfreeze_from and unfreeze_from != "none":
            order = ["stem", "layer1", "layer2", "layer3", "layer4"]
            if unfreeze_from not in order:
                raise ValueError(
                    f"unfreeze_from must be one of {order + ['none']}, got {unfreeze_from!r}"
                )
            start = order.index(unfreeze_from)
            for stage_name in order[start:]:
                stage = getattr(self, stage_name)
                for param in stage.parameters():
                    param.requires_grad = True
                self._trainable_modules.append(stage)

    def trainable_parameters(self):
        """Backbone parameters that require grad (possibly empty)."""
        for module in self._trainable_modules:
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    @property
    def out_channels(self) -> dict[str, int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        g = self.avgpool(f4).flatten(1)

        return {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
            "global": g,
        }
