from __future__ import annotations

from models.backbones.resnet_backbone import ResNetBackbone


def build_backbone(cfg):
    name = cfg.model.backbone.name
    pretrained = bool(cfg.model.backbone.pretrained)
    frozen = bool(cfg.model.backbone.frozen)
    unfreeze_from = str(getattr(cfg.model.backbone, "unfreeze_from", "none"))

    if name in {"resnet50", "wide_resnet50_2"}:
        return ResNetBackbone(
            name=name,
            pretrained=pretrained,
            frozen=frozen,
            unfreeze_from=unfreeze_from,
        )

    raise ValueError(f"Backbone non supportato: {name}")
