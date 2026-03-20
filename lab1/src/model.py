from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def create_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model: torch.nn.Module, num_blocks: int) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    for block in model.features[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def get_gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.features[-1]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    num_classes: int,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

