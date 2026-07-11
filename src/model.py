import torch
import torch.nn as nn

import timm


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Load pretrained ViT from timm and replace the classification head.
    All encoder weights start pretrained (ImageNet-21k or ImageNet-1k).
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {model_name}")
    print(f"Params : {total/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")
    return model


def model_size_mb(model: nn.Module) -> float:
    """Return model size in MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e6
