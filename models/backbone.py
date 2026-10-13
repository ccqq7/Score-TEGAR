from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:  # pragma: no cover
    ResNet50_Weights = None


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("TORCH_HOME", str(project_root / ".torch_cache"))
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
        else:  # pragma: no cover
            model = resnet50(pretrained=pretrained)

        # MLRSNet uses RGB inputs, so the default 3-channel stem is already compatible.
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = model.fc.in_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return torch.flatten(features, 1)
