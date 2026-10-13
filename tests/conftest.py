from __future__ import annotations

import importlib.util
import sys
import types


if importlib.util.find_spec("torchvision") is None:
    # Strict-protocol unit tests exercise cached features only.  Keep their
    # imports testable in the lightweight CPU environment without pretending
    # that raw-image torchvision functionality is available.
    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    models = types.ModuleType("torchvision.models")

    class _InterpolationMode:
        BICUBIC = "bicubic"

    class _UnavailableTransform:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torchvision is required for raw-image transforms")

    transforms.InterpolationMode = _InterpolationMode
    for name in (
        "Compose",
        "Resize",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "PILToTensor",
        "ConvertImageDtype",
        "Normalize",
    ):
        setattr(transforms, name, _UnavailableTransform)

    def _unavailable_resnet50(*args, **kwargs):
        raise RuntimeError("torchvision is required for ResNet image models")

    models.resnet50 = _unavailable_resnet50
    models.ResNet50_Weights = None
    torchvision.transforms = transforms
    torchvision.models = models
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.models"] = models
