from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class RandomRotate90:
    def __call__(self, image: Image.Image) -> Image.Image:
        angle = int(torch.randint(0, 4, (1,)).item()) * 90
        return image.rotate(angle)


def build_clip_image_transform(
    image_size: int = 224,
    is_train: bool = False,
    augment: bool = True,
) -> transforms.Compose:
    steps: list[object] = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    ]
    if is_train and augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotate90(),
            ]
        )
    steps.extend(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=CLIP_IMAGE_MEAN, std=CLIP_IMAGE_STD),
        ]
    )
    return transforms.Compose(steps)


def _import_open_clip():
    try:
        import open_clip  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency depends on local environment
        raise ImportError(
            "open_clip_torch is required for the GeoRSCLIP pipeline. "
            "Install it with `pip install open_clip_torch` and retry."
        ) from exc
    return open_clip


def resolve_clip_pretrained_spec(
    pretrained_path: str | None = None,
    default_pretrained: str | None = None,
) -> str:
    if pretrained_path:
        candidate = Path(pretrained_path)
        if candidate.is_file():
            return str(candidate.resolve())
        looks_like_local_path = candidate.suffix.lower() in {".pt", ".bin", ".ckpt", ".safetensors"} or pretrained_path.startswith(
            ("./", ".\\", "../", "..\\", "/", "\\")
        ) or (":" in pretrained_path[:3])
        if looks_like_local_path and default_pretrained:
            return default_pretrained
        return pretrained_path
    if default_pretrained:
        return default_pretrained
    raise ValueError("Provide either a CLIP pretrained checkpoint path or a default pretrained tag.")


def _is_local_checkpoint_spec(spec: str) -> bool:
    candidate = Path(spec)
    if candidate.is_file():
        return True
    return candidate.suffix.lower() in {".pt", ".bin", ".ckpt", ".safetensors"} or spec.startswith(
        ("./", ".\\", "../", "..\\", "/", "\\")
    ) or (":" in spec[:3])


class CLIPVisualBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained_path: str | None = None,
        default_pretrained: str | None = None,
    ) -> None:
        super().__init__()
        open_clip = _import_open_clip()
        pretrained_spec = resolve_clip_pretrained_spec(pretrained_path, default_pretrained)
        self.model_name = model_name
        self.pretrained_spec = pretrained_spec
        if _is_local_checkpoint_spec(pretrained_spec):
            checkpoint_path = Path(pretrained_spec).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")
            previous_disable = logging.root.manager.disable
            logging.disable(logging.WARNING)
            try:
                self.model = open_clip.create_model(model_name, pretrained=None)
            finally:
                logging.disable(previous_disable)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            msg = self.model.load_state_dict(checkpoint, strict=False)
            if getattr(msg, "missing_keys", None) or getattr(msg, "unexpected_keys", None):
                print(f"Loaded CLIP checkpoint with non-strict match: {msg}")
            print(f"Loaded local CLIP checkpoint: {checkpoint_path}")
        else:
            self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_spec)
        self.visual = self.model.visual
        self.feat_dim = int(getattr(self.model, "text_projection", torch.empty(512, 512)).shape[-1])
        if self.feat_dim <= 0:
            self.feat_dim = 512
        self.patch_hidden_dim = int(getattr(getattr(self.visual, "conv1", None), "weight", torch.empty(768, 3, 32, 32)).shape[0])
        if self.patch_hidden_dim <= 0:
            self.patch_hidden_dim = self.feat_dim
        positional_embedding = getattr(self.visual, "positional_embedding", None)
        self.num_patches = int(positional_embedding.shape[0] - 1) if positional_embedding is not None else 0
        self._patch_features: torch.Tensor | None = None
        self._hook_registered = False
        self._patch_warning_emitted = False
        self._proj = getattr(self.visual, "proj", None)
        self._try_register_hook()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _try_register_hook(self) -> None:
        if hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks"):
            self.visual.transformer.resblocks[-1].register_forward_hook(self._hook_fn)
            self._hook_registered = True
            return
        if hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "blocks"):
            self.visual.trunk.blocks[-1].register_forward_hook(self._hook_fn)
            self._hook_registered = True
            return
        print("WARNING: CLIPVisualBackbone could not register a patch-token hook. Falling back to CLS-only mode.")

    def _hook_fn(self, module, inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            self._patch_features = None
            return
        if self.num_patches > 0:
            seq_len = self.num_patches + 1
            if output.shape[1] == seq_len:
                self._patch_features = output
                return
            if output.shape[0] == seq_len:
                self._patch_features = output.permute(1, 0, 2)
                return
        if output.shape[1] > output.shape[0]:
            self._patch_features = output
        else:
            self._patch_features = output.permute(1, 0, 2)

    def _project_patch_tokens(self) -> torch.Tensor | None:
        if not self._hook_registered or self._patch_features is None:
            return None
        if self._patch_features.ndim != 3 or self._patch_features.shape[1] <= 1:
            return None
        patch_tokens = self._patch_features[:, 1:, :]
        ln_post = getattr(self.visual, "ln_post", None)
        if ln_post is not None:
            patch_tokens = ln_post(patch_tokens)
        proj = self._proj
        if isinstance(proj, (torch.Tensor, nn.Parameter)):
            patch_tokens = patch_tokens @ proj
        elif proj is not None and callable(proj):
            patch_tokens = proj(patch_tokens)
        elif patch_tokens.shape[-1] != self.feat_dim:
            return None
        return F.normalize(patch_tokens, dim=-1)

    def forward(self, images: torch.Tensor, return_patches: bool = False):
        self._patch_features = None
        with torch.inference_mode():
            if hasattr(self.model, "encode_image"):
                cls_features = self.model.encode_image(images, normalize=False)
            else:  # pragma: no cover - open_clip exposes encode_image
                cls_features = self.visual(images)
        cls_features = F.normalize(cls_features, dim=-1)
        if not return_patches:
            self._patch_features = None
            return cls_features

        patch_tokens = self._project_patch_tokens()
        if patch_tokens is None and not self._patch_warning_emitted:
            print("WARNING: Patch tokens are unavailable for this CLIP backbone. Falling back to global attention.")
            self._patch_warning_emitted = True
        self._patch_features = None
        return cls_features, patch_tokens


class CLIPTextEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained_path: str | None = None,
        default_pretrained: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        open_clip = _import_open_clip()
        pretrained_spec = resolve_clip_pretrained_spec(pretrained_path, default_pretrained)
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if _is_local_checkpoint_spec(pretrained_spec):
            checkpoint_path = Path(pretrained_spec).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")
            previous_disable = logging.root.manager.disable
            logging.disable(logging.WARNING)
            try:
                self.model = open_clip.create_model(model_name, pretrained=None)
            finally:
                logging.disable(previous_disable)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            msg = self.model.load_state_dict(checkpoint, strict=False)
            if getattr(msg, "missing_keys", None) or getattr(msg, "unexpected_keys", None):
                print(f"Loaded CLIP checkpoint with non-strict match: {msg}")
            print(f"Loaded local CLIP checkpoint: {checkpoint_path}")
        else:
            self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_spec)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_text(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        features: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                batch_features = self.model.encode_text(tokens)
                batch_features = F.normalize(batch_features, dim=-1)
                features.append(batch_features.cpu())
        return torch.cat(features, dim=0)
