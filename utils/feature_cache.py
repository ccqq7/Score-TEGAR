from __future__ import annotations

import re
from pathlib import Path


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def resolve_clip_cache_tag(
    clip_model: str,
    clip_pretrained: str | None,
    clip_default_pretrained: str | None,
    split_seed: int,
    image_size: int,
) -> str:
    if clip_pretrained:
        pretrained_token = Path(clip_pretrained).stem
    else:
        pretrained_token = clip_default_pretrained or "default"
    return "_".join(
        [
            _sanitize_token(clip_model),
            _sanitize_token(pretrained_token),
            f"seed{int(split_seed)}",
            f"img{int(image_size)}",
        ]
    )


def resolve_feature_cache_dir(
    feature_cache_root: str | Path,
    clip_model: str,
    clip_pretrained: str | None,
    clip_default_pretrained: str | None,
    split_seed: int,
    image_size: int,
) -> Path:
    root = Path(feature_cache_root).resolve()
    tag = resolve_clip_cache_tag(
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        clip_default_pretrained=clip_default_pretrained,
        split_seed=split_seed,
        image_size=image_size,
    )
    return root / tag


def resolve_feature_cache_path(
    feature_cache_root: str | Path,
    split: str,
    clip_model: str,
    clip_pretrained: str | None,
    clip_default_pretrained: str | None,
    split_seed: int,
    image_size: int,
) -> Path:
    return resolve_feature_cache_dir(
        feature_cache_root=feature_cache_root,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        clip_default_pretrained=clip_default_pretrained,
        split_seed=split_seed,
        image_size=image_size,
    ) / f"{split}.pt"
