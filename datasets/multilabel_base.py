from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from models.clip_backbone import build_clip_image_transform


def split_indices_random(
    num_samples: int,
    rng: np.random.Generator,
    split_ratio: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"Split ratio must sum to 1.0, got {split_ratio}")
    indices = rng.permutation(int(num_samples))
    n_train = int(num_samples * split_ratio[0])
    n_val = int(num_samples * split_ratio[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


class MultiLabelImageDataset(Dataset):
    def __init__(
        self,
        *,
        samples: list[dict[str, Any]],
        label_names: Sequence[str],
        split: str,
        image_size: int = 224,
        clip_preprocess=None,
        clip_augment_train: bool = True,
        train_fraction: float = 1.0,
        split_seed: int = 42,
        unseen_labels: Sequence[str] | None = None,
        return_metadata: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split}")
        if not (0.0 < float(train_fraction) <= 1.0):
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        if not samples:
            raise RuntimeError(f"No records prepared for split={split}")

        self.samples = list(samples)
        self.label_names = list(label_names)
        self.node_count = int(len(self.label_names))
        self.split = str(split)
        self.return_metadata = bool(return_metadata)
        self.unseen_labels = [label for label in (unseen_labels or []) if label in self.label_names]
        self.unseen_indices = [self.label_names.index(label) for label in self.unseen_labels]
        self.seen_labels = [label for label in self.label_names if label not in self.unseen_labels]

        if split == "train" and float(train_fraction) < 1.0:
            n_total = len(self.samples)
            n_keep = max(1, int(n_total * float(train_fraction)))
            rng_fraction = np.random.default_rng(int(split_seed) + 1000)
            keep_indices = sorted(rng_fraction.choice(n_total, size=n_keep, replace=False).tolist())
            self.samples = [self.samples[idx] for idx in keep_indices]
            print(f"  Train fraction={float(train_fraction):.2f}: {n_keep}/{n_total} samples kept")

        self.transform = clip_preprocess or build_clip_image_transform(
            image_size=int(image_size),
            is_train=(split == "train"),
            augment=bool(clip_augment_train),
        )
        self._image_size = int(image_size)
        self._image_load_warning_count = 0
        self._image_load_warning_limit = 20
        self.allow_image_fallback = True

    def _build_fallback_image_tensor(self, image_path: Path, reason: str) -> torch.Tensor:
        if not self.allow_image_fallback:
            raise RuntimeError(
                f"Failed to load image '{image_path}': {reason}. "
                "Image fallback is disabled for auditable feature-cache generation."
            )
        if self._image_load_warning_count < self._image_load_warning_limit:
            print(f"[WARN] Failed to load image '{image_path}': {reason}. Using a black fallback image.")
        self._image_load_warning_count += 1
        fallback = Image.new("RGB", (self._image_size, self._image_size), color=(0, 0, 0))
        return self.transform(fallback)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        record = self.samples[index]
        image_path = Path(record["image_path"])
        if not image_path.is_file():
            image_tensor = self._build_fallback_image_tensor(image_path, "missing image file")
        else:
            try:
                with Image.open(image_path) as image:
                    image_tensor = self.transform(image.convert("RGB"))
            except (UnidentifiedImageError, OSError) as exc:
                image_tensor = self._build_fallback_image_tensor(image_path, str(exc))

        label_tensor = torch.tensor(record["labels"], dtype=torch.float32)

        if not self.return_metadata:
            return image_tensor, label_tensor

        metadata = {
            "image_path": str(image_path),
            "image_name": str(record.get("image_name", image_path.name)),
            "scene_category": str(record.get("scene_category", "unknown")),
        }
        return image_tensor, label_tensor, metadata

    def get_label_mask(self) -> torch.Tensor:
        mask = torch.ones(self.node_count, dtype=torch.float32)
        for idx in self.unseen_indices:
            mask[idx] = 0.0
        return mask


class FeatureCacheDataset(Dataset):
    def __init__(
        self,
        *,
        feature_cache_path: str | Path,
        unseen_labels: Sequence[str] | None = None,
        return_metadata: bool = False,
    ) -> None:
        cache_path = Path(feature_cache_path).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Feature cache not found: {cache_path}")

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.features = payload["features"].float()
        self.labels = payload["labels"].float()
        self.label_names = list(payload["label_names"])
        self.node_count = int(len(self.label_names))
        self.return_metadata = bool(return_metadata)
        self.unseen_labels = [label for label in (unseen_labels or []) if label in self.label_names]
        self.unseen_indices = [self.label_names.index(label) for label in self.unseen_labels]
        self.seen_labels = [label for label in self.label_names if label not in self.unseen_labels]
        self.split = str(payload.get("split", "unknown"))

        metadata: Mapping[str, Any] = payload.get("metadata", {})
        self.image_paths = list(metadata.get("image_path", []))
        self.image_names = list(metadata.get("image_name", []))
        self.scene_categories = list(metadata.get("scene_category", []))

        if self.features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(self.features.shape)}")
        if self.labels.ndim != 2:
            raise ValueError(f"Expected 2D labels, got shape {tuple(self.labels.shape)}")
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Feature/label count mismatch: {self.features.shape[0]} vs {self.labels.shape[0]}"
            )
        if self.labels.shape[1] != self.node_count:
            raise ValueError(
                f"Label dimension mismatch: {self.labels.shape[1]} vs node_count={self.node_count}"
            )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        feature_tensor = self.features[index]
        label_tensor = self.labels[index]
        if not self.return_metadata:
            return feature_tensor, label_tensor

        metadata = {
            "image_path": self.image_paths[index] if index < len(self.image_paths) else "",
            "image_name": self.image_names[index] if index < len(self.image_names) else f"{index:06d}",
            "scene_category": self.scene_categories[index]
            if index < len(self.scene_categories)
            else "unknown",
        }
        return feature_tensor, label_tensor, metadata

    def get_label_mask(self) -> torch.Tensor:
        mask = torch.ones(self.node_count, dtype=torch.float32)
        for idx in self.unseen_indices:
            mask[idx] = 0.0
        return mask
