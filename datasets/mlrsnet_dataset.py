from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from models.clip_backbone import build_clip_image_transform
from utils.kg_builder import KGDefinition
from utils.mlrsnet_io import (
    IMAGE_DIR_CANDIDATES,
    LABEL_DIR_CANDIDATES,
    find_existing_subdir,
    read_scene_csv,
    split_indices_per_scene,
)


class MLRSNetDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        kg_path: str | Path,
        split: str = "train",
        image_size: int = 224,
        clip_preprocess=None,
        clip_augment_train: bool = True,
        split_seed: int = 42,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        train_fraction: float = 1.0,
        unseen_labels: list[str] | None = None,
        return_metadata: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split}")
        if not (0.0 < float(train_fraction) <= 1.0):
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")

        data_root = Path(data_root)
        self.images_dir = find_existing_subdir(data_root, IMAGE_DIR_CANDIDATES)
        self.labels_dir = find_existing_subdir(data_root, LABEL_DIR_CANDIDATES)
        self.return_metadata = return_metadata
        self.split = split

        kg_definition = KGDefinition(kg_path)
        self.label_names: list[str] = list(kg_definition.nodes)
        self.label_to_index: dict[str, int] = dict(kg_definition.label_to_index)
        self.node_count: int = int(kg_definition.node_count)
        self.merged_labels: list[dict[str, str]] = list(kg_definition.payload.get("merged_labels", []))
        self.unseen_labels = [label for label in (unseen_labels or []) if label in self.label_names]
        self.unseen_indices = [self.label_names.index(label) for label in self.unseen_labels]
        self.seen_labels = [label for label in self.label_names if label not in self.unseen_labels]

        self.samples: list[dict[str, Any]] = []
        rng = np.random.default_rng(split_seed)
        csv_files = sorted(self.labels_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {self.labels_dir}")

        for csv_file in csv_files:
            scene_category = csv_file.stem
            scene_frame = read_scene_csv(csv_file)
            missing_labels = [label for label in self.label_names if label not in scene_frame.columns]
            if missing_labels:
                raise ValueError(f"Missing labels in {csv_file}: {missing_labels}")

            # KG construction already merges duplicated labels, e.g. tanks into stadium.
            scene_frame = scene_frame[["image", *self.label_names]].copy()
            train_idx, val_idx, test_idx = split_indices_per_scene(len(scene_frame), rng, split_ratio)
            selected_indices = {"train": train_idx, "val": val_idx, "test": test_idx}[split]

            scene_dir = self.images_dir / scene_category
            for idx in selected_indices:
                row = scene_frame.iloc[int(idx)]
                self.samples.append(
                    {
                        "image_path": scene_dir / str(row["image"]),
                        "image_name": str(row["image"]),
                        "scene_category": scene_category,
                        "labels": row[self.label_names].to_numpy(dtype=np.float32),
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No records prepared for split={split}")

        if split == "train" and float(train_fraction) < 1.0:
            n_total = len(self.samples)
            n_keep = max(1, int(n_total * float(train_fraction)))
            rng_fraction = np.random.RandomState(split_seed + 1000)
            indices = rng_fraction.choice(n_total, size=n_keep, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"  Train fraction={float(train_fraction):.2f}: {n_keep}/{n_total} samples kept")

        self.transform = clip_preprocess or build_clip_image_transform(
            image_size=image_size,
            is_train=(split == "train"),
            augment=clip_augment_train,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        record = self.samples[index]
        image_path = Path(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(record["labels"], dtype=torch.float32)

        if not self.return_metadata:
            return image_tensor, label_tensor

        metadata = {
            "image_path": str(image_path),
            "image_name": record["image_name"],
            "scene_category": record["scene_category"],
        }
        return image_tensor, label_tensor, metadata

    def get_label_mask(self) -> torch.Tensor:
        mask = torch.ones(len(self.label_names), dtype=torch.float32)
        for idx in self.unseen_indices:
            mask[idx] = 0.0
        return mask


class MLRSNetFeatureDataset(Dataset):
    def __init__(
        self,
        feature_cache_path: str | Path,
        unseen_labels: list[str] | None = None,
        return_metadata: bool = False,
    ) -> None:
        feature_cache_path = Path(feature_cache_path).resolve()
        if not feature_cache_path.is_file():
            raise FileNotFoundError(f"Feature cache not found: {feature_cache_path}")

        payload = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
        self.features = payload["features"].float()
        self.labels = payload["labels"].float()
        self.label_names = list(payload["label_names"])
        self.node_count = int(len(self.label_names))
        self.return_metadata = return_metadata
        self.unseen_labels = [label for label in (unseen_labels or []) if label in self.label_names]
        self.unseen_indices = [self.label_names.index(label) for label in self.unseen_labels]
        self.seen_labels = [label for label in self.label_names if label not in self.unseen_labels]

        metadata = payload.get("metadata", {})
        self.image_paths = list(metadata.get("image_path", []))
        self.image_names = list(metadata.get("image_name", []))
        self.scene_categories = list(metadata.get("scene_category", []))
        self.split = str(payload.get("split", "unknown"))

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
            "scene_category": self.scene_categories[index] if index < len(self.scene_categories) else "unknown",
        }
        return feature_tensor, label_tensor, metadata

    def get_label_mask(self) -> torch.Tensor:
        mask = torch.ones(len(self.label_names), dtype=torch.float32)
        for idx in self.unseen_indices:
            mask[idx] = 0.0
        return mask
