from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.multilabel_base import FeatureCacheDataset, MultiLabelImageDataset
from utils.kg_builder import KGDefinition


BIGEARTHNET_LABELS: list[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

SPLIT_NAME_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

METADATA_CANDIDATES: tuple[str, ...] = (
    "metadata.parquet",
    "metadata.json",
    "metadata.jsonl",
    "metadata.ndjson",
)

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
)


def _normalise_label_name(label: str) -> str:
    label = str(label).strip()
    label = label.replace("Land principally occupied by agriculture with significant areas of natural vegetation", "Land principally occupied by agriculture, with significant areas of natural vegetation")
    label = label.replace("Moors heathland and sclerophyllous vegetation", "Moors, heathland and sclerophyllous vegetation")
    label = label.replace("Transitional woodland/shrub", "Transitional woodland, shrub")
    label = label.replace("Beaches dunes sands", "Beaches, dunes, sands")
    return label


def _metadata_path(data_root: str | Path) -> Path:
    data_root = Path(data_root)
    for candidate_name in METADATA_CANDIDATES:
        metadata_path = data_root / candidate_name
        if metadata_path.is_file():
            return metadata_path
    raise FileNotFoundError(
        f"BigEarthNet metadata not found under {data_root}; "
        f"checked {list(METADATA_CANDIDATES)}"
    )


def _split_image_dir(data_root: str | Path, split_name: str) -> Path:
    data_root = Path(data_root)
    candidates = [
        data_root / "image" / split_name,
        data_root / split_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"BigEarthNet image split dir not found for split={split_name}; "
        f"checked {[str(item) for item in candidates]}"
    )


def _load_metadata_frame(metadata_path: Path) -> pd.DataFrame:
    suffix = metadata_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(metadata_path).copy()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(metadata_path, lines=True).copy()
    if suffix == ".json":
        try:
            return pd.read_json(metadata_path).copy()
        except ValueError:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in ("records", "items", "data"):
                    if isinstance(payload.get(key), list):
                        return pd.DataFrame(payload[key]).copy()
            if isinstance(payload, list):
                return pd.DataFrame(payload).copy()
            raise
    raise ValueError(f"Unsupported BigEarthNet metadata suffix: {metadata_path}")


def _infer_split_extensions(data_root: str | Path, split_names: list[str]) -> dict[str, str]:
    extension_map: dict[str, str] = {}
    for split_name in sorted(set(split_names)):
        split_dir = _split_image_dir(data_root, split_name)
        chosen_ext = None
        for extension in IMAGE_EXTENSIONS:
            if next(split_dir.glob(f"*{extension}"), None) is not None:
                chosen_ext = extension
                break
        extension_map[split_name] = chosen_ext or ".tif"
    return extension_map


def load_bigearthnet_labels(data_root: str | Path) -> tuple[pd.DataFrame, list[str], list[Path]]:
    metadata_path = _metadata_path(data_root)
    frame = _load_metadata_frame(metadata_path)
    if "patch_id" not in frame.columns or "labels" not in frame.columns:
        raise ValueError(f"Unexpected BigEarthNet metadata schema: {list(frame.columns)}")

    frame["split"] = frame["split"].astype(str).str.lower()
    split_extensions = _infer_split_extensions(data_root, frame["split"].tolist())
    frame["image"] = [
        f"{patch_id}{split_extensions.get(split_name, '.tif')}"
        for patch_id, split_name in zip(frame["patch_id"].astype(str), frame["split"].astype(str))
    ]
    if "country" in frame.columns:
        frame["scene_category"] = frame["country"].astype(str)
    else:
        frame["scene_category"] = "bigearthnet"

    label_columns = list(BIGEARTHNET_LABELS)
    normalised_rows: list[list[str]] = []
    for raw_labels in frame["labels"].tolist():
        labels = [_normalise_label_name(item) for item in list(raw_labels)]
        normalised_rows.append(labels)

    for label in label_columns:
        frame[label] = [1 if label in row_labels else 0 for row_labels in normalised_rows]

    frame = frame[["image", "split", "scene_category", *label_columns]].copy()
    frame[label_columns] = frame[label_columns].astype(np.int32)
    return frame, label_columns, [metadata_path]


class BigEarthNetDataset(MultiLabelImageDataset):
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
        del split_ratio
        data_root = Path(data_root)
        image_split = SPLIT_NAME_MAP.get(split)
        if image_split is None:
            raise ValueError(f"split must be one of train/val/test, got {split}")
        image_dir = _split_image_dir(data_root, image_split)

        kg_definition = KGDefinition(kg_path)
        label_names = list(kg_definition.nodes)
        frame, all_label_columns, _ = load_bigearthnet_labels(data_root)
        missing = [label for label in label_names if label not in all_label_columns]
        if missing:
            raise ValueError(f"Missing KG labels in BigEarthNet metadata: {missing}")
        split_frame = frame[frame["split"] == image_split].copy()
        labels_matrix = split_frame[label_names].to_numpy(dtype=np.float32)
        samples: list[dict[str, Any]] = []
        for row_idx, row in enumerate(split_frame[["image", "scene_category"]].itertuples(index=False)):
            samples.append(
                {
                    "image_path": image_dir / str(row.image),
                    "image_name": str(row.image),
                    "scene_category": str(getattr(row, "scene_category", "bigearthnet")),
                    "labels": labels_matrix[row_idx],
                }
            )

        super().__init__(
            samples=samples,
            label_names=label_names,
            split=split,
            image_size=image_size,
            clip_preprocess=clip_preprocess,
            clip_augment_train=clip_augment_train,
            train_fraction=train_fraction,
            split_seed=split_seed,
            unseen_labels=unseen_labels,
            return_metadata=return_metadata,
        )


class BigEarthNetFeatureDataset(FeatureCacheDataset):
    pass
