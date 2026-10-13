from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.multilabel_base import FeatureCacheDataset, MultiLabelImageDataset, split_indices_random
from utils.kg_builder import KGDefinition


def _label_csv_path(data_root: str | Path) -> Path:
    data_root = Path(data_root)
    csv_path = data_root / "China-MAS-50k_label.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"China-MAS label CSV not found: {csv_path}")
    return csv_path


def _image_root(data_root: str | Path) -> Path:
    data_root = Path(data_root)
    candidates = [
        data_root / "China-MAS-50k_image",
        data_root / "images",
        data_root / "image",
    ]
    for image_dir in candidates:
        if image_dir.is_dir():
            return image_dir
    raise FileNotFoundError(
        f"China-MAS image dir not found under {data_root}; "
        f"checked {[str(item) for item in candidates]}"
    )


def load_chinamas_labels(data_root: str | Path) -> tuple[pd.DataFrame, list[str], list[Path]]:
    csv_path = _label_csv_path(data_root)
    frame = pd.read_csv(csv_path).copy()
    if "nim" in frame.columns:
        frame.rename(columns={"nim": "image"}, inplace=True)
    elif "image" not in frame.columns:
        raise ValueError(f"Unexpected China-MAS schema: {list(frame.columns)}")
    frame["image"] = frame["image"].astype(str)
    frame["scene_category"] = "china-mas"
    label_columns = [column for column in frame.columns if column not in {"image", "scene_category"}]
    frame[label_columns] = frame[label_columns].astype(np.int32)
    return frame[["image", "scene_category", *label_columns]].copy(), label_columns, [csv_path]


class ChinaMASDataset(MultiLabelImageDataset):
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
        image_dir = _image_root(data_root)
        kg_definition = KGDefinition(kg_path)
        label_names = list(kg_definition.nodes)
        frame, all_label_columns, _ = load_chinamas_labels(data_root)
        missing = [label for label in label_names if label not in all_label_columns]
        if missing:
            raise ValueError(f"Missing KG labels in China-MAS CSV: {missing}")

        rng = np.random.default_rng(int(split_seed))
        train_idx, val_idx, test_idx = split_indices_random(len(frame), rng, tuple(split_ratio))
        selected = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
        split_frame = frame.iloc[selected].copy()

        labels_matrix = split_frame[label_names].to_numpy(dtype=np.float32)
        samples: list[dict[str, Any]] = []
        for row_idx, row in enumerate(split_frame[["image", "scene_category"]].itertuples(index=False)):
            samples.append(
                {
                    "image_path": image_dir / str(row.image),
                    "image_name": str(row.image),
                    "scene_category": str(getattr(row, "scene_category", "china-mas")),
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


class ChinaMASFeatureDataset(FeatureCacheDataset):
    pass
