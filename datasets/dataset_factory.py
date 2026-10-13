from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from datasets.bigearthnet_dataset import (
    BigEarthNetDataset,
    BigEarthNetFeatureDataset,
    load_bigearthnet_labels,
)
from datasets.chinamas_dataset import (
    ChinaMASDataset,
    ChinaMASFeatureDataset,
    load_chinamas_labels,
)
from datasets.mlrsnet_dataset import MLRSNetDataset, MLRSNetFeatureDataset
from utils.mlrsnet_io import load_all_label_tables


DATASET_LOADERS: dict[str, Callable[[str | Path], tuple[pd.DataFrame, list[str], list[Path]]]] = {
    "mlrsnet": load_all_label_tables,
    "bigearthnet": load_bigearthnet_labels,
    "chinamas": load_chinamas_labels,
}

IMAGE_DATASET_CLASSES = {
    "mlrsnet": MLRSNetDataset,
    "bigearthnet": BigEarthNetDataset,
    "chinamas": ChinaMASDataset,
}

FEATURE_DATASET_CLASSES = {
    "mlrsnet": MLRSNetFeatureDataset,
    "bigearthnet": BigEarthNetFeatureDataset,
    "chinamas": ChinaMASFeatureDataset,
}


def normalize_dataset_name(dataset_name: str | None) -> str:
    if not dataset_name:
        return "mlrsnet"
    normalized = str(dataset_name).strip().lower().replace("_", "-")
    aliases = {
        "mlrsnet": "mlrsnet",
        "bigearthnet": "bigearthnet",
        "ben": "bigearthnet",
        "china-mas": "chinamas",
        "chinamas": "chinamas",
        "china-mas-50k": "chinamas",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Expected one of {sorted(set(aliases.values()))}."
        )
    return aliases[normalized]


def load_dataset_labels(dataset_name: str | None, data_root: str | Path) -> tuple[pd.DataFrame, list[str], list[Path]]:
    dataset_key = normalize_dataset_name(dataset_name)
    return DATASET_LOADERS[dataset_key](data_root)


def create_image_dataset(dataset_name: str | None, **kwargs):
    dataset_key = normalize_dataset_name(dataset_name)
    return IMAGE_DATASET_CLASSES[dataset_key](**kwargs)


def create_feature_dataset(dataset_name: str | None, **kwargs):
    dataset_key = normalize_dataset_name(dataset_name)
    return FEATURE_DATASET_CLASSES[dataset_key](**kwargs)
