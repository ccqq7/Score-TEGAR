from __future__ import annotations

import importlib
from types import ModuleType

from datasets.dataset_factory import normalize_dataset_name


LABEL_CONFIG_MODULES = {
    "mlrsnet": "utils.label_config",
    "bigearthnet": "configs.bigearthnet_label_config",
    "chinamas": "configs.chinamas_label_config",
}


def load_label_config(dataset_name: str | None) -> ModuleType:
    dataset_key = normalize_dataset_name(dataset_name)
    return importlib.import_module(LABEL_CONFIG_MODULES[dataset_key])
