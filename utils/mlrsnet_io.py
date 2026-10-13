from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_DIR_CANDIDATES = ("Labels", "labels")
IMAGE_DIR_CANDIDATES = ("Images", "images")


def find_existing_subdir(root: str | Path, candidates: tuple[str, ...]) -> Path:
    root = Path(root)
    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find any of {candidates} under {root}")


def detect_csv_header(csv_path: str | Path) -> bool:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)
    if not first_row:
        return False
    if first_row[0].strip().lower() == "image":
        return True
    return not all(cell.strip() in {"0", "1"} for cell in first_row[1:])


def read_scene_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    has_header = detect_csv_header(csv_path)
    if has_header:
        frame = pd.read_csv(csv_path)
    else:
        frame = pd.read_csv(csv_path, header=None)
        num_columns = frame.shape[1]
        frame.columns = ["image"] + [f"label_{idx}" for idx in range(1, num_columns)]

    frame = frame.copy()
    frame.rename(columns={frame.columns[0]: "image"}, inplace=True)
    frame["image"] = frame["image"].astype(str)
    label_columns = [column for column in frame.columns if column != "image"]
    frame[label_columns] = frame[label_columns].astype(np.int32)
    return frame


def load_all_label_tables(data_root: str | Path) -> tuple[pd.DataFrame, list[str], list[Path]]:
    label_dir = find_existing_subdir(data_root, LABEL_DIR_CANDIDATES)
    csv_files = sorted(label_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {label_dir}")

    combined_frames: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None
    for csv_file in csv_files:
        scene_frame = read_scene_csv(csv_file)
        if reference_columns is None:
            reference_columns = list(scene_frame.columns)
        elif list(scene_frame.columns) != reference_columns:
            raise ValueError(f"CSV schema mismatch detected in {csv_file}")
        scene_frame.insert(1, "scene_category", csv_file.stem)
        combined_frames.append(scene_frame)

    combined = pd.concat(combined_frames, ignore_index=True)
    label_columns = [column for column in combined.columns if column not in {"image", "scene_category"}]
    return combined, label_columns, csv_files


def split_indices_per_scene(
    num_samples: int,
    rng: np.random.Generator,
    split_ratio: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"Split ratio must sum to 1.0, got {split_ratio}")

    indices = rng.permutation(num_samples)
    n_train = int(num_samples * split_ratio[0])
    n_val = int(num_samples * split_ratio[1])
    n_test = num_samples - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx
