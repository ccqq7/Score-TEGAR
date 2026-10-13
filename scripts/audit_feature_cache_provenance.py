from __future__ import annotations

"""Fail-closed provenance audit for frozen image-feature caches.

The auditor never mutates a cache.  It verifies that cached rows can be bound
to the current source tree and checkpoint, then recomputes a small,
deterministic sample with the same non-augmenting CLIP image transform used to
build the caches.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow ``python scripts/audit_feature_cache_provenance.py`` from any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.dataset_factory import create_image_dataset, normalize_dataset_name
from models.clip_backbone import CLIPVisualBackbone, build_clip_image_transform


SCHEMA_VERSION = 1
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_MIN_COSINE = 0.99999
DEFAULT_MAX_ABS = 1e-3
DEFAULT_MAX_L2 = 5e-3


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _normalise_metadata_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/")


def _dataset_relative_key(
    dataset: str,
    split: str,
    image_name: str,
    scene_category: str,
) -> str:
    image_name = _normalise_metadata_path(image_name).split("/")[-1]
    if not image_name:
        raise ValueError("empty image_name")
    if dataset == "mlrsnet":
        scene = _normalise_metadata_path(scene_category).split("/")[-1]
        if not scene or scene == "unknown":
            raise ValueError("MLRSNet requires a non-empty scene_category")
        return f"{scene}/{image_name}"
    if dataset == "bigearthnet":
        image_split = {"train": "train", "val": "validation", "test": "test"}[split]
        return f"{image_split}/{image_name}"
    if dataset == "chinamas":
        return image_name
    raise ValueError(f"Unsupported dataset: {dataset}")


def _metadata_path_has_key(dataset: str, metadata_path: str, key: str) -> bool:
    """Check the old absolute path without depending on its drive or root name."""
    old_path = _normalise_metadata_path(metadata_path).casefold()
    key = _normalise_metadata_path(key).casefold()
    if not old_path:
        return False
    if dataset == "chinamas":
        return old_path.split("/")[-1] == key
    return old_path == key or old_path.endswith(f"/{key}")


def _expected_rows(dataset_obj: Any, dataset: str, split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for sample in dataset_obj.samples:
        image_path = Path(sample["image_path"]).resolve()
        image_name = str(sample.get("image_name", image_path.name))
        scene_category = str(sample.get("scene_category", "unknown"))
        key = _dataset_relative_key(dataset, split, image_name, scene_category)
        folded = key.casefold()
        if folded in rows:
            raise ValueError(f"ambiguous source key in current dataset: {key}")
        rows[folded] = {
            "key": key,
            "image_path": image_path,
            "image_name": image_name,
            "scene_category": scene_category,
            "labels": torch.as_tensor(sample["labels"], dtype=torch.float32),
        }
    return rows


def adjacent_exact_duplicate_stats(features: torch.Tensor, chunk_size: int = 8192) -> dict[str, int]:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {tuple(features.shape)}")
    count = int(features.shape[0])
    if count == 0:
        return {"adjacent_duplicate_pairs": 0, "longest_exact_run": 0}

    adjacent_pairs = 0
    current_run = 1
    longest_run = 1
    previous = features[0]
    for start in range(1, count, int(chunk_size)):
        block = features[start : min(count, start + int(chunk_size))]
        left = torch.cat([previous.unsqueeze(0), block[:-1]], dim=0)
        duplicate_flags = torch.eq(block, left).all(dim=1).cpu().tolist()
        for is_duplicate in duplicate_flags:
            if is_duplicate:
                adjacent_pairs += 1
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        previous = block[-1]
    return {
        "adjacent_duplicate_pairs": int(adjacent_pairs),
        "longest_exact_run": int(longest_run),
    }


def _select_sample_indices(split: str, image_names: Sequence[str], sample_count: int) -> list[int]:
    ranked: list[tuple[str, int]] = []
    for index, image_name in enumerate(image_names):
        digest = hashlib.sha256(f"{split}:{image_name}".encode("utf-8")).hexdigest()
        ranked.append((digest, index))
    ranked.sort()
    return [index for _, index in ranked[: min(max(0, int(sample_count)), len(ranked))]]


def load_backbone(model_name: str, checkpoint: Path, device: torch.device) -> CLIPVisualBackbone:
    backbone = CLIPVisualBackbone(
        model_name=model_name,
        pretrained_path=str(checkpoint),
        default_pretrained=None,
    ).to(device)
    if device.type == "cuda":
        # Match build_feature_cache.py.  ViT GEMM kernels can otherwise differ
        # enough between contiguous and channels-last inputs to trip a strict
        # max-absolute-error check even when the cache is bit-exact under its
        # original build path.
        backbone = backbone.to(memory_format=torch.channels_last)
    backbone.eval()
    return backbone


def recompute_features(
    backbone: Any,
    image_paths: Sequence[Path],
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    transform = build_clip_image_transform(image_size=int(image_size), is_train=False, augment=False)
    tensors: list[torch.Tensor] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensors.append(transform(image.convert("RGB")))

    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), max(1, int(batch_size))):
            batch = torch.stack(tensors[start : start + max(1, int(batch_size))])
            if device.type == "cuda":
                batch = batch.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                batch = batch.to(device)
            outputs.append(backbone(batch, return_patches=False).detach().float().cpu())
    if not outputs:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def _resolve_cache_dir(cache_root: Path, splits: Sequence[str]) -> Path:
    cache_root = cache_root.resolve()
    if all((cache_root / f"{split}.pt").is_file() for split in splits):
        return cache_root
    candidates = sorted(
        {
            path.parent.resolve()
            for path in cache_root.rglob(f"{splits[0]}.pt")
            if all((path.parent / f"{split}.pt").is_file() for split in splits)
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one cache directory under {cache_root} containing "
            f"{list(splits)}, found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _critical_source_hashes(dataset: str) -> dict[str, str]:
    dataset_file = {
        "mlrsnet": "datasets/mlrsnet_dataset.py",
        "bigearthnet": "datasets/bigearthnet_dataset.py",
        "chinamas": "datasets/chinamas_dataset.py",
    }[dataset]
    relative_paths = (
        "build_feature_cache.py",
        "models/clip_backbone.py",
        "datasets/dataset_factory.py",
        dataset_file,
        "utils/feature_cache.py",
        "scripts/audit_feature_cache_provenance.py",
    )
    return {
        relative_path: sha256_file(REPOSITORY_ROOT / relative_path)
        for relative_path in relative_paths
        if (REPOSITORY_ROOT / relative_path).is_file()
    }


def _environment_payload(device: torch.device) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "device_requested": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and cuda_available else None,
        "pid": os.getpid(),
    }


def audit_feature_cache_provenance(
    *,
    dataset: str,
    data_root: str | Path,
    kg_path: str | Path,
    cache_root: str | Path,
    checkpoint: str | Path,
    samples_per_split: int = 3,
    splits: Sequence[str] = DEFAULT_SPLITS,
    allow_legacy_checkpoint_binding: bool = False,
    min_cosine: float = DEFAULT_MIN_COSINE,
    max_abs: float = DEFAULT_MAX_ABS,
    max_l2: float = DEFAULT_MAX_L2,
    device: str | torch.device | None = None,
    recompute_batch_size: int = 32,
    clip_model: str | None = None,
    image_size: int | None = None,
) -> dict[str, Any]:
    dataset = normalize_dataset_name(dataset)
    split_names = tuple(str(split) for split in splits)
    invalid_splits = [split for split in split_names if split not in DEFAULT_SPLITS]
    if invalid_splits:
        raise ValueError(f"Unsupported splits: {invalid_splits}")
    if not split_names:
        raise ValueError("At least one split is required")
    if int(samples_per_split) < 1:
        raise ValueError("samples_per_split must be >= 1")

    data_root = Path(data_root).resolve()
    kg_path = Path(kg_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not kg_path.is_file():
        raise FileNotFoundError(f"KG definition not found: {kg_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cache_dir = _resolve_cache_dir(Path(cache_root), split_names)
    checkpoint_sha = sha256_file(checkpoint)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "passed": False,
        "errors": [],
        "warnings": [],
        "inputs": {
            "dataset": dataset,
            "data_root": _json_path(data_root),
            "kg_path": _json_path(kg_path),
            "cache_root": _json_path(cache_root),
            "resolved_cache_dir": _json_path(cache_dir),
            "checkpoint": _json_path(checkpoint),
            "splits": list(split_names),
            "samples_per_split": int(samples_per_split),
            "allow_legacy_checkpoint_binding": bool(allow_legacy_checkpoint_binding),
            "recompute_batch_size": int(recompute_batch_size),
            "recompute_memory_format": (
                "channels_last" if resolved_device.type == "cuda" else "contiguous"
            ),
        },
        "thresholds": {
            "min_cosine": float(min_cosine),
            "max_abs": float(max_abs),
            "max_l2": float(max_l2),
        },
        "environment": _environment_payload(resolved_device),
        "hashes": {
            "checkpoint_sha256": checkpoint_sha,
            "kg_sha256": sha256_file(kg_path),
            "cache_sha256": {},
            "critical_source_sha256": _critical_source_hashes(dataset),
        },
        "splits": {},
        "cross_split": {},
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]
    all_keys: dict[str, str] = {}
    cross_split_overlaps: list[dict[str, str]] = []
    samples_to_recompute: list[dict[str, Any]] = []
    inferred_model_names: set[str] = set()
    inferred_image_sizes: set[int] = set()
    source_inventory = hashlib.sha256()

    for split in split_names:
        cache_path = cache_dir / f"{split}.pt"
        report["hashes"]["cache_sha256"][split] = sha256_file(cache_path)
        split_report: dict[str, Any] = {
            "cache_path": _json_path(cache_path),
            "errors": [],
            "warnings": [],
            "checkpoint_binding": {},
            "shape": {},
            "metadata": {},
            "duplicates": {},
            "samples": [],
        }
        report["splits"][split] = split_report
        split_errors: list[str] = split_report["errors"]
        split_warnings: list[str] = split_report["warnings"]

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            split_errors.append(f"cannot load cache: {exc}")
            errors.append(f"{split}: cannot load cache: {exc}")
            continue

        embedded_sha = payload.get("clip_pretrained_sha256")
        split_report["checkpoint_binding"] = {
            "embedded_sha256": embedded_sha,
            "actual_sha256": checkpoint_sha,
        }
        if embedded_sha is None or not str(embedded_sha).strip():
            message = f"{split}: cache has no embedded checkpoint SHA-256"
            if allow_legacy_checkpoint_binding:
                split_warnings.append(message + " (explicit legacy override accepted)")
                warnings.append(message + " (explicit legacy override accepted)")
                split_report["checkpoint_binding"]["status"] = "legacy_override"
            else:
                split_errors.append(message)
                errors.append(message)
                split_report["checkpoint_binding"]["status"] = "missing"
        elif str(embedded_sha).casefold() != checkpoint_sha.casefold():
            message = f"{split}: embedded checkpoint SHA-256 does not match current checkpoint"
            split_errors.append(message)
            errors.append(message)
            split_report["checkpoint_binding"]["status"] = "mismatch"
        else:
            split_report["checkpoint_binding"]["status"] = "matched"

        if str(payload.get("split", split)) != split:
            split_errors.append(f"payload split={payload.get('split')!r}, expected {split!r}")
        model_name = clip_model or payload.get("clip_model")
        current_image_size = image_size or payload.get("image_size")
        if not model_name:
            split_errors.append("clip_model is absent from cache and was not supplied")
        else:
            inferred_model_names.add(str(model_name))
        if current_image_size is None:
            split_errors.append("image_size is absent from cache and was not supplied")
        else:
            inferred_image_sizes.add(int(current_image_size))

        features = payload.get("features")
        labels = payload.get("labels")
        label_names = payload.get("label_names")
        metadata = payload.get("metadata")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            split_errors.append(f"features must be a 2D tensor, got {type(features).__name__}")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            split_errors.append(f"labels must be a 2D tensor, got {type(labels).__name__}")
        if not isinstance(label_names, (list, tuple)) or not label_names:
            split_errors.append("label_names must be a non-empty list")
        if not isinstance(metadata, dict):
            split_errors.append("metadata must be a dictionary")
            metadata = {}

        structural_ok = (
            isinstance(features, torch.Tensor)
            and features.ndim == 2
            and isinstance(labels, torch.Tensor)
            and labels.ndim == 2
            and isinstance(label_names, (list, tuple))
            and bool(label_names)
        )
        if not structural_ok:
            errors.extend(f"{split}: {message}" for message in split_errors if f"{split}: {message}" not in errors)
            del payload
            continue

        row_count = int(features.shape[0])
        split_report["shape"] = {
            "features": list(features.shape),
            "labels": list(labels.shape),
            "label_names": len(label_names),
            "feature_dtype": str(features.dtype),
            "label_dtype": str(labels.dtype),
        }
        if labels.shape[0] != row_count:
            split_errors.append(f"feature/label row mismatch: {row_count} vs {labels.shape[0]}")
        if labels.shape[1] != len(label_names):
            split_errors.append(f"label width/name mismatch: {labels.shape[1]} vs {len(label_names)}")
        if not bool(torch.isfinite(features).all()):
            split_errors.append("features contain non-finite values")
        if not bool(torch.isfinite(labels).all()):
            split_errors.append("labels contain non-finite values")
        elif not bool(torch.logical_or(labels == 0, labels == 1).all()):
            split_errors.append("labels contain values other than 0 or 1")

        metadata_lists: dict[str, list[Any]] = {}
        for key in ("image_path", "image_name", "scene_category"):
            value = metadata.get(key, [])
            if not isinstance(value, (list, tuple)):
                split_errors.append(f"metadata.{key} must be a list")
                value = []
            metadata_lists[key] = list(value)
            if len(value) != row_count:
                split_errors.append(f"metadata.{key} length={len(value)}, expected {row_count}")
        split_report["metadata"] = {key: len(value) for key, value in metadata_lists.items()}
        split_report["duplicates"] = adjacent_exact_duplicate_stats(features)

        split_seed = int(payload.get("split_seed", 42))
        split_ratio_raw = payload.get("split_ratio", (0.7, 0.15, 0.15))
        try:
            split_ratio = tuple(float(value) for value in split_ratio_raw)
            dataset_obj = create_image_dataset(
                dataset,
                data_root=data_root,
                kg_path=kg_path,
                split=split,
                image_size=int(current_image_size or 224),
                split_seed=split_seed,
                split_ratio=split_ratio,
                train_fraction=1.0,
                unseen_labels=[],
                clip_augment_train=False,
                return_metadata=True,
            )
            expected = _expected_rows(dataset_obj, dataset, split)
        except Exception as exc:
            split_errors.append(f"cannot reconstruct current source split: {exc}")
            expected = {}

        if expected and len(expected) != row_count:
            split_errors.append(f"cache/source row mismatch: {row_count} vs {len(expected)}")
        if expected and list(label_names) != list(dataset_obj.label_names):
            split_errors.append("cache label_names do not match current KG label order")

        metadata_complete = all(len(values) == row_count for values in metadata_lists.values())
        mapped_rows: list[dict[str, Any]] = []
        seen_in_split: set[str] = set()
        if expected and metadata_complete and labels.shape[0] == row_count:
            for index in range(row_count):
                image_name = str(metadata_lists["image_name"][index])
                scene_category = str(metadata_lists["scene_category"][index])
                old_path = str(metadata_lists["image_path"][index])
                try:
                    key = _dataset_relative_key(dataset, split, image_name, scene_category)
                except ValueError as exc:
                    split_errors.append(f"row {index}: {exc}")
                    continue
                folded = key.casefold()
                if not _metadata_path_has_key(dataset, old_path, key):
                    split_errors.append(f"row {index}: metadata path does not end in dataset-relative key {key!r}")
                    continue
                if folded in seen_in_split:
                    split_errors.append(f"duplicate source within {split}: {key}")
                    continue
                seen_in_split.add(folded)
                current = expected.get(folded)
                if current is None:
                    split_errors.append(f"row {index}: source key absent from current dataset: {key}")
                    continue
                current_path: Path = current["image_path"]
                if not current_path.is_file():
                    split_errors.append(f"row {index}: mapped source image is missing: {current_path}")
                    continue
                if labels.shape[1:] == current["labels"].shape and not torch.equal(
                    labels[index].float(), current["labels"].float()
                ):
                    split_errors.append(f"row {index}: cached labels do not match current source labels for {key}")
                previous_split = all_keys.get(folded)
                if previous_split is not None and previous_split != split:
                    cross_split_overlaps.append({"key": key, "first_split": previous_split, "second_split": split})
                else:
                    all_keys[folded] = split
                stat = current_path.stat()
                source_inventory.update(
                    f"{split}\0{key}\0{stat.st_size}\n".encode("utf-8")
                )
                mapped_rows.append(
                    {
                        "index": index,
                        "key": key,
                        "image_name": image_name,
                        "current_path": current_path,
                    }
                )

        split_report["metadata"]["mapped_existing"] = len(mapped_rows)
        if len(mapped_rows) == row_count and row_count > 0:
            sample_indices = set(
                _select_sample_indices(
                    split,
                    metadata_lists["image_name"],
                    samples_per_split,
                )
            )
            mapped_by_index = {row["index"]: row for row in mapped_rows}
            for index in sorted(sample_indices):
                mapped = mapped_by_index.get(index)
                if mapped is None:
                    continue
                samples_to_recompute.append(
                    {
                        "split": split,
                        "index": index,
                        "image_name": mapped["image_name"],
                        "key": mapped["key"],
                        "image_path": mapped["current_path"],
                        "cached_feature": features[index].detach().float().cpu().clone(),
                        "image_size": int(current_image_size),
                    }
                )

        for message in split_errors:
            prefixed = f"{split}: {message}"
            if prefixed not in errors and message not in errors:
                errors.append(prefixed)
        del payload, features, labels

    report["cross_split"] = {
        "unique_source_count": len(all_keys),
        "overlap_count": len(cross_split_overlaps),
        "overlaps": cross_split_overlaps[:100],
        "overlaps_truncated": max(0, len(cross_split_overlaps) - 100),
    }
    if cross_split_overlaps:
        errors.append(f"cross-split source overlap detected: {len(cross_split_overlaps)}")
    report["hashes"]["source_inventory_manifest_sha256"] = source_inventory.hexdigest()

    if len(inferred_model_names) > 1:
        errors.append(f"inconsistent clip_model across caches: {sorted(inferred_model_names)}")
    if len(inferred_image_sizes) > 1:
        errors.append(f"inconsistent image_size across caches: {sorted(inferred_image_sizes)}")

    # Recompute only after structural and binding checks pass.  This prevents a
    # plausible sample match from masking a broken full-cache provenance chain.
    if not errors:
        resolved_model = next(iter(inferred_model_names))
        backbone = None
        try:
            backbone = load_backbone(resolved_model, checkpoint, resolved_device)
            grouped: dict[int, list[dict[str, Any]]] = {}
            for sample in samples_to_recompute:
                grouped.setdefault(int(sample["image_size"]), []).append(sample)
            for current_image_size, samples in grouped.items():
                recalculated = recompute_features(
                    backbone,
                    [sample["image_path"] for sample in samples],
                    current_image_size,
                    resolved_device,
                    recompute_batch_size,
                )
                if recalculated.shape[0] != len(samples):
                    errors.append(
                        f"recompute returned {recalculated.shape[0]} rows for {len(samples)} samples"
                    )
                    continue
                for offset, sample in enumerate(samples):
                    cached = sample.pop("cached_feature")
                    current = recalculated[offset].float().cpu()
                    if current.shape != cached.shape:
                        metrics = {
                            "passed": False,
                            "error": f"feature shape mismatch: {list(current.shape)} vs {list(cached.shape)}",
                        }
                        errors.append(f"{sample['split']}:{sample['image_name']}: {metrics['error']}")
                    else:
                        cosine = float(F.cosine_similarity(current.unsqueeze(0), cached.unsqueeze(0)).item())
                        delta = current - cached
                        max_abs_value = float(delta.abs().max().item())
                        l2_value = float(torch.linalg.vector_norm(delta).item())
                        passed = (
                            cosine >= float(min_cosine)
                            and max_abs_value <= float(max_abs)
                            and l2_value <= float(max_l2)
                        )
                        metrics = {
                            "cosine": cosine,
                            "max_abs": max_abs_value,
                            "l2": l2_value,
                            "passed": bool(passed),
                        }
                        if not passed:
                            errors.append(
                                f"{sample['split']}:{sample['image_name']}: recomputed feature exceeds thresholds"
                            )
                    image_path = Path(sample.pop("image_path"))
                    sample["source_image_sha256"] = sha256_file(image_path)
                    sample["current_path"] = _json_path(image_path)
                    sample["metrics"] = metrics
                    report["splits"][sample["split"]]["samples"].append(sample)
        except Exception as exc:
            errors.append(f"feature recomputation failed: {type(exc).__name__}: {exc}")
        finally:
            if backbone is not None:
                del backbone

    sampled_source_digest = hashlib.sha256()
    sampled_source_count = 0
    for split in split_names:
        for sample in report["splits"][split]["samples"]:
            sampled_source_digest.update(
                f"{split}\0{sample['key']}\0{sample['source_image_sha256']}\n".encode("utf-8")
            )
            sampled_source_count += 1
    report["hashes"]["sampled_source_images_sha256"] = (
        sampled_source_digest.hexdigest() if sampled_source_count else None
    )
    report["hashes"]["sampled_source_image_count"] = sampled_source_count

    report["status"] = "passed" if not errors else "failed"
    report["passed"] = not errors
    return report


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mlrsnet", "bigearthnet", "chinamas"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--kg", required=True, dest="kg_path")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=DEFAULT_SPLITS)
    parser.add_argument("--allow-legacy-checkpoint-binding", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    parser.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    parser.add_argument("--max-l2", type=float, default=DEFAULT_MAX_L2)
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu.")
    parser.add_argument("--recompute-batch-size", type=int, default=32)
    parser.add_argument("--clip-model", default=None, help="Override only for a legacy cache lacking clip_model.")
    parser.add_argument("--image-size", type=int, default=None, help="Override only for a legacy cache lacking image_size.")
    parser.add_argument("--output", default=None, help="Default: <resolved-cache-dir>/feature_cache_provenance_audit.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path: Path | None = Path(args.output).resolve() if args.output else None
    try:
        report = audit_feature_cache_provenance(
            dataset=args.dataset,
            data_root=args.data_root,
            kg_path=args.kg_path,
            cache_root=args.cache_root,
            checkpoint=args.checkpoint,
            samples_per_split=args.samples_per_split,
            splits=args.splits,
            allow_legacy_checkpoint_binding=args.allow_legacy_checkpoint_binding,
            min_cosine=args.min_cosine,
            max_abs=args.max_abs,
            max_l2=args.max_l2,
            device=args.device,
            recompute_batch_size=args.recompute_batch_size,
            clip_model=args.clip_model,
            image_size=args.image_size,
        )
        if output_path is None:
            output_path = Path(report["inputs"]["resolved_cache_dir"]) / "feature_cache_provenance_audit.json"
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        if output_path is None:
            output_path = Path(args.cache_root).resolve() / "feature_cache_provenance_audit.json"

    _write_report(report, output_path)
    print(f"Feature-cache provenance audit: {report['status'].upper()}")
    print(f"JSON sidecar: {output_path}")
    if report.get("errors"):
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
