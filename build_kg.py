from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a typed KG definition from multi-label remote sensing datasets.")
    parser.add_argument(
        "--dataset",
        default="mlrsnet",
        choices=["mlrsnet", "bigearthnet", "chinamas"],
        help="Dataset name used to select the metadata loader.",
    )
    parser.add_argument("--data_root", default=None, help="Dataset root directory for the dataset-loader path.")
    parser.add_argument(
        "--image_split_manifest",
        default=None,
        help=(
            "JSON/CSV image-split manifest. Required with --data_root in strict mode; "
            "the embedded split metadata in --train_feature_cache is also accepted as the manifest."
        ),
    )
    parser.add_argument(
        "--train_feature_cache",
        default=None,
        help="Train-only .pt feature-cache payload containing labels, label_names, split, and metadata.",
    )
    parser.add_argument(
        "--allowed_seen_labels_file",
        default=None,
        help="Outer-split JSON containing seen_labels (and preferably label_names), or one seen label per line.",
    )
    parser.add_argument("--allowed_seen_labels", nargs="*", default=None)
    parser.add_argument(
        "--legacy-full-dataset",
        action="store_true",
        help="Explicitly reproduce the legacy all-images/all-labels graph construction.",
    )
    parser.add_argument("--output_dir", default="./data", help="Directory to save KG outputs.")
    parser.add_argument("--pmi_threshold", type=float, default=2.5, help="PMI threshold for often_cooccur.")
    parser.add_argument("--min_cooccur", type=int, default=50, help="Minimum cooccurrence for often_cooccur.")
    parser.add_argument(
        "--exclusion_pmi_threshold",
        type=float,
        default=-2.0,
        help="PMI threshold for statistical_exclusion.",
    )
    parser.add_argument(
        "--exclusion_min_freq",
        type=int,
        default=100,
        help="Minimum frequency for exclusion relation candidates.",
    )
    parser.add_argument(
        "--hier_conditional_high",
        type=float,
        default=0.8,
        help="High conditional probability threshold for hierarchical relations.",
    )
    parser.add_argument(
        "--hier_conditional_low",
        type=float,
        default=0.3,
        help="Low reverse conditional probability threshold for hierarchical relations.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    return parser.parse_args()


def normalize_dataset_name(dataset_name: str | None) -> str:
    normalized = str(dataset_name or "mlrsnet").strip().lower().replace("_", "-")
    aliases = {
        "mlrsnet": "mlrsnet",
        "bigearthnet": "bigearthnet",
        "ben": "bigearthnet",
        "chinamas": "chinamas",
        "china-mas": "chinamas",
        "china-mas-50k": "chinamas",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return aliases[normalized]


def load_dataset_labels(dataset_name: str, data_root: str | Path):
    # Dataset modules import the image preprocessing stack; keep that optional for
    # strict feature-cache construction, which only needs label tensors.
    from datasets.dataset_factory import load_dataset_labels as dataset_loader

    return dataset_loader(dataset_name, data_root)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_sample_id(value: object) -> str:
    return str(value).strip().replace("\\", "/")


def _sample_id(image: object, scene_category: object | None = None) -> str:
    image_id = _normalise_sample_id(image)
    if scene_category is None or not str(scene_category).strip():
        return image_id
    return f"{_normalise_sample_id(scene_category)}/{image_id}"


def load_allowed_seen_labels(
    labels_file: str | Path | None,
    inline_labels: list[str] | None,
) -> tuple[list[str], list[str] | None, dict[str, object]]:
    if labels_file and inline_labels:
        raise ValueError("Use either --allowed_seen_labels_file or --allowed_seen_labels, not both.")
    if labels_file:
        path = Path(labels_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Allowed-seen label file not found: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                seen = payload
                all_labels = None
            elif isinstance(payload, dict):
                seen = payload.get("seen_labels", payload.get("allowed_seen_labels"))
                all_labels = payload.get("label_names", payload.get("labels", payload.get("nodes")))
                if not isinstance(seen, list):
                    raise ValueError(f"JSON must contain seen_labels or allowed_seen_labels: {path}")
                if all_labels is not None and not isinstance(all_labels, list):
                    raise ValueError(f"label_names/labels/nodes must be a list: {path}")
            else:
                raise ValueError(f"Unsupported allowed-seen payload: {path}")
        else:
            seen = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            all_labels = None
        provenance = {
            "kind": "file",
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    elif inline_labels:
        seen = list(inline_labels)
        all_labels = None
        provenance = {"kind": "inline"}
    else:
        raise ValueError(
            "Strict mode requires --allowed_seen_labels_file or --allowed_seen_labels. "
            "Use --legacy-full-dataset explicitly only for legacy reproduction."
        )

    seen_labels = [str(label).strip() for label in seen if str(label).strip()]
    full_labels = [str(label).strip() for label in all_labels] if all_labels is not None else None
    if not seen_labels or len(set(seen_labels)) != len(seen_labels):
        raise ValueError("Allowed seen labels must be non-empty and unique.")
    if full_labels is not None:
        if len(set(full_labels)) != len(full_labels):
            raise ValueError("Full label vocabulary must be unique.")
        missing = sorted(set(seen_labels) - set(full_labels))
        if missing:
            raise ValueError(f"Seen labels missing from full label vocabulary: {missing}")
    provenance.update(
        {
            "seen_labels_sha256": _canonical_json_sha256(seen_labels),
            "label_names_sha256": _canonical_json_sha256(full_labels) if full_labels is not None else None,
        }
    )
    return seen_labels, full_labels, provenance


def _manifest_train_items(payload: object) -> list[object]:
    if isinstance(payload, dict) and isinstance(payload.get("splits"), dict):
        items = payload["splits"].get("train")
    elif isinstance(payload, dict) and isinstance(payload.get("train"), list):
        items = payload["train"]
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        items = [item for item in payload["records"] if str(item.get("split", "")).lower() == "train"]
    elif isinstance(payload, list):
        items = [item for item in payload if not isinstance(item, dict) or str(item.get("split", "train")).lower() == "train"]
    else:
        items = None
    if not isinstance(items, list) or not items:
        raise ValueError("Image-split manifest contains no train items.")
    return items


def load_train_sample_ids(manifest_path: str | Path) -> tuple[set[str], dict[str, object]]:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image-split manifest not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "split" not in frame.columns or "image" not in frame.columns:
            raise ValueError("CSV split manifest requires split and image columns.")
        train_frame = frame[frame["split"].astype(str).str.lower() == "train"]
        items: list[object] = train_frame.to_dict(orient="records")
    else:
        items = _manifest_train_items(json.loads(path.read_text(encoding="utf-8")))

    sample_ids: set[str] = set()
    for item in items:
        if isinstance(item, str):
            sample_ids.add(_normalise_sample_id(item))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported train manifest item: {item!r}")
        if item.get("sample_id"):
            sample_ids.add(_normalise_sample_id(item["sample_id"]))
        elif item.get("image") or item.get("image_name"):
            image = item.get("image", item.get("image_name"))
            sample_ids.add(_sample_id(image, item.get("scene_category")))
        elif item.get("image_path"):
            sample_ids.add(_normalise_sample_id(item["image_path"]))
        else:
            raise ValueError(f"Train manifest item lacks sample_id/image/image_path: {item}")
    if len(sample_ids) != len(items):
        raise ValueError("Train image-split manifest contains duplicate sample identifiers.")
    return sample_ids, {
        "kind": "external_image_split_manifest",
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_sample_count": len(sample_ids),
        "train_sample_ids_sha256": _canonical_json_sha256(sorted(sample_ids)),
    }


def merge_identical_labels(
    frame: pd.DataFrame,
    label_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[dict[str, str]]]:
    signature_to_label: dict[bytes, str] = {}
    columns_to_drop: list[str] = []
    merged_labels: list[dict[str, str]] = []

    for label in label_columns:
        signature = frame[label].to_numpy(np.uint8).tobytes()
        if signature not in signature_to_label:
            signature_to_label[signature] = label
            continue

        kept_label = signature_to_label[signature]
        if {kept_label, label} == {"stadium", "tanks"}:
            kept_label = "stadium"
        columns_to_drop.append(label if kept_label != label else kept_label)
        merged_labels.append(
            {
                "kept": kept_label,
                "removed": label if kept_label != label else kept_label,
                "reason": "identical_columns",
            }
        )

    deduped_frame = frame.drop(columns=columns_to_drop)
    deduped_labels = [label for label in label_columns if label not in columns_to_drop]
    return deduped_frame, deduped_labels, merged_labels


def _frame_sample_candidates(row: pd.Series) -> set[str]:
    image = _normalise_sample_id(row["image"])
    candidates = {image}
    scene = row.get("scene_category")
    if scene is not None and str(scene).strip():
        candidates.add(_sample_id(image, scene))
    if row.get("image_path") is not None and str(row.get("image_path")).strip():
        candidates.add(_normalise_sample_id(row["image_path"]))
    return candidates


def filter_frame_to_train_manifest(
    frame: pd.DataFrame,
    train_sample_ids: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    selected_indices: list[int] = []
    matched_ids: set[str] = set()
    for row_index, row in frame.iterrows():
        candidates = _frame_sample_candidates(row)
        matches = candidates & train_sample_ids
        if matches:
            selected_indices.append(row_index)
            matched_ids.update(matches)
    missing = sorted(train_sample_ids - matched_ids)
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"Image-split manifest has {len(missing)} train IDs not found in dataset metadata; examples={preview}"
        )
    selected = frame.loc[selected_indices].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("Image-split manifest selected no training samples.")
    canonical_ids = [
        _sample_id(row.image, getattr(row, "scene_category", None))
        for row in selected[[column for column in ("image", "scene_category") if column in selected.columns]].itertuples(index=False)
    ]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("Selected training metadata contains duplicate canonical sample identifiers.")
    return selected, canonical_ids


def load_strict_frame_from_dataset(
    dataset_name: str,
    data_root: str | Path,
    image_split_manifest: str | Path,
    seen_labels: list[str],
    full_labels: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[Path], dict[str, object]]:
    frame, source_label_columns, csv_files = load_dataset_labels(dataset_name, data_root)
    node_labels = list(full_labels) if full_labels is not None else list(source_label_columns)
    missing_nodes = sorted(set(node_labels) - set(source_label_columns))
    missing_seen = sorted(set(seen_labels) - set(source_label_columns))
    if missing_nodes or missing_seen:
        raise ValueError(
            f"Labels missing from dataset metadata: nodes={missing_nodes}, seen={missing_seen}"
        )
    train_ids, manifest_provenance = load_train_sample_ids(image_split_manifest)
    train_frame, canonical_ids = filter_frame_to_train_manifest(frame, train_ids)
    strict_frame = train_frame[[column for column in ("image", "scene_category") if column in train_frame.columns] + seen_labels].copy()
    source = {
        "kind": "dataset_loader",
        "data_root": str(Path(data_root).resolve()),
        "source_files": [str(path.resolve()) for path in csv_files],
        "image_split_manifest": manifest_provenance,
        "train_sample_count": len(strict_frame),
        "train_sample_ids_sha256": _canonical_json_sha256(canonical_ids),
    }
    return strict_frame, seen_labels, node_labels, csv_files, source


def load_strict_frame_from_feature_cache(
    cache_path: str | Path,
    seen_labels: list[str],
    full_labels: list[str] | None,
    image_split_manifest: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[Path], dict[str, object]]:
    import torch

    path = Path(cache_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Train feature cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Feature cache must contain a dictionary payload.")
    if str(payload.get("split", "")).lower() != "train":
        raise ValueError(f"Feature cache must declare split='train', got {payload.get('split')!r}")
    cache_labels = [str(label) for label in payload.get("label_names", [])]
    labels_tensor = payload.get("labels")
    if labels_tensor is None or getattr(labels_tensor, "ndim", None) != 2:
        raise ValueError("Feature cache requires a 2-D labels tensor.")
    if labels_tensor.shape[1] != len(cache_labels):
        raise ValueError("Feature-cache labels width does not match label_names.")
    node_labels = list(full_labels) if full_labels is not None else list(cache_labels)
    missing_nodes = sorted(set(node_labels) - set(cache_labels))
    missing_seen = sorted(set(seen_labels) - set(cache_labels))
    if missing_nodes or missing_seen:
        raise ValueError(f"Labels missing from feature cache: nodes={missing_nodes}, seen={missing_seen}")

    metadata = payload.get("metadata", {})
    sample_count = int(labels_tensor.shape[0])
    image_names = list(metadata.get("image_name", []))
    image_paths = list(metadata.get("image_path", []))
    scenes = list(metadata.get("scene_category", []))
    if image_names and len(image_names) != sample_count:
        raise ValueError("Feature-cache metadata image_name length mismatch.")
    if not image_names and image_paths:
        image_names = [Path(str(value)).name for value in image_paths]
    if not image_names:
        image_names = [f"cache_row_{index}" for index in range(sample_count)]
    if not scenes:
        scenes = [""] * sample_count
    if len(scenes) != sample_count:
        raise ValueError("Feature-cache metadata scene_category length mismatch.")

    seen_indices = [cache_labels.index(label) for label in seen_labels]
    seen_matrix = labels_tensor[:, seen_indices].detach().cpu().numpy().astype(np.int32, copy=False)
    frame = pd.DataFrame(seen_matrix, columns=seen_labels)
    frame.insert(0, "image", [str(value) for value in image_names])
    frame.insert(1, "scene_category", [str(value) for value in scenes])

    if image_split_manifest:
        train_ids, external_manifest = load_train_sample_ids(image_split_manifest)
        frame, canonical_ids = filter_frame_to_train_manifest(frame, train_ids)
        manifest_provenance = external_manifest
        seen_matrix = frame[seen_labels].to_numpy(dtype=np.int32)
    else:
        canonical_ids = [_sample_id(image, scene) for image, scene in zip(image_names, scenes)]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("Feature-cache train metadata contains duplicate canonical sample identifiers.")
        manifest_provenance = {
            "kind": "embedded_train_feature_cache_manifest",
            "declared_split": "train",
            "train_sample_count": sample_count,
            "train_sample_ids_sha256": _canonical_json_sha256(canonical_ids),
        }

    source = {
        "kind": "train_feature_cache",
        "path": str(path),
        "file_size": int(path.stat().st_size),
        "file_mtime_ns": int(path.stat().st_mtime_ns),
        "image_split_manifest": manifest_provenance,
        "train_sample_count": len(frame),
        "train_sample_ids_sha256": _canonical_json_sha256(canonical_ids),
        "seen_label_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(seen_matrix).tobytes()
        ).hexdigest(),
    }
    return frame, seen_labels, node_labels, [path], source


def compute_statistics(frame: pd.DataFrame, label_columns: list[str]) -> dict[str, np.ndarray | int]:
    matrix = frame[label_columns].to_numpy(dtype=np.int32)
    total_images = matrix.shape[0]
    label_counts = matrix.sum(axis=0).astype(np.int64)
    cooccurrence = (matrix.T @ matrix).astype(np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = cooccurrence / label_counts[np.newaxis, :]

    label_prob = label_counts / total_images
    pair_prob = cooccurrence / total_images
    denom = label_prob[:, np.newaxis] * label_prob[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(pair_prob / denom)

    return {
        "matrix": matrix,
        "total_images": total_images,
        "label_counts": label_counts,
        "cooccurrence": cooccurrence,
        "conditional": conditional,
        "pmi": pmi,
    }


def extract_often_cooccur(
    labels: list[str],
    counts: np.ndarray,
    cooccurrence: np.ndarray,
    pmi: np.ndarray,
    pmi_threshold: float,
    min_cooccur: int,
) -> tuple[list[list[str]], pd.DataFrame]:
    rows = []
    edges: list[list[str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if cooccurrence[i, j] > min_cooccur and pmi[i, j] > pmi_threshold:
                edge = [labels[i], labels[j]]
                edges.append(edge)
                rows.append(
                    {
                        "label_a": edge[0],
                        "label_b": edge[1],
                        "count_a": int(counts[i]),
                        "count_b": int(counts[j]),
                        "cooccurrence": int(cooccurrence[i, j]),
                        "pmi": float(pmi[i, j]),
                    }
                )
    relation_df = pd.DataFrame(rows)
    if not relation_df.empty:
        relation_df = relation_df.sort_values(["pmi", "cooccurrence"], ascending=[False, False]).reset_index(drop=True)
    return edges, relation_df


def extract_statistical_exclusion(
    labels: list[str],
    counts: np.ndarray,
    cooccurrence: np.ndarray,
    pmi: np.ndarray,
    exclusion_pmi_threshold: float,
    min_freq: int,
) -> tuple[list[list[str]], pd.DataFrame]:
    rows = []
    edges: list[list[str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if counts[i] <= min_freq or counts[j] <= min_freq:
                continue
            pair_pmi = pmi[i, j]
            if cooccurrence[i, j] == 0:
                criterion = "zero_cooccurrence"
            elif pair_pmi < exclusion_pmi_threshold:
                criterion = "negative_pmi"
            else:
                continue
            edge = [labels[i], labels[j]]
            edges.append(edge)
            rows.append(
                {
                    "label_a": edge[0],
                    "label_b": edge[1],
                    "count_a": int(counts[i]),
                    "count_b": int(counts[j]),
                    "cooccurrence": int(cooccurrence[i, j]),
                    "pmi": float(pair_pmi),
                    "criterion": criterion,
                }
            )
    relation_df = pd.DataFrame(rows)
    if not relation_df.empty:
        relation_df = relation_df.sort_values(
            ["criterion", "cooccurrence", "label_a", "label_b"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return edges, relation_df


def extract_hierarchical(
    labels: list[str],
    counts: np.ndarray,
    cooccurrence: np.ndarray,
    conditional: np.ndarray,
    pmi: np.ndarray,
    often_edges: list[list[str]],
    conditional_high: float,
    conditional_low: float,
) -> tuple[list[list[str]], pd.DataFrame]:
    often_pairs = {tuple(sorted(edge)) for edge in often_edges}
    rows = []
    edges: list[list[str]] = []
    for parent_idx in range(len(labels)):
        for child_idx in range(len(labels)):
            if parent_idx == child_idx:
                continue
            if conditional[parent_idx, child_idx] <= conditional_high:
                continue
            if conditional[child_idx, parent_idx] >= conditional_low:
                continue
            undirected_pair = tuple(sorted((labels[parent_idx], labels[child_idx])))
            if undirected_pair in often_pairs:
                continue
            edge = [labels[child_idx], labels[parent_idx]]
            edges.append(edge)
            rows.append(
                {
                    "child": edge[0],
                    "parent": edge[1],
                    "count_child": int(counts[child_idx]),
                    "count_parent": int(counts[parent_idx]),
                    "cooccurrence": int(cooccurrence[parent_idx, child_idx]),
                    "p_parent_given_child": float(conditional[parent_idx, child_idx]),
                    "p_child_given_parent": float(conditional[child_idx, parent_idx]),
                    "pmi": float(pmi[parent_idx, child_idx]),
                }
            )
    relation_df = pd.DataFrame(rows)
    if not relation_df.empty:
        relation_df = relation_df.sort_values(
            ["p_parent_given_child", "p_child_given_parent", "cooccurrence"],
            ascending=[False, True, False],
        ).reset_index(drop=True)
    return edges, relation_df


def compute_cluster_order(matrix: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(matrix.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if safe.shape[0] < 3:
        return np.arange(safe.shape[0])
    distances = pdist(safe, metric="euclidean")
    if len(distances) == 0 or np.allclose(distances, 0.0):
        return np.arange(safe.shape[0])
    linkage_matrix = linkage(distances, method="average")
    return leaves_list(linkage_matrix)


def plot_frequency(frequency_df: pd.DataFrame, output_path: Path, dpi: int, title: str) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = frequency_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 15))
    ax.barh(plot_df["label"], plot_df["count"], color="#4c72b0", edgecolor="black", linewidth=0.25)
    ax.set_title(title)
    ax.set_xlabel("Image count")
    ax.set_ylabel("Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    cmap: str,
    transform: str,
    dpi: int,
) -> None:
    order = compute_cluster_order(matrix)
    ordered = matrix[np.ix_(order, order)].astype(float)
    ordered_labels = [labels[idx] for idx in order]

    if transform == "log1p":
        plot_values = np.log1p(np.clip(ordered, 0, None))
        cbar_label = "log1p(count)"
    else:
        finite = ordered[np.isfinite(ordered)]
        lower = float(np.percentile(finite, 2)) if finite.size else -1.0
        upper = float(np.percentile(finite, 98)) if finite.size else 1.0
        if np.isclose(lower, upper):
            lower -= 1.0
            upper += 1.0
        plot_values = np.nan_to_num(ordered, nan=0.0, posinf=upper, neginf=lower)
        plot_values = np.clip(plot_values, lower, upper)
        cbar_label = "PMI (log2, clipped)"

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        plot_values,
        cmap=cmap,
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        square=True,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_graph_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    dataset_name = normalize_dataset_name(args.dataset)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.legacy_full_dataset:
        if not args.data_root:
            raise ValueError("--legacy-full-dataset requires --data_root.")
        frame, label_columns, csv_files = load_dataset_labels(dataset_name, args.data_root)
        cleaned_frame, cleaned_labels, merged_labels = merge_identical_labels(frame, label_columns)
        statistics_frame = cleaned_frame
        statistics_labels = cleaned_labels
        node_labels = cleaned_labels
        allowed_seen_labels = cleaned_labels
        allowed_provenance: dict[str, object] = {
            "kind": "legacy_all_labels",
            "seen_labels_sha256": _canonical_json_sha256(cleaned_labels),
        }
        source_provenance: dict[str, object] = {
            "kind": "legacy_full_dataset",
            "data_root": str(Path(args.data_root).resolve()),
            "source_files": [str(path.resolve()) for path in csv_files],
            "sample_count": len(statistics_frame),
        }
    else:
        allowed_seen_labels, full_labels, allowed_provenance = load_allowed_seen_labels(
            args.allowed_seen_labels_file,
            args.allowed_seen_labels,
        )
        if bool(args.train_feature_cache) == bool(args.data_root):
            raise ValueError(
                "Strict mode requires exactly one source: --train_feature_cache or --data_root."
            )
        if args.train_feature_cache:
            statistics_frame, statistics_labels, node_labels, csv_files, source_provenance = (
                load_strict_frame_from_feature_cache(
                    cache_path=args.train_feature_cache,
                    seen_labels=allowed_seen_labels,
                    full_labels=full_labels,
                    image_split_manifest=args.image_split_manifest,
                )
            )
        else:
            if not args.image_split_manifest:
                raise ValueError("Strict --data_root construction requires --image_split_manifest.")
            statistics_frame, statistics_labels, node_labels, csv_files, source_provenance = (
                load_strict_frame_from_dataset(
                    dataset_name=dataset_name,
                    data_root=args.data_root,
                    image_split_manifest=args.image_split_manifest,
                    seen_labels=allowed_seen_labels,
                    full_labels=full_labels,
                )
            )
        merged_labels = []

    print(f"Dataset={dataset_name}, mode={'legacy' if args.legacy_full_dataset else 'strict'}")
    print(f"Statistics source: {source_provenance['kind']}")
    print(
        f"Training samples: {len(statistics_frame)}, statistic labels: {len(statistics_labels)}, "
        f"graph nodes: {len(node_labels)}"
    )

    stats = compute_statistics(statistics_frame, statistics_labels)
    total_images = int(stats["total_images"])
    label_counts = stats["label_counts"]
    cooccurrence = stats["cooccurrence"]
    conditional = stats["conditional"]
    pmi = stats["pmi"]

    frequency_df = (
        pd.DataFrame({"label": statistics_labels, "count": label_counts})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    top_label = frequency_df.iloc[0]
    tail_label = frequency_df.iloc[-1]
    ratio = float(top_label["count"] / max(int(tail_label["count"]), 1))
    print(
        f"Label frequency: max={int(top_label['count'])} ({top_label['label']}), "
        f"min={int(tail_label['count'])} ({tail_label['label']}), ratio={ratio:.1f}x"
    )

    print("\nBuilding co-occurrence matrix...")
    print("Building PMI matrix...")

    often_edges, often_df = extract_often_cooccur(
        statistics_labels,
        label_counts,
        cooccurrence,
        pmi,
        args.pmi_threshold,
        args.min_cooccur,
    )
    exclusion_edges, exclusion_df = extract_statistical_exclusion(
        statistics_labels,
        label_counts,
        cooccurrence,
        pmi,
        args.exclusion_pmi_threshold,
        args.exclusion_min_freq,
    )
    hierarchical_edges, hierarchical_df = extract_hierarchical(
        statistics_labels,
        label_counts,
        cooccurrence,
        conditional,
        pmi,
        often_edges,
        args.hier_conditional_high,
        args.hier_conditional_low,
    )

    possible_pairs = len(statistics_labels) * (len(statistics_labels) - 1) // 2
    total_typed_edges = len(often_edges) + len(exclusion_edges) + len(hierarchical_edges)

    print("\nExtracting relations:")
    print(
        f"  often_cooccur:          {len(often_edges):4d} pairs "
        f"(PMI > {args.pmi_threshold}, co-occur > {args.min_cooccur})"
    )
    print(
        f"  statistical_exclusion:  {len(exclusion_edges):4d} pairs "
        f"(co-occur = 0 or PMI < {args.exclusion_pmi_threshold})"
    )
    print(
        f"  hierarchical:           {len(hierarchical_edges):4d} pairs "
        f"(P(parent|child) > {args.hier_conditional_high}, P(child|parent) < {args.hier_conditional_low})"
    )
    print(
        f"  Total typed edges:      {total_typed_edges:4d} / {possible_pairs} "
        f"possible pairs ({(total_typed_edges / max(possible_pairs, 1)):.1%})"
    )

    frequency_df.to_csv(output_dir / "label_frequency.csv", index=False, encoding="utf-8-sig")
    plot_frequency(frequency_df, output_dir / "label_frequency.png", args.dpi, f"{dataset_name} Label Frequency")
    plot_heatmap(
        cooccurrence,
        statistics_labels,
        output_dir / "cooccurrence_heatmap.png",
        f"{dataset_name} Co-occurrence Heatmap",
        "YlOrRd",
        "log1p",
        args.dpi,
    )
    plot_heatmap(
        pmi,
        statistics_labels,
        output_dir / "pmi_heatmap.png",
        f"{dataset_name} PMI Heatmap",
        "coolwarm",
        "pmi",
        args.dpi,
    )

    relation_edges = {
        "often_cooccur": often_edges,
        "statistical_exclusion": exclusion_edges,
        "hierarchical": hierarchical_edges,
    }
    all_edge_endpoints = {
        endpoint
        for edges in relation_edges.values()
        for edge in edges
        for endpoint in edge
    }
    endpoint_violations = sorted(all_edge_endpoints - set(allowed_seen_labels))
    strict_mode = not args.legacy_full_dataset
    leakage_violations: list[str] = []
    if strict_mode and endpoint_violations:
        leakage_violations.append(f"empirical edge endpoints outside allowed seen labels: {endpoint_violations}")
    if strict_mode and set(statistics_labels) != set(allowed_seen_labels):
        leakage_violations.append("statistics label set differs from allowed seen label set")
    declared_split = source_provenance.get("image_split_manifest", {}).get("declared_split")
    if strict_mode and source_provenance["kind"] == "train_feature_cache" and declared_split != "train" and not args.image_split_manifest:
        leakage_violations.append("feature-cache source did not declare train split")

    leakage_audit = {
        "schema_version": 1,
        "protocol_mode": "strict_train_seen_only_v1" if strict_mode else "legacy_full_dataset",
        "passed": bool(strict_mode and not leakage_violations),
        "statistics_scope": "train_samples_seen_labels_only" if strict_mode else "all_samples_all_labels",
        "train_sample_count": total_images,
        "node_count": len(node_labels),
        "statistics_label_count": len(statistics_labels),
        "allowed_seen_label_count": len(allowed_seen_labels),
        "unseen_annotation_columns_used": False if strict_mode else True,
        "all_empirical_edge_endpoints_seen": not endpoint_violations,
        "endpoint_violations": endpoint_violations,
        "violations": leakage_violations,
    }
    if strict_mode and leakage_violations:
        raise RuntimeError(f"Leakage audit failed: {leakage_violations}")

    relation_criteria = {
        "often_cooccur": f"PMI > {args.pmi_threshold} and joint count > {args.min_cooccur}",
        "statistical_exclusion": (
            f"both counts > {args.exclusion_min_freq} and "
            f"(joint count = 0 or PMI < {args.exclusion_pmi_threshold})"
        ),
        "hierarchical": (
            f"P(parent|child) > {args.hier_conditional_high} and "
            f"P(child|parent) < {args.hier_conditional_low}"
        ),
    }
    edge_provenance = {
        "schema_version": 1,
        "protocol_mode": leakage_audit["protocol_mode"],
        "statistics_scope": leakage_audit["statistics_scope"],
        "source": source_provenance,
        "allowed_seen_labels": allowed_seen_labels,
        "allowed_seen_labels_provenance": allowed_provenance,
        "node_labels_sha256": _canonical_json_sha256(node_labels),
        "statistics_labels_sha256": _canonical_json_sha256(statistics_labels),
        "relations": {
            name: {
                "criterion": relation_criteria[name],
                "edge_count": len(edges),
                "edges_sha256": _canonical_json_sha256(edges),
            }
            for name, edges in relation_edges.items()
        },
    }

    count_lookup = {label: int(count) for label, count in zip(statistics_labels, label_counts)}
    kg_payload = {
        "schema_version": 2,
        "protocol_mode": leakage_audit["protocol_mode"],
        "dataset": dataset_name,
        "nodes": node_labels,
        "node_count": len(node_labels),
        "edges": relation_edges,
        "edge_counts": {
            "often_cooccur": len(often_edges),
            "statistical_exclusion": len(exclusion_edges),
            "hierarchical": len(hierarchical_edges),
        },
        "label_to_index": {label: idx for idx, label in enumerate(node_labels)},
        "label_frequency": {label: count_lookup.get(label, 0) for label in node_labels},
        "label_frequency_scope": "train_seen_only; unseen nodes stored as 0" if strict_mode else "full_dataset",
        "thresholds_used": {
            "often_pmi": args.pmi_threshold,
            "often_min_cooccur": args.min_cooccur,
            "exclusion_pmi": args.exclusion_pmi_threshold,
            "exclusion_min_freq": args.exclusion_min_freq,
            "hier_conditional_high": args.hier_conditional_high,
            "hier_conditional_low": args.hier_conditional_low,
        },
        "total_images": total_images,
        "merged_labels": merged_labels,
        "hierarchical_edge_format": "child_to_parent",
        "edge_provenance": edge_provenance,
        "leakage_audit": leakage_audit,
    }
    kg_path = output_dir / "kg_definition.json"
    edge_provenance_path = output_dir / "edge_provenance.json"
    leakage_audit_path = output_dir / "leakage_audit.json"
    with kg_path.open("w", encoding="utf-8") as f:
        json.dump(kg_payload, f, ensure_ascii=False, indent=2)
    edge_provenance_path.write_text(
        json.dumps(edge_provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    leakage_audit_path.write_text(
        json.dumps(leakage_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance_columns = {
        "source_split": "train" if strict_mode else "all",
        "statistics_scope": leakage_audit["statistics_scope"],
        "allowed_seen_labels_sha256": allowed_provenance["seen_labels_sha256"],
    }
    for relation_frame in (often_df, exclusion_df, hierarchical_df):
        if not relation_frame.empty:
            for column, value in provenance_columns.items():
                relation_frame[column] = value

    if not often_df.empty:
        often_df.to_csv(output_dir / "relations_often_cooccur.csv", index=False, encoding="utf-8-sig")
    if not exclusion_df.empty:
        exclusion_df.to_csv(
            output_dir / "relations_statistical_exclusion.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not hierarchical_df.empty:
        hierarchical_df.to_csv(output_dir / "relations_hierarchical.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {kg_path}")
    print(f"Saved: {edge_provenance_path}")
    print(f"Saved: {leakage_audit_path}")
    print(f"Saved: {output_dir / 'label_frequency.csv'}")
    print(f"Saved: {output_dir / 'label_frequency.png'}")
    print(f"Saved: {output_dir / 'cooccurrence_heatmap.png'}")
    print(f"Saved: {output_dir / 'pmi_heatmap.png'}")
    return {
        "kg": kg_path,
        "edge_provenance": edge_provenance_path,
        "leakage_audit": leakage_audit_path,
    }


def main() -> None:
    args = parse_args()
    build_graph_artifacts(args)


if __name__ == "__main__":
    main()
