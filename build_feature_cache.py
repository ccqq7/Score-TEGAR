from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.dataset_factory import create_image_dataset, normalize_dataset_name
from models.clip_backbone import CLIPVisualBackbone
from utils.feature_cache import resolve_feature_cache_path


def _checkpoint_sha256(path_like: str | None) -> str | None:
    if not path_like:
        return None
    path = Path(path_like).resolve()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen CLIP visual features for multi-label datasets.")
    parser.add_argument("--config", default="./configs/default.yaml")
    parser.add_argument("--dataset", default=None, choices=["mlrsnet", "bigearthnet", "chinamas"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--kg_path", default=None)
    parser.add_argument("--clip_model", default=None)
    parser.add_argument("--clip_pretrained", default=None)
    parser.add_argument("--clip_default_pretrained", default=None)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None, help="Optional subset of splits to cache.")
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests; saves only the first N samples per split.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.dataset is not None:
        config["data"]["dataset"] = args.dataset
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.kg_path is not None:
        config["data"]["kg_path"] = args.kg_path
    if args.clip_model is not None:
        config["clip"]["model_name"] = args.clip_model
    if args.clip_pretrained is not None:
        config["clip"]["pretrained_path"] = args.clip_pretrained
    if args.clip_default_pretrained is not None:
        config["clip"]["default_pretrained"] = args.clip_default_pretrained
    if args.feature_cache_dir is not None:
        config["data"]["feature_cache_dir"] = args.feature_cache_dir
    if args.batch_size is not None:
        config["data"]["feature_cache_batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["feature_cache_build_num_workers"] = args.num_workers
    if args.splits is not None and len(args.splits) > 0:
        config["data"]["feature_cache_splits"] = list(args.splits)
    if args.max_samples_per_split is not None:
        config["data"]["feature_cache_max_samples_per_split"] = int(args.max_samples_per_split)
    return config


def build_split_cache(split: str, config: dict, backbone: CLIPVisualBackbone, device: torch.device) -> Path:
    dataset_name = normalize_dataset_name(config["data"].get("dataset", "mlrsnet"))
    image_size = int(config["clip"].get("image_size", config["data"].get("image_size", 224)))
    split_seed = int(config["data"]["split_seed"])
    split_ratio = tuple(config["data"]["split_ratio"])
    feature_cache_root = config["data"]["feature_cache_dir"]
    cache_path = resolve_feature_cache_path(
        feature_cache_root=feature_cache_root,
        split=split,
        clip_model=config["clip"]["model_name"],
        clip_pretrained=config["clip"].get("pretrained_path"),
        clip_default_pretrained=config["clip"].get("default_pretrained"),
        split_seed=split_seed,
        image_size=image_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_image_dataset(
        dataset_name,
        data_root=config["data"]["root"],
        kg_path=config["data"]["kg_path"],
        split=split,
        image_size=image_size,
        split_seed=split_seed,
        split_ratio=split_ratio,
        train_fraction=1.0,
        unseen_labels=[],
        clip_augment_train=False,
        return_metadata=True,
    )
    dataset.allow_image_fallback = bool(
        config["data"].get("feature_cache_allow_image_fallback", False)
    )
    if not dataset.allow_image_fallback:
        missing_paths = [
            str(Path(sample["image_path"]))
            for sample in dataset.samples
            if not Path(sample["image_path"]).is_file()
        ]
        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            raise FileNotFoundError(
                f"Refusing to build {dataset_name}/{split} cache with "
                f"{len(missing_paths)} missing images. First missing paths:\n{preview}"
            )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("feature_cache_batch_size", 1024)),
        shuffle=False,
        num_workers=max(0, int(config["data"].get("feature_cache_build_num_workers", 0))),
        pin_memory=device.type == "cuda",
    )

    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    metadata = {"image_path": [], "image_name": [], "scene_category": []}
    max_samples = config["data"].get("feature_cache_max_samples_per_split")
    max_samples = None if max_samples is None else max(1, int(max_samples))
    n_saved = 0

    backbone.eval()
    with torch.inference_mode():
        for images, labels, batch_metadata in tqdm(dataloader, desc=f"Cache {split}", dynamic_ncols=True):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last if images.ndim == 4 and device.type == "cuda" else torch.contiguous_format)
            features = backbone(images, return_patches=False).cpu()
            all_features.append(features)
            all_labels.append(labels.cpu())
            metadata["image_path"].extend(batch_metadata["image_path"])
            metadata["image_name"].extend(batch_metadata["image_name"])
            metadata["scene_category"].extend(batch_metadata["scene_category"])
            n_saved += int(features.shape[0])
            if max_samples is not None and n_saved >= max_samples:
                break

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    if max_samples is not None:
        features_tensor = features_tensor[:max_samples]
        labels_tensor = labels_tensor[:max_samples]
        for key in metadata:
            metadata[key] = metadata[key][:max_samples]

    payload = {
        "split": split,
        "features": features_tensor,
        "labels": labels_tensor,
        "label_names": dataset.label_names,
        "metadata": metadata,
        "data_root": str(Path(config["data"]["root"]).resolve()),
        "kg_path": str(Path(config["data"]["kg_path"]).resolve()),
        "clip_model": config["clip"]["model_name"],
        "clip_pretrained": config["clip"].get("pretrained_path"),
        "clip_pretrained_sha256": _checkpoint_sha256(config["clip"].get("pretrained_path")),
        "clip_default_pretrained": config["clip"].get("default_pretrained"),
        "split_seed": split_seed,
        "split_ratio": list(split_ratio),
        "image_size": image_size,
        "image_fallback_allowed": bool(dataset.allow_image_fallback),
        "missing_image_count": 0,
    }
    torch.save(payload, cache_path)
    print(f"Saved {split} feature cache: {cache_path}")
    return cache_path


def main() -> None:
    args = parse_args()
    config = load_config(args)
    print(f"Dataset={normalize_dataset_name(config['data'].get('dataset', 'mlrsnet'))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(config["training"].get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))

    backbone = CLIPVisualBackbone(
        model_name=config["clip"]["model_name"],
        pretrained_path=config["clip"].get("pretrained_path"),
        default_pretrained=config["clip"].get("default_pretrained"),
    ).to(device)
    if device.type == "cuda":
        backbone = backbone.to(memory_format=torch.channels_last)

    splits = tuple(config["data"].get("feature_cache_splits") or ("train", "val", "test"))
    for split in splits:
        build_split_cache(split, config, backbone, device)


if __name__ == "__main__":
    main()
