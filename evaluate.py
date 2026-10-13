from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.dataset_factory import create_feature_dataset, create_image_dataset, normalize_dataset_name
from models.classifier import create_model
from utils.feature_cache import resolve_feature_cache_path
from utils.kg_builder import KGDefinition
from utils.metrics import compute_map, compute_per_class_ap, compute_zero_shot_metrics, summarize_frequency_groups, decision_threshold_to_logit


DELTA_MODES = {"zs_homo_delta", "zs_tegar_delta", "zs_tegar_no_prop", "zs_homo_gat", "zs_tegar"}
V3_MODES = {"zs_homo_v3", "zs_tegar_v3", "zs_tegar_no_patch"}
GENERATOR_MODES = {"zs_proj_only", "zs_homo_gen", "zs_tegar_gen"}
PROTO_REGRESSION_MODES = {"zs_proto_only", "zs_homo_proto", "zs_tegar_proto"}
LOGIT_REFINER_MODES = {
    "zs_logit_tegar",
    "zs_logit_homo",
    "zs_logit_none",
    "zs_logit_tegar_no_gate",
    "zs_logit_tegar_no_uncertainty_gate",
    "zs_logit_tegar_no_experts",
    "zs_logit_tegar_no_consensus",
    "zs_logit_tegar_no_exclusion_neg",
    "zs_logit_tegar_reg",
    "zs_logit_homo_reg",
    "zs_logit_tegar_fixed_temp",
    "zs_logit_tegar_k2",
    "zs_logit_tegar_k8",
    "zs_logit_homo_pm",
}
CONTROLLED_FEATURE_MODES = {
    "zs_linear_probe",
    "zs_clip_adapter",
    "zs_feature_tegar",
    "zs_gcnz_anchor",
}
FEATURE_CACHE_MODES = {
    "zs_clip",
    *GENERATOR_MODES,
    *PROTO_REGRESSION_MODES,
    *LOGIT_REFINER_MODES,
    *CONTROLLED_FEATURE_MODES,
}






def _normalise_path(path_like: str | Path) -> str:
    path_str = str(Path(path_like).resolve())
    if sys.platform.startswith("win") and not path_str.startswith("\\\\?\\"):
        if path_str.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path_str[2:]
        return "\\\\?\\" + path_str
    return path_str


def _ensure_directory(path_like: str | Path) -> None:
    os.makedirs(_normalise_path(path_like), exist_ok=True)


def _path_exists(path_like: str | Path) -> bool:
    return os.path.exists(_normalise_path(path_like))


def assert_finite_payload(value: object, *, context: str) -> None:
    """Fail before invalid model outputs or artifacts can be reported as results."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite tensor detected in {context}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite array detected in {context}")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite value detected in {context}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_payload(item, context=f"{context}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_finite_payload(item, context=f"{context}[{index}]")


def _save_figure(fig: plt.Figure, output_path: str | Path, **kwargs) -> None:
    _ensure_directory(Path(output_path).parent)
    fig.savefig(_normalise_path(output_path), **kwargs)


def _write_csv(frame: pd.DataFrame, output_path: str | Path, **kwargs) -> None:
    _ensure_directory(Path(output_path).parent)
    frame.to_csv(_normalise_path(output_path), **kwargs)


def _write_text(output_path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    _ensure_directory(Path(output_path).parent)
    with open(_normalise_path(output_path), "w", encoding=encoding) as handle:
        handle.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot MLRSNet models in CLIP space.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument(
        "--mode",
        choices=[
            "zs_clip",
            "zs_proj_only",
            "zs_homo_gen",
            "zs_tegar_gen",
            "zs_proto_only",
            "zs_homo_proto",
            "zs_tegar_proto",
            "zs_logit_none",
            "zs_logit_homo",
            "zs_logit_tegar",
            "zs_logit_tegar_no_gate",
            "zs_logit_tegar_no_uncertainty_gate",
            "zs_logit_tegar_no_experts",
            "zs_logit_tegar_no_consensus",
            "zs_logit_tegar_no_exclusion_neg",
            "zs_logit_tegar_reg",
            "zs_logit_homo_reg",
            "zs_logit_tegar_fixed_temp",
            "zs_logit_tegar_k2",
            "zs_logit_tegar_k8",
            "zs_logit_homo_pm",
            "zs_feature_tegar",
            "zs_gcnz_anchor",
            "zs_linear_probe",
            "zs_clip_adapter",
            "zs_homo_v3",
            "zs_tegar_v3",
            "zs_tegar_no_patch",
            "zs_homo_delta",
            "zs_tegar_delta",
            "zs_tegar_no_prop",
            "zs_homo_gat",
            "zs_tegar",
        ],
        default=None,
    )
    parser.add_argument("--weight_type", choices=["best", "last"], default="best")
    parser.add_argument("--config", default="./configs/default.yaml")
    parser.add_argument("--dataset", default=None, choices=["mlrsnet", "bigearthnet", "chinamas"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--kg_path", default=None)
    parser.add_argument("--text_anchors", default=None)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument(
        "--require_feature_cache",
        action="store_true",
        help=(
            "Fail closed unless evaluation resolves to the cached-feature pipeline. "
            "Strict protocol runners should always enable this flag."
        ),
    )
    parser.add_argument(
        "--strict_protocol",
        action="store_true",
        help=(
            "Enable fail-closed evaluation contracts used by run_strict_protocol.py, "
            "including an exact checkpoint state_dict match."
        ),
    )
    parser.add_argument(
        "--expected_feature_cache_sha256",
        default=None,
        help=(
            "SHA-256 locked in the prepared strict protocol manifest for the selected split. "
            "The resolved cache is hashed and checked before its payload is loaded."
        ),
    )
    parser.add_argument("--unseen_labels_file", default=None)
    parser.add_argument("--clip_model", default=None)
    parser.add_argument("--clip_pretrained", default=None)
    parser.add_argument("--clip_default_pretrained", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Locked global threshold selected on inner validation; never tune it on outer test.",
    )
    parser.add_argument("--decision_threshold_space", choices=["probability", "logit"], default="probability")
    parser.add_argument(
        "--threshold_manifest",
        default=None,
        help="Optional JSON containing decision_threshold and an inner-validation provenance record.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not _path_exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    if args.run_dir is None:
        return None
    if args.mode is None:
        raise ValueError("When using --run_dir, you must also provide --mode.")
    run_dir = Path(args.run_dir).resolve()
    if not os.path.isdir(_normalise_path(run_dir)):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoint_path = run_dir / f"{args.weight_type}_{args.mode}.pt"
    if not _path_exists(checkpoint_path):
        raise FileNotFoundError(f"Resolved checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_config(args: argparse.Namespace, checkpoint_payload: dict | None = None) -> dict:
    if checkpoint_payload is not None and checkpoint_payload.get("config") is not None:
        config = copy.deepcopy(checkpoint_payload["config"])
    else:
        with Path(args.config).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.kg_path is not None:
        config["data"]["kg_path"] = args.kg_path
    if args.text_anchors is not None:
        config["data"]["text_anchors_path"] = args.text_anchors
    if args.feature_cache_dir is not None:
        config["data"]["feature_cache_dir"] = args.feature_cache_dir
    if args.unseen_labels_file is not None:
        config["data"]["unseen_labels_file"] = args.unseen_labels_file
    if args.clip_model is not None:
        config["clip"]["model_name"] = args.clip_model
    if args.clip_pretrained is not None:
        config["clip"]["pretrained_path"] = args.clip_pretrained
    if args.clip_default_pretrained is not None:
        config["clip"]["default_pretrained"] = args.clip_default_pretrained
    if args.batch_size is not None:
        config["eval"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.output_dir is not None:
        config["eval"]["output_dir"] = args.output_dir
    if args.mode is not None:
        config["training"]["mode"] = args.mode
    if args.dataset is not None:
        config["data"]["dataset"] = args.dataset
    return config


def load_zero_shot_split(config: dict) -> dict:
    split_path = config["data"].get("unseen_labels_file")
    if not split_path:
        raise ValueError("data.unseen_labels_file must be set for zero-shot evaluation.")
    split_path = Path(split_path)
    if not _path_exists(split_path):
        raise FileNotFoundError(f"Zero-shot split file not found: {split_path}")
    with open(_normalise_path(split_path), "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_num_workers(config: dict, context: str = "DataLoader") -> int:
    return resolve_num_workers_for_cache(config, use_feature_cache=False, context=context)


def resolve_num_workers_for_cache(
    config: dict,
    *,
    use_feature_cache: bool,
    context: str = "DataLoader",
) -> int:
    requested_key = "feature_num_workers" if use_feature_cache else "num_workers"
    force_key = "force_single_worker_on_windows_feature" if use_feature_cache else "force_single_worker_on_windows"
    requested = max(0, int(config["data"].get(requested_key, 0)))
    force_single = bool(config["data"].get(force_key, True))
    if sys.platform.startswith("win") and force_single and requested > 0:
        print(
            f"{context}: Windows detected, forcing num_workers=0 "
            f"(requested {requested}) to avoid torch worker spawn/pagefile crashes."
        )
        return 0
    return requested


def configure_runtime(config: dict, device: torch.device) -> bool:
    deterministic = bool(config["training"].get("deterministic", False))
    deterministic_warn_only = bool(
        config["training"].get("deterministic_warn_only", True)
    )
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if deterministic and device.type == "cuda" and cublas_workspace_config not in {
        ":4096:8",
        ":16:8",
    }:
        raise RuntimeError(
            "Deterministic CUDA execution requires CUBLAS_WORKSPACE_CONFIG "
            "to be set before process start to ':4096:8' or ':16:8'."
        )
    try:
        torch.use_deterministic_algorithms(
            deterministic,
            warn_only=deterministic_warn_only,
        )
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    allow_tf32 = bool(config["training"].get("allow_tf32", True))
    if device.type == "cuda":
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = (
            False
            if deterministic
            else bool(config["training"].get("cudnn_benchmark", True))
        )
    elif deterministic:
        torch.backends.cudnn.benchmark = False
    matmul_precision = str(config["training"].get("matmul_precision", "high"))
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        pass
    warn_only_enabled = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: deterministic_warn_only if deterministic else False,
    )
    get_matmul_precision = getattr(torch, "get_float32_matmul_precision", lambda: matmul_precision)
    config["training"]["runtime_state"] = {
        "deterministic_requested": deterministic,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only_enabled": bool(warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "float32_matmul_precision": str(get_matmul_precision()),
        "cublas_workspace_config": cublas_workspace_config,
    }
    return bool(config["training"].get("channels_last", True)) and device.type == "cuda"


def move_inputs_to_device(inputs: torch.Tensor, device: torch.device, use_channels_last: bool) -> torch.Tensor:
    if use_channels_last and inputs.ndim == 4:
        return inputs.to(device=device, non_blocking=True, memory_format=torch.channels_last)
    return inputs.to(device=device, non_blocking=True)


def forward_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    return_aux: bool,
    use_feature_cache: bool,
):
    if use_feature_cache:
        if not hasattr(model, "forward_from_features"):
            raise RuntimeError(f"Model {type(model).__name__} does not support cached-feature forward.")
        return model.forward_from_features(inputs, return_aux=return_aux)
    return model(inputs, return_aux=return_aux)


def assert_feature_cache_requirement(
    *,
    require_feature_cache: bool,
    use_feature_cache: bool,
    mode: str,
) -> None:
    if require_feature_cache and not use_feature_cache:
        raise RuntimeError(
            "Evaluation was invoked with --require_feature_cache, but the resolved "
            f"configuration would use raw images for mode={mode!r}. Ensure "
            "data.use_feature_cache=true and select a cache-compatible mode."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evaluation_feature_cache_path(config: dict, split: str) -> Path:
    return resolve_feature_cache_path(
        feature_cache_root=config["data"]["feature_cache_dir"],
        split=split,
        clip_model=config["clip"]["model_name"],
        clip_pretrained=config["clip"].get("pretrained_path"),
        clip_default_pretrained=config["clip"].get("default_pretrained"),
        split_seed=int(config["data"]["split_seed"]),
        image_size=int(config["clip"].get("image_size", config["data"].get("image_size", 224))),
    )


def verify_locked_feature_cache(
    config: dict,
    *,
    split: str,
    expected_sha256: str,
) -> Path:
    """Verify the protocol-locked cache before any torch payload deserialization."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected feature-cache SHA-256 must be exactly 64 hexadecimal characters")
    cache_path = resolve_evaluation_feature_cache_path(config, split)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Protocol-locked feature cache not found: {cache_path}")
    actual = _sha256_file(cache_path)
    if actual != expected:
        raise RuntimeError(
            "Protocol-locked feature cache SHA-256 mismatch before loading: "
            f"split={split!r} path={cache_path} expected={expected} actual={actual}"
        )
    return cache_path


def load_checkpoint_model_state(
    model: torch.nn.Module,
    checkpoint_payload: dict,
    *,
    strict_protocol: bool,
):
    """Load exact state for strict runs while preserving legacy compatibility."""

    model_state = checkpoint_payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint does not contain a model_state mapping")
    return model.load_state_dict(model_state, strict=bool(strict_protocol))


def build_loader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    config: dict,
) -> dict:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 4))
    return kwargs


def build_dataloader(
    config: dict,
    kg_path: Path,
    split: str,
    unseen_labels: list[str],
    *,
    use_feature_cache: bool,
) -> DataLoader:
    dataset_name = normalize_dataset_name(config["data"].get("dataset", "mlrsnet"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_feature_cache:
        feature_cache_path = resolve_evaluation_feature_cache_path(config, split)
        dataset = create_feature_dataset(
            dataset_name,
            feature_cache_path=feature_cache_path,
            unseen_labels=unseen_labels,
            return_metadata=True,
        )
    else:
        dataset = create_image_dataset(
            dataset_name,
            data_root=config["data"]["root"],
            kg_path=kg_path,
            split=split,
            image_size=int(config["clip"].get("image_size", config["data"].get("image_size", 224))),
            split_seed=int(config["data"]["split_seed"]),
            split_ratio=tuple(config["data"]["split_ratio"]),
            unseen_labels=unseen_labels,
            clip_augment_train=False,
            return_metadata=True,
        )
    num_workers = resolve_num_workers_for_cache(config, use_feature_cache=use_feature_cache, context=f"{split} DataLoader")
    return DataLoader(
        dataset,
        **build_loader_kwargs(
            batch_size=int(config["eval"].get("batch_size", config["training"]["batch_size"])),
            num_workers=num_workers,
            device=device,
            config=config,
        ),
    )


def plot_tsne(
    initial_mean: torch.Tensor,
    final_mean: torch.Tensor,
    label_names: list[str],
    output_path: Path,
    perplexity: float = 8.0,
    random_state: int = 42,
) -> None:
    output_path = Path(output_path)
    _ensure_directory(output_path.parent)
    try:
        from sklearn.manifold import TSNE
    except Exception:  # pragma: no cover - sklearn depends on local environment
        return

    initial_np = initial_mean.numpy()
    final_np = final_mean.numpy()
    points = np.concatenate([initial_np, final_np], axis=0)
    if points.shape[0] < 4:
        return
    effective_perplexity = max(2.0, min(float(perplexity), float(points.shape[0] - 1)))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(random_state),
    ).fit_transform(points)

    before = embedding[: len(label_names)]
    after = embedding[len(label_names) :]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(before[:, 0], before[:, 1], c="#1f77b4", marker="o", label="before TEGAR", alpha=0.8)
    ax.scatter(after[:, 0], after[:, 1], c="#d62728", marker="^", label="after TEGAR", alpha=0.8)
    for idx, label in enumerate(label_names):
        ax.plot([before[idx, 0], after[idx, 0]], [before[idx, 1], after[idx, 1]], color="#7f7f7f", alpha=0.5)
        ax.text(after[idx, 0], after[idx, 1], label, fontsize=8)
    ax.set_title("Unseen Label Representation Shift")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def plot_unseen_neighbor_scatter(unseen_df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    _ensure_directory(output_path.parent)
    neighbor_column = "n_seen_neighbors" if "n_seen_neighbors" in unseen_df.columns else "seen_neighbor_count"
    if unseen_df.empty or neighbor_column not in unseen_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(unseen_df[neighbor_column], unseen_df["ap"], c="#2ca02c", s=60)
    for row in unseen_df.itertuples(index=False):
        neighbor_value = getattr(row, neighbor_column)
        ax.text(neighbor_value + 0.03, row.ap, row.label, fontsize=8)
    ax.set_title("Unseen AP vs Seen Neighbor Count")
    ax.set_xlabel("Seen neighbors in KG")
    ax.set_ylabel("AP")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def instantiate_model(
    config: dict,
    num_labels: int,
    device: torch.device,
    use_channels_last: bool,
    seen_indices: list[int] | None = None,
    unseen_indices: list[int] | None = None,
    use_feature_cache: bool = False,
) -> torch.nn.Module:
    model_config = config["model"]
    model = create_model(
        mode=config["training"]["mode"],
        num_labels=num_labels,
        kg_path=config["data"]["kg_path"],
        tegar_layers=int(model_config.get("tegar_layers", model_config.get("layers", 2))),
        tegar_dim=int(model_config.get("tegar_dim", model_config.get("hidden_dim", 128))),
        pretrained=False,
        clip_model_name=config["clip"]["model_name"],
        clip_pretrained_path=config["clip"].get("pretrained_path"),
        clip_default_pretrained=config["clip"].get("default_pretrained"),
        text_anchors_path=config["data"]["text_anchors_path"],
        exclusion_beta_init=float(model_config.get("exclusion_beta_init", 0.1)),
        dropout=float(model_config.get("dropout", 0.1)),
        spatial_heads=int(model_config.get("spatial_heads", 4)),
        cross_attn_heads=int(model_config.get("cross_attn_heads", 4)),
        delta_scale_init=float(model_config.get("delta_scale_init", 1.0)),
        use_cached_features=use_feature_cache,
        exclusion_mode=model_config.get("exclusion_mode"),
        disabled_relations=tuple(model_config.get("disabled_relations") or ()),
        strict_exclusion_score=bool(model_config.get("strict_exclusion_score", True)),
        gate_initial_activation=float(model_config.get("gate_initial_activation", 0.5)),
        gate_control=model_config.get("gate_control"),
        gate_shuffle_seed=int(model_config.get("gate_shuffle_seed", 20260826)),
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if hasattr(model, "set_seen_unseen_indices"):
        model.set_seen_unseen_indices(seen_indices or [], unseen_indices or [])
    return model


def build_label_degrees(kg_definition: KGDefinition) -> dict[str, int]:
    degrees = {label: 0 for label in kg_definition.nodes}
    for edges in kg_definition.edges.values():
        for src, dst in edges:
            if src in degrees:
                degrees[src] += 1
            if dst in degrees:
                degrees[dst] += 1
    return degrees


def evaluate_checkpoint_or_model(
    args: argparse.Namespace,
    checkpoint_path: Path | None,
    verbose: bool = True,
) -> dict:
    checkpoint_payload = (
        torch.load(_normalise_path(checkpoint_path), map_location="cpu", weights_only=False)
        if checkpoint_path
        else None
    )
    if checkpoint_payload is not None:
        assert_finite_payload(checkpoint_payload, context=f"checkpoint {checkpoint_path}")
    threshold_source = "fixed_config"
    if args.threshold_manifest:
        threshold_payload = json.loads(Path(args.threshold_manifest).read_text(encoding="utf-8"))
        assert_finite_payload(threshold_payload, context=f"threshold manifest {args.threshold_manifest}")
        if "decision_threshold" not in threshold_payload:
            raise ValueError("threshold manifest must contain decision_threshold")
        if threshold_payload.get("source") not in {"inner_validation", "strict_inner_validation"}:
            raise ValueError("threshold manifest is not marked as inner-validation-derived")
        decision_threshold = float(threshold_payload["decision_threshold"])
        threshold_space = threshold_payload.get("decision_threshold_space", "probability")
        threshold_source = "inner_validation_manifest"
    elif args.decision_threshold is not None:
        decision_threshold = float(args.decision_threshold)
        threshold_space = getattr(args, "decision_threshold_space", "probability")
        threshold_source = "locked_cli"
    elif checkpoint_payload is not None and "decision_threshold" in checkpoint_payload:
        decision_threshold = float(checkpoint_payload["decision_threshold"])
        threshold_space = checkpoint_payload.get("decision_threshold_space", "probability")
        threshold_source = "checkpoint_inner_validation"
    else:
        decision_threshold = 0.5
        threshold_space = "probability"
    decision_threshold_to_logit(decision_threshold, threshold_space)
    config = load_config(args, checkpoint_payload)
    dataset_name = normalize_dataset_name(config["data"].get("dataset", "mlrsnet"))
    mode_name = str(config["training"]["mode"])
    use_feature_cache = bool(config["data"].get("use_feature_cache", True)) and mode_name in FEATURE_CACHE_MODES
    assert_feature_cache_requirement(
        require_feature_cache=bool(getattr(args, "require_feature_cache", False)),
        use_feature_cache=use_feature_cache,
        mode=mode_name,
    )
    strict_protocol = bool(getattr(args, "strict_protocol", False))
    expected_cache_sha256 = getattr(args, "expected_feature_cache_sha256", None)
    if strict_protocol:
        if not bool(getattr(args, "require_feature_cache", False)):
            raise RuntimeError("Strict evaluation requires --require_feature_cache")
        if not expected_cache_sha256:
            raise RuntimeError(
                "Strict evaluation requires --expected_feature_cache_sha256 from the prepared protocol manifest"
            )
    if expected_cache_sha256:
        if not use_feature_cache:
            raise RuntimeError("A locked feature-cache SHA-256 cannot be verified on a raw-image pipeline")
        verify_locked_feature_cache(
            config,
            split=args.split,
            expected_sha256=expected_cache_sha256,
        )
    split_payload = load_zero_shot_split(config)
    unseen_labels = list(split_payload["unseen_labels"])
    seen_labels = list(split_payload["seen_labels"])

    kg_path = Path(config["data"]["kg_path"])
    kg_definition = KGDefinition(kg_path)
    num_labels = int(kg_definition.node_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_channels_last = configure_runtime(config, device)

    dataloader = build_dataloader(config, kg_path, args.split, unseen_labels, use_feature_cache=use_feature_cache)
    seen_label_set = set(seen_labels)
    unseen_label_set = set(unseen_labels)
    seen_indices = [idx for idx, label in enumerate(dataloader.dataset.label_names) if label in seen_label_set]
    unseen_indices = [idx for idx, label in enumerate(dataloader.dataset.label_names) if label in unseen_label_set]
    model = instantiate_model(
        config,
        num_labels,
        device,
        use_channels_last,
        seen_indices,
        unseen_indices,
        use_feature_cache=use_feature_cache,
    )
    if checkpoint_payload is not None:
        load_result = load_checkpoint_model_state(
            model,
            checkpoint_payload,
            strict_protocol=strict_protocol,
        )
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        if verbose and (missing_keys or unexpected_keys):
            print(
                "Checkpoint loaded with non-strict matching: "
                f"missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
            )
    model.eval()

    run_name = (
        checkpoint_payload.get("run_name")
        if checkpoint_payload is not None
        else f"{config['training']['mode']}_direct"
    ) or (checkpoint_path.parent.name if checkpoint_path else f"{config['training']['mode']}_direct")
    weight_label = args.weight_type if checkpoint_path is not None else "direct"
    output_dir = Path(config["eval"]["output_dir"]).resolve() / run_name / args.split / weight_label
    _ensure_directory(output_dir)
    if verbose:
        print(f"Input pipeline: {'cached CLIP features' if use_feature_cache else 'raw images + frozen CLIP backbone'}")
        print(f"Dataset: {dataset_name}")

    all_logits: list[np.ndarray] = []
    all_clip_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_delta: list[np.ndarray] = []
    gate_rows: list[dict[str, object]] = []
    unseen_indices = [dataloader.dataset.label_names.index(label) for label in unseen_labels]
    initial_sum = None
    final_sum = None
    representation_count = 0
    used_spatial_attn_batches = 0
    total_batches = 0
    enhanced_anchor_sim_total = 0.0
    evidence_cls_sim_total = 0.0
    classifier_anchor_sim_total = 0.0
    aux_batches = 0

    iterator = dataloader
    if verbose:
        iterator = tqdm(dataloader, desc=f"Evaluate {run_name}", dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for batch_idx, (images, targets, metadata) in enumerate(iterator, start=1):
            images = move_inputs_to_device(images, device, use_channels_last)
            logits, aux = forward_model(model, images, return_aux=True, use_feature_cache=use_feature_cache)
            assert_finite_payload(logits, context=f"evaluation logits batch {batch_idx}")
            all_logits.append(logits.detach().cpu().numpy())
            clip_logits = aux.get("clip_logits")
            if clip_logits is not None:
                assert_finite_payload(clip_logits, context=f"evaluation CLIP logits batch {batch_idx}")
                all_clip_logits.append(clip_logits.detach().cpu().numpy())
            delta = aux.get("delta")
            if delta is not None:
                assert_finite_payload(delta, context=f"evaluation delta batch {batch_idx}")
                all_delta.append(delta.detach().cpu().numpy())
            all_targets.append(targets.numpy())
            total_batches += 1
            if bool(aux.get("used_spatial_attn", False)):
                used_spatial_attn_batches += 1
            if "enhanced_anchor_sim" in aux or "evidence_cls_sim" in aux:
                enhanced_anchor_sim_total += float(aux.get("enhanced_anchor_sim", 0.0))
                evidence_cls_sim_total += float(aux.get("evidence_cls_sim", 0.0))
                classifier_anchor_sim_total += float(aux.get("classifier_anchor_sim", 0.0))
                aux_batches += 1
            elif "classifier_anchor_sim" in aux:
                classifier_anchor_sim_total += float(aux.get("classifier_anchor_sim", 0.0))
                aux_batches += 1

            gate_values = aux.get("gate_values")
            if gate_values is not None:
                if gate_values.ndim == 3:
                    gate_means = gate_values.mean(dim=1).detach().cpu().numpy()
                else:
                    gate_means = gate_values.detach().cpu().numpy()
                relation_names = list(
                    aux.get(
                        "relation_names",
                        ["often_cooccur", "statistical_exclusion", "hierarchical"][: gate_means.shape[1]],
                    )
                )
                if len(relation_names) != gate_means.shape[1]:
                    relation_names = [f"relation_{index}" for index in range(gate_means.shape[1])]
                for idx, (image_name, scene_category) in enumerate(zip(metadata["image_name"], metadata["scene_category"])):
                    row = {"image_name": image_name, "scene_category": scene_category}
                    row.update(
                        {
                            f"gate_{relation_name}": float(gate_means[idx, relation_index])
                            for relation_index, relation_name in enumerate(relation_names)
                        }
                    )
                    gate_rows.append(row)

            initial_nodes = aux.get("initial_nodes")
            final_nodes = aux.get("final_nodes")
            if initial_nodes is not None and final_nodes is not None and unseen_indices:
                batch_initial = initial_nodes[:, unseen_indices, :].sum(dim=0).detach().cpu()
                batch_final = final_nodes[:, unseen_indices, :].sum(dim=0).detach().cpu()
                initial_sum = batch_initial if initial_sum is None else initial_sum + batch_initial
                final_sum = batch_final if final_sum is None else final_sum + batch_final
                representation_count += int(images.size(0))

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

    raw_logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    clip_logits_np = np.concatenate(all_clip_logits, axis=0) if all_clip_logits else raw_logits_np.copy()
    delta_mix_alpha = 1.0
    if checkpoint_payload is not None and config["training"]["mode"] in LOGIT_REFINER_MODES:
        delta_mix_alpha = float(checkpoint_payload.get("delta_mix_alpha", 1.0))
    if config["training"]["mode"] in LOGIT_REFINER_MODES and abs(delta_mix_alpha - 1.0) > 1e-8:
        logits_np = clip_logits_np + delta_mix_alpha * (raw_logits_np - clip_logits_np)
        if verbose:
            print(f"Applied CLIP/refiner mixing alpha={delta_mix_alpha:.2f}")
    else:
        logits_np = raw_logits_np
    delta_np = np.concatenate(all_delta, axis=0) if all_delta else np.zeros_like(raw_logits_np)
    per_class_ap = compute_per_class_ap(logits_np, targets_np, dataloader.dataset.label_names)
    overall_map = compute_map(per_class_ap)
    raw_per_class_ap = compute_per_class_ap(raw_logits_np, targets_np, dataloader.dataset.label_names)
    raw_overall_map = compute_map(raw_per_class_ap)
    clip_per_class_ap = compute_per_class_ap(clip_logits_np, targets_np, dataloader.dataset.label_names)
    clip_overall_map = compute_map(clip_per_class_ap)
    group_summary = summarize_frequency_groups(
        per_class_ap,
        kg_definition.label_frequency,
        rare_k=int(config["eval"]["rare_k"]),
        freq_k=int(config["eval"]["freq_k"]),
    )
    zero_shot_metrics = compute_zero_shot_metrics(
        logits_np,
        targets_np,
        dataloader.dataset.label_names,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        decision_threshold=decision_threshold,
        decision_threshold_space=threshold_space,
    )
    raw_zero_shot_metrics = compute_zero_shot_metrics(
        raw_logits_np,
        targets_np,
        dataloader.dataset.label_names,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        decision_threshold=decision_threshold,
        decision_threshold_space=threshold_space,
    )
    clip_zero_shot_metrics = compute_zero_shot_metrics(
        clip_logits_np,
        targets_np,
        dataloader.dataset.label_names,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        decision_threshold=decision_threshold,
        decision_threshold_space=threshold_space,
    )

    # --- Calibration: apply saved bias or search for optimal bias ---
    calibration_bias = 0.0
    if checkpoint_payload is not None:
        calibration_bias = float(checkpoint_payload.get("calibration_bias", 0.0))
    if calibration_bias > 0:
        calibrated_logits = logits_np.copy()
        calibrated_logits[:, seen_indices] -= calibration_bias
        calibrated_metrics = compute_zero_shot_metrics(
            calibrated_logits, targets_np, dataloader.dataset.label_names,
            seen_labels=seen_labels, unseen_labels=unseen_labels,
            decision_threshold=decision_threshold,
            decision_threshold_space=threshold_space,
        )
        if verbose:
            print(
                f"Calibration (bias={calibration_bias:.2f}): "
                f"seen={calibrated_metrics['seen_mAP']:.4f} | "
                f"unseen={calibrated_metrics['unseen_mAP']:.4f} | "
                f"HM={calibrated_metrics['harmonic_mean']:.4f}"
            )
    else:
        calibrated_metrics = zero_shot_metrics
    label_degrees = build_label_degrees(kg_definition)
    unseen_set = set(unseen_labels)
    per_class_export = per_class_ap.copy()
    per_class_export["is_unseen"] = per_class_export["label"].isin(unseen_set)
    per_class_export["n_positive"] = per_class_export["positive_count"]
    per_class_export["kg_degree"] = per_class_export["label"].map(label_degrees).fillna(0).astype(int)
    clip_ap_lookup = dict(zip(clip_per_class_ap["label"], clip_per_class_ap["ap"]))
    delta_mean = delta_np.mean(axis=0)
    delta_std = delta_np.std(axis=0)
    per_class_export["clip_ap"] = per_class_export["label"].map(clip_ap_lookup)
    per_class_export["ap_gain"] = per_class_export["ap"] - per_class_export["clip_ap"]
    per_class_export["mean_delta"] = per_class_export["label"].map(
        {label: float(delta_mean[idx]) for idx, label in enumerate(dataloader.dataset.label_names)}
    )
    per_class_export["std_delta"] = per_class_export["label"].map(
        {label: float(delta_std[idx]) for idx, label in enumerate(dataloader.dataset.label_names)}
    )

    _write_csv(per_class_export, output_dir / "per_class_ap.csv", index=False, encoding="utf-8-sig")
    _write_csv(
        per_class_export,
        output_dir / f"per_class_ap_{config['training']['mode']}_{args.split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if config["training"]["mode"] in DELTA_MODES:
        _write_csv(
            per_class_export[
            ["label", "is_unseen", "mean_delta", "std_delta", "clip_ap", "ap", "ap_gain", "n_positive", "kg_degree"]
        ].rename(columns={"ap": "final_ap"}),
            output_dir / "delta_analysis.csv",
            index=False,
            encoding="utf-8-sig",
        )
    _write_csv(
        zero_shot_metrics["seen_per_class"],
        output_dir / f"seen_per_class_ap_{config['training']['mode']}_{args.split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_csv(
        zero_shot_metrics["unseen_per_class"],
        output_dir / f"unseen_per_class_ap_{config['training']['mode']}_{args.split}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    unseen_info_df = pd.DataFrame(split_payload.get("unseen_info", []))
    unseen_ap_df = zero_shot_metrics["unseen_per_class"].copy()
    if not unseen_info_df.empty:
        unseen_ap_df = unseen_ap_df.merge(unseen_info_df, on="label", how="left")
    unseen_ap_df["kg_degree"] = unseen_ap_df["label"].map(label_degrees).fillna(0).astype(int)
    unseen_ap_df["n_positive"] = unseen_ap_df["positive_count"]
    if "clip_nearest_seen" in unseen_ap_df.columns:
        unseen_ap_df["clip_nearest_seen"] = unseen_ap_df["clip_nearest_seen"].apply(
            lambda items: "/".join(
                f"{item['label']}({float(item['similarity']):.2f})"
                for item in (items or [])
            )
            if isinstance(items, list)
            else ""
        )
    if "nearest_seen" in unseen_ap_df.columns:
        unseen_ap_df["nearest_seen"] = unseen_ap_df["nearest_seen"].apply(
            lambda items: "/".join(f"{label}({float(score):.2f})" for label, score in (items or []))
            if isinstance(items, list)
            else ""
        )
    if "n_seen_neighbors" not in unseen_ap_df.columns and "seen_neighbor_count" in unseen_ap_df.columns:
        unseen_ap_df["n_seen_neighbors"] = unseen_ap_df["seen_neighbor_count"]
    _write_csv(
        unseen_ap_df,
        output_dir / f"unseen_ap_analysis_{config['training']['mode']}_{args.split}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unseen_export_columns = [
        column
        for column in ["label", "ap", "n_positive", "kg_degree", "n_seen_neighbors", "nearest_seen", "clip_nearest_seen"]
        if column in unseen_ap_df.columns
    ]
    _write_csv(
        unseen_ap_df[unseen_export_columns],
        output_dir / "unseen_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_csv(unseen_ap_df[unseen_export_columns], output_dir / "unseen_analysis.csv", index=False, encoding="utf-8-sig")
    plot_unseen_neighbor_scatter(
        unseen_ap_df,
        output_dir / f"unseen_ap_vs_neighbors_{config['training']['mode']}_{args.split}.png",
    )

    if config["training"]["mode"] in {"zs_tegar_delta", "zs_tegar"}:
        delta_mean_lookup = {label: float(delta_mean[idx]) for idx, label in enumerate(dataloader.dataset.label_names)}
        propagation_rows: list[dict[str, object]] = []
        seen_set = set(seen_labels)
        for edge_type, edges in kg_definition.edges.items():
            for src, dst in edges:
                if src in unseen_set and dst in seen_set:
                    propagation_rows.append(
                        {
                            "unseen_label": src,
                            "seen_neighbor": dst,
                            "edge_type": edge_type,
                            "neighbor_mean_delta": delta_mean_lookup.get(dst, 0.0),
                            "unseen_mean_delta": delta_mean_lookup.get(src, 0.0),
                        }
                    )
                if dst in unseen_set and src in seen_set:
                    propagation_rows.append(
                        {
                            "unseen_label": dst,
                            "seen_neighbor": src,
                            "edge_type": edge_type,
                            "neighbor_mean_delta": delta_mean_lookup.get(src, 0.0),
                            "unseen_mean_delta": delta_mean_lookup.get(dst, 0.0),
                        }
                    )
        if propagation_rows:
            _write_csv(
                pd.DataFrame(propagation_rows),
                output_dir / "delta_propagation.csv",
                index=False,
                encoding="utf-8-sig",
            )

    if gate_rows:
        gate_df = pd.DataFrame(gate_rows)
        _write_csv(
            gate_df,
            output_dir / f"gate_per_image_{config['training']['mode']}_{args.split}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        gate_scene_df = (
            gate_df.groupby("scene_category")
            .agg(
                gate_cooccur=("gate_often_cooccur", "mean"),
                gate_exclusion=("gate_statistical_exclusion", "mean"),
                gate_hierarchical=("gate_hierarchical", "mean"),
                n_samples=("scene_category", "size"),
            )
            .reset_index()
        )
        _write_csv(
            gate_scene_df,
            output_dir / f"gate_by_scene_{config['training']['mode']}_{args.split}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _write_csv(gate_scene_df, output_dir / "gate_analysis.csv", index=False, encoding="utf-8-sig")

    if representation_count > 0 and initial_sum is not None and final_sum is not None:
        initial_mean = initial_sum / float(representation_count)
        final_mean = final_sum / float(representation_count)
        plot_tsne(
            initial_mean=initial_mean,
            final_mean=final_mean,
            label_names=unseen_labels,
            output_path=output_dir / f"tsne_unseen_before_after_{config['training']['mode']}_{args.split}.png",
            perplexity=float(config["eval"].get("tsne_perplexity", 8.0)),
            random_state=int(config["eval"].get("tsne_random_state", 42)),
        )
        plot_tsne(
            initial_mean=initial_mean,
            final_mean=final_mean,
            label_names=unseen_labels,
            output_path=output_dir / "tsne_visualization.png",
            perplexity=float(config["eval"].get("tsne_perplexity", 8.0)),
            random_state=int(config["eval"].get("tsne_random_state", 42)),
        )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "mode": config["training"]["mode"],
        "run_name": run_name,
        "split": args.split,
        "weight_type": weight_label,
        "overall_mAP": float(overall_map),
        "seen_mAP": float(zero_shot_metrics["seen_mAP"]),
        "unseen_mAP": float(zero_shot_metrics["unseen_mAP"]),
        "harmonic_mean": float(zero_shot_metrics["harmonic_mean"]),
        "decision_threshold": float(decision_threshold),
        "decision_threshold_space": threshold_space,
        "decision_threshold_source": threshold_source,
        "seen_macro_F1": float(zero_shot_metrics["seen_macro_F1"]),
        "seen_micro_F1": float(zero_shot_metrics["seen_micro_F1"]),
        "seen_macro_recall": float(zero_shot_metrics["seen_macro_recall"]),
        "unseen_macro_F1": float(zero_shot_metrics["unseen_macro_F1"]),
        "unseen_micro_F1": float(zero_shot_metrics["unseen_micro_F1"]),
        "unseen_macro_recall": float(zero_shot_metrics["unseen_macro_recall"]),
        "seen_macro_precision": float(zero_shot_metrics["seen_macro_precision"]),
        "unseen_macro_precision": float(zero_shot_metrics["unseen_macro_precision"]),
        "seen_positive_prediction_rate": float(zero_shot_metrics["seen_positive_prediction_rate"]),
        "unseen_positive_prediction_rate": float(zero_shot_metrics["unseen_positive_prediction_rate"]),
        "f1_harmonic_mean": float(zero_shot_metrics["f1_harmonic_mean"]),
        "raw_overall_mAP": float(raw_overall_map),
        "raw_seen_mAP": float(raw_zero_shot_metrics["seen_mAP"]),
        "raw_unseen_mAP": float(raw_zero_shot_metrics["unseen_mAP"]),
        "raw_harmonic_mean": float(raw_zero_shot_metrics["harmonic_mean"]),
        "clip_overall_mAP": float(clip_overall_map),
        "clip_seen_mAP": float(clip_zero_shot_metrics["seen_mAP"]),
        "clip_unseen_mAP": float(clip_zero_shot_metrics["unseen_mAP"]),
        "clip_harmonic_mean": float(clip_zero_shot_metrics["harmonic_mean"]),
        "n_labels": int(num_labels),
        "n_seen": int(len(seen_labels)),
        "n_unseen": int(len(unseen_labels)),
        "AP_rare": float(group_summary["AP_rare"]),
        "AP_freq": float(group_summary["AP_freq"]),
        "rare_labels": group_summary["rare_labels"],
        "freq_labels": group_summary["freq_labels"],
        "seen_labels": seen_labels,
        "unseen_labels": unseen_labels,
        "train_best_metric_name": checkpoint_payload.get("best_metric_name") if checkpoint_payload else None,
        "train_best_score": checkpoint_payload.get("best_score") if checkpoint_payload else None,
        "train_best_map": checkpoint_payload.get("best_map") if checkpoint_payload else None,
        "used_spatial_attn_batches": int(used_spatial_attn_batches),
        "total_batches": int(total_batches),
        "used_spatial_attn_fraction": float(used_spatial_attn_batches / max(total_batches, 1)),
        "enhanced_anchor_sim": float(enhanced_anchor_sim_total / max(aux_batches, 1)) if aux_batches > 0 else None,
        "evidence_cls_sim": float(evidence_cls_sim_total / max(aux_batches, 1)) if aux_batches > 0 else None,
        "classifier_anchor_sim": float(classifier_anchor_sim_total / max(aux_batches, 1)) if aux_batches > 0 else None,
        "mean_abs_delta": float(np.abs(delta_np).mean()),
        "std_delta": float(delta_np.std()),
        "delta_mix_alpha": float(delta_mix_alpha),
        "calibration_bias": float(calibration_bias),
        "calibrated_seen_mAP": float(calibrated_metrics["seen_mAP"]),
        "calibrated_unseen_mAP": float(calibrated_metrics["unseen_mAP"]),
        "calibrated_harmonic_mean": float(calibrated_metrics["harmonic_mean"]),
        "per_class_ap": {row.label: float(row.ap) if pd.notna(row.ap) else None for row in per_class_ap.itertuples(index=False)},
    }
    assert_finite_payload(summary, context="evaluation summary")
    summary_path = output_dir / f"summary_{config['training']['mode']}_{args.split}.json"
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    _write_text(summary_path, summary_payload, encoding="utf-8")
    _write_text(output_dir / "zeroshot_metrics.json", summary_payload, encoding="utf-8")

    if verbose:
        print(f"Mode: {config['training']['mode']}")
        print(f"overall mAP: {summary['overall_mAP']:.4f}")
        print(f"seen mAP: {summary['seen_mAP']:.4f}")
        print(f"unseen mAP: {summary['unseen_mAP']:.4f}")
        print(f"harmonic mean: {summary['harmonic_mean']:.4f}")
        print(
            f"clip baseline: seen={summary['clip_seen_mAP']:.4f} | "
            f"unseen={summary['clip_unseen_mAP']:.4f} | HM={summary['clip_harmonic_mean']:.4f}"
        )
        if summary.get("enhanced_anchor_sim") is not None:
            print(
                f"aux sims: enhanced_anchor={float(summary['enhanced_anchor_sim']):.4f} | "
                f"evidence_cls={float(summary['evidence_cls_sim']):.4f} | "
                f"used_patches={float(summary['used_spatial_attn_fraction']):.2f}"
            )
        if summary.get("classifier_anchor_sim") is not None:
            print(f"classifier_anchor_sim: {float(summary['classifier_anchor_sim']):.4f}")
        print(f"mean|delta|: {summary['mean_abs_delta']:.4f}")
        print(f"Saved evaluation outputs to: {output_dir}")

    return summary


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args)
    evaluate_checkpoint_or_model(args, checkpoint_path, verbose=True)


if __name__ == "__main__":
    main()
