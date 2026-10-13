from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.dataset_factory import create_feature_dataset, create_image_dataset, normalize_dataset_name
from datasets.mlrsnet_dataset import MLRSNetDataset
from models.classifier import create_model
from utils.feature_cache import resolve_feature_cache_path
from utils.kg_builder import KGDefinition
from utils.losses import AsymmetricLoss
from utils.metrics import (
    assert_no_forbidden_labels,
    build_nested_label_protocol,
    compute_map,
    compute_per_class_ap,
    compute_zero_shot_metrics,
    search_decision_threshold,
)
from utils.training_viz import TrainingVisualizer


RESIDUAL_DELTA_MODES = {"zs_homo_delta", "zs_tegar_delta", "zs_tegar_no_prop", "zs_homo_gat", "zs_tegar"}
V3_MODES = {"zs_tegar_v3", "zs_homo_v3", "zs_tegar_no_patch"}
GENERATOR_MODES = {"zs_proj_only", "zs_homo_gen", "zs_tegar_gen"}
PROTO_REGRESSION_MODES = {"zs_proto_only", "zs_homo_proto", "zs_tegar_proto"}
LOGIT_REFINER_BASE_MODES = {
    "zs_logit_tegar",
    "zs_logit_homo",
    "zs_logit_none",
    "zs_logit_tegar_no_gate",
    "zs_logit_tegar_no_uncertainty_gate",
    "zs_logit_tegar_no_experts",
    "zs_logit_tegar_no_consensus",
    "zs_logit_tegar_no_exclusion_neg",
    "zs_logit_tegar_fixed_temp",
    "zs_logit_tegar_k2",
    "zs_logit_tegar_k8",
    "zs_logit_homo_pm",
}
LOGIT_REFINER_REG_MODES = {"zs_logit_tegar_reg", "zs_logit_homo_reg"}
LOGIT_REFINER_MODES = LOGIT_REFINER_BASE_MODES | LOGIT_REFINER_REG_MODES
CONTROLLED_FEATURE_MODES = {"zs_feature_tegar", "zs_gcnz_anchor", "zs_linear_probe", "zs_clip_adapter"}
FEATURE_CACHE_MODES = {
    "zs_clip",
    *GENERATOR_MODES,
    *PROTO_REGRESSION_MODES,
    *LOGIT_REFINER_MODES,
    *CONTROLLED_FEATURE_MODES,
}


def should_search_legacy_calibration_bias(mode_name: str, legacy_outer_validation: bool) -> bool:
    """Return whether the legacy mAP-based seen-logit bias search is allowed."""
    return bool(legacy_outer_validation and mode_name not in LOGIT_REFINER_MODES)


def assert_strict_nonlogit_calibration_bias_zero(
    mode_name: str,
    legacy_outer_validation: bool,
    calibration_bias: float,
) -> None:
    """Reject leakage-prone legacy calibration in strict non-logit runs."""
    if (
        not legacy_outer_validation
        and mode_name not in LOGIT_REFINER_MODES
        and abs(float(calibration_bias)) > 1e-12
    ):
        raise AssertionError(
            "Strict non-logit protocol requires calibration_bias=0; "
            "use the inner-validation decision threshold for F1-HM selection instead."
        )


def _normalise_output_path(output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(path.resolve())
    if sys.platform.startswith("win") and not path_str.startswith("\\\\?\\"):
        if path_str.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path_str[2:]
        return "\\\\?\\" + path_str
    return path_str


def assert_finite_payload(value: object, *, context: str) -> None:
    """Reject non-finite numeric values without silently accepting corrupt artifacts."""
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


def assert_finite_gradients(model: torch.nn.Module, *, context: str) -> None:
    """Check unscaled gradients; AMP-scaled overflow is handled by GradScaler instead."""
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Non-finite gradient detected in {context}: {name}")


def assert_finite_model_state(
    model: torch.nn.Module,
    *,
    context: str,
    trainable_only: bool = False,
) -> None:
    state = (
        ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
        if trainable_only
        else model.state_dict().items()
    )
    for name, tensor in state:
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Non-finite model state detected in {context}: {name}")


def _torch_save(payload: object, output_path: str | Path) -> None:
    assert_finite_payload(payload, context=f"checkpoint {Path(output_path)}")
    target_path = Path(output_path).resolve()
    temporary_path = target_path.with_name(
        f".{target_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    # Save to a unique sibling first so the existing checkpoint remains valid
    # while PyTorch serialises the replacement. Windows indexers/antivirus can
    # briefly hold a just-written destination open, so retry only the atomic
    # replacement instead of truncating the checkpoint in place.
    with open(_normalise_output_path(temporary_path), "wb") as checkpoint_file:
        torch.save(payload, checkpoint_file)

    normalised_temporary = _normalise_output_path(temporary_path)
    normalised_target = _normalise_output_path(target_path)
    max_replace_attempts = 20
    for attempt in range(max_replace_attempts):
        try:
            os.replace(normalised_temporary, normalised_target)
            return
        except PermissionError:
            if attempt + 1 == max_replace_attempts:
                raise
            time.sleep(min(0.05 * (2**attempt), 1.0))


def seed_everything(training_seed: int) -> None:
    random.seed(int(training_seed))
    np.random.seed(int(training_seed))
    torch.manual_seed(int(training_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(training_seed))


def redact_hidden_targets(targets: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    """Remove hidden-label annotation values before augmentation or loss code sees them."""
    visible = label_mask.to(device=targets.device, dtype=torch.bool).view(1, -1)
    if targets.ndim != 2 or visible.size(1) != targets.size(1):
        raise ValueError(
            f"Target/mask shape mismatch: targets={tuple(targets.shape)}, mask={tuple(label_mask.shape)}"
        )
    return targets.masked_fill(~visible, 0.0)


def resolve_protocol_seeds(split_payload: dict, config: dict) -> dict[str, int]:
    outer_split_seed = int(split_payload.get("seed", config["data"].get("split_seed", 42)))
    configured_training_seed = config["training"].get("training_seed")
    configured_inner_seed = config["training"].get("inner_split_seed")
    return {
        "outer_split_seed": outer_split_seed,
        "training_seed": int(outer_split_seed if configured_training_seed is None else configured_training_seed),
        "inner_split_seed": int(
            outer_split_seed + 1_000_003 if configured_inner_seed is None else configured_inner_seed
        ),
    }


# ---------------------------------------------------------------------------
# Label Smoothing for multi-label targets
# ---------------------------------------------------------------------------
def smooth_labels(targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Apply label smoothing to binary targets.

    Smooths 1 -> 1 - smoothing/2 and 0 -> smoothing/2,
    preventing overconfident predictions.
    """
    if smoothing <= 0.0:
        return targets
    return targets * (1.0 - smoothing) + smoothing / 2.0


# ---------------------------------------------------------------------------
# Mixup data augmentation for multi-label
# ---------------------------------------------------------------------------
def mixup_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup augmentation: linearly interpolate between random pairs.

    For multi-label, targets are also mixed (convex combination).
    """
    if alpha <= 0.0:
        return images, targets
    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_targets = lam * targets + (1.0 - lam) * targets[index]
    return mixed_images, mixed_targets


# ---------------------------------------------------------------------------
# Cosine schedule with warmup
# ---------------------------------------------------------------------------
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing LR schedule with linear warmup."""
    import math

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max(1e-6, float(epoch + 1) / float(max(1, warmup_epochs)))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Legacy seen/unseen calibration-bias search (post-training)
# ---------------------------------------------------------------------------
def search_calibration_bias(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    seen_indices: list[int],
    unseen_indices: list[int],
    label_names: list[str],
    seen_labels: list[str],
    unseen_labels: list[str],
    forbidden_labels: list[str] | None = None,
    use_channels_last: bool = False,
    use_feature_cache: bool = False,
) -> tuple[float, float]:
    """Legacy-only grid search for a calibration bias on seen classes.

    In generalized zero-shot learning, models are biased toward seen classes.
    Subtracting a bias from seen-class logits improves harmonic mean.
    Strict non-logit runs must not call this mAP-based search because a
    class-wise constant shift cannot change within-class AP rankings.
    Returns (best_bias, best_harmonic_mean).
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = move_inputs_to_device(images, device, use_channels_last)
            outputs = forward_model(model, images, return_aux=False, use_feature_cache=use_feature_cache)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            assert_finite_payload(logits, context="calibration search logits")
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    best_hm = -1.0
    best_bias = 0.0

    for bias in np.arange(0.0, 5.0, 0.1):
        calibrated = logits_np.copy()
        calibrated[:, seen_indices] -= bias
        metrics = compute_zero_shot_metrics(
            calibrated, targets_np, label_names,
            seen_labels=seen_labels, unseen_labels=unseen_labels,
            forbidden_labels=forbidden_labels,
        )
        hm = float(metrics["harmonic_mean"])
        if hm > best_hm:
            best_hm = hm
            best_bias = float(bias)

    return best_bias, best_hm


def search_logit_mix_alpha(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
    seen_labels: list[str],
    unseen_labels: list[str],
    forbidden_labels: list[str] | None = None,
    use_channels_last: bool = False,
    use_feature_cache: bool = False,
) -> tuple[float, float]:
    """Search a residual mixing factor between raw CLIP logits and graph-refined logits.

    This is useful for logit-refiner modes where the learned delta may help only on
    uncertain labels. Searching alpha on the validation split avoids overusing a
    noisy residual and provides a safe fallback to the CLIP baseline.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_clip_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = move_inputs_to_device(images, device, use_channels_last)
            outputs = forward_model(model, images, return_aux=True, use_feature_cache=use_feature_cache)
            if isinstance(outputs, tuple):
                logits, aux = outputs
            else:
                logits, aux = outputs, {}
            clip_logits = aux.get("clip_logits")
            if clip_logits is None:
                continue
            assert_finite_payload(logits, context="mix-alpha search logits")
            assert_finite_payload(clip_logits, context="mix-alpha search CLIP logits")
            all_logits.append(logits.detach().cpu().numpy())
            all_clip_logits.append(clip_logits.detach().cpu().numpy())
            all_targets.append(targets.numpy())

    if not all_logits or not all_clip_logits:
        return 1.0, 0.0

    logits_np = np.concatenate(all_logits, axis=0)
    clip_logits_np = np.concatenate(all_clip_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    delta_logits_np = logits_np - clip_logits_np

    raw_metrics = compute_zero_shot_metrics(
        logits_np,
        targets_np,
        label_names,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        forbidden_labels=forbidden_labels,
    )
    best_alpha = 1.0
    best_hm = float(raw_metrics["harmonic_mean"])

    for alpha in np.arange(0.0, 5.01, 0.1):
        mixed_logits = clip_logits_np + float(alpha) * delta_logits_np
        metrics = compute_zero_shot_metrics(
            mixed_logits,
            targets_np,
            label_names,
            seen_labels=seen_labels,
            unseen_labels=unseen_labels,
            forbidden_labels=forbidden_labels,
        )
        hm = float(metrics["harmonic_mean"])
        if hm > best_hm:
            best_hm = hm
            best_alpha = float(alpha)

    return best_alpha, best_hm


def search_model_decision_threshold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    label_names: list[str],
    seen_labels: list[str],
    unseen_labels: list[str],
    seen_indices: list[int],
    forbidden_labels: list[str] | None = None,
    calibration_bias: float = 0.0,
    delta_mix_alpha: float | None = None,
    use_channels_last: bool = False,
    use_feature_cache: bool = False,
    candidate_thresholds: tuple[float, ...] | list[float] | None = None,
    threshold_strategy: str = "exact_logit",
) -> tuple[float, dict]:
    """Select the final decision threshold from inner validation predictions."""
    model.eval()
    logits_batches: list[np.ndarray] = []
    clip_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    forbidden_indices = [label_names.index(label) for label in (forbidden_labels or []) if label in label_names]
    with torch.no_grad():
        for images, targets in dataloader:
            images = move_inputs_to_device(images, device, use_channels_last)
            outputs = forward_model(model, images, return_aux=True, use_feature_cache=use_feature_cache)
            if isinstance(outputs, tuple):
                logits, aux = outputs
            else:
                logits, aux = outputs, {}
            assert_finite_payload(logits, context="threshold search logits")
            logits_batches.append(logits.detach().cpu().numpy())
            clip_logits = aux.get("clip_logits")
            if clip_logits is not None:
                assert_finite_payload(clip_logits, context="threshold search CLIP logits")
                clip_batches.append(clip_logits.detach().cpu().numpy())
            safe_targets = targets.detach().cpu().numpy().copy()
            if forbidden_indices:
                safe_targets[:, forbidden_indices] = 0.0
            target_batches.append(safe_targets)

    logits_np = np.concatenate(logits_batches, axis=0)
    targets_np = np.concatenate(target_batches, axis=0)
    if delta_mix_alpha is not None and clip_batches:
        clip_np = np.concatenate(clip_batches, axis=0)
        logits_np = clip_np + float(delta_mix_alpha) * (logits_np - clip_np)
    elif calibration_bias:
        logits_np = logits_np.copy()
        logits_np[:, seen_indices] -= float(calibration_bias)
    return search_decision_threshold(
        logits_np,
        targets_np,
        label_names,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        forbidden_labels=forbidden_labels,
        candidate_thresholds=candidate_thresholds,
        strategy=threshold_strategy,
    )






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train zero-shot MLRSNet models in CLIP space.")
    parser.add_argument("--config", default="./configs/default.yaml")
    parser.add_argument("--dataset", default=None, choices=["mlrsnet", "bigearthnet", "chinamas"])
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
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--kg_path", default=None)
    parser.add_argument("--text_anchors", default=None)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--unseen_labels_file", default=None)
    parser.add_argument("--clip_model", default=None)
    parser.add_argument("--clip_pretrained", default=None)
    parser.add_argument("--clip_default_pretrained", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--graph_dim", type=int, default=None)
    parser.add_argument("--graph_layers", type=int, default=None)
    parser.add_argument("--cross_attn_heads", type=int, default=None)
    parser.add_argument("--delta_scale_init", type=float, default=None)
    parser.add_argument("--lambda_reg", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--disable_auto_tune_batch_size", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--early_stop_warmup_evals", type=int, default=None)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--inner_split_seed", type=int, default=None)
    parser.add_argument("--inner_surrogate_fraction", type=float, default=None)
    parser.add_argument(
        "--legacy_outer_unseen_validation",
        action="store_true",
        help="Explicitly restore the deprecated protocol that selects on true outer-unseen validation labels.",
    )
    parser.add_argument(
        "--protocol_stage",
        choices=["inner_select", "outer_retrain"],
        default=None,
        help="Nested-protocol stage. outer_retrain disables validation-driven decisions.",
    )
    parser.add_argument("--fixed_epochs", type=int, default=None)
    parser.add_argument(
        "--selection_manifest",
        default=None,
        help="Inner threshold_manifest.json to carry locked post-processing into outer_retrain.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = copy.deepcopy(config)
    if args.dataset is not None:
        updated["data"]["dataset"] = args.dataset
    if args.mode is not None:
        updated["training"]["mode"] = args.mode
    if args.data_root is not None:
        updated["data"]["root"] = args.data_root
    if args.kg_path is not None:
        updated["data"]["kg_path"] = args.kg_path
    if args.text_anchors is not None:
        updated["data"]["text_anchors_path"] = args.text_anchors
    if args.feature_cache_dir is not None:
        updated["data"]["feature_cache_dir"] = args.feature_cache_dir
    if args.unseen_labels_file is not None:
        updated["data"]["unseen_labels_file"] = args.unseen_labels_file
    if args.clip_model is not None:
        updated["clip"]["model_name"] = args.clip_model
    if args.clip_pretrained is not None:
        updated["clip"]["pretrained_path"] = args.clip_pretrained
    if args.clip_default_pretrained is not None:
        updated["clip"]["default_pretrained"] = args.clip_default_pretrained
    if args.batch_size is not None:
        updated["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        updated["training"]["epochs"] = args.epochs
    if args.lr is not None:
        updated["training"]["lr"] = args.lr
    if args.graph_dim is not None:
        updated["model"]["tegar_dim"] = args.graph_dim
    if args.graph_layers is not None:
        updated["model"]["tegar_layers"] = args.graph_layers
    if args.cross_attn_heads is not None:
        updated["model"]["cross_attn_heads"] = args.cross_attn_heads
    if args.delta_scale_init is not None:
        updated["model"]["delta_scale_init"] = args.delta_scale_init
    if args.lambda_reg is not None:
        updated["training"]["lambda_reg"] = args.lambda_reg
    if args.num_workers is not None:
        updated["data"]["num_workers"] = args.num_workers
    if args.train_fraction is not None:
        updated["data"]["train_fraction"] = args.train_fraction
    if args.output_dir is not None:
        updated["training"]["output_dir"] = args.output_dir
    if args.max_train_batches is not None:
        updated["training"]["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        updated["training"]["max_val_batches"] = args.max_val_batches
    if args.disable_tensorboard:
        updated["training"]["tensorboard"] = False
    if args.disable_early_stop:
        updated["training"]["early_stop"] = False
    if args.disable_auto_tune_batch_size:
        updated["training"]["auto_tune_batch_size"] = False
    if args.early_stop_patience is not None:
        updated["training"]["early_stop_patience"] = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        updated["training"]["early_stop_min_delta"] = args.early_stop_min_delta
    if args.early_stop_warmup_evals is not None:
        updated["training"]["early_stop_warmup_evals"] = args.early_stop_warmup_evals
    if args.training_seed is not None:
        updated["training"]["training_seed"] = int(args.training_seed)
    if args.inner_split_seed is not None:
        updated["training"]["inner_split_seed"] = int(args.inner_split_seed)
    if args.inner_surrogate_fraction is not None:
        updated["training"]["inner_surrogate_fraction"] = float(args.inner_surrogate_fraction)
    if args.legacy_outer_unseen_validation:
        updated["training"]["legacy_outer_unseen_validation"] = True
    if args.protocol_stage is not None:
        updated["training"]["protocol_stage"] = str(args.protocol_stage)
    if args.fixed_epochs is not None:
        updated["training"]["fixed_epochs"] = int(args.fixed_epochs)
    if args.selection_manifest is not None:
        updated["training"]["selection_manifest"] = str(args.selection_manifest)
    return updated


def load_zero_shot_split(config: dict) -> dict:
    split_path = config["data"].get("unseen_labels_file")
    if not split_path:
        raise ValueError("data.unseen_labels_file must be set for zero-shot training.")
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Zero-shot split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_visual_prototypes(config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    proto_path = Path(config["data"].get("visual_prototypes_path", ""))
    if not proto_path.is_file():
        raise FileNotFoundError(
            f"Visual prototypes not found at {proto_path}. Run build_visual_prototypes.py first."
        )
    payload = torch.load(proto_path, map_location=device, weights_only=False)
    prototypes = payload["prototypes"].float().to(device)
    mask = payload["prototype_mask"].bool().to(device)
    return prototypes, mask


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


def build_loader_kwargs(
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    config: dict,
) -> dict:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 4))
    return kwargs


def configure_optimizer(model: torch.nn.Module, config: dict) -> AdamW:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check the model freeze settings.")
    optimizer_kwargs = {
        "lr": float(config["training"]["lr"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    if torch.cuda.is_available() and bool(config["training"].get("fused_optimizer", True)):
        optimizer_kwargs["fused"] = True
    try:
        return AdamW(trainable_params, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        return AdamW(trainable_params, **optimizer_kwargs)


def round_batch_size(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(value))
    return max(multiple, (int(value) // multiple) * multiple)


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def cleanup_after_oom(device: torch.device, model: torch.nn.Module | None = None) -> None:
    if model is not None:
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.empty_cache()


def current_free_gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return float(free_bytes / (1024 ** 3))


def compute_training_losses(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    mode_name: str,
    reg_weight: float,
    seen_indices: list[int],
    use_amp: bool,
    use_feature_cache: bool,
    asl_criterion: AsymmetricLoss | None = None,
    label_smoothing: float = 0.0,
    visual_prototypes: torch.Tensor | None = None,
    prototype_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits, aux = forward_model(model, inputs, return_aux=True, use_feature_cache=use_feature_cache)
        loss_reg = logits.new_zeros(())
        mean_abs_delta = logits.new_zeros(())
        if mode_name in PROTO_REGRESSION_MODES:
            classifiers = aux.get("classifiers")
            if classifiers is None or visual_prototypes is None or prototype_mask is None:
                raise RuntimeError("Prototype regression mode requires classifiers and loaded visual prototypes.")
            if hasattr(model, "output_norm"):
                classifiers = model.output_norm(classifiers)
            seen_classifiers = classifiers[:, prototype_mask, :]
            seen_classifiers_mean = seen_classifiers.mean(dim=0)
            seen_classifiers_norm = F.normalize(seen_classifiers_mean, dim=-1)
            target_protos = visual_prototypes[prototype_mask]
            loss_reg = F.mse_loss(seen_classifiers_norm, target_protos)
            loss_bce = logits.new_zeros(())
            loss = loss_reg
        elif mode_name in LOGIT_REFINER_MODES:
            smoothed_targets = smooth_labels(targets, label_smoothing) if label_smoothing > 0 else targets
            if asl_criterion is not None:
                loss_per_label = asl_criterion(logits, smoothed_targets)
            else:
                loss_per_label = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction="none")
            loss_bce = (loss_per_label * label_mask.unsqueeze(0)).sum() / (
                label_mask.sum().clamp_min(1.0) * inputs.size(0)
            )
            delta = aux.get("delta")
            if delta is not None:
                mean_abs_delta = delta.abs().mean()
                if mode_name in LOGIT_REFINER_REG_MODES and reg_weight > 0.0:
                    loss_reg = delta.pow(2).mean()
            else:
                mean_abs_delta = logits.new_zeros(())
            if mode_name in LOGIT_REFINER_REG_MODES and reg_weight > 0.0:
                loss = loss_bce + reg_weight * loss_reg
            else:
                loss_reg = logits.new_zeros(())
                loss = loss_bce
        else:
            smoothed_targets = smooth_labels(targets, label_smoothing) if label_smoothing > 0 else targets
            if asl_criterion is not None:
                loss_per_label = asl_criterion(logits, smoothed_targets)
                loss_bce = (loss_per_label * label_mask.unsqueeze(0)).sum() / (
                    label_mask.sum().clamp_min(1.0) * inputs.size(0)
                )
            else:
                loss_per_label = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction="none")
                loss_bce = (loss_per_label * label_mask.unsqueeze(0)).sum() / (
                    label_mask.sum().clamp_min(1.0) * inputs.size(0)
                )
            if mode_name in RESIDUAL_DELTA_MODES:
                delta = aux.get("delta")
                if delta is not None:
                    mean_abs_delta = delta.abs().mean()
                    if reg_weight > 0.0:
                        loss_reg = delta.pow(2).mean()
            elif mode_name in V3_MODES or mode_name in GENERATOR_MODES:
                loss_reg = logits.new_zeros(())
            else:
                aligned_nodes = aux.get("aligned_nodes")
                initial_nodes = aux.get("initial_nodes")
                if (
                    reg_weight > 0.0
                    and aligned_nodes is not None
                    and initial_nodes is not None
                    and hasattr(model, "compute_anchor_regularization")
                ):
                    loss_reg = model.compute_anchor_regularization(aligned_nodes, initial_nodes, seen_indices)
            loss = loss_bce + reg_weight * loss_reg
    return loss, loss_bce, loss_reg, mean_abs_delta, aux


def try_probe_batch(
    *,
    model: torch.nn.Module,
    dataset: MLRSNetDataset,
    device: torch.device,
    batch_size: int,
    label_mask: torch.Tensor,
    mode_name: str,
    reg_weight: float,
    seen_indices: list[int],
    use_amp: bool,
    use_channels_last: bool,
    use_feature_cache: bool,
    asl_criterion: AsymmetricLoss | None = None,
    label_smoothing: float = 0.0,
    visual_prototypes: torch.Tensor | None = None,
    prototype_mask: torch.Tensor | None = None,
) -> tuple[bool, float]:
    sample_image, sample_target = dataset[0]
    repeat_shape = (batch_size,) + (1,) * sample_image.ndim
    images = sample_image.unsqueeze(0).repeat(*repeat_shape)
    images = move_inputs_to_device(images, device, use_channels_last)
    targets = sample_target.unsqueeze(0).repeat(batch_size, 1).to(device, non_blocking=True)
    targets = redact_hidden_targets(targets, label_mask)

    cleanup_after_oom(device, model=model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        loss, _loss_bce, _loss_reg, _mean_abs_delta, _aux = compute_training_losses(
            model=model,
            inputs=images,
            targets=targets,
            label_mask=label_mask,
            mode_name=mode_name,
            reg_weight=reg_weight,
            seen_indices=seen_indices,
            use_amp=use_amp,
            use_feature_cache=use_feature_cache,
            asl_criterion=asl_criterion,
            label_smoothing=label_smoothing,
            visual_prototypes=visual_prototypes,
            prototype_mask=prototype_mask,
        )
        assert_finite_payload(loss, context="synthetic batch-size probe loss")
        loss.backward()
        assert_finite_gradients(model, context="synthetic batch-size probe backward")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_gb = get_gpu_memory_gb(device)
        cleanup_after_oom(device, model=model)
        del images, targets, loss
        return True, peak_gb
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        cleanup_after_oom(device, model=model)
        del images, targets
        return False, 0.0


def try_actual_batch(
    *,
    model: torch.nn.Module,
    dataset: MLRSNetDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    config: dict,
    label_mask: torch.Tensor,
    mode_name: str,
    reg_weight: float,
    seen_indices: list[int],
    use_amp: bool,
    use_channels_last: bool,
    use_feature_cache: bool,
    asl_criterion: AsymmetricLoss | None = None,
    label_smoothing: float = 0.0,
    visual_prototypes: torch.Tensor | None = None,
    prototype_mask: torch.Tensor | None = None,
) -> tuple[bool, float]:
    loader = DataLoader(
        dataset,
        **build_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            device=device,
            config=config,
        ),
    )

    cleanup_after_oom(device, model=model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        images, targets = next(iter(loader))
        images = move_inputs_to_device(images, device, use_channels_last)
        targets = targets.to(device, non_blocking=True)
        targets = redact_hidden_targets(targets, label_mask)
        loss, _loss_bce, _loss_reg, _mean_abs_delta, _aux = compute_training_losses(
            model=model,
            inputs=images,
            targets=targets,
            label_mask=label_mask,
            mode_name=mode_name,
            reg_weight=reg_weight,
            seen_indices=seen_indices,
            use_amp=use_amp,
            use_feature_cache=use_feature_cache,
            asl_criterion=asl_criterion,
            label_smoothing=label_smoothing,
            visual_prototypes=visual_prototypes,
            prototype_mask=prototype_mask,
        )
        assert_finite_payload(loss, context="actual batch-size probe loss")
        loss.backward()
        assert_finite_gradients(model, context="actual batch-size probe backward")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_gb = get_gpu_memory_gb(device)
        cleanup_after_oom(device, model=model)
        del images, targets, loss, loader
        return True, peak_gb
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        cleanup_after_oom(device, model=model)
        del loader
        return False, 0.0


def resolve_training_batch_size(
    *,
    model: torch.nn.Module,
    dataset: MLRSNetDataset,
    device: torch.device,
    config: dict,
    label_mask: torch.Tensor,
    mode_name: str,
    reg_weight: float,
    seen_indices: list[int],
    use_amp: bool,
    use_channels_last: bool,
    num_workers: int,
    use_feature_cache: bool,
    asl_criterion: AsymmetricLoss | None = None,
    label_smoothing: float = 0.0,
    visual_prototypes: torch.Tensor | None = None,
    prototype_mask: torch.Tensor | None = None,
) -> int:
    requested = max(1, int(config["training"]["batch_size"]))
    if device.type != "cuda" or not bool(config["training"].get("auto_tune_batch_size", True)):
        return requested

    multiple = max(1, int(config["training"].get("auto_tune_multiple", 256)))
    min_batch = max(1, int(config["training"].get("auto_tune_min_batch_size", min(256, requested))))
    max_batch = max(requested, int(config["training"].get("auto_tune_max_batch_size", requested)))
    probe_batch = min(max_batch, max(min_batch, int(config["training"].get("auto_tune_probe_batch_size", 256))))
    target_gpu_gb = float(config["training"].get("auto_tune_target_gpu_gb", 10.0))
    safety_margin_gb = float(config["training"].get("auto_tune_safety_margin_gb", 2.0))

    probe_batch = round_batch_size(probe_batch, multiple)
    max_batch = round_batch_size(max_batch, multiple)
    min_batch = round_batch_size(min_batch, multiple)
    requested = round_batch_size(requested, multiple)

    free_gpu_gb = current_free_gpu_memory_gb(device)
    target_by_free = max(float(min_batch), free_gpu_gb - safety_margin_gb)
    effective_target_gb = min(target_gpu_gb, target_by_free)

    if effective_target_gb <= 0.0:
        print(
            f"Auto batch tuner: free GPU memory is too low ({free_gpu_gb:.2f}G). "
            f"Keeping requested batch_size={requested}."
        )
        return requested

    ok, probe_peak_gb = try_probe_batch(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=probe_batch,
        label_mask=label_mask,
        mode_name=mode_name,
        reg_weight=reg_weight,
        seen_indices=seen_indices,
        use_amp=use_amp,
        use_channels_last=use_channels_last,
        use_feature_cache=use_feature_cache,
        asl_criterion=asl_criterion,
        label_smoothing=label_smoothing,
        visual_prototypes=visual_prototypes,
        prototype_mask=prototype_mask,
    )
    if not ok or probe_peak_gb <= 0.0:
        fallback = min(requested, probe_batch)
        print(
            f"Auto batch tuner: probe at batch_size={probe_batch} failed. "
            f"Falling back to batch_size={fallback}."
        )
        return fallback

    estimated = int(round(probe_batch * (effective_target_gb / max(probe_peak_gb, 1e-6))))
    candidate = max(min_batch, min(max_batch, round_batch_size(estimated, multiple)))

    best_candidate = None
    best_peak_gb = 0.0
    while candidate >= min_batch:
        ok, candidate_peak_gb = try_actual_batch(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=candidate,
            num_workers=num_workers,
            config=config,
            label_mask=label_mask,
            mode_name=mode_name,
            reg_weight=reg_weight,
            seen_indices=seen_indices,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
            use_feature_cache=use_feature_cache,
            asl_criterion=asl_criterion,
            label_smoothing=label_smoothing,
            visual_prototypes=visual_prototypes,
            prototype_mask=prototype_mask,
        )
        if ok:
            best_candidate = candidate
            best_peak_gb = candidate_peak_gb
            break
        next_candidate = round_batch_size(max(min_batch, int(candidate * 0.5)), multiple)
        if next_candidate >= candidate:
            break
        candidate = next_candidate

    if best_candidate is None:
        safe_fallback = min(requested, probe_batch)
        print(f"Auto batch tuner: could not verify larger batches. Using batch_size={safe_fallback}.")
        return safe_fallback

    target_floor_gb = effective_target_gb * 0.90
    upward_candidate = best_candidate + multiple
    while upward_candidate <= max_batch and best_peak_gb < target_floor_gb:
        ok, upward_peak_gb = try_actual_batch(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=upward_candidate,
            num_workers=num_workers,
            config=config,
            label_mask=label_mask,
            mode_name=mode_name,
            reg_weight=reg_weight,
            seen_indices=seen_indices,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
            use_feature_cache=use_feature_cache,
            asl_criterion=asl_criterion,
            label_smoothing=label_smoothing,
            visual_prototypes=visual_prototypes,
            prototype_mask=prototype_mask,
        )
        if not ok:
            break
        best_candidate = upward_candidate
        best_peak_gb = upward_peak_gb
        upward_candidate += multiple

    print(
        "Auto batch tuner: "
        f"probe={probe_batch} -> {probe_peak_gb:.2f}G, "
        f"selected batch_size={best_candidate} -> {best_peak_gb:.2f}G "
        f"(target {effective_target_gb:.2f}G, free {free_gpu_gb:.2f}G)."
    )
    return best_candidate


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    seen_labels: list[str],
    unseen_labels: list[str],
    max_batches: int | None = None,
    show_progress: bool = True,
    use_channels_last: bool = False,
    use_feature_cache: bool = False,
    decision_threshold: float = 0.5,
    forbidden_labels: list[str] | None = None,
    decision_threshold_space: str = "probability",
) -> dict[str, float | pd.DataFrame | list[str]]:
    model.eval()
    label_names = list(dataloader.dataset.label_names)
    evaluated_labels = list(dict.fromkeys([*seen_labels, *unseen_labels]))
    assert_no_forbidden_labels(evaluated_labels, forbidden_labels, context="training validation")
    evaluated_indices = [label_names.index(label) for label in evaluated_labels]
    forbidden_indices = [label_names.index(label) for label in (forbidden_labels or []) if label in label_names]
    all_logits: list[np.ndarray] = []
    all_clip_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    enhanced_anchor_sim_total = 0.0
    evidence_cls_sim_total = 0.0
    classifier_anchor_sim_total = 0.0
    aux_batches = 0
    used_patches_batches = 0
    total_batches = 0

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Validation", dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(iterator, start=1):
            images = move_inputs_to_device(images, device, use_channels_last)
            targets = targets.to(device, non_blocking=True)
            outputs = forward_model(model, images, return_aux=True, use_feature_cache=use_feature_cache)
            if isinstance(outputs, tuple):
                logits, aux = outputs
            else:
                logits, aux = outputs, {}
            assert_finite_payload(logits, context=f"validation logits batch {batch_idx}")
            all_logits.append(logits.detach().cpu().numpy())
            clip_logits = aux.get("clip_logits")
            if clip_logits is not None:
                assert_finite_payload(clip_logits, context=f"validation CLIP logits batch {batch_idx}")
                all_clip_logits.append(clip_logits.detach().cpu().numpy())
            safe_targets = targets.detach().cpu().numpy().copy()
            if forbidden_indices:
                safe_targets[:, forbidden_indices] = 0.0
            all_targets.append(safe_targets)
            total_batches += 1
            if "enhanced_anchor_sim" in aux or "evidence_cls_sim" in aux:
                enhanced_anchor_sim_total += float(aux.get("enhanced_anchor_sim", 0.0))
                evidence_cls_sim_total += float(aux.get("evidence_cls_sim", 0.0))
                classifier_anchor_sim_total += float(aux.get("classifier_anchor_sim", 0.0))
                aux_batches += 1
            elif "classifier_anchor_sim" in aux:
                classifier_anchor_sim_total += float(aux.get("classifier_anchor_sim", 0.0))
                aux_batches += 1
            if bool(aux.get("used_patches", False)):
                used_patches_batches += 1
            if max_batches is not None and batch_idx >= max_batches:
                break

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    per_class_ap = compute_per_class_ap(
        logits_np[:, evaluated_indices],
        targets_np[:, evaluated_indices],
        evaluated_labels,
    )
    metrics = {
        "overall_mAP": compute_map(per_class_ap),
        "per_class_ap": per_class_ap,
    }
    metrics.update(
        compute_zero_shot_metrics(
            logits_np,
            targets_np,
            label_names,
            seen_labels=seen_labels,
            unseen_labels=unseen_labels,
            decision_threshold=decision_threshold,
            decision_threshold_space=decision_threshold_space,
            forbidden_labels=forbidden_labels,
        )
    )
    if all_clip_logits:
        clip_logits_np = np.concatenate(all_clip_logits, axis=0)
        metrics.update(
            {
                "clip_per_class_ap": compute_per_class_ap(
                    clip_logits_np[:, evaluated_indices],
                    targets_np[:, evaluated_indices],
                    evaluated_labels,
                ),
            }
        )
        metrics.update(
            {
                f"clip_{key}": value
                for key, value in compute_zero_shot_metrics(
                    clip_logits_np,
                    targets_np,
                    label_names,
                    seen_labels=seen_labels,
                    unseen_labels=unseen_labels,
                    decision_threshold=decision_threshold,
                    decision_threshold_space=decision_threshold_space,
                    forbidden_labels=forbidden_labels,
                ).items()
                if key in {
                    "seen_mAP", "unseen_mAP", "harmonic_mean", "overall_mAP",
                    "seen_macro_F1", "seen_micro_F1", "seen_macro_recall",
                    "unseen_macro_F1", "unseen_micro_F1", "unseen_macro_recall",
                    "f1_harmonic_mean", "decision_threshold",
                }
            }
        )
    if aux_batches > 0:
        metrics["enhanced_anchor_sim"] = float(enhanced_anchor_sim_total / aux_batches)
        metrics["evidence_cls_sim"] = float(evidence_cls_sim_total / aux_batches)
        metrics["classifier_anchor_sim"] = float(classifier_anchor_sim_total / aux_batches)
    if total_batches > 0:
        metrics["used_patches_fraction"] = float(used_patches_batches / total_batches)
    for metric_name in (
        "overall_mAP",
        "seen_mAP",
        "unseen_mAP",
        "harmonic_mean",
        "seen_macro_F1",
        "seen_micro_F1",
        "seen_macro_recall",
        "unseen_macro_F1",
        "unseen_micro_F1",
        "unseen_macro_recall",
        "f1_harmonic_mean",
    ):
        assert_finite_payload(metrics.get(metric_name), context=f"validation metric {metric_name}")
    return metrics


def get_gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))


def create_run_directory(base_output_dir: str | Path, mode: str) -> Path:
    base_output_dir = Path(base_output_dir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_output_dir / f"{mode}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_output_dir / f"{mode}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def compute_regularization_weight(config: dict, epoch: int, mode: str) -> float:
    if mode in V3_MODES or mode in GENERATOR_MODES or mode in PROTO_REGRESSION_MODES:
        return 0.0
    if mode in LOGIT_REFINER_REG_MODES:
        return float(config["training"].get("lambda_delta_reg", 0.01))
    if mode in LOGIT_REFINER_MODES:
        return 0.0
    if mode not in RESIDUAL_DELTA_MODES:
        base_weight = float(config["training"].get("lambda_anchor", 0.0))
        if base_weight <= 0.0:
            return 0.0
        total_epochs = max(1, int(config["training"]["epochs"]))
        warmup_ratio = float(config["training"].get("lambda_anchor_warmup_ratio", 0.1))
        warmup_scale = float(config["training"].get("lambda_anchor_warmup_scale", 2.0))
        warmup_epochs = max(1, int(round(total_epochs * warmup_ratio)))
        if epoch <= warmup_epochs:
            return base_weight * warmup_scale
        return base_weight
    return float(config["training"].get("lambda_reg", 0.01))


def metric_value(metrics: dict, metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f"Metric '{metric_name}' not found in metrics: {list(metrics.keys())}")
    numeric = float(value)
    assert_finite_payload(numeric, context=f"selection metric {metric_name}")
    return numeric


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    dataset_name = normalize_dataset_name(config["data"].get("dataset", "mlrsnet"))
    if str(config["training"]["mode"]) in LOGIT_REFINER_MODES:
        config["training"]["selection_metric"] = "harmonic_mean"
        config["training"]["early_stop_patience"] = int(
            config["training"].get("logit_early_stop_patience", config["training"].get("early_stop_patience", 15))
        )
        config["training"]["early_stop_min_delta"] = float(
            config["training"].get("logit_early_stop_min_delta", 0.001)
        )
        config["training"]["early_stop_warmup_evals"] = int(
            config["training"].get("logit_early_stop_warmup_evals", config["training"].get("early_stop_warmup_evals", 5))
        )
    split_payload = load_zero_shot_split(config)
    outer_unseen_labels = list(split_payload["unseen_labels"])
    outer_seen_labels = list(split_payload["seen_labels"])
    seed_manifest = resolve_protocol_seeds(split_payload, config)
    outer_split_seed = seed_manifest["outer_split_seed"]
    training_seed = seed_manifest["training_seed"]
    inner_split_seed = seed_manifest["inner_split_seed"]
    legacy_outer_validation = bool(config["training"].get("legacy_outer_unseen_validation", False))
    protocol_stage = str(config["training"].get("protocol_stage", "inner_select"))
    locked_postprocessing: dict[str, object] = {}
    if protocol_stage == "outer_retrain":
        if legacy_outer_validation:
            raise ValueError("outer_retrain cannot be combined with legacy_outer_unseen_validation")
        fixed_epochs = config["training"].get("fixed_epochs")
        if fixed_epochs is None or int(fixed_epochs) <= 0:
            raise ValueError("outer_retrain requires --fixed_epochs with a positive value selected by inner validation")
        config["training"]["epochs"] = int(fixed_epochs)
        config["training"]["early_stop"] = False
        selection_manifest_path = config["training"].get("selection_manifest")
        if selection_manifest_path:
            selection_manifest_path = Path(selection_manifest_path).resolve()
            if not selection_manifest_path.is_file():
                raise FileNotFoundError(f"Inner selection manifest not found: {selection_manifest_path}")
            locked_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
            if locked_payload.get("threshold_source") != "inner_validation":
                raise ValueError("selection_manifest must come from strict inner validation")
            locked_calibration_bias = float(locked_payload.get("selected_calibration_bias", 0.0))
            assert_strict_nonlogit_calibration_bias_zero(
                mode_name=str(config["training"]["mode"]),
                legacy_outer_validation=legacy_outer_validation,
                calibration_bias=locked_calibration_bias,
            )
            locked_postprocessing = {
                "decision_threshold": float(locked_payload["decision_threshold"]),
                "decision_threshold_space": locked_payload.get("decision_threshold_space", "probability"),
                "decision_threshold_strategy": locked_payload.get("decision_threshold_strategy", "probability_grid"),
                "selected_mix_alpha": locked_payload.get("selected_mix_alpha"),
                "selected_calibration_bias": locked_calibration_bias,
                "selection_manifest": str(selection_manifest_path),
            }
        protocol = {
            "mode": "strict_outer_retrain",
            "outer_seen_labels": outer_seen_labels,
            "outer_unseen_labels": outer_unseen_labels,
            "inner_seen_labels": outer_seen_labels,
            "surrogate_unseen_labels": [],
            "training_hidden_labels": outer_unseen_labels,
            "selection_seen_labels": outer_seen_labels,
            "selection_unseen_labels": [],
            "inner_seed": inner_split_seed,
            "surrogate_fraction": 0.0,
        }
    else:
        protocol = build_nested_label_protocol(
            outer_seen_labels,
            outer_unseen_labels,
            inner_seed=inner_split_seed,
            surrogate_fraction=float(config["training"].get("inner_surrogate_fraction", 0.2)),
            legacy_outer_unseen_validation=legacy_outer_validation,
        )
    seen_labels = list(protocol["selection_seen_labels"])
    unseen_labels = list(protocol["selection_unseen_labels"])
    training_hidden_labels = list(protocol["training_hidden_labels"])
    forbidden_validation_labels = [] if legacy_outer_validation else list(outer_unseen_labels)
    config["training"]["protocol_mode"] = str(protocol["mode"])
    config["training"]["protocol_stage"] = protocol_stage
    config["training"]["seed_manifest"] = seed_manifest
    seed_everything(training_seed)

    kg_path = Path(config["data"]["kg_path"])
    if not kg_path.is_file():
        raise FileNotFoundError(f"KG definition not found at {kg_path}")
    kg_definition = KGDefinition(kg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    use_channels_last = configure_runtime(config, device)
    use_feature_cache = bool(config["data"].get("use_feature_cache", True)) and str(config["training"]["mode"]) in FEATURE_CACHE_MODES
    if not legacy_outer_validation and not use_feature_cache:
        raise RuntimeError(
            "Strict inner-surrogate protocol requires data.use_feature_cache=true and a cache-compatible mode; "
            "raw-image training is disabled for the A-track rerun."
        )
    image_size = int(config["clip"].get("image_size", config["data"].get("image_size", 224)))
    split_seed = int(config["data"]["split_seed"])
    split_ratio = tuple(config["data"]["split_ratio"])

    if use_feature_cache:
        train_cache_path = resolve_feature_cache_path(
            feature_cache_root=config["data"]["feature_cache_dir"],
            split="train",
            clip_model=config["clip"]["model_name"],
            clip_pretrained=config["clip"].get("pretrained_path"),
            clip_default_pretrained=config["clip"].get("default_pretrained"),
            split_seed=split_seed,
            image_size=image_size,
        )
        val_cache_path = None
        if protocol_stage != "outer_retrain":
            val_cache_path = resolve_feature_cache_path(
                feature_cache_root=config["data"]["feature_cache_dir"],
                split="val",
                clip_model=config["clip"]["model_name"],
                clip_pretrained=config["clip"].get("pretrained_path"),
                clip_default_pretrained=config["clip"].get("default_pretrained"),
                split_seed=split_seed,
                image_size=image_size,
            )
        if not train_cache_path.is_file() or (val_cache_path is not None and not val_cache_path.is_file()):
            raise FileNotFoundError(
                "Feature cache is enabled but cache files are missing. "
                f"Expected: train={train_cache_path}, val={val_cache_path}. "
                "Run build_feature_cache.py first."
            )
        train_dataset = create_feature_dataset(
            dataset_name,
            feature_cache_path=train_cache_path,
            unseen_labels=training_hidden_labels,
            return_metadata=False,
        )
        val_dataset = None
        if val_cache_path is not None:
            val_dataset = create_feature_dataset(
                dataset_name,
                feature_cache_path=val_cache_path,
                unseen_labels=training_hidden_labels,
                return_metadata=False,
            )
    else:
        train_dataset = create_image_dataset(
            dataset_name,
            data_root=config["data"]["root"],
            kg_path=kg_path,
            split="train",
            image_size=image_size,
            split_seed=split_seed,
            split_ratio=split_ratio,
            train_fraction=float(config["data"].get("train_fraction", 1.0)),
            unseen_labels=training_hidden_labels,
            clip_augment_train=bool(config["data"].get("clip_augment_train", True)),
        )
        val_dataset = None
        if protocol_stage != "outer_retrain":
            val_dataset = create_image_dataset(
                dataset_name,
                data_root=config["data"]["root"],
                kg_path=kg_path,
                split="val",
                image_size=image_size,
                split_seed=split_seed,
                split_ratio=split_ratio,
                unseen_labels=training_hidden_labels,
                clip_augment_train=False,
            )
    num_labels = int(train_dataset.node_count)

    label_mask = train_dataset.get_label_mask().to(device)
    seen_label_set = set(seen_labels)
    unseen_label_set = set(unseen_labels)
    training_hidden_label_set = set(training_hidden_labels)
    seen_indices = [idx for idx, label in enumerate(train_dataset.label_names) if label in seen_label_set]
    unseen_indices = [idx for idx, label in enumerate(train_dataset.label_names) if label in unseen_label_set]
    training_hidden_indices = [
        idx for idx, label in enumerate(train_dataset.label_names) if label in training_hidden_label_set
    ]
    num_workers = resolve_num_workers_for_cache(config, use_feature_cache=use_feature_cache, context="Training DataLoader")

    if str(config["training"]["mode"]) == "zs_clip":
        raise ValueError("zs_clip is evaluation-only. Use evaluate.py for the pure CLIP baseline.")

    model = create_model(
        mode=config["training"]["mode"],
        num_labels=num_labels,
        kg_path=str(kg_path),
        tegar_layers=int(config["model"]["tegar_layers"]),
        tegar_dim=int(config["model"]["tegar_dim"]),
        pretrained=False,
        clip_model_name=config["clip"]["model_name"],
        clip_pretrained_path=config["clip"].get("pretrained_path"),
        clip_default_pretrained=config["clip"].get("default_pretrained"),
        text_anchors_path=config["data"]["text_anchors_path"],
        exclusion_beta_init=float(config["model"].get("exclusion_beta_init", 0.1)),
        dropout=float(config["model"].get("dropout", 0.1)),
        spatial_heads=int(config["model"].get("spatial_heads", 4)),
        cross_attn_heads=int(config["model"].get("cross_attn_heads", 4)),
        delta_scale_init=float(config["model"].get("delta_scale_init", 1.0)),
        use_cached_features=use_feature_cache,
        exclusion_mode=config["model"].get("exclusion_mode"),
        disabled_relations=tuple(config["model"].get("disabled_relations", ())),
        strict_exclusion_score=bool(config["model"].get("strict_exclusion_score", True)),
        gate_initial_activation=float(config["model"].get("gate_initial_activation", 0.5)),
        gate_control=config["model"].get("gate_control"),
        gate_shuffle_seed=int(config["model"].get("gate_shuffle_seed", 20260826)),
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if hasattr(model, "set_seen_unseen_indices"):
        model.set_seen_unseen_indices(seen_indices, training_hidden_indices)

    visual_prototypes = None
    prototype_mask = None
    if str(config["training"]["mode"]) in PROTO_REGRESSION_MODES:
        visual_prototypes, prototype_mask = load_visual_prototypes(config, device)
        print(f"Loaded visual prototypes: {int(prototype_mask.sum().item())} seen classes")

    # Initialize ASL loss before batch-size probing because the probe reuses the
    # full training loss path.
    asl_gamma_neg = float(config["training"].get("asl_gamma_neg", 4))
    asl_gamma_pos = float(config["training"].get("asl_gamma_pos", 0))
    asl_clip_val = float(config["training"].get("asl_clip", 0.05))
    use_asl = bool(config["training"].get("use_asl", True))
    asl_criterion = AsymmetricLoss(
        gamma_neg=asl_gamma_neg,
        gamma_pos=asl_gamma_pos,
        clip=asl_clip_val,
        reduction="none",
    ) if use_asl else None
    label_smoothing = float(config["training"].get("label_smoothing", 0.1))
    mixup_alpha = float(config["training"].get("mixup_alpha", 0.2))
    if use_feature_cache and mixup_alpha > 0.0:
        print(
            "Disabling mixup: training is using cached CLIP features, so mixup would "
            "interpolate feature vectors instead of images."
        )
        mixup_alpha = 0.0

    reg_weight_epoch1 = compute_regularization_weight(config, 1, str(config["training"]["mode"]))
    tuned_batch_size = resolve_training_batch_size(
        model=model,
        dataset=train_dataset,
        device=device,
        config=config,
        label_mask=label_mask,
        mode_name=str(config["training"]["mode"]),
        reg_weight=reg_weight_epoch1,
        seen_indices=seen_indices,
        use_amp=use_amp,
        use_channels_last=use_channels_last,
        num_workers=num_workers,
        use_feature_cache=use_feature_cache,
        asl_criterion=asl_criterion,
        label_smoothing=label_smoothing,
        visual_prototypes=visual_prototypes,
        prototype_mask=prototype_mask,
    )
    config["training"]["batch_size"] = int(tuned_batch_size)
    val_batch_size = min(
        tuned_batch_size,
        int(config["eval"].get("batch_size", config["training"]["batch_size"])),
    )
    config["eval"]["batch_size"] = int(val_batch_size)

    train_loader = DataLoader(
        train_dataset,
        **build_loader_kwargs(
            batch_size=tuned_batch_size,
            shuffle=True,
            num_workers=num_workers,
            device=device,
            config=config,
        ),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            **build_loader_kwargs(
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=num_workers,
                device=device,
                config=config,
            ),
        )

    optimizer = configure_optimizer(model, config)

    # Cosine schedule with warmup (improves early training stability)
    warmup_epochs = int(config["training"].get("warmup_epochs", 5))
    total_epochs = max(1, int(config["training"]["epochs"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = create_run_directory(config["training"]["output_dir"], config["training"]["mode"])
    config["training"]["output_dir"] = str(output_dir)
    best_checkpoint_path = output_dir / f"best_{config['training']['mode']}.pt"
    last_checkpoint_path = output_dir / f"last_{config['training']['mode']}.pt"
    visualizer = TrainingVisualizer(
        output_dir=output_dir,
        enable_tensorboard=bool(config["training"].get("tensorboard", True)),
        run_name=output_dir.name,
    )
    protocol_manifest = {
        **protocol,
        **seed_manifest,
        "outer_split_file": str(Path(config["data"]["unseen_labels_file"]).resolve()),
        "feature_cache_required": not legacy_outer_validation,
        "feature_cache_used": bool(use_feature_cache),
        "validation_forbidden_labels": forbidden_validation_labels,
        "threshold_source": "inner_validation" if not legacy_outer_validation else "legacy_outer_validation",
        "locked_postprocessing": locked_postprocessing,
    }
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    visualizer.write_run_info(
        {
            "config": config,
            "device": str(device),
            "num_labels": num_labels,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset) if val_dataset is not None else 0,
            "seen_labels": seen_labels,
            "unseen_labels": unseen_labels,
            "outer_seen_labels": outer_seen_labels,
            "outer_unseen_labels": outer_unseen_labels,
            "surrogate_unseen_labels": list(protocol["surrogate_unseen_labels"]),
            "training_hidden_labels": training_hidden_labels,
            "protocol": protocol_manifest,
            "seeds": seed_manifest,
            "kg_path": str(kg_path),
            "active_labels": train_dataset.label_names,
            "run_dir": str(output_dir),
            "run_name": output_dir.name,
            "use_feature_cache": bool(use_feature_cache),
        }
    )
    print(f"Run directory: {output_dir}")
    print(
        f"Protocol: {protocol['mode']} | outer={len(outer_seen_labels)}/{len(outer_unseen_labels)} "
        f"seen/unseen | inner={len(seen_labels)}/{len(unseen_labels)} seen/surrogate-unseen"
    )
    print(
        f"Seeds: outer_split={outer_split_seed}, inner_split={inner_split_seed}, training={training_seed}"
    )
    print(f"Input pipeline: {'cached CLIP features' if use_feature_cache else 'raw images + frozen CLIP backbone'}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch sizes: train={tuned_batch_size}, val={val_batch_size if val_loader is not None else 'disabled'}")

    selection_metric_name = str(config["training"].get("selection_metric", "harmonic_mean"))
    best_score = -float("inf")
    val_history: list[float] = []
    early_stop_enabled = bool(config["training"].get("early_stop", True))
    early_stop_patience = int(config["training"].get("early_stop_patience", 8))
    early_stop_min_delta = float(config["training"].get("early_stop_min_delta", 0.002))
    early_stop_warmup_evals = int(config["training"].get("early_stop_warmup_evals", 3))
    batch_log_interval = max(1, int(config["training"].get("batch_log_interval", 10)))
    global_step = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        running_total_loss = 0.0
        running_bce_loss = 0.0
        running_reg_loss = 0.0
        running_mean_abs_delta = 0.0
        running_enhanced_anchor_sim = 0.0
        running_evidence_cls_sim = 0.0
        running_classifier_anchor_sim = 0.0
        running_used_patches = 0.0
        sample_count = 0
        reg_weight = compute_regularization_weight(config, epoch, str(config["training"]["mode"]))

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{int(config['training']['epochs']):03d}",
            dynamic_ncols=True,
            leave=True,
        )
        for batch_idx, (images, targets) in enumerate(progress, start=1):
            images = move_inputs_to_device(images, device, use_channels_last)
            targets = targets.to(device, non_blocking=True)
            targets = redact_hidden_targets(targets, label_mask)

            # Mixup data augmentation
            if mixup_alpha > 0 and model.training:
                images, targets = mixup_data(images, targets, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            loss, loss_bce, loss_reg, mean_abs_delta, aux = compute_training_losses(
                model=model,
                inputs=images,
                targets=targets,
                label_mask=label_mask,
                mode_name=str(config["training"]["mode"]),
                reg_weight=reg_weight,
                seen_indices=seen_indices,
                use_amp=use_amp,
                use_feature_cache=use_feature_cache,
                asl_criterion=asl_criterion,
                label_smoothing=label_smoothing,
                visual_prototypes=visual_prototypes,
                prototype_mask=prototype_mask,
            )

            assert_finite_payload(
                {
                    "loss": loss,
                    "bce_loss": loss_bce,
                    "reg_loss": loss_reg,
                    "mean_abs_delta": mean_abs_delta,
                },
                context=f"training epoch {epoch} batch {batch_idx}",
            )
            scaler.scale(loss).backward()
            # GradScaler deliberately skips an optimizer step when *scaled* gradients
            # overflow. Inspect gradients only when they are genuinely unscaled so a
            # recoverable AMP scale reduction is not misclassified as model corruption.
            if not scaler.is_enabled():
                assert_finite_gradients(
                    model,
                    context=f"training epoch {epoch} batch {batch_idx}",
                )
            scaler.step(optimizer)
            scaler.update()
            assert_finite_model_state(
                model,
                context=f"training epoch {epoch} batch {batch_idx} optimizer step",
                trainable_only=True,
            )

            batch_size = images.size(0)
            enhanced_anchor_sim_value = float(aux.get("enhanced_anchor_sim", 0.0))
            evidence_cls_sim_value = float(aux.get("evidence_cls_sim", 0.0))
            classifier_anchor_sim_value = float(aux.get("classifier_anchor_sim", 0.0))
            used_patches_value = 1.0 if bool(aux.get("used_patches", False)) else 0.0
            running_total_loss += float(loss.detach().cpu().item()) * batch_size
            running_bce_loss += float(loss_bce.detach().cpu().item()) * batch_size
            running_reg_loss += float(loss_reg.detach().cpu().item()) * batch_size
            running_mean_abs_delta += float(mean_abs_delta.detach().cpu().item()) * batch_size
            running_enhanced_anchor_sim += enhanced_anchor_sim_value * batch_size
            running_evidence_cls_sim += evidence_cls_sim_value * batch_size
            running_classifier_anchor_sim += classifier_anchor_sim_value * batch_size
            running_used_patches += used_patches_value * batch_size
            sample_count += batch_size
            global_step += 1

            if (
                batch_idx == 1
                or batch_idx % batch_log_interval == 0
                or (
                    config["training"].get("max_train_batches") is not None
                    and batch_idx >= int(config["training"]["max_train_batches"])
                )
                or batch_idx == len(train_loader)
            ):
                loss_value = float(loss.detach().cpu().item())
                bce_value = float(loss_bce.detach().cpu().item())
                reg_value = float(loss_reg.detach().cpu().item())
                mean_abs_delta_value = float(mean_abs_delta.detach().cpu().item())
                avg_loss = running_total_loss / max(sample_count, 1)
                gpu_gb = get_gpu_memory_gb(device)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{avg_loss:.4f}",
                    bce=f"{(running_bce_loss / max(sample_count, 1)):.4f}",
                    reg=f"{(running_reg_loss / max(sample_count, 1)):.4f}",
                    delta=f"{(running_mean_abs_delta / max(sample_count, 1)):.4f}",
                    eas=f"{(running_enhanced_anchor_sim / max(sample_count, 1)):.3f}",
                    ecs=f"{(running_evidence_cls_sim / max(sample_count, 1)):.3f}",
                    cas=f"{(running_classifier_anchor_sim / max(sample_count, 1)):.3f}",
                    gpu=f"{gpu_gb:.2f}G",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
                visualizer.log_batch(
                    global_step,
                    {
                        "loss": loss_value,
                        "bce_loss": bce_value,
                        "reg_loss": reg_value,
                        "mean_abs_delta": mean_abs_delta_value,
                        "enhanced_anchor_sim": enhanced_anchor_sim_value,
                        "evidence_cls_sim": evidence_cls_sim_value,
                        "classifier_anchor_sim": classifier_anchor_sim_value,
                        "used_patches": used_patches_value,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "gpu_gb": gpu_gb,
                    },
                )
            if config["training"].get("max_train_batches") is not None and batch_idx >= int(
                config["training"]["max_train_batches"]
            ):
                break

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
                category=UserWarning,
            )
            scheduler.step()
        epoch_time = time.perf_counter() - epoch_start
        train_loss = running_total_loss / max(sample_count, 1)
        train_bce_loss = running_bce_loss / max(sample_count, 1)
        train_reg_loss = running_reg_loss / max(sample_count, 1)
        train_mean_abs_delta = running_mean_abs_delta / max(sample_count, 1)
        train_enhanced_anchor_sim = running_enhanced_anchor_sim / max(sample_count, 1)
        train_evidence_cls_sim = running_evidence_cls_sim / max(sample_count, 1)
        train_classifier_anchor_sim = running_classifier_anchor_sim / max(sample_count, 1)
        train_used_patches_fraction = running_used_patches / max(sample_count, 1)

        epoch_metrics: dict[str, float | int | bool | None] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_bce_loss": float(train_bce_loss),
            "train_reg_loss": float(train_reg_loss),
            "reg_weight": float(reg_weight),
            "mean_abs_delta": float(train_mean_abs_delta),
            "train_enhanced_anchor_sim": float(train_enhanced_anchor_sim),
            "train_evidence_cls_sim": float(train_evidence_cls_sim),
            "train_classifier_anchor_sim": float(train_classifier_anchor_sim),
            "train_used_patches_fraction": float(train_used_patches_fraction),
            "delta_scale": float(
                getattr(model, "logit_temperature", torch.tensor(0.0)).detach().abs().cpu().item()
            ),
            "lr_main": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(epoch_time),
            "val_mAP": None,
            "val_seen_mAP": None,
            "val_unseen_mAP": None,
            "val_harmonic_mean": None,
            "val_clip_mAP": None,
            "val_clip_seen_mAP": None,
            "val_clip_unseen_mAP": None,
            "val_clip_harmonic_mean": None,
            "improved": False,
            "early_stop_triggered": False,
            "selection_metric": None,
        }

        should_validate = (
            protocol_stage != "outer_retrain"
            and (
                epoch % int(config["training"].get("eval_interval", 1)) == 0
                or epoch == int(config["training"]["epochs"])
            )
        )
        if should_validate:
            if val_loader is None:
                raise AssertionError("Validation loader is required during inner_select")
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                seen_labels=seen_labels,
                unseen_labels=unseen_labels,
                max_batches=config["training"].get("max_val_batches"),
                show_progress=True,
                use_channels_last=use_channels_last,
                use_feature_cache=use_feature_cache,
                forbidden_labels=forbidden_validation_labels,
            )
            score = metric_value(val_metrics, selection_metric_name)
            val_history.append(score)
            improved = score > (best_score + early_stop_min_delta)
            if improved:
                best_score = score
                _torch_save(
                    {
                        "epoch": epoch,
                        "mode": config["training"]["mode"],
                        "model_state": model.state_dict(),
                        "config": config,
                        "best_score": float(best_score),
                        "best_map": float(val_metrics["overall_mAP"]),
                        "best_metric_name": selection_metric_name,
                        "val_overall_map": float(val_metrics["overall_mAP"]),
                        "val_seen_map": float(val_metrics["seen_mAP"]),
                        "val_unseen_map": float(val_metrics["unseen_mAP"]),
                        "val_harmonic_mean": float(val_metrics["harmonic_mean"]),
                        "run_name": output_dir.name,
                    },
                    best_checkpoint_path,
                )

            epoch_metrics.update(
                {
                    "val_mAP": float(val_metrics["overall_mAP"]),
                    "val_seen_mAP": float(val_metrics["seen_mAP"]),
                    "val_unseen_mAP": float(val_metrics["unseen_mAP"]),
                    "val_harmonic_mean": float(val_metrics["harmonic_mean"]),
                    "val_clip_mAP": float(val_metrics.get("clip_overall_mAP", val_metrics["overall_mAP"])),
                    "val_clip_seen_mAP": float(val_metrics.get("clip_seen_mAP", val_metrics["seen_mAP"])),
                    "val_clip_unseen_mAP": float(val_metrics.get("clip_unseen_mAP", val_metrics["unseen_mAP"])),
                    "val_clip_harmonic_mean": float(val_metrics.get("clip_harmonic_mean", val_metrics["harmonic_mean"])),
                    "val_enhanced_anchor_sim": float(val_metrics.get("enhanced_anchor_sim", 0.0)),
                    "val_evidence_cls_sim": float(val_metrics.get("evidence_cls_sim", 0.0)),
                    "val_classifier_anchor_sim": float(val_metrics.get("classifier_anchor_sim", 0.0)),
                    "val_used_patches_fraction": float(val_metrics.get("used_patches_fraction", 0.0)),
                    "improved": improved,
                    "selection_metric": float(score),
                }
            )

            if str(config["training"]["mode"]) in V3_MODES:
                verbose_line = (
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | bce={train_bce_loss:.4f} | "
                    f"lr={float(optimizer.param_groups[0]['lr']):.2e} | "
                    f"Val: seen={val_metrics['seen_mAP']:.4f} | unseen={val_metrics['unseen_mAP']:.4f} | "
                    f"HM={val_metrics['harmonic_mean']:.4f} | "
                    f"clip_seen={float(val_metrics.get('clip_seen_mAP', val_metrics['seen_mAP'])):.4f} | "
                    f"clip_unseen={float(val_metrics.get('clip_unseen_mAP', val_metrics['unseen_mAP'])):.4f} | "
                    f"Aux: enhanced_anchor_sim={float(val_metrics.get('enhanced_anchor_sim', 0.0)):.4f} | "
                    f"evidence_cls_sim={float(val_metrics.get('evidence_cls_sim', 0.0)):.4f} | "
                    f"used_patches={float(val_metrics.get('used_patches_fraction', 0.0)):.2f}"
                )
            elif str(config["training"]["mode"]) in GENERATOR_MODES or str(config["training"]["mode"]) in PROTO_REGRESSION_MODES:
                verbose_line = (
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | bce={train_bce_loss:.4f} | "
                    f"Val: seen={val_metrics['seen_mAP']:.4f} | unseen={val_metrics['unseen_mAP']:.4f} | "
                    f"HM={val_metrics['harmonic_mean']:.4f} | "
                    f"Aux: classifier_anchor_sim={float(val_metrics.get('classifier_anchor_sim', 0.0)):.4f} | "
                    f"clip_seen={float(val_metrics.get('clip_seen_mAP', val_metrics['seen_mAP'])):.4f} | "
                    f"clip_unseen={float(val_metrics.get('clip_unseen_mAP', val_metrics['unseen_mAP'])):.4f}"
                )
            else:
                verbose_line = (
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | bce={train_bce_loss:.4f} | "
                    f"reg={train_reg_loss:.4f} | delta_scale={float(epoch_metrics['delta_scale']):.4f} | "
                    f"mean|delta|={train_mean_abs_delta:.4f} | "
                    f"Val: seen={val_metrics['seen_mAP']:.4f} | unseen={val_metrics['unseen_mAP']:.4f} | "
                    f"HM={val_metrics['harmonic_mean']:.4f} | "
                    f"clip_seen={float(val_metrics.get('clip_seen_mAP', val_metrics['seen_mAP'])):.4f} | "
                    f"clip_unseen={float(val_metrics.get('clip_unseen_mAP', val_metrics['unseen_mAP'])):.4f}"
                )
            print(verbose_line)

            if (
                early_stop_enabled
                and len(val_history) >= max(early_stop_warmup_evals, early_stop_patience)
                and len(val_history[-early_stop_patience:]) == early_stop_patience
            ):
                recent = val_history[-early_stop_patience:]
                if max(recent) - min(recent) <= early_stop_min_delta:
                    epoch_metrics["early_stop_triggered"] = True
                    _torch_save(
                        {
                            "epoch": epoch,
                            "mode": config["training"]["mode"],
                            "model_state": model.state_dict(),
                            "config": config,
                            "best_score": float(best_score),
                            "best_map": float(val_metrics["overall_mAP"]),
                            "best_metric_name": selection_metric_name,
                            "val_overall_map": float(val_metrics["overall_mAP"]),
                            "val_seen_map": float(val_metrics["seen_mAP"]),
                            "val_unseen_map": float(val_metrics["unseen_mAP"]),
                            "val_harmonic_mean": float(val_metrics["harmonic_mean"]),
                            "run_name": output_dir.name,
                        },
                        last_checkpoint_path,
                    )
                    visualizer.log_epoch(epoch_metrics)
                    print(
                        f"Early stopping at epoch {epoch}: recent {selection_metric_name} range "
                        f"{max(recent) - min(recent):.6f} <= {early_stop_min_delta:.6f}"
                    )
                    break

        _torch_save(
            {
                "epoch": epoch,
                "mode": config["training"]["mode"],
                "model_state": model.state_dict(),
                "config": config,
                "best_score": float(best_score) if math.isfinite(float(best_score)) else None,
                "best_map": float(epoch_metrics["val_mAP"]) if epoch_metrics["val_mAP"] is not None else None,
                "best_metric_name": selection_metric_name,
                "val_overall_map": epoch_metrics["val_mAP"],
                "val_seen_map": epoch_metrics["val_seen_mAP"],
                "val_unseen_map": epoch_metrics["val_unseen_mAP"],
                "val_harmonic_mean": epoch_metrics["val_harmonic_mean"],
                "run_name": output_dir.name,
            },
            last_checkpoint_path,
        )
        visualizer.log_epoch(epoch_metrics)

    if protocol_stage == "outer_retrain":
        fixed_epoch_payload = {
            "epoch": int(config["training"]["epochs"]),
            "mode": config["training"]["mode"],
            "model_state": model.state_dict(),
            "config": config,
            "protocol": protocol_manifest,
            **locked_postprocessing,
            "best_metric_name": "fixed_epochs_from_inner_selection",
            "best_score": None,
            "run_name": output_dir.name,
        }
        _torch_save(fixed_epoch_payload, best_checkpoint_path)

    visualizer.close()

    # ---------- Post-training calibration / mixing search ----------
    if protocol_stage != "outer_retrain" and best_checkpoint_path.is_file():
        if val_loader is None:
            raise AssertionError("Inner selection requires a validation loader")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        raw_best_hm = float(best_ckpt.get("val_harmonic_mean", best_score))
        mode_name = str(config["training"]["mode"])
        selected_mix_alpha: float | None = None
        selected_calibration_bias = 0.0
        calibration_policy = "not_applicable_logit_refiner"

        if mode_name in LOGIT_REFINER_MODES:
            print("\nSearching for optimal CLIP/refiner mixing alpha...")
            mix_alpha, mix_hm = search_logit_mix_alpha(
                model=model,
                dataloader=val_loader,
                device=device,
                label_names=train_dataset.label_names,
                seen_labels=seen_labels,
                unseen_labels=unseen_labels,
                forbidden_labels=forbidden_validation_labels,
                use_channels_last=use_channels_last,
                use_feature_cache=use_feature_cache,
            )
            print(
                f"Mix result: alpha={mix_alpha:.2f}, mixed HM={mix_hm:.4f} "
                f"(raw HM={raw_best_hm:.4f})"
            )
            if mix_hm > raw_best_hm + 1e-6:
                selected_mix_alpha = float(mix_alpha)
                best_ckpt["delta_mix_alpha"] = mix_alpha
                best_ckpt["calibrated_harmonic_mean"] = mix_hm
                _torch_save(best_ckpt, best_checkpoint_path)
                print("Mixing alpha improved HM and was saved to checkpoint.")
            else:
                selected_mix_alpha = 1.0
                print("Mixing alpha did not improve HM; checkpoint left unchanged.")
        elif should_search_legacy_calibration_bias(mode_name, legacy_outer_validation):
            calibration_policy = "legacy_outer_unseen_map_hm_search"
            print("\nSearching for optimal calibration bias...")
            cal_bias, cal_hm = search_calibration_bias(
                model=model,
                dataloader=val_loader,
                device=device,
                seen_indices=seen_indices,
                unseen_indices=unseen_indices,
                label_names=train_dataset.label_names,
                seen_labels=seen_labels,
                unseen_labels=unseen_labels,
                forbidden_labels=forbidden_validation_labels,
                use_channels_last=use_channels_last,
                use_feature_cache=use_feature_cache,
            )
            print(
                f"Calibration result: bias={cal_bias:.2f}, calibrated HM={cal_hm:.4f} "
                f"(raw HM={raw_best_hm:.4f})"
            )
            if cal_hm > raw_best_hm + 1e-6:
                selected_calibration_bias = float(cal_bias)
                best_ckpt["calibration_bias"] = cal_bias
                best_ckpt["calibrated_harmonic_mean"] = cal_hm
                _torch_save(best_ckpt, best_checkpoint_path)
                print("Calibration bias improved HM and was saved to checkpoint.")
            else:
                print("Calibration bias did not improve HM; checkpoint left unchanged.")
        else:
            calibration_policy = "disabled_strict_non_logit"
            print(
                "\nStrict non-logit protocol: legacy mAP-based calibration-bias search is disabled; "
                "calibration_bias=0 and only the inner-validation F1-HM threshold is selected."
            )

        assert_strict_nonlogit_calibration_bias_zero(
            mode_name=mode_name,
            legacy_outer_validation=legacy_outer_validation,
            calibration_bias=selected_calibration_bias,
        )

        decision_threshold, threshold_metrics = search_model_decision_threshold(
            model=model,
            dataloader=val_loader,
            device=device,
            label_names=train_dataset.label_names,
            seen_labels=seen_labels,
            unseen_labels=unseen_labels,
            seen_indices=seen_indices,
            forbidden_labels=forbidden_validation_labels,
            calibration_bias=selected_calibration_bias,
            delta_mix_alpha=selected_mix_alpha,
            use_channels_last=use_channels_last,
            use_feature_cache=use_feature_cache,
            candidate_thresholds=config["training"].get("decision_threshold_grid"),
            threshold_strategy=config["training"].get("decision_threshold_strategy", "exact_logit"),
        )
        scalar_threshold_metrics = {
            key: float(value)
            for key, value in threshold_metrics.items()
            if isinstance(value, (float, int, np.floating, np.integer))
        }
        best_ckpt["decision_threshold"] = float(decision_threshold)
        best_ckpt["decision_threshold_space"] = threshold_metrics.get("decision_threshold_space", "probability")
        best_ckpt["decision_threshold_strategy"] = threshold_metrics.get("decision_threshold_strategy", "probability_grid")
        best_ckpt["threshold_source"] = "inner_validation"
        best_ckpt["selected_mix_alpha"] = selected_mix_alpha
        best_ckpt["selected_calibration_bias"] = float(selected_calibration_bias)
        best_ckpt["calibration_policy"] = calibration_policy
        best_ckpt["inner_validation_threshold_metrics"] = scalar_threshold_metrics
        best_ckpt["protocol"] = protocol_manifest
        _torch_save(best_ckpt, best_checkpoint_path)
        threshold_manifest = {
            "source": "strict_inner_validation" if not legacy_outer_validation else "inner_validation",
            "decision_threshold": float(decision_threshold),
            "decision_threshold_space": best_ckpt["decision_threshold_space"],
            "decision_threshold_strategy": best_ckpt["decision_threshold_strategy"],
            "threshold_source": "inner_validation",
            "objective": "harmonic_mean_of_seen_and_surrogate_unseen_macro_F1",
            "selected_mix_alpha": selected_mix_alpha,
            "selected_calibration_bias": float(selected_calibration_bias),
            "calibration_policy": calibration_policy,
            "inner_seen_labels": seen_labels,
            "surrogate_unseen_labels": unseen_labels,
            "forbidden_outer_unseen_labels": forbidden_validation_labels,
            "metrics": scalar_threshold_metrics,
            "seeds": seed_manifest,
            "best_checkpoint": str(best_checkpoint_path),
        }
        (output_dir / "threshold_manifest.json").write_text(
            json.dumps(threshold_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Inner-validation decision threshold={decision_threshold:.8g} ({best_ckpt['decision_threshold_space']}), "
            f"F1-HM={float(threshold_metrics['f1_harmonic_mean']):.4f}"
        )

    print(f"Training finished. Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
