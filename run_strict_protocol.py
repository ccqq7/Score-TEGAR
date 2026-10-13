from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from scripts.capture_repro_manifest import capture_repro_manifest
from utils.feature_cache import resolve_feature_cache_path
from utils.metrics import build_nested_label_protocol, search_decision_threshold


PROJECT_ROOT = Path(__file__).resolve().parent
METHOD_TO_MODE = {
    "linear_probe": "zs_linear_probe",
    "clip_adapter": "zs_clip_adapter",
    "feature_tegar": "zs_feature_tegar",
    "score_graph": "zs_logit_homo",
    "score_graph_pm": "zs_logit_homo_pm",
    "score_tegar": "zs_logit_tegar",
    "gcnz_anchor": "zs_gcnz_anchor",
}
EMPIRICAL_RELATIONS = ("often_cooccur", "statistical_exclusion", "hierarchical")
REPRO_SOURCE_FILES = (
    "run_strict_protocol.py",
    "scripts/capture_repro_manifest.py",
    "train.py",
    "evaluate.py",
    "datasets/dataset_factory.py",
    "datasets/multilabel_base.py",
    "datasets/mlrsnet_dataset.py",
    "datasets/bigearthnet_dataset.py",
    "datasets/chinamas_dataset.py",
    "models/classifier.py",
    "models/clip_backbone.py",
    "models/baselines.py",
    "models/tegar.py",
    "models/logit_refiner.py",
    "models/controlled_baselines.py",
    "utils/feature_cache.py",
    "utils/kg_builder.py",
    "utils/losses.py",
    "utils/metrics.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and run the leakage-free JAG revision protocol.")
    parser.add_argument("--config", default="./configs/strict_a.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--phase", choices=["prepare", "audit", "pilot", "full"], default="prepare")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--pilot_outer_seed", type=int, default=42)
    parser.add_argument("--pilot_training_seed", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hashed_file(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required artifact not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _validate_hashed_file(record: dict, *, role: str) -> Path:
    path = Path(record.get("path", "")).resolve()
    if not path.is_file():
        raise AssertionError(f"{role} is missing: {path}")
    actual = _sha256_file(path)
    if actual != record.get("sha256"):
        raise AssertionError(f"{role} hash mismatch: {path}")
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise AssertionError(f"{role} size mismatch: {path}")
    return path


def _assert_finite_payload(value: object, *, context: str) -> None:
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
            _assert_finite_payload(item, context=f"{context}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_payload(item, context=f"{context}[{index}]")


def _load_finite_json(path: Path, *, role: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{role} must contain a JSON object: {path}")
    _assert_finite_payload(payload, context=f"{role} {path}")
    return payload


def _load_finite_checkpoint(path: Path, *, role: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(f"{role} must contain a checkpoint dictionary: {path}")
    _assert_finite_payload(payload, context=f"{role} {path}")
    return payload


def _validate_numeric_artifact(path: Path, *, role: str) -> None:
    role_key = role.lower()
    if any(name in role_key for name in ("inner_checkpoint", "outer_checkpoint")):
        _load_finite_checkpoint(path, role=role)
    elif any(name in role_key for name in ("threshold_manifest", "locked_selection_manifest", "test_metrics")):
        _load_finite_json(path, role=role)


def _run(command: list[str], stage: str) -> None:
    print(f"\n[{stage}]\n{' '.join(command)}", flush=True)
    environment = os.environ.copy()
    # PyTorch requires this variable to exist before the CUDA process starts;
    # setting it inside train.py/evaluate.py can occur after CUDA initialization.
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"Stage failed ({stage}), return code {completed.returncode}")


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload["protocol"].get("legacy_mode"):
        raise ValueError("run_strict_protocol.py refuses legacy_mode=true")
    if not payload["training"].get("use_feature_cache", False):
        raise ValueError("Strict revision runs must use the existing offline feature caches.")
    if payload["training"].get("allow_raw_images", False):
        raise ValueError("Raw-image training is disabled for this cached-feature revision.")
    return payload


def _cache_path(config: dict, dataset_name: str, split: str) -> Path:
    dataset = config["datasets"][dataset_name]
    clip = config["clip"]
    return resolve_feature_cache_path(
        feature_cache_root=_resolve(dataset["feature_cache_root"]),
        split=split,
        clip_model=clip["model_name"],
        clip_pretrained=str(_resolve(clip["pretrained_path"])),
        clip_default_pretrained=clip.get("default_pretrained"),
        split_seed=int(config["protocol"]["image_split_seed"]),
        image_size=int(clip["image_size"]),
    )


def _load_cache_header(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path.resolve()),
        "split": str(payload.get("split", "unknown")),
        "label_names": list(payload["label_names"]),
        "feature_shape": list(payload["features"].shape),
        "label_shape": list(payload["labels"].shape),
        "feature_dtype": str(payload["features"].dtype),
        "clip_model": payload.get("clip_model"),
        "clip_pretrained": payload.get("clip_pretrained"),
        "clip_pretrained_sha256": payload.get("clip_pretrained_sha256"),
        "clip_default_pretrained": payload.get("clip_default_pretrained"),
        "split_seed": payload.get("split_seed"),
        "image_size": payload.get("image_size"),
        "sha256": _sha256_file(path),
    }


def _declared_checkpoint_matches(declared: object, expected: Path) -> bool:
    if not isinstance(declared, str) or not declared.strip():
        return False
    declared_path = Path(declared)
    if declared_path.is_absolute():
        return declared_path.resolve() == expected.resolve()
    declared_norm = declared.replace("\\", "/").lstrip("./").casefold()
    expected_norm = str(expected.resolve()).replace("\\", "/").casefold()
    return expected_norm.endswith(f"/{declared_norm}") or expected_norm == declared_norm


def _validate_cache_header(
    header: dict,
    *,
    config: dict,
    dataset_name: str,
    split: str,
) -> str:
    """Fail closed on cache/config mismatches and report checkpoint binding strength."""
    clip = config["clip"]
    protocol = config["protocol"]
    expected_path = _cache_path(config, dataset_name, split).resolve()
    if Path(header.get("path", "")).resolve() != expected_path:
        raise AssertionError(f"Cache path mismatch for {dataset_name}/{split}")
    if header.get("split") != split:
        raise AssertionError(f"Cache split declaration mismatch for {dataset_name}/{split}")
    if header.get("clip_model") != clip["model_name"]:
        raise AssertionError(f"Cache CLIP model mismatch for {dataset_name}/{split}")
    if int(header.get("split_seed", -1)) != int(protocol["image_split_seed"]):
        raise AssertionError(f"Cache split seed mismatch for {dataset_name}/{split}")
    if int(header.get("image_size", -1)) != int(clip["image_size"]):
        raise AssertionError(f"Cache image size mismatch for {dataset_name}/{split}")
    if header.get("clip_default_pretrained") != clip.get("default_pretrained"):
        raise AssertionError(f"Cache fallback checkpoint mismatch for {dataset_name}/{split}")
    expected_checkpoint = _resolve(clip["pretrained_path"])
    if not _declared_checkpoint_matches(header.get("clip_pretrained"), expected_checkpoint):
        raise AssertionError(f"Cache checkpoint declaration mismatch for {dataset_name}/{split}")

    feature_shape = list(header.get("feature_shape", []))
    label_shape = list(header.get("label_shape", []))
    label_names = list(header.get("label_names", []))
    if len(feature_shape) != 2 or len(label_shape) != 2:
        raise AssertionError(f"Cache tensors must be rank-2 for {dataset_name}/{split}")
    if feature_shape[0] != label_shape[0] or label_shape[1] != len(label_names):
        raise AssertionError(f"Cache tensor dimensions disagree for {dataset_name}/{split}")

    embedded_hash = header.get("clip_pretrained_sha256")
    current_hash = _sha256_file(expected_checkpoint)
    if embedded_hash is not None:
        if embedded_hash != current_hash:
            raise AssertionError(f"Cache embedded checkpoint hash mismatch for {dataset_name}/{split}")
        return "embedded_sha256_match"
    return "declared_path_match_only_no_embedded_sha256"


def _write_json(path: Path, payload: dict) -> None:
    _assert_finite_payload(payload, context=f"JSON artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _new_run_directory(base_dir: Path, mode: str, before: set[Path]) -> Path:
    candidates = [
        path.resolve()
        for path in base_dir.glob(f"{mode}_*")
        if path.is_dir() and path.resolve() not in before
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one new {mode} run under {base_dir}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    scalar = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (float, int, np.floating, np.integer))
    }
    _assert_finite_payload(scalar, context="scalar metrics")
    return scalar


def _resolved_train_config(
    *,
    strict_config: dict,
    dataset_name: str,
    graph_path: Path,
    anchor_path: Path,
    outer_split_path: Path,
    output_dir: Path,
    mode: str,
    training_seed: int,
    inner_seed: int,
    protocol_stage: str,
    epochs: int,
    fixed_epochs: int | None,
    selection_manifest: Path | None,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict:
    base = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    dataset = strict_config["datasets"][dataset_name]
    clip = strict_config["clip"]
    model = strict_config["model"]
    training = strict_config["training"]

    base["clip"].update(
        {
            "model_name": clip["model_name"],
            "pretrained_path": str(_resolve(clip["pretrained_path"])),
            "default_pretrained": clip.get("default_pretrained"),
            "image_size": int(clip["image_size"]),
        }
    )
    base["data"].update(
        {
            "dataset": dataset_name,
            "root": str(_resolve(dataset["data_root"])),
            "kg_path": str(graph_path),
            "text_anchors_path": str(anchor_path),
            "feature_cache_dir": str(_resolve(dataset["feature_cache_root"])),
            "use_feature_cache": True,
            "unseen_labels_file": str(outer_split_path),
            "split_seed": int(strict_config["protocol"]["image_split_seed"]),
            "num_workers": 0,
            "feature_num_workers": 0,
            "force_single_worker_on_windows_feature": True,
        }
    )
    base["model"].update(
        {
            "tegar_dim": int(model["hidden_dim"]),
            "tegar_layers": int(model["layers"]),
            "dropout": float(model["dropout"]),
            "exclusion_beta_init": float(model["exclusion_beta_init"]),
            "gate_initial_activation": float(model.get("gate_initial_activation", 0.5)),
            "gate_control": str(model.get("gate_control", "learned")),
            "gate_shuffle_seed": int(model.get("gate_shuffle_seed", 20260826)),
            "exclusion_mode": str(model.get("exclusion_mode", "negative")),
            "disabled_relations": list(model.get("disabled_relations", [])),
            "strict_exclusion_score": bool(model.get("strict_exclusion_score", True)),
        }
    )
    base["training"].update(
        {
            "mode": mode,
            "batch_size": int(training["batch_size"]),
            "epochs": int(epochs),
            "lr": float(training["lr"]),
            "weight_decay": float(training["weight_decay"]),
            "warmup_epochs": min(int(training["warmup_epochs"]), max(0, int(epochs) - 1)),
            "use_asl": bool(training["use_asl"]),
            "asl_gamma_neg": float(training["asl_gamma_neg"]),
            "asl_gamma_pos": float(training["asl_gamma_pos"]),
            "asl_clip": float(training["asl_clip"]),
            "label_smoothing": float(training["label_smoothing"]),
            "training_seed": int(training_seed),
            "inner_split_seed": int(inner_seed),
            "inner_surrogate_fraction": float(strict_config["protocol"]["inner_surrogate_fraction"]),
            "decision_threshold_grid": list(strict_config["protocol"]["threshold_grid"]),
            "decision_threshold_strategy": strict_config["protocol"].get("threshold_strategy", "exact_logit"),
            "protocol_stage": protocol_stage,
            "fixed_epochs": int(fixed_epochs) if fixed_epochs is not None else None,
            "selection_manifest": str(selection_manifest) if selection_manifest is not None else None,
            "legacy_outer_unseen_validation": False,
            "output_dir": str(output_dir),
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "tensorboard": False,
            "deterministic": bool(training.get("deterministic", True)),
            "deterministic_warn_only": bool(
                training.get("deterministic_warn_only", False)
            ),
            "auto_tune_batch_size": False,
        }
    )
    base["eval"]["batch_size"] = int(training["batch_size"])
    return base


def _select_clip_threshold(
    *,
    config: dict,
    dataset_name: str,
    anchor_path: Path,
    inner_split_path: Path,
    output_path: Path,
    max_val_batches: int | None,
) -> dict:
    val_payload = torch.load(_cache_path(config, dataset_name, "val"), map_location="cpu", weights_only=False)
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    label_names = list(val_payload["label_names"])
    if list(anchor_payload["label_names"]) != label_names:
        raise ValueError(f"{dataset_name}: CLIP threshold selection label order differs from basic anchors")
    features = val_payload["features"].float()
    targets = val_payload["labels"].float()
    if max_val_batches is not None:
        limit = min(len(features), int(max_val_batches) * int(config["training"]["batch_size"]))
        features = features[:limit]
        targets = targets[:limit]
    logits = 100.0 * (F.normalize(features, dim=-1) @ F.normalize(anchor_payload["anchors"].float(), dim=-1).T)
    _assert_finite_payload(logits, context=f"{dataset_name} CLIP threshold logits")
    inner_split = json.loads(inner_split_path.read_text(encoding="utf-8"))
    threshold, metrics = search_decision_threshold(
        logits.numpy(),
        targets.numpy(),
        label_names,
        seen_labels=list(inner_split["selection_seen_labels"]),
        unseen_labels=list(inner_split["selection_unseen_labels"]),
        candidate_thresholds=list(config["protocol"]["threshold_grid"]),
        strategy=config["protocol"].get("threshold_strategy", "exact_logit"),
        forbidden_labels=list(inner_split["outer_unseen_labels"]),
    )
    manifest = {
        "schema_version": 1,
        "method": "clip_zs",
        "decision_threshold": float(threshold),
        "decision_threshold_space": metrics.get("decision_threshold_space", "probability"),
        "decision_threshold_strategy": metrics.get("decision_threshold_strategy", "probability_grid"),
        "source": "strict_inner_validation",
        "threshold_source": "inner_validation",
        "objective": "harmonic_mean_of_seen_and_surrogate_unseen_macro_F1",
        "inner_split": str(inner_split_path),
        "forbidden_outer_unseen_labels": list(inner_split["outer_unseen_labels"]),
        "selected_mix_alpha": None,
        "selected_calibration_bias": 0.0,
        "metrics": _scalar_metrics(metrics),
    }
    _write_json(output_path, manifest)
    return manifest


def _locate_eval_metrics(eval_root: Path) -> Path:
    candidates = sorted(eval_root.rglob("zeroshot_metrics.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Evaluation did not create zeroshot_metrics.json under {eval_root}")
    metrics_path = candidates[-1]
    _load_finite_json(metrics_path, role="test_metrics")
    return metrics_path


def _job_directory(
    output_root: Path,
    *,
    phase: str,
    dataset_name: str,
    outer_seed: int,
    training_seed: int,
    method: str,
) -> Path:
    if phase not in {"pilot", "full"}:
        raise ValueError(f"Experiments require phase=pilot or phase=full, got {phase!r}")
    phase_root = output_root / phase
    return phase_root / dataset_name / f"outer_{outer_seed}" / f"train_{training_seed}" / method


def _resolved_job_spec(
    *,
    phase: str,
    dataset_name: str,
    split_record: dict,
    method: str,
    training_seed: int,
    strict_config_path: Path,
    strict_config: dict,
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
) -> dict:
    payload = {
        "schema_version": 1,
        "phase": phase,
        "dataset": dataset_name,
        "outer_seed": int(split_record["outer_seed"]),
        "inner_seed": int(split_record["inner_seed"]),
        "training_seed": int(training_seed),
        "method": method,
        "strict_config": str(strict_config_path.resolve()),
        "strict_config_sha256": _sha256_file(strict_config_path),
        "resolved_strict_config_sha256": _sha256_json(strict_config),
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_test_batches": max_test_batches,
        "feature_cache_only": True,
    }
    return payload


def _configured_training_seeds(configured_seeds: Sequence[int]) -> list[int]:
    """Validate the configured initialization seeds without changing their order."""
    configured = [int(seed) for seed in configured_seeds]
    if not configured:
        raise ValueError("Strict protocol declares no training seeds")
    if len(configured) != len(set(configured)):
        raise ValueError(f"Strict protocol declares duplicate training seeds: {configured}")
    return configured


def _validate_existing_full_job(
    *,
    result_path: Path,
    expected_job_spec: dict,
    resolved_job_config_path: Path,
) -> dict:
    """Fail closed: reuse only an intact full job with identical inputs."""
    expected_phase = expected_job_spec.get("phase")
    if expected_phase != "full":
        raise ValueError("--skip_existing is only valid for phase=full")
    payload = _load_finite_json(result_path, role="completed run manifest")
    if payload.get("phase") != expected_phase:
        raise AssertionError(
            f"Refusing to reuse phase={payload.get('phase')!r} result as phase={expected_phase!r}: "
            f"{result_path}"
        )
    if payload.get("strict_config_sha256") != expected_job_spec["strict_config_sha256"]:
        raise AssertionError(f"Strict config changed since completed run: {result_path}")
    if payload.get("resolved_strict_config_sha256") != expected_job_spec["resolved_strict_config_sha256"]:
        raise AssertionError(f"Resolved strict config changed since completed run: {result_path}")
    if not resolved_job_config_path.is_file():
        raise AssertionError(f"Resolved job config is missing: {resolved_job_config_path}")
    resolved_job_spec = json.loads(resolved_job_config_path.read_text(encoding="utf-8"))
    expected_hash = _sha256_json(expected_job_spec)
    if _sha256_json(resolved_job_spec) != expected_hash:
        raise AssertionError(f"Resolved job config no longer matches requested job: {resolved_job_config_path}")
    if payload.get("resolved_job_config_sha256") != _sha256_file(resolved_job_config_path):
        raise AssertionError(f"Resolved job config file hash mismatch: {resolved_job_config_path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise AssertionError(f"Completed run has no auditable artifact registry: {result_path}")
    for role, record in artifacts.items():
        if not isinstance(record, dict):
            raise AssertionError(f"Invalid artifact record {role!r}: {result_path}")
        artifact_path = _validate_hashed_file(record, role=f"completed job artifact {role}")
        _validate_numeric_artifact(artifact_path, role=role)
    repro_record = payload.get("reproducibility_manifest")
    if not isinstance(repro_record, dict):
        raise AssertionError(f"Completed run lacks a reproducibility manifest: {result_path}")
    repro_path = _validate_hashed_file(repro_record, role="reproducibility manifest")
    repro = _load_finite_json(repro_path, role="reproducibility manifest")
    for role, record in repro.get("artifacts", {}).items():
        artifact_path = _validate_hashed_file(record, role=f"reproducibility input {role}")
        _validate_numeric_artifact(artifact_path, role=role)
    checkpoint = repro.get("checkpoint")
    if checkpoint is not None:
        checkpoint_path = _validate_hashed_file(checkpoint, role="reproducibility checkpoint")
        _load_finite_checkpoint(checkpoint_path, role="reproducibility checkpoint")
    protocol_config = repro.get("protocol_config")
    if not isinstance(protocol_config, dict):
        raise AssertionError(f"Reproducibility manifest lacks protocol config identity: {repro_path}")
    _validate_hashed_file(protocol_config, role="reproducibility protocol config")
    source_files = repro.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise AssertionError(f"Reproducibility manifest lacks source hashes: {repro_path}")
    expected_source_paths = {
        str((PROJECT_ROOT / relative_path).resolve())
        for relative_path in REPRO_SOURCE_FILES
    }
    recorded_source_paths = {str(Path(path).resolve()) for path in source_files}
    if recorded_source_paths != expected_source_paths:
        missing = sorted(expected_source_paths.difference(recorded_source_paths))
        unexpected = sorted(recorded_source_paths.difference(expected_source_paths))
        raise AssertionError(
            "Reproducibility source registry differs from the current strict source contract: "
            f"missing={missing} unexpected={unexpected}"
        )
    for source_path, record in source_files.items():
        _validate_hashed_file(
            {"path": source_path, **record},
            role="reproducibility source file",
        )
    return payload


def _capture_job_reproducibility(
    *,
    config: dict,
    strict_config_path: Path,
    dataset_name: str,
    split_record: dict,
    anchor_path: Path,
    resolved_job_config_path: Path,
    result: dict,
    job_dir: Path,
) -> tuple[dict[str, dict], dict]:
    artifacts: dict[str, Path] = {
        "resolved_job_config": resolved_job_config_path,
        "train_cache": _cache_path(config, dataset_name, "train"),
        "val_cache": _cache_path(config, dataset_name, "val"),
        "test_cache": _cache_path(config, dataset_name, "test"),
        "outer_split": Path(split_record["outer_split"]),
        "inner_split": Path(split_record["inner_split"]),
        "outer_graph": Path(split_record["outer_graph"]),
        "inner_graph": Path(split_record["inner_graph"]),
        "text_anchors": anchor_path,
        "georsclip_checkpoint": _resolve(config["clip"]["pretrained_path"]),
        "threshold_manifest": Path(result["threshold_manifest"]),
        "test_metrics": Path(result["test_metrics"]),
    }
    checkpoint: Path | None = None
    if result["method"] != "clip_zs":
        checkpoint = Path(result["outer_checkpoint"])
        artifacts.update(
            {
                "inner_resolved_config": job_dir / "inner" / "resolved_config.yaml",
                "outer_resolved_config": job_dir / "outer" / "resolved_config.yaml",
                "inner_checkpoint": Path(result["inner_checkpoint"]),
                "outer_checkpoint": checkpoint,
                "locked_selection_manifest": job_dir / "locked_selection_manifest.json",
            }
        )
    repro_path = job_dir / "reproducibility_manifest.json"
    source_files = [PROJECT_ROOT / path for path in REPRO_SOURCE_FILES]
    capture_payload = capture_repro_manifest(
        output=repro_path,
        protocol_config=strict_config_path,
        artifacts=artifacts,
        checkpoint=checkpoint,
        source_files=source_files,
    )
    for role, path in artifacts.items():
        _validate_numeric_artifact(Path(path), role=role)
    hashed_artifacts = {role: _hashed_file(path) for role, path in artifacts.items()}
    return hashed_artifacts, {
        "path": str(repro_path.resolve()),
        "size_bytes": repro_path.stat().st_size,
        "sha256": _sha256_file(repro_path),
        "source_tree_sha256": _sha256_json(capture_payload["source_files"]),
    }


def _run_clip_job(
    *,
    config: dict,
    dataset_name: str,
    split_record: dict,
    anchor_path: Path,
    job_dir: Path,
    expected_test_cache_sha256: str,
    max_val_batches: int | None,
    max_test_batches: int | None,
) -> dict:
    threshold_path = job_dir / "inner" / "threshold_manifest.json"
    threshold_manifest = _select_clip_threshold(
        config=config,
        dataset_name=dataset_name,
        anchor_path=anchor_path,
        inner_split_path=Path(split_record["inner_split"]),
        output_path=threshold_path,
        max_val_batches=max_val_batches,
    )
    eval_root = job_dir / "test"
    dataset = config["datasets"][dataset_name]
    clip = config["clip"]
    command = [
        sys.executable,
        "evaluate.py",
        "--mode",
        "zs_clip",
        "--config",
        str(PROJECT_ROOT / "configs" / "default.yaml"),
        "--dataset",
        dataset_name,
        "--data_root",
        str(_resolve(dataset["data_root"])),
        "--kg_path",
        str(Path(split_record["outer_graph"])),
        "--text_anchors",
        str(anchor_path),
        "--feature_cache_dir",
        str(_resolve(dataset["feature_cache_root"])),
        "--unseen_labels_file",
        str(Path(split_record["outer_split"])),
        "--clip_model",
        str(clip["model_name"]),
        "--clip_pretrained",
        str(_resolve(clip["pretrained_path"])),
        "--clip_default_pretrained",
        str(clip["default_pretrained"]),
        "--split",
        "test",
        "--output_dir",
        str(eval_root),
        "--threshold_manifest",
        str(threshold_path),
        "--require_feature_cache",
        "--strict_protocol",
        "--expected_feature_cache_sha256",
        expected_test_cache_sha256,
    ]
    if max_test_batches is not None:
        command.extend(["--max_batches", str(max_test_batches)])
    _run(command, f"{dataset_name}: clip_zs outer={split_record['outer_seed']}")
    metrics_path = _locate_eval_metrics(eval_root)
    return {
        "method": "clip_zs",
        "threshold_manifest": str(threshold_path),
        "decision_threshold": threshold_manifest["decision_threshold"],
        "decision_threshold_space": threshold_manifest.get("decision_threshold_space", "probability"),
        "test_metrics": str(metrics_path),
    }


def _copy_inner_selection_to_outer_checkpoint(
    *,
    inner_checkpoint: Path,
    outer_checkpoint: Path,
    threshold_path: Path,
    locked_manifest_path: Path,
) -> dict:
    inner = _load_finite_checkpoint(inner_checkpoint, role="inner_checkpoint")
    outer = _load_finite_checkpoint(outer_checkpoint, role="outer_checkpoint")
    threshold = _load_finite_json(threshold_path, role="threshold_manifest")
    copied = {
        "decision_threshold": float(threshold["decision_threshold"]),
        "decision_threshold_space": threshold.get("decision_threshold_space", "probability"),
        "decision_threshold_strategy": threshold.get("decision_threshold_strategy", "probability_grid"),
        "delta_mix_alpha": float(inner.get("delta_mix_alpha", threshold.get("selected_mix_alpha", 1.0) or 1.0)),
        "calibration_bias": float(inner.get("calibration_bias", threshold.get("selected_calibration_bias", 0.0))),
    }
    outer.update(copied)
    outer["selection_source"] = "strict_inner_validation"
    outer["inner_checkpoint_sha256"] = _sha256_file(inner_checkpoint)
    outer["threshold_manifest"] = str(threshold_path)
    _assert_finite_payload(outer, context=f"locked outer checkpoint {outer_checkpoint}")
    torch.save(outer, outer_checkpoint)
    locked = {
        "schema_version": 1,
        **copied,
        "source": "strict_inner_validation",
        "inner_checkpoint": str(inner_checkpoint),
        "inner_checkpoint_sha256": outer["inner_checkpoint_sha256"],
        "outer_checkpoint": str(outer_checkpoint),
        "outer_checkpoint_sha256": _sha256_file(outer_checkpoint),
        "threshold_manifest": str(threshold_path),
    }
    _write_json(locked_manifest_path, locked)
    return locked


def _run_trainable_job(
    *,
    config: dict,
    dataset_name: str,
    split_record: dict,
    anchor_path: Path,
    method: str,
    training_seed: int,
    job_dir: Path,
    expected_test_cache_sha256: str,
    pilot: bool,
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
) -> dict:
    mode = METHOD_TO_MODE[method]
    inner_seed = int(split_record["inner_seed"])
    inner_base = job_dir / "inner" / "runs"
    inner_epochs = min(2, int(config["training"]["epochs"])) if pilot else int(config["training"]["epochs"])
    inner_config = _resolved_train_config(
        strict_config=config,
        dataset_name=dataset_name,
        graph_path=Path(split_record["inner_graph"]),
        anchor_path=anchor_path,
        outer_split_path=Path(split_record["outer_split"]),
        output_dir=inner_base,
        mode=mode,
        training_seed=training_seed,
        inner_seed=inner_seed,
        protocol_stage="inner_select",
        epochs=inner_epochs,
        fixed_epochs=None,
        selection_manifest=None,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    inner_config_path = job_dir / "inner" / "resolved_config.yaml"
    _write_yaml(inner_config_path, inner_config)
    before = {path.resolve() for path in inner_base.glob(f"{mode}_*") if path.is_dir()}
    _run([sys.executable, "train.py", "--config", str(inner_config_path)], f"{dataset_name}: {method} inner")
    inner_run = _new_run_directory(inner_base, mode, before)
    inner_checkpoint = inner_run / f"best_{mode}.pt"
    threshold_path = inner_run / "threshold_manifest.json"
    if not inner_checkpoint.is_file() or not threshold_path.is_file():
        raise FileNotFoundError(f"Inner selection outputs missing for {method}: {inner_run}")
    inner_payload = _load_finite_checkpoint(inner_checkpoint, role="inner_checkpoint")
    _load_finite_json(threshold_path, role="threshold_manifest")
    fixed_epochs = int(inner_payload["epoch"])

    outer_base = job_dir / "outer" / "runs"
    outer_config = _resolved_train_config(
        strict_config=config,
        dataset_name=dataset_name,
        graph_path=Path(split_record["outer_graph"]),
        anchor_path=anchor_path,
        outer_split_path=Path(split_record["outer_split"]),
        output_dir=outer_base,
        mode=mode,
        training_seed=training_seed,
        inner_seed=inner_seed,
        protocol_stage="outer_retrain",
        epochs=fixed_epochs,
        fixed_epochs=fixed_epochs,
        selection_manifest=threshold_path,
        max_train_batches=max_train_batches,
        max_val_batches=None,
    )
    outer_config_path = job_dir / "outer" / "resolved_config.yaml"
    _write_yaml(outer_config_path, outer_config)
    before = {path.resolve() for path in outer_base.glob(f"{mode}_*") if path.is_dir()}
    _run([sys.executable, "train.py", "--config", str(outer_config_path)], f"{dataset_name}: {method} outer")
    outer_run = _new_run_directory(outer_base, mode, before)
    outer_checkpoint = outer_run / f"best_{mode}.pt"
    if not outer_checkpoint.is_file():
        raise FileNotFoundError(f"Outer checkpoint missing for {method}: {outer_checkpoint}")
    _load_finite_checkpoint(outer_checkpoint, role="outer_checkpoint")
    locked_path = job_dir / "locked_selection_manifest.json"
    locked = _copy_inner_selection_to_outer_checkpoint(
        inner_checkpoint=inner_checkpoint,
        outer_checkpoint=outer_checkpoint,
        threshold_path=threshold_path,
        locked_manifest_path=locked_path,
    )

    eval_root = job_dir / "test"
    command = [
        sys.executable,
        "evaluate.py",
        "--checkpoint",
        str(outer_checkpoint),
        "--dataset",
        dataset_name,
        "--kg_path",
        str(Path(split_record["outer_graph"])),
        "--text_anchors",
        str(anchor_path),
        "--feature_cache_dir",
        str(_resolve(config["datasets"][dataset_name]["feature_cache_root"])),
        "--unseen_labels_file",
        str(Path(split_record["outer_split"])),
        "--split",
        "test",
        "--output_dir",
        str(eval_root),
        "--threshold_manifest",
        str(threshold_path),
        "--require_feature_cache",
        "--strict_protocol",
        "--expected_feature_cache_sha256",
        expected_test_cache_sha256,
    ]
    if max_test_batches is not None:
        command.extend(["--max_batches", str(max_test_batches)])
    _run(command, f"{dataset_name}: {method} locked outer test")
    metrics_path = _locate_eval_metrics(eval_root)
    return {
        "method": method,
        "mode": mode,
        "inner_run": str(inner_run),
        "inner_checkpoint": str(inner_checkpoint),
        "selected_epoch": fixed_epochs,
        "threshold_manifest": str(threshold_path),
        "outer_run": str(outer_run),
        "outer_checkpoint": str(outer_checkpoint),
        "locked_selection": locked,
        "test_metrics": str(metrics_path),
    }


def run_experiments(
    *,
    config: dict,
    strict_config_path: Path,
    manifests: dict[str, dict],
    methods: list[str],
    phase: str,
    pilot_outer_seed: int,
    pilot_training_seed: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
    skip_existing: bool,
) -> None:
    if phase not in {"pilot", "full"}:
        raise ValueError(f"Experiments require phase=pilot or phase=full, got {phase!r}")
    if phase == "full" and any(
        limit is not None for limit in (max_train_batches, max_val_batches, max_test_batches)
    ):
        raise ValueError(f"phase={phase} forbids --max_train_batches, --max_val_batches, and --max_test_batches")
    if skip_existing and phase != "full":
        raise ValueError("--skip_existing is only valid for phase=full")
    unknown = sorted(set(methods).difference({"clip_zs", *METHOD_TO_MODE}))
    if unknown:
        raise ValueError(f"Unsupported strict methods: {unknown}")
    selected_training_seeds = _configured_training_seeds(config["protocol"]["training_seeds"])
    output_root = _resolve(config["outputs"]["root"])
    pilot = phase == "pilot"
    train_limit = 1 if pilot and max_train_batches is None else max_train_batches
    val_limit = 1 if pilot and max_val_batches is None else max_val_batches
    test_limit = 1 if pilot and max_test_batches is None else max_test_batches

    for dataset_name, manifest in manifests.items():
        anchor_path = Path(manifest["text_anchors"])
        test_cache_record = manifest.get("cache_headers", {}).get("test")
        if not isinstance(test_cache_record, dict) or not test_cache_record.get("sha256"):
            raise ValueError(
                f"{dataset_name}: prepared protocol manifest lacks the locked test-cache SHA-256"
            )
        expected_test_cache_sha256 = str(test_cache_record["sha256"])
        split_records = list(manifest["splits"])
        if pilot:
            split_records = [record for record in split_records if int(record["outer_seed"]) == pilot_outer_seed]
            if not split_records:
                raise ValueError(f"{dataset_name}: pilot outer seed {pilot_outer_seed} not prepared")
            training_seeds = [int(pilot_training_seed)]
        else:
            training_seeds = list(selected_training_seeds)
        for split_record in split_records:
            for training_seed in training_seeds:
                for method in methods:
                    if method == "clip_zs" and training_seed != training_seeds[0]:
                        continue
                    job_dir = _job_directory(
                        output_root,
                        phase=phase,
                        dataset_name=dataset_name,
                        outer_seed=int(split_record["outer_seed"]),
                        training_seed=training_seed,
                        method=method,
                    )
                    result_path = job_dir / "run_manifest.json"
                    resolved_job_config_path = job_dir / "resolved_job_config.json"
                    job_spec = _resolved_job_spec(
                        phase=phase,
                        dataset_name=dataset_name,
                        split_record=split_record,
                        method=method,
                        training_seed=training_seed,
                        strict_config_path=strict_config_path,
                        strict_config=config,
                        max_train_batches=train_limit,
                        max_val_batches=val_limit,
                        max_test_batches=test_limit,
                    )
                    if skip_existing and result_path.is_file():
                        _validate_existing_full_job(
                            result_path=result_path,
                            expected_job_spec=job_spec,
                            resolved_job_config_path=resolved_job_config_path,
                        )
                        print(f"[skip] {dataset_name} outer={split_record['outer_seed']} train={training_seed} {method}")
                        continue
                    job_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(resolved_job_config_path, job_spec)
                    if method == "clip_zs":
                        result = _run_clip_job(
                            config=config,
                            dataset_name=dataset_name,
                            split_record=split_record,
                            anchor_path=anchor_path,
                            job_dir=job_dir,
                            expected_test_cache_sha256=expected_test_cache_sha256,
                            max_val_batches=val_limit,
                            max_test_batches=test_limit,
                        )
                    else:
                        result = _run_trainable_job(
                            config=config,
                            dataset_name=dataset_name,
                            split_record=split_record,
                            anchor_path=anchor_path,
                            method=method,
                            training_seed=training_seed,
                            job_dir=job_dir,
                            expected_test_cache_sha256=expected_test_cache_sha256,
                            pilot=pilot,
                            max_train_batches=train_limit,
                            max_val_batches=val_limit,
                            max_test_batches=test_limit,
                        )
                    artifacts, repro_manifest = _capture_job_reproducibility(
                        config=config,
                        strict_config_path=strict_config_path,
                        dataset_name=dataset_name,
                        split_record=split_record,
                        anchor_path=anchor_path,
                        resolved_job_config_path=resolved_job_config_path,
                        result=result,
                        job_dir=job_dir,
                    )
                    _write_json(
                        result_path,
                        {
                            "schema_version": 2,
                            "phase": phase,
                            "dataset": dataset_name,
                            "outer_seed": int(split_record["outer_seed"]),
                            "training_seed": training_seed,
                            "feature_cache_only": True,
                            "raw_images_read": False,
                            "anchor_sha256": manifest["text_anchors_sha256"],
                            "outer_graph_sha256": split_record["outer_graph_sha256"],
                            "inner_graph_sha256": split_record["inner_graph_sha256"],
                            "strict_config": str(strict_config_path.resolve()),
                            "strict_config_sha256": job_spec["strict_config_sha256"],
                            "resolved_strict_config_sha256": job_spec["resolved_strict_config_sha256"],
                            "resolved_job_config": str(resolved_job_config_path.resolve()),
                            "resolved_job_config_sha256": _sha256_file(resolved_job_config_path),
                            "artifacts": artifacts,
                            "reproducibility_manifest": repro_manifest,
                            **result,
                        },
                    )


def _build_empirical_graph(
    *,
    dataset_name: str,
    train_cache: Path,
    allowed_labels_file: Path,
    output_dir: Path,
) -> Path:
    _run(
        [
            sys.executable,
            "build_kg.py",
            "--dataset",
            dataset_name,
            "--train_feature_cache",
            str(train_cache),
            "--allowed_seen_labels_file",
            str(allowed_labels_file),
            "--output_dir",
            str(output_dir),
        ],
        f"{dataset_name}: empirical graph {output_dir.parent.name}/{output_dir.name}",
    )
    return output_dir / "kg_definition.json"


def _build_basic_anchors(config: dict, dataset_name: str, kg_path: Path, output_path: Path) -> None:
    clip = config["clip"]
    _run(
        [
            sys.executable,
            "build_text_anchors.py",
            "--dataset",
            dataset_name,
            "--kg_path",
            str(kg_path),
            "--clip_model",
            str(clip["model_name"]),
            "--clip_pretrained",
            str(_resolve(clip["pretrained_path"])),
            "--clip_default_pretrained",
            str(clip["default_pretrained"]),
            "--output",
            str(output_path),
        ],
        f"{dataset_name}: basic text anchors",
    )


def _augment_semantic(
    *,
    config: dict,
    kg_path: Path,
    anchor_path: Path,
    split_path: Path,
    output_path: Path,
) -> None:
    _run(
        [
            sys.executable,
            "build_semantic_affinity.py",
            "--kg_path",
            str(kg_path),
            "--text_anchors",
            str(anchor_path),
            "--outer_split",
            str(split_path),
            "--top_k",
            str(config["protocol"]["semantic_affinity_top_k"]),
            "--output",
            str(output_path),
        ],
        f"semantic affinity: {output_path.parent.name}",
    )


def prepare_dataset(
    config: dict,
    dataset_name: str,
    artifact_root: Path,
    *,
    strict_config_path: Path,
) -> dict:
    dataset_config = config["datasets"][dataset_name]
    dataset_root = artifact_root / dataset_name
    split_dir = dataset_root / "outer_splits"
    train_cache = _cache_path(config, dataset_name, "train")
    cache_headers = {}
    cache_checkpoint_bindings = {}
    for split in ("train", "val", "test"):
        header = _load_cache_header(_cache_path(config, dataset_name, split))
        cache_headers[split] = header
        cache_checkpoint_bindings[split] = _validate_cache_header(
            header,
            config=config,
            dataset_name=dataset_name,
            split=split,
        )
    label_names = cache_headers["train"]["label_names"]
    if any(cache_headers[split]["label_names"] != label_names for split in ("val", "test")):
        raise ValueError(f"{dataset_name}: cache label orders differ across image splits")
    if cache_headers["train"]["split"] != "train":
        raise ValueError(f"{dataset_name}: train cache does not declare split=train")

    split_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "select_unseen.py",
            "--dataset",
            dataset_name,
            "--label_names_file",
            str(train_cache),
            "--n_unseen",
            str(dataset_config["n_unseen"]),
            "--seeds",
            *[str(seed) for seed in config["protocol"]["outer_split_seeds"]],
            "--output_dir",
            str(split_dir),
        ],
        f"{dataset_name}: outer label splits",
    )

    outer_seeds = [int(seed) for seed in config["protocol"]["outer_split_seeds"]]
    first_outer = split_dir / f"split_seed{outer_seeds[0]}.json"
    first_raw_graph = _build_empirical_graph(
        dataset_name=dataset_name,
        train_cache=train_cache,
        allowed_labels_file=first_outer,
        output_dir=dataset_root / f"outer_{outer_seeds[0]}" / "outer_empirical",
    )
    anchor_path = dataset_root / f"text_anchors_{dataset_name}_basic.pt"
    _build_basic_anchors(config, dataset_name, first_raw_graph, anchor_path)

    split_records = []
    for outer_seed in outer_seeds:
        outer_dir = dataset_root / f"outer_{outer_seed}"
        outer_split_path = split_dir / f"split_seed{outer_seed}.json"
        outer_split = json.loads(outer_split_path.read_text(encoding="utf-8"))
        if outer_seed == outer_seeds[0]:
            outer_graph_raw = first_raw_graph
        else:
            # A prepare run is also the cache-invalidation boundary.  Reusing a
            # graph merely because the file exists can mix a newly selected
            # feature cache with empirical edges built from an older cache.
            outer_graph_raw = _build_empirical_graph(
                dataset_name=dataset_name,
                train_cache=train_cache,
                allowed_labels_file=outer_split_path,
                output_dir=outer_dir / "outer_empirical",
            )
        outer_graph = outer_dir / "outer_graph.json"
        _augment_semantic(
            config=config,
            kg_path=outer_graph_raw,
            anchor_path=anchor_path,
            split_path=outer_split_path,
            output_path=outer_graph,
        )

        inner_seed = int(config["protocol"]["inner_seed_offset"]) + outer_seed
        nested = build_nested_label_protocol(
            list(outer_split["seen_labels"]),
            list(outer_split["unseen_labels"]),
            inner_seed=inner_seed,
            surrogate_fraction=float(config["protocol"]["inner_surrogate_fraction"]),
        )
        inner_split_path = outer_dir / "inner_split.json"
        inner_graph_unseen = list(
            dict.fromkeys([*nested["surrogate_unseen_labels"], *nested["outer_unseen_labels"]])
        )
        inner_payload = {
            "schema_version": 1,
            "protocol_mode": "strict_inner_surrogate_v1",
            "dataset": dataset_name,
            "outer_seed": outer_seed,
            "inner_seed": inner_seed,
            "label_names": label_names,
            "seen_labels": list(nested["inner_seen_labels"]),
            "unseen_labels": inner_graph_unseen,
            "selection_seen_labels": list(nested["selection_seen_labels"]),
            "selection_unseen_labels": list(nested["selection_unseen_labels"]),
            "outer_unseen_labels": list(nested["outer_unseen_labels"]),
            "training_hidden_labels": list(nested["training_hidden_labels"]),
        }
        _write_json(inner_split_path, inner_payload)
        inner_graph_raw = _build_empirical_graph(
            dataset_name=dataset_name,
            train_cache=train_cache,
            allowed_labels_file=inner_split_path,
            output_dir=outer_dir / "inner_empirical",
        )
        inner_graph = outer_dir / "inner_graph.json"
        _augment_semantic(
            config=config,
            kg_path=inner_graph_raw,
            anchor_path=anchor_path,
            split_path=inner_split_path,
            output_path=inner_graph,
        )
        split_records.append(
            {
                "outer_seed": outer_seed,
                "inner_seed": inner_seed,
                "outer_split": str(outer_split_path),
                "inner_split": str(inner_split_path),
                "outer_split_sha256": _sha256_file(outer_split_path),
                "inner_split_sha256": _sha256_file(inner_split_path),
                "outer_graph": str(outer_graph),
                "inner_graph": str(inner_graph),
                "outer_graph_sha256": _sha256_file(outer_graph),
                "inner_graph_sha256": _sha256_file(inner_graph),
            }
        )

    dataset_manifest = {
        "schema_version": 2,
        "strict_config": str(strict_config_path.resolve()),
        "strict_config_sha256": _sha256_file(strict_config_path),
        "resolved_strict_config_sha256": _sha256_json(config),
        "protocol": config["protocol"],
        "dataset": dataset_name,
        "data_root": str(_resolve(dataset_config["data_root"])),
        "feature_cache_only": True,
        "cache_headers": cache_headers,
        "cache_checkpoint_bindings": cache_checkpoint_bindings,
        "georsclip_checkpoint": _hashed_file(_resolve(config["clip"]["pretrained_path"])),
        "text_anchors": str(anchor_path),
        "text_anchors_sha256": _sha256_file(anchor_path),
        "cooccur_prompts_used": False,
        "splits": split_records,
    }
    _write_json(dataset_root / "protocol_manifest.json", dataset_manifest)
    return dataset_manifest


def _audit_graph(
    *,
    graph_path: Path,
    expected_sha256: str,
    split_path: Path,
    anchor_path: Path,
    train_cache_path: Path,
    expected_nodes: list[str],
) -> None:
    if not graph_path.is_file():
        raise AssertionError(f"Graph is missing: {graph_path}")
    if _sha256_file(graph_path) != expected_sha256:
        raise AssertionError(f"Graph hash mismatch: {graph_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    seen = set(split["seen_labels"])
    if list(graph.get("nodes", [])) != expected_nodes:
        raise AssertionError(f"Graph node order differs from feature caches: {graph_path}")
    for relation in EMPIRICAL_RELATIONS:
        edges = graph.get("edges", {}).get(relation, [])
        endpoints: set[str] = set()
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise AssertionError(f"Malformed empirical edge in {relation}: {graph_path}")
            endpoints.update(str(endpoint) for endpoint in edge)
        leaked = sorted(endpoints - seen)
        if leaked:
            raise AssertionError(f"Empirical relation {relation} has forbidden endpoints {leaked}: {graph_path}")
    provenance = graph.get("edge_provenance", {})
    if set(provenance.get("allowed_seen_labels", [])) != seen:
        raise AssertionError(f"Graph provenance allowed-seen set differs from split: {graph_path}")
    source = provenance.get("source", {})
    if source.get("kind") != "train_feature_cache":
        raise AssertionError(f"Graph was not built from a train feature cache: {graph_path}")
    source_path = Path(source.get("path", "")).resolve()
    if source_path != train_cache_path.resolve():
        raise AssertionError(f"Graph train-cache source differs from current strict cache: {graph_path}")
    declared_split = source.get("image_split_manifest", {}).get("declared_split")
    if declared_split != "train":
        raise AssertionError(f"Graph source does not declare split=train: {graph_path}")
    semantic = graph.get("relation_provenance", {}).get("semantic_affinity", {})
    if semantic.get("annotation_independent") is not True:
        raise AssertionError(f"Semantic-affinity provenance is not annotation-independent: {graph_path}")
    if semantic.get("text_anchors_sha256") != _sha256_file(anchor_path):
        raise AssertionError(f"Graph semantic-affinity anchor hash mismatch: {graph_path}")
    if semantic.get("outer_split_sha256") != _sha256_file(split_path):
        raise AssertionError(f"Graph semantic-affinity split hash mismatch: {graph_path}")


def audit_manifest(
    manifest: dict,
    *,
    config: dict,
    strict_config_path: Path,
    dataset_name: str | None = None,
) -> None:
    if not manifest.get("feature_cache_only"):
        raise AssertionError("Strict manifest is not feature-cache-only")
    if manifest.get("cooccur_prompts_used"):
        raise AssertionError("Strict manifest enabled co-occurrence prompts")
    expected_dataset = dataset_name or str(manifest.get("dataset"))
    if manifest.get("dataset") != expected_dataset or expected_dataset not in config["datasets"]:
        raise AssertionError(f"Protocol manifest dataset mismatch: expected {expected_dataset!r}")
    current_config_hash = _sha256_file(strict_config_path)
    if manifest.get("strict_config_sha256") != current_config_hash:
        raise AssertionError("Strict config file hash differs from the prepared protocol manifest")
    if manifest.get("resolved_strict_config_sha256") != _sha256_json(config):
        raise AssertionError("Resolved strict config differs from the prepared protocol manifest")
    if _sha256_json(manifest.get("protocol")) != _sha256_json(config["protocol"]):
        raise AssertionError("Protocol section differs from the current strict config")

    cache_headers = manifest.get("cache_headers", {})
    cache_checkpoint_bindings = manifest.get("cache_checkpoint_bindings", {})
    _validate_hashed_file(
        manifest.get("georsclip_checkpoint", {}),
        role=f"{expected_dataset} GeoRSCLIP checkpoint",
    )
    expected_nodes: list[str] | None = None
    for split_name in ("train", "val", "test"):
        header = cache_headers.get(split_name)
        if not isinstance(header, dict):
            raise AssertionError(f"Missing cache header for split={split_name}")
        expected_path = _cache_path(config, expected_dataset, split_name).resolve()
        recorded_path = Path(header.get("path", "")).resolve()
        if recorded_path != expected_path:
            raise AssertionError(f"Cache path changed for {expected_dataset}/{split_name}")
        if not expected_path.is_file() or _sha256_file(expected_path) != header.get("sha256"):
            raise AssertionError(f"Cache hash mismatch for {expected_dataset}/{split_name}: {expected_path}")
        binding = _validate_cache_header(
            header,
            config=config,
            dataset_name=expected_dataset,
            split=split_name,
        )
        if cache_checkpoint_bindings.get(split_name) != binding:
            raise AssertionError(f"Cache checkpoint binding changed for {expected_dataset}/{split_name}")
        labels = list(header.get("label_names", []))
        if expected_nodes is None:
            expected_nodes = labels
        elif labels != expected_nodes:
            raise AssertionError(f"Cache label order mismatch for {expected_dataset}/{split_name}")
    assert expected_nodes is not None

    anchor_path = Path(manifest["text_anchors"])
    if not anchor_path.is_file():
        raise AssertionError(f"Text anchors are missing: {anchor_path}")
    anchor_hash = _sha256_file(anchor_path)
    if anchor_hash != manifest["text_anchors_sha256"]:
        raise AssertionError("Text-anchor hash changed after manifest creation")
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if list(anchor_payload.get("label_names", [])) != expected_nodes:
        raise AssertionError("Text-anchor label order differs from feature caches")

    records = list(manifest.get("splits", []))
    expected_outer_seeds = [int(seed) for seed in config["protocol"]["outer_split_seeds"]]
    if [int(record.get("outer_seed", -1)) for record in records] != expected_outer_seeds:
        raise AssertionError("Prepared outer split seeds differ from the strict config")
    for split in manifest["splits"]:
        outer_split_path = Path(split["outer_split"])
        inner_split_path = Path(split["inner_split"])
        for key, path in (("outer_split_sha256", outer_split_path), ("inner_split_sha256", inner_split_path)):
            if not path.is_file() or _sha256_file(path) != split.get(key):
                raise AssertionError(f"Split hash mismatch: {path}")
        outer_payload = json.loads(outer_split_path.read_text(encoding="utf-8"))
        inner_payload = json.loads(inner_split_path.read_text(encoding="utf-8"))
        outer_seen = set(outer_payload.get("seen_labels", []))
        outer_unseen = set(outer_payload.get("unseen_labels", []))
        if outer_seen & outer_unseen or outer_seen | outer_unseen != set(expected_nodes):
            raise AssertionError(f"Outer split is not a disjoint complete label partition: {outer_split_path}")
        if len(outer_unseen) != int(config["datasets"][expected_dataset]["n_unseen"]):
            raise AssertionError(f"Outer split unseen count differs from strict config: {outer_split_path}")
        if set(inner_payload.get("outer_unseen_labels", [])) != outer_unseen:
            raise AssertionError(f"Inner split changed the real outer-unseen labels: {inner_split_path}")
        if set(inner_payload.get("seen_labels", [])) & set(inner_payload.get("unseen_labels", [])):
            raise AssertionError(f"Inner split seen/unseen sets overlap: {inner_split_path}")
        _audit_graph(
            graph_path=Path(split["outer_graph"]),
            expected_sha256=str(split["outer_graph_sha256"]),
            split_path=outer_split_path,
            anchor_path=anchor_path,
            train_cache_path=_cache_path(config, expected_dataset, "train"),
            expected_nodes=expected_nodes,
        )
        _audit_graph(
            graph_path=Path(split["inner_graph"]),
            expected_sha256=str(split["inner_graph_sha256"]),
            split_path=inner_split_path,
            anchor_path=anchor_path,
            train_cache_path=_cache_path(config, expected_dataset, "train"),
            expected_nodes=expected_nodes,
        )


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.phase == "full" and any(
        limit is not None
        for limit in (args.max_train_batches, args.max_val_batches, args.max_test_batches)
    ):
        raise ValueError(f"phase={args.phase} forbids all max_*_batches options")
    if args.skip_existing and args.phase != "full":
        raise ValueError("--skip_existing is only valid for phase=full")
    _configured_training_seeds(config["protocol"]["training_seeds"])
    artifact_root = _resolve(config["outputs"]["artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or (["mlrsnet"] if args.phase == "pilot" else list(config["datasets"]))
    manifests = {}
    for dataset_name in datasets:
        if dataset_name not in config["datasets"]:
            raise ValueError(f"Unknown dataset in strict config: {dataset_name}")
        manifest_path = artifact_root / dataset_name / "protocol_manifest.json"
        if args.phase == "audit" or (args.phase in {"pilot", "full"} and manifest_path.is_file()):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = prepare_dataset(
                config,
                dataset_name,
                artifact_root,
                strict_config_path=config_path,
            )
        audit_manifest(
            manifest,
            config=config,
            strict_config_path=config_path,
            dataset_name=dataset_name,
        )
        manifests[dataset_name] = manifest
    _write_json(
        artifact_root / "strict_protocol_manifest.json",
        {
            "schema_version": 1,
            "config": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "resolved_config_sha256": _sha256_json(config),
            "datasets": manifests,
        },
    )
    if args.phase in {"pilot", "full"}:
        run_experiments(
            config=config,
            strict_config_path=config_path,
            manifests=manifests,
            methods=list(args.methods or config["methods"]),
            phase=args.phase,
            pilot_outer_seed=int(args.pilot_outer_seed),
            pilot_training_seed=int(args.pilot_training_seed),
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
            skip_existing=bool(args.skip_existing),
        )


if __name__ == "__main__":
    main()
