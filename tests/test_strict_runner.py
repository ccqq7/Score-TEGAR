from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from run_strict_protocol import (
    PROJECT_ROOT,
    REPRO_SOURCE_FILES,
    _audit_graph,
    _copy_inner_selection_to_outer_checkpoint,
    _hashed_file,
    _job_directory,
    _run_clip_job,
    _run,
    _resolved_train_config,
    _configured_training_seeds,
    _sha256_file,
    _sha256_json,
    _validate_cache_header,
    _validate_existing_full_job,
    audit_manifest,
    parse_args,
    prepare_dataset,
    run_experiments,
)


def test_resolved_config_preserves_training_seed_zero(tmp_path) -> None:
    strict = yaml.safe_load((PROJECT_ROOT / "configs" / "strict_a.yaml").read_text(encoding="utf-8"))
    resolved = _resolved_train_config(
        strict_config=strict,
        dataset_name="mlrsnet",
        graph_path=tmp_path / "inner_graph.json",
        anchor_path=tmp_path / "anchors.pt",
        outer_split_path=tmp_path / "outer_split.json",
        output_dir=tmp_path / "runs",
        mode="zs_logit_tegar",
        training_seed=0,
        inner_seed=100042,
        protocol_stage="inner_select",
        epochs=2,
        fixed_epochs=None,
        selection_manifest=None,
        max_train_batches=1,
        max_val_batches=1,
    )
    assert resolved["data"]["use_feature_cache"] is True
    assert resolved["training"]["training_seed"] == 0
    assert resolved["training"]["inner_split_seed"] == 100042
    assert resolved["training"]["protocol_stage"] == "inner_select"
    assert resolved["training"]["max_train_batches"] == 1
    assert resolved["training"]["decision_threshold_grid"] == strict["protocol"]["threshold_grid"]
    assert resolved["training"]["deterministic"] is True
    assert resolved["training"]["deterministic_warn_only"] is False
    assert resolved["model"]["gate_control"] == strict["model"].get("gate_control", "learned")
    assert resolved["model"]["gate_shuffle_seed"] == int(
        strict["model"].get("gate_shuffle_seed", 20260826)
    )


def test_strict_runner_sets_cublas_workspace_before_subprocess(monkeypatch) -> None:
    captured = {}

    class _Completed:
        returncode = 0

    def fake_run(command, *, cwd, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        return _Completed()

    monkeypatch.setattr("run_strict_protocol.subprocess.run", fake_run)
    _run(["python", "train.py"], "test")

    assert captured["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_strict_runner_passes_protocol_locked_test_cache_hash_to_evaluate(monkeypatch, tmp_path) -> None:
    captured = {}
    expected = "a" * 64

    def fake_select(**kwargs):
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return {"decision_threshold": 0.5}

    monkeypatch.setattr("run_strict_protocol._select_clip_threshold", fake_select)
    monkeypatch.setattr(
        "run_strict_protocol._run",
        lambda command, _stage: captured.setdefault("command", command),
    )
    monkeypatch.setattr(
        "run_strict_protocol._locate_eval_metrics",
        lambda _root: tmp_path / "metrics.json",
    )
    config = {
        "datasets": {
            "mlrsnet": {
                "data_root": str(tmp_path / "raw"),
                "feature_cache_root": str(tmp_path / "cache"),
            }
        },
        "clip": {
            "model_name": "ViT-B-32",
            "pretrained_path": str(tmp_path / "model.pt"),
            "default_pretrained": "default",
        },
    }
    _run_clip_job(
        config=config,
        dataset_name="mlrsnet",
        split_record={
            "outer_seed": 42,
            "outer_graph": str(tmp_path / "graph.json"),
            "outer_split": str(tmp_path / "split.json"),
            "inner_split": str(tmp_path / "inner.json"),
        },
        anchor_path=tmp_path / "anchors.pt",
        job_dir=tmp_path / "job",
        expected_test_cache_sha256=expected,
        max_val_batches=1,
        max_test_batches=1,
    )

    command = captured["command"]
    assert "--strict_protocol" in command
    assert "--require_feature_cache" in command
    hash_index = command.index("--expected_feature_cache_sha256")
    assert command[hash_index + 1] == expected


def test_cache_header_validation_records_legacy_checkpoint_binding(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "pretrained" / "georsclip" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-georsclip")
    cache = tmp_path / "train.pt"
    cache.write_bytes(b"cache")
    config = {
        "clip": {
            "model_name": "ViT-B-32",
            "pretrained_path": str(checkpoint),
            "default_pretrained": "laion2b_s34b_b79k",
            "image_size": 224,
        },
        "protocol": {"image_split_seed": 42},
        "datasets": {"mlrsnet": {"feature_cache_root": str(tmp_path)}},
    }
    monkeypatch.setattr("run_strict_protocol._cache_path", lambda *_args, **_kwargs: cache)
    header = {
        "path": str(cache),
        "split": "train",
        "label_names": ["a", "b"],
        "feature_shape": [3, 512],
        "label_shape": [3, 2],
        "clip_model": "ViT-B-32",
        "clip_pretrained": ".\\pretrained\\georsclip\\model.pt",
        "clip_pretrained_sha256": None,
        "clip_default_pretrained": "laion2b_s34b_b79k",
        "split_seed": 42,
        "image_size": 224,
    }

    binding = _validate_cache_header(
        header,
        config=config,
        dataset_name="mlrsnet",
        split="train",
    )
    assert binding == "declared_path_match_only_no_embedded_sha256"

    header["image_size"] = 256
    with pytest.raises(AssertionError, match="image size"):
        _validate_cache_header(
            header,
            config=config,
            dataset_name="mlrsnet",
            split="train",
        )


@pytest.mark.parametrize(
    ("threshold", "space"),
    [(0.35, None), (26.7, "logit"), (-0.25, "logit")],
)
def test_inner_selection_is_copied_to_outer_checkpoint(tmp_path, threshold, space) -> None:
    inner_path = tmp_path / "inner.pt"
    outer_path = tmp_path / "outer.pt"
    threshold_path = tmp_path / "threshold.json"
    locked_path = tmp_path / "locked.json"
    torch.save(
        {
            "model_state": {"weight": torch.tensor([1.0])},
            "delta_mix_alpha": 0.6,
            "calibration_bias": 0.0,
        },
        inner_path,
    )
    torch.save({"model_state": {"weight": torch.tensor([2.0])}}, outer_path)
    threshold_path.write_text(
        json.dumps(
            {
                "decision_threshold": threshold,
                **({"decision_threshold_space": space,
                    "decision_threshold_strategy": "exact_logit"} if space else {}),
                "selected_mix_alpha": 0.6,
                "selected_calibration_bias": 0.0,
            }
        ),
        encoding="utf-8",
    )

    locked = _copy_inner_selection_to_outer_checkpoint(
        inner_checkpoint=inner_path,
        outer_checkpoint=outer_path,
        threshold_path=threshold_path,
        locked_manifest_path=locked_path,
    )
    outer = torch.load(outer_path, map_location="cpu", weights_only=False)
    assert outer["decision_threshold"] == threshold
    assert outer["decision_threshold_space"] == (space or "probability")
    assert locked["decision_threshold_space"] == (space or "probability")
    assert locked["decision_threshold_strategy"] == ("exact_logit" if space else "probability_grid")
    # Exercise the locked value through the metric entry point used by evaluate.py.
    # Scores above 1 and below 0 must retain their logit interpretation on reload.
    import numpy as np
    from utils.metrics import compute_zero_shot_metrics, decision_threshold_to_logit

    cutoff = decision_threshold_to_logit(threshold, space or "probability")
    metrics = compute_zero_shot_metrics(
        np.array([[cutoff + 0.1, cutoff - 0.1], [cutoff - 0.1, cutoff + 0.1]]),
        np.array([[1, 0], [0, 1]]), ["seen", "unseen"],
        seen_labels=["seen"], unseen_labels=["unseen"],
        decision_threshold=outer["decision_threshold"],
        decision_threshold_space=outer["decision_threshold_space"],
    )
    assert metrics["seen_macro_F1"] == 1.0
    assert metrics["unseen_macro_recall"] == 1.0
    assert outer["delta_mix_alpha"] == 0.6
    assert outer["calibration_bias"] == 0.0
    assert torch.equal(outer["model_state"]["weight"], torch.tensor([2.0]))
    assert locked["source"] == "strict_inner_validation"
    assert locked_path.is_file()


def test_pilot_and_full_job_directories_are_disjoint(tmp_path) -> None:
    kwargs = {
        "dataset_name": "mlrsnet",
        "outer_seed": 42,
        "training_seed": 0,
        "method": "score_tegar",
    }
    pilot = _job_directory(tmp_path, phase="pilot", **kwargs)
    full = _job_directory(tmp_path, phase="full", **kwargs)
    assert pilot != full
    assert "pilot" in pilot.parts
    assert "full" in full.parts
    assert not pilot.is_relative_to(full)
    assert not full.is_relative_to(pilot)


@pytest.mark.parametrize("phase", ["prepare", "audit", "pilot", "full"])
def test_cli_retains_standard_protocol_phases(monkeypatch, phase) -> None:
    monkeypatch.setattr("sys.argv", ["run_strict_protocol.py", "--phase", phase])
    assert parse_args().phase == phase


@pytest.mark.parametrize("seeds", [[], [0, 0]])
def test_configured_training_seeds_reject_empty_or_duplicate_values(seeds) -> None:
    with pytest.raises(ValueError, match="training seeds"):
        _configured_training_seeds(seeds)


def test_full_phase_runs_all_configured_initializations_for_every_split(
    monkeypatch,
    tmp_path,
) -> None:
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text("protocol: {training_seeds: [0, 1, 2]}\n", encoding="utf-8")
    anchor_path = tmp_path / "anchors.pt"
    anchor_path.write_bytes(b"anchors")
    config = {
        "protocol": {"training_seeds": [0, 1, 2]},
        "outputs": {"root": str(tmp_path / "results")},
    }
    outer_seeds = [42, 123, 456, 789, 1024]
    manifests = {
        "mlrsnet": {
            "text_anchors": str(anchor_path),
            "text_anchors_sha256": "a" * 64,
            "cache_headers": {"test": {"sha256": "b" * 64}},
            "splits": [
                {
                    "outer_seed": outer_seed,
                    "inner_seed": 100000 + outer_seed,
                    "outer_graph_sha256": "c" * 64,
                    "inner_graph_sha256": "d" * 64,
                }
                for outer_seed in outer_seeds
            ],
        }
    }
    monkeypatch.setattr(
        "run_strict_protocol._run_trainable_job",
        lambda **_kwargs: {"method": "score_tegar"},
    )
    monkeypatch.setattr(
        "run_strict_protocol._capture_job_reproducibility",
        lambda **_kwargs: ({}, {}),
    )
    run_experiments(
        config=config,
        strict_config_path=strict_path,
        manifests=manifests,
        methods=["score_tegar"],
        phase="full",
        pilot_outer_seed=42,
        pilot_training_seed=0,
        max_train_batches=None,
        max_val_batches=None,
        max_test_batches=None,
        skip_existing=False,
    )

    run_manifests = sorted((tmp_path / "results" / "full").glob("**/run_manifest.json"))
    assert len(run_manifests) == 15
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in run_manifests]
    assert {payload["outer_seed"] for payload in payloads} == set(outer_seeds)
    assert {payload["training_seed"] for payload in payloads} == {0, 1, 2}


def test_full_phase_rejects_all_batch_limits(tmp_path) -> None:
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text("protocol: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="phase=full forbids"):
        run_experiments(
            config={},
            strict_config_path=strict_path,
            manifests={},
            methods=[],
            phase="full",
            pilot_outer_seed=42,
            pilot_training_seed=0,
            max_train_batches=1,
            max_val_batches=None,
            max_test_batches=None,
            skip_existing=False,
        )


def _valid_existing_job(
    tmp_path,
    *,
    expected_phase: str = "full",
    result_phase: str | None = None,
) -> tuple[dict, object, object]:
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text("protocol: {name: strict}\n", encoding="utf-8")
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    repro = tmp_path / "repro.json"
    source_files = {
        str((PROJECT_ROOT / relative_path).resolve()): {
            "size_bytes": (PROJECT_ROOT / relative_path).resolve().stat().st_size,
            "sha256": _sha256_file((PROJECT_ROOT / relative_path).resolve()),
        }
        for relative_path in REPRO_SOURCE_FILES
    }
    repro.write_text(
        json.dumps(
            {
                "protocol_config": _hashed_file(strict_path),
                "checkpoint": None,
                "artifacts": {"test_metrics": _hashed_file(artifact)},
                "source_files": source_files,
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "schema_version": 1,
        "phase": expected_phase,
        "dataset": "mlrsnet",
        "outer_seed": 42,
        "inner_seed": 100042,
        "training_seed": 0,
        "method": "score_tegar",
        "strict_config": str(strict_path.resolve()),
        "strict_config_sha256": _sha256_file(strict_path),
        "resolved_strict_config_sha256": _sha256_json({"protocol": {"name": "strict"}}),
        "max_train_batches": None,
        "max_val_batches": None,
        "max_test_batches": None,
        "feature_cache_only": True,
    }
    resolved_path = tmp_path / "resolved_job_config.json"
    resolved_path.write_text(json.dumps(spec), encoding="utf-8")
    payload = {
        "phase": result_phase or expected_phase,
        "strict_config_sha256": spec["strict_config_sha256"],
        "resolved_strict_config_sha256": spec["resolved_strict_config_sha256"],
        "resolved_job_config_sha256": _sha256_file(resolved_path),
        "artifacts": {"test_metrics": _hashed_file(artifact)},
        "reproducibility_manifest": _hashed_file(repro),
    }
    result_path = tmp_path / "run_manifest.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return spec, resolved_path, result_path


def test_skip_existing_accepts_only_intact_full_job(tmp_path) -> None:
    spec, resolved_path, result_path = _valid_existing_job(tmp_path)
    reused = _validate_existing_full_job(
        result_path=result_path,
        expected_job_spec=spec,
        resolved_job_config_path=resolved_path,
    )
    assert reused["phase"] == "full"


def test_skip_existing_refuses_hash_valid_nonfinite_checkpoint(tmp_path) -> None:
    spec, resolved_path, result_path = _valid_existing_job(tmp_path)
    checkpoint_path = tmp_path / "outer_checkpoint.pt"
    torch.save(
        {"model_state": {"weight": torch.tensor([float("nan")])}},
        checkpoint_path,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["artifacts"]["outer_checkpoint"] = _hashed_file(checkpoint_path)
    repro_path = Path(result["reproducibility_manifest"]["path"])
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    repro["checkpoint"] = _hashed_file(checkpoint_path)
    repro["artifacts"]["outer_checkpoint"] = _hashed_file(checkpoint_path)
    repro_path.write_text(json.dumps(repro), encoding="utf-8")
    result["reproducibility_manifest"] = _hashed_file(repro_path)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(FloatingPointError, match="Non-finite tensor"):
        _validate_existing_full_job(
            result_path=result_path,
            expected_job_spec=spec,
            resolved_job_config_path=resolved_path,
        )


def test_skip_existing_refuses_hash_valid_nonfinite_metrics(tmp_path) -> None:
    spec, resolved_path, result_path = _valid_existing_job(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"harmonic_mean": NaN}', encoding="utf-8")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["artifacts"]["test_metrics"] = _hashed_file(metrics_path)
    repro_path = Path(result["reproducibility_manifest"]["path"])
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    repro["artifacts"]["test_metrics"] = _hashed_file(metrics_path)
    repro_path.write_text(json.dumps(repro), encoding="utf-8")
    result["reproducibility_manifest"] = _hashed_file(repro_path)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(FloatingPointError, match="Non-finite value"):
        _validate_existing_full_job(
            result_path=result_path,
            expected_job_spec=spec,
            resolved_job_config_path=resolved_path,
        )


def test_skip_existing_refuses_pilot_contamination(tmp_path) -> None:
    spec, resolved_path, result_path = _valid_existing_job(tmp_path, result_phase="pilot")
    with pytest.raises(AssertionError, match="phase='pilot'"):
        _validate_existing_full_job(
            result_path=result_path,
            expected_job_spec=spec,
            resolved_job_config_path=resolved_path,
        )


def test_skip_existing_refuses_stale_resolved_config_or_artifact(tmp_path) -> None:
    spec, resolved_path, result_path = _valid_existing_job(tmp_path)
    stale = dict(spec)
    stale["training_seed"] = 2
    resolved_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(AssertionError, match="Resolved job config"):
        _validate_existing_full_job(
            result_path=result_path,
            expected_job_spec=spec,
            resolved_job_config_path=resolved_path,
        )

    spec, resolved_path, result_path = _valid_existing_job(tmp_path)
    (tmp_path / "metrics.json").write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(AssertionError, match="hash mismatch"):
        _validate_existing_full_job(
            result_path=result_path,
            expected_job_spec=spec,
            resolved_job_config_path=resolved_path,
        )


def test_skip_existing_refuses_changed_reproducibility_source(tmp_path) -> None:
    spec, resolved_path, result_path = _valid_existing_job(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    repro_path = Path(payload["reproducibility_manifest"]["path"])
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    source_path = next(iter(repro["source_files"]))
    repro["source_files"][source_path]["sha256"] = "0" * 64
    repro_path.write_text(json.dumps(repro), encoding="utf-8")
    payload["reproducibility_manifest"] = _hashed_file(repro_path)
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match="reproducibility source file hash mismatch"):
        _validate_existing_full_job(
            result_path=result_path,
            expected_job_spec=spec,
            resolved_job_config_path=resolved_path,
        )

def test_audit_rejects_stale_manifest_config_hash_before_trusting_passed(tmp_path) -> None:
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text("protocol: {name: current}\ndatasets: {mlrsnet: {}}\n", encoding="utf-8")
    config = yaml.safe_load(strict_path.read_text(encoding="utf-8"))
    stale_manifest = {
        "dataset": "mlrsnet",
        "feature_cache_only": True,
        "cooccur_prompts_used": False,
        "strict_config_sha256": "0" * 64,
        "leakage_audit": {"passed": True},
    }
    with pytest.raises(AssertionError, match="Strict config file hash"):
        audit_manifest(
            stale_manifest,
            config=config,
            strict_config_path=strict_path,
            dataset_name="mlrsnet",
        )


def test_graph_audit_recomputes_empirical_endpoints(tmp_path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"seen_labels": ["seen"], "unseen_labels": ["unseen"]}), encoding="utf-8")
    anchor_path = tmp_path / "anchors.pt"
    torch.save({"label_names": ["seen", "unseen"], "anchors": torch.eye(2)}, anchor_path)
    cache_path = tmp_path / "train.pt"
    torch.save({"split": "train"}, cache_path)
    graph = {
        "nodes": ["seen", "unseen"],
        "edges": {
            "often_cooccur": [["seen", "unseen"]],
            "statistical_exclusion": [],
            "hierarchical": [],
        },
        "edge_provenance": {
            "allowed_seen_labels": ["seen"],
            "source": {
                "kind": "train_feature_cache",
                "path": str(cache_path),
                "image_split_manifest": {"declared_split": "train"},
            },
        },
        "relation_provenance": {
            "semantic_affinity": {
                "annotation_independent": True,
                "text_anchors_sha256": _sha256_file(anchor_path),
                "outer_split_sha256": _sha256_file(split_path),
            }
        },
        "leakage_audit": {"passed": True},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    with pytest.raises(AssertionError, match="forbidden endpoints"):
        _audit_graph(
            graph_path=graph_path,
            expected_sha256=_sha256_file(graph_path),
            split_path=split_path,
            anchor_path=anchor_path,
            train_cache_path=cache_path,
            expected_nodes=["seen", "unseen"],
        )


def test_prepare_rebuilds_existing_stale_empirical_graphs(monkeypatch, tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_root = artifact_root / "mlrsnet"
    stale_targets = [
        dataset_root / "outer_123" / "outer_empirical" / "kg_definition.json",
        dataset_root / "outer_123" / "inner_empirical" / "kg_definition.json",
    ]
    for target in stale_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"marker": "stale"}), encoding="utf-8")

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    strict_config_path = tmp_path / "strict.yaml"
    strict_config_path.write_text("protocol: {name: test}\n", encoding="utf-8")
    data_root = tmp_path / "raw"
    data_root.mkdir()
    cache_paths = {}
    for split in ("train", "val", "test"):
        cache_path = tmp_path / "cache" / f"{split}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(f"cache:{split}".encode("utf-8"))
        cache_paths[split] = cache_path

    label_names = ["a", "b", "c", "d"]
    config = {
        "protocol": {
            "name": "test",
            "outer_split_seeds": [42, 123],
            "inner_seed_offset": 100000,
            "inner_surrogate_fraction": 0.5,
            "semantic_affinity_top_k": 1,
            "image_split_seed": 42,
        },
        "datasets": {
            "mlrsnet": {
                "data_root": str(data_root),
                "feature_cache_root": str(tmp_path / "cache"),
                "n_unseen": 1,
            }
        },
        "clip": {
            "model_name": "ViT-B-32",
            "pretrained_path": str(checkpoint),
            "default_pretrained": "unused",
            "image_size": 224,
        },
    }

    monkeypatch.setattr(
        "run_strict_protocol._cache_path",
        lambda _config, _dataset, split: cache_paths[split],
    )
    monkeypatch.setattr(
        "run_strict_protocol._load_cache_header",
        lambda path: {
            "path": str(path),
            "split": Path(path).stem,
            "label_names": label_names,
            "feature_shape": [4, 8],
            "label_shape": [4, len(label_names)],
        },
    )
    monkeypatch.setattr(
        "run_strict_protocol._validate_cache_header",
        lambda *_args, **_kwargs: "matched_sha256",
    )

    empirical_build_counts: dict[Path, int] = {}

    def fake_run(command, _stage):
        script = command[1]
        if script == "select_unseen.py":
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            seeds_start = command.index("--seeds") + 1
            seeds_end = command.index("--output_dir")
            for seed_text in command[seeds_start:seeds_end]:
                seed = int(seed_text)
                unseen = "d" if seed == 42 else "c"
                (output_dir / f"split_seed{seed}.json").write_text(
                    json.dumps(
                        {
                            "label_names": label_names,
                            "seen_labels": [label for label in label_names if label != unseen],
                            "unseen_labels": [unseen],
                        }
                    ),
                    encoding="utf-8",
                )
            return
        if script == "build_kg.py":
            output_dir = Path(command[command.index("--output_dir") + 1])
            target = (output_dir / "kg_definition.json").resolve()
            empirical_build_counts[target] = empirical_build_counts.get(target, 0) + 1
            output_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({"nodes": label_names, "edges": {}, "marker": "fresh"}),
                encoding="utf-8",
            )
            return
        if script == "build_text_anchors.py":
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"label_names": label_names, "anchors": torch.eye(len(label_names))}, output_path)
            return
        if script == "build_semantic_affinity.py":
            kg_path = Path(command[command.index("--kg_path") + 1])
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(kg_path.read_text(encoding="utf-8"), encoding="utf-8")
            return
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("run_strict_protocol._run", fake_run)

    manifest = prepare_dataset(
        config,
        "mlrsnet",
        artifact_root,
        strict_config_path=strict_config_path,
    )

    assert len(manifest["splits"]) == 2
    for target in stale_targets:
        resolved_target = target.resolve()
        assert empirical_build_counts[resolved_target] == 1
        assert json.loads(target.read_text(encoding="utf-8"))["marker"] == "fresh"
