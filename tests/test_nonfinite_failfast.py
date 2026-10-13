from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import evaluate
import train
from run_strict_protocol import (
    _copy_inner_selection_to_outer_checkpoint,
    _load_finite_checkpoint,
    _load_finite_json,
)


def test_training_checkpoint_save_rejects_nonfinite_model_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    payload = {"model_state": {"weight": torch.tensor([float("nan")])}}

    with pytest.raises(FloatingPointError, match="Non-finite tensor"):
        train._torch_save(payload, checkpoint_path)

    assert not checkpoint_path.exists()


def test_training_checkpoint_save_repeatedly_overwrites_unicode_path(tmp_path) -> None:
    checkpoint_path = tmp_path / "中文实验" / "last_checkpoint.pt"

    train._torch_save({"epoch": 1, "weight": torch.tensor([1.0])}, checkpoint_path)
    train._torch_save({"epoch": 2, "weight": torch.tensor([2.0])}, checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2
    assert torch.equal(payload["weight"], torch.tensor([2.0]))


def test_training_checkpoint_save_retries_transient_windows_replace_lock(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "中文实验" / "best_checkpoint.pt"
    train._torch_save({"epoch": 1}, checkpoint_path)
    real_replace = train.os.replace
    replace_attempts = 0

    def transiently_locked_replace(source, destination) -> None:
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts < 3:
            raise PermissionError("simulated transient checkpoint lock")
        real_replace(source, destination)

    monkeypatch.setattr(train.os, "replace", transiently_locked_replace)
    monkeypatch.setattr(train.time, "sleep", lambda _seconds: None)

    train._torch_save({"epoch": 2}, checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert replace_attempts == 3
    assert payload["epoch"] == 2
    assert list(checkpoint_path.parent.glob("*.tmp")) == []


def test_unscaled_gradient_check_rejects_nonfinite_gradient() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, float("inf"))

    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        train.assert_finite_gradients(model, context="unit test")


@pytest.mark.parametrize(
    "payload",
    [
        torch.tensor([0.0, float("nan")]),
        np.array([0.0, float("inf")]),
        {"metric": float("-inf")},
    ],
)
def test_evaluation_finite_guard_rejects_tensor_array_and_metric(payload) -> None:
    with pytest.raises(FloatingPointError, match="Non-finite"):
        evaluate.assert_finite_payload(payload, context="unit test")


def test_strict_checkpoint_and_json_loaders_reject_nonfinite_artifacts(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics_path = tmp_path / "metrics.json"
    torch.save({"model_state": {"weight": torch.tensor([float("inf")])}}, checkpoint_path)
    metrics_path.write_text('{"harmonic_mean": NaN}', encoding="utf-8")

    with pytest.raises(FloatingPointError, match="Non-finite tensor"):
        _load_finite_checkpoint(checkpoint_path, role="outer_checkpoint")
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        _load_finite_json(metrics_path, role="test_metrics")


def test_locking_inner_selection_rejects_nonfinite_threshold(tmp_path) -> None:
    inner_path = tmp_path / "inner.pt"
    outer_path = tmp_path / "outer.pt"
    threshold_path = tmp_path / "threshold.json"
    locked_path = tmp_path / "locked.json"
    torch.save({"model_state": {"weight": torch.tensor([1.0])}}, inner_path)
    torch.save({"model_state": {"weight": torch.tensor([2.0])}}, outer_path)
    threshold_path.write_text(
        json.dumps(
            {
                "source": "strict_inner_validation",
                "decision_threshold": float("nan"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FloatingPointError, match="threshold_manifest"):
        _copy_inner_selection_to_outer_checkpoint(
            inner_checkpoint=inner_path,
            outer_checkpoint=outer_path,
            threshold_path=threshold_path,
            locked_manifest_path=locked_path,
        )

    assert not locked_path.exists()
