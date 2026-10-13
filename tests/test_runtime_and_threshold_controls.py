from __future__ import annotations

import numpy as np
import pytest
import torch

import evaluate
import train


class _FeatureModel(torch.nn.Module):
    def forward_from_features(self, features: torch.Tensor, return_aux: bool = False):
        logits = features[:, :2]
        if return_aux:
            return logits, {}
        return logits


def test_deterministic_runtime_enables_algorithms_disables_benchmark_and_records_state() -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_precision = getattr(torch, "get_float32_matmul_precision", lambda: "high")()
    config = {
        "training": {
            "deterministic": True,
            "deterministic_warn_only": True,
            "cudnn_benchmark": True,
            "channels_last": True,
            "matmul_precision": "high",
        }
    }
    try:
        channels_last = train.configure_runtime(config, torch.device("cpu"))
        state = config["training"]["runtime_state"]
        assert channels_last is False
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert state["deterministic_requested"] is True
        assert state["deterministic_algorithms_enabled"] is True
        assert state["deterministic_warn_only_enabled"] is True
        assert state["cudnn_benchmark"] is False
        assert state["float32_matmul_precision"] == "high"
        assert "cublas_workspace_config" in state
    finally:
        try:
            torch.use_deterministic_algorithms(
                previous_enabled,
                warn_only=previous_warn_only,
            )
        except TypeError:
            torch.use_deterministic_algorithms(previous_enabled)
        torch.backends.cudnn.benchmark = previous_benchmark
        try:
            torch.set_float32_matmul_precision(previous_precision)
        except Exception:
            pass


def test_threshold_grid_is_forwarded_without_modification(monkeypatch) -> None:
    captured = {}

    def fake_search(logits, targets, label_names, **kwargs):
        captured["logits"] = logits
        captured["targets"] = targets
        captured["label_names"] = label_names
        captured.update(kwargs)
        return 0.25, {"harmonic_mean_f1": 0.75}

    monkeypatch.setattr(train, "search_decision_threshold", fake_search)
    grid = [0.11, 0.25, 0.73]
    features = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    threshold, metrics = train.search_model_decision_threshold(
        model=_FeatureModel(),
        dataloader=[(features, targets)],
        device=torch.device("cpu"),
        label_names=["seen", "unseen"],
        seen_labels=["seen"],
        unseen_labels=["unseen"],
        seen_indices=[0],
        use_feature_cache=True,
        candidate_thresholds=grid,
    )

    assert threshold == 0.25
    assert metrics["harmonic_mean_f1"] == 0.75
    assert captured["candidate_thresholds"] is grid
    assert np.array_equal(captured["logits"], features.numpy())


def test_threshold_grid_defaults_to_none(monkeypatch) -> None:
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return 0.5, {}

    monkeypatch.setattr(train, "search_decision_threshold", fake_search)
    features = torch.tensor([[1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0]])
    train.search_model_decision_threshold(
        model=_FeatureModel(),
        dataloader=[(features, targets)],
        device=torch.device("cpu"),
        label_names=["seen", "unseen"],
        seen_labels=["seen"],
        unseen_labels=["unseen"],
        seen_indices=[0],
        use_feature_cache=True,
    )

    assert captured["candidate_thresholds"] is None


def test_deterministic_cuda_requires_preconfigured_cublas_workspace(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    config = {
        "training": {
            "deterministic": True,
            "deterministic_warn_only": False,
        }
    }
    with pytest.raises(RuntimeError, match="requires CUBLAS_WORKSPACE_CONFIG"):
        train.configure_runtime(config, torch.device("cuda"))


def test_evaluate_model_reconstruction_forwards_gate_initial_activation(monkeypatch) -> None:
    captured = {}

    class _Model(torch.nn.Module):
        pass

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return _Model()

    monkeypatch.setattr(evaluate, "create_model", fake_create_model)
    config = {
        "training": {"mode": "zs_logit_tegar"},
        "model": {
            "tegar_dim": 17,
            "tegar_layers": 1,
            "gate_initial_activation": 0.3,
            "gate_control": "shuffled",
            "gate_shuffle_seed": 73,
        },
        "data": {
            "kg_path": "unused-kg.json",
            "text_anchors_path": "unused-anchors.pt",
        },
        "clip": {"model_name": "ViT-B-32"},
    }

    evaluate.instantiate_model(
        config=config,
        num_labels=4,
        device=torch.device("cpu"),
        use_channels_last=False,
        use_feature_cache=True,
    )

    assert captured["tegar_dim"] == 17
    assert captured["gate_initial_activation"] == 0.3
    assert captured["gate_control"] == "shuffled"
    assert captured["gate_shuffle_seed"] == 73
