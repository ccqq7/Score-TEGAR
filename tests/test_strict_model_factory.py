from __future__ import annotations

import json

import torch

from evaluate import FEATURE_CACHE_MODES
from models.classifier import create_model
from models.controlled_baselines import (
    CachedCLIPAdapter,
    CachedLinearProbe,
    FeatureTEGAR,
    GCNZAnchor,
)
from models.logit_refiner import ProbabilitySpaceRefiner


def _strict_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels = ["a", "b", "c", "d"]
    kg_path = tmp_path / "kg.json"
    kg_path.write_text(
        json.dumps(
            {
                "nodes": labels,
                "node_count": len(labels),
                "label_to_index": {label: index for index, label in enumerate(labels)},
                "label_frequency": {label: 1 for label in labels},
                "edges": {
                    "often_cooccur": [["a", "b"]],
                    "statistical_exclusion": [["a", "c"]],
                    "hierarchical": [["a", "d"]],
                    "semantic_affinity": [["b", "c"], ["c", "d"]],
                },
            }
        ),
        encoding="utf-8",
    )
    anchors_path = tmp_path / "anchors.pt"
    torch.save({"anchors": torch.randn(len(labels), 8), "label_names": labels}, anchors_path)
    return labels, kg_path, anchors_path


def _create(mode: str, tmp_path, **kwargs):
    labels, kg_path, anchors_path = _strict_inputs(tmp_path)
    tegar_dim = kwargs.pop("tegar_dim", 8)
    tegar_layers = kwargs.pop("tegar_layers", 1)
    model = create_model(
        mode=mode,
        num_labels=len(labels),
        kg_path=str(kg_path),
        text_anchors_path=str(anchors_path),
        tegar_layers=tegar_layers,
        tegar_dim=tegar_dim,
        dropout=0.0,
        use_cached_features=True,
        **kwargs,
    )
    return model


def test_factory_exposes_all_strict_cached_feature_modes(tmp_path) -> None:
    expected_types = {
        "zs_linear_probe": CachedLinearProbe,
        "zs_clip_adapter": CachedCLIPAdapter,
        "zs_feature_tegar": FeatureTEGAR,
        "zs_gcnz_anchor": GCNZAnchor,
        "zs_logit_homo_pm": ProbabilitySpaceRefiner,
    }
    features = torch.randn(3, 8)
    for mode, expected_type in expected_types.items():
        model = _create(mode, tmp_path / mode)
        assert isinstance(model, expected_type)
        assert mode in FEATURE_CACHE_MODES
        logits, aux = model.forward_from_features(features, return_aux=True)
        assert logits.shape == (3, 4)
        assert isinstance(aux, dict)


def test_factory_forwards_strict_ablation_parameters(tmp_path) -> None:
    disabled_relations = ("hierarchical",)
    for mode in ("zs_feature_tegar", "zs_logit_tegar", "zs_logit_homo_pm"):
        model = _create(
            mode,
            tmp_path / mode,
            exclusion_mode="disabled",
            disabled_relations=disabled_relations,
            strict_exclusion_score=True,
        )
        graph = model.graph
        assert graph.exclusion_mode == "disabled"
        assert graph.disabled_relations == frozenset(disabled_relations)
        assert torch.count_nonzero(graph._adjacency()["hierarchical"]) == 0
        if isinstance(model, ProbabilitySpaceRefiner):
            assert model.strict_exclusion_score is True


def test_legacy_factory_call_remains_valid(tmp_path) -> None:
    model = _create("zs_logit_tegar", tmp_path)
    assert model.exclusion_mode == "negative"
    assert model.graph.disabled_relations == frozenset()


def test_factory_forwards_hidden_dimension_to_all_controlled_graph_modes(tmp_path) -> None:
    hidden_dim = 13
    logit = _create("zs_logit_tegar", tmp_path / "logit", tegar_dim=hidden_dim)
    feature = _create("zs_feature_tegar", tmp_path / "feature", tegar_dim=hidden_dim)
    gcnz = _create(
        "zs_gcnz_anchor",
        tmp_path / "gcnz",
        tegar_dim=hidden_dim,
        tegar_layers=2,
    )

    assert logit.node_encoder[0].out_features == hidden_dim
    assert feature.anchor_encoder[0].out_features == hidden_dim
    assert gcnz.layers[0].out_features == hidden_dim


def test_factory_forwards_gate_initial_activation_to_cached_tegar_modes(tmp_path) -> None:
    initial_activation = 0.2
    for mode in ("zs_logit_tegar", "zs_logit_homo_pm", "zs_feature_tegar"):
        model = _create(
            mode,
            tmp_path / mode,
            gate_initial_activation=initial_activation,
        )
        for layer in model.graph.layers:
            assert torch.allclose(
                torch.sigmoid(layer.gate.bias),
                torch.full_like(layer.gate.bias, initial_activation),
                atol=1e-7,
                rtol=0.0,
            )


def test_factory_forwards_gate_control_and_seed_to_cached_tegar_modes(tmp_path) -> None:
    for mode in ("zs_logit_tegar", "zs_logit_homo_pm", "zs_feature_tegar"):
        model = _create(
            mode,
            tmp_path / mode,
            gate_control="shuffled",
            gate_shuffle_seed=73,
        )
        assert model.graph.gate_control == "shuffled"
        assert model.graph.gate_shuffle_seed == 73
        for layer in model.graph.layers:
            assert layer.gate_control == "shuffled"
            assert layer.gate_shuffle_seed == 73
            assert layer.gate_permutation != tuple(range(len(layer.relation_names)))
