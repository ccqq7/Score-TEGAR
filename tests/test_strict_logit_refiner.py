from __future__ import annotations

import json

import torch

from models.logit_refiner import ProbabilitySpaceRefiner


def _inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels = ["a", "b", "c", "d"]
    kg = tmp_path / "kg.json"
    kg.write_text(
        json.dumps(
            {
                "nodes": labels,
                "node_count": len(labels),
                "label_to_index": {label: index for index, label in enumerate(labels)},
                "label_frequency": {label: 1 for label in labels},
                "edges": {
                    "often_cooccur": [["a", "b"]],
                    "statistical_exclusion": [["a", "c"], ["b", "d"]],
                    "hierarchical": [["a", "d"]],
                    "semantic_affinity": [["a", "b"], ["b", "c"], ["c", "d"]],
                },
            }
        ),
        encoding="utf-8",
    )
    anchors = tmp_path / "anchors.pt"
    torch.save({"anchors": torch.randn(len(labels), 8), "label_names": labels}, anchors)
    return kg, anchors


def _model(tmp_path, graph_type: str, exclusion_mode: str, **kwargs):
    kg, anchors = _inputs(tmp_path)
    return ProbabilitySpaceRefiner(
        num_labels=4,
        kg_path=str(kg),
        text_anchors_path=str(anchors),
        graph_type=graph_type,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        use_cached_features=True,
        exclusion_mode=exclusion_mode,
        strict_exclusion_score=True,
        scene_expert_count=2,
        **kwargs,
    )


def test_applied_exclusion_contribution_is_monotonic(tmp_path) -> None:
    features = torch.randn(5, 8)
    for mode, comparison in (
        ("negative", lambda value: torch.all(value <= 1e-7)),
        ("positive", lambda value: torch.all(value >= -1e-7)),
        ("disabled", lambda value: torch.allclose(value, torch.zeros_like(value))),
    ):
        model = _model(tmp_path / mode, "tegar", mode)
        _, aux = model.forward_from_features(features, return_aux=True)
        contribution = aux["applied_exclusion_score_contribution"]
        assert comparison(contribution)


def test_parameter_matched_refiners_have_exact_trainable_count(tmp_path) -> None:
    typed = _model(tmp_path / "typed", "tegar", "negative")
    control = _model(tmp_path / "control", "homo_pm", "negative")
    typed_count = sum(parameter.numel() for parameter in typed.parameters() if parameter.requires_grad)
    control_count = sum(parameter.numel() for parameter in control.parameters() if parameter.requires_grad)
    assert typed_count == control_count
    assert control.graph.legacy_latent_exclusion is False


def test_disabled_relation_head_bias_cannot_leak_or_receive_gradients(tmp_path) -> None:
    model = _model(
        tmp_path / "disabled",
        "tegar",
        "negative",
        disabled_relations=("hierarchical",),
    )
    model.eval()
    features = torch.randn(4, 8)

    before = model.forward_from_features(features)
    disabled_heads = model.relation_delta_heads["hierarchical"]
    with torch.no_grad():
        for head in disabled_heads:
            head.weight.fill_(123.0)
            head.bias.fill_(456.0)
    after = model.forward_from_features(features)
    assert torch.equal(before, after)

    model.zero_grad(set_to_none=True)
    model.forward_from_features(features).sum().backward()
    for head in disabled_heads:
        assert head.weight.grad is None or torch.count_nonzero(head.weight.grad) == 0
        assert head.bias.grad is None or torch.count_nonzero(head.bias.grad) == 0

    active_head = model.relation_delta_heads["often_cooccur"][0]
    assert active_head.weight.grad is not None
    assert torch.count_nonzero(active_head.weight.grad) > 0
    assert active_head.bias.grad is not None

    disabled_index = model.relation_names.index("hierarchical")
    active_index = model.relation_names.index("often_cooccur")
    shared_gate = model.graph.layers[0].gate
    assert shared_gate.weight.grad is not None
    assert shared_gate.bias.grad is not None
    assert torch.equal(
        shared_gate.weight.grad[disabled_index],
        torch.zeros_like(shared_gate.weight.grad[disabled_index]),
    )
    assert torch.equal(
        shared_gate.bias.grad[disabled_index],
        torch.zeros_like(shared_gate.bias.grad[disabled_index]),
    )
    assert torch.count_nonzero(shared_gate.weight.grad[active_index]) > 0
