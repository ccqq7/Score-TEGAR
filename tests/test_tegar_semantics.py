from __future__ import annotations

import json

import pytest
import torch

from models.tegar import (
    EMPIRICAL_RELATION_NAMES,
    ParameterMatchedHomogeneousGAT,
    TEGAR,
    TEGARLayer,
    count_trainable_parameters,
    parameter_count_report,
)


def _write_kg(tmp_path, *, include_semantic: bool = True) -> str:
    nodes = ["a", "b", "c"]
    edges = {
        "often_cooccur": [["a", "b"]],
        "statistical_exclusion": [["b", "c"]],
        "hierarchical": [["c", "a"]],
    }
    if include_semantic:
        edges["semantic_affinity"] = [["a", "c"]]
    payload = {
        "nodes": nodes,
        "node_count": len(nodes),
        "edges": edges,
        "label_to_index": {name: idx for idx, name in enumerate(nodes)},
        "label_frequency": {name: 10 for name in nodes},
    }
    path = tmp_path / ("kg_semantic.json" if include_semantic else "kg_legacy.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _layer_adjacency(relation_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    adjacency = {name: torch.zeros(3, 3) for name in relation_names}
    adjacency["statistical_exclusion"][0, 1] = 1.0
    adjacency["statistical_exclusion"][1, 2] = 1.0
    adjacency["statistical_exclusion"][2, 0] = 1.0
    return adjacency


@pytest.mark.parametrize(
    ("mode", "expected_sign"),
    [("negative", -1), ("positive", 1), ("disabled", 0)],
)
def test_exclusion_score_contribution_has_strict_mode_sign(mode: str, expected_sign: int) -> None:
    relation_names = (*EMPIRICAL_RELATION_NAMES, "semantic_affinity")
    layer = TEGARLayer(
        hidden_dim=5,
        visual_dim=4,
        dropout=0.0,
        exclusion_beta_init=0.2,
        exclusion_mode=mode,
        relation_names=relation_names,
    )
    adjacency = _layer_adjacency(relation_names)

    for seed in range(8):
        torch.manual_seed(seed)
        hidden = 20.0 * torch.randn(2, 3, 5)
        visual = 20.0 * torch.randn(2, 4)
        _, aux = layer(hidden, visual, adjacency)

        evidence = aux["exclusion_evidence"]
        beta = aux["exclusion_beta"]
        gate = aux["exclusion_gate"]
        contribution = aux["exclusion_score_contribution"]

        assert evidence.shape == (2, 3)
        assert gate.shape == (2, 1)
        assert torch.all(torch.isfinite(evidence))
        assert torch.all(evidence >= 0)
        assert beta.ndim == 0 and beta > 0
        if expected_sign < 0:
            assert torch.all(contribution <= 0)
        elif expected_sign > 0:
            assert torch.all(contribution >= 0)
        else:
            assert torch.count_nonzero(contribution) == 0


def test_negative_exclusion_score_branch_backpropagates_through_beta_and_evidence() -> None:
    layer = TEGARLayer(
        hidden_dim=5,
        visual_dim=4,
        dropout=0.0,
        exclusion_beta_init=0.2,
        exclusion_mode="negative",
    )
    hidden = torch.randn(2, 3, 5, requires_grad=True)
    visual = torch.randn(2, 4)
    _, aux = layer(hidden, visual, _layer_adjacency(tuple(EMPIRICAL_RELATION_NAMES)))

    (-aux["exclusion_score_contribution"].sum()).backward()

    assert layer.beta_raw.grad is not None
    assert torch.isfinite(layer.beta_raw.grad)
    assert layer.exclusion_evidence_proj.weight.grad is not None
    assert torch.all(torch.isfinite(layer.exclusion_evidence_proj.weight.grad))


def test_strict_mode_blocks_arbitrary_signed_latent_exclusion_path() -> None:
    hidden = torch.randn(2, 3, 5)
    visual = torch.randn(2, 4)
    adjacency = _layer_adjacency(tuple(EMPIRICAL_RELATION_NAMES))
    strict = TEGARLayer(hidden_dim=5, visual_dim=4, dropout=0.0)
    legacy = TEGARLayer(
        hidden_dim=5,
        visual_dim=4,
        dropout=0.0,
        legacy_latent_exclusion=True,
    )

    _, strict_aux = strict(hidden, visual, adjacency)
    _, legacy_aux = legacy(hidden, visual, adjacency)

    assert torch.count_nonzero(
        strict_aux["relation_messages"]["statistical_exclusion"]
    ) == 0
    assert torch.count_nonzero(
        strict_aux["raw_relation_messages"]["statistical_exclusion"]
    ) > 0
    assert torch.count_nonzero(
        legacy_aux["relation_messages"]["statistical_exclusion"]
    ) > 0
    assert strict_aux["legacy_latent_exclusion"] is False
    assert legacy_aux["legacy_latent_exclusion"] is True


def test_exclusion_beta_remains_strictly_positive_under_finite_underflow_input() -> None:
    layer = TEGARLayer(hidden_dim=5, visual_dim=4, dropout=0.0)
    with torch.no_grad():
        layer.beta_raw.fill_(-1e6)
    hidden = torch.randn(2, 3, 5)
    visual = torch.randn(2, 4)
    _, aux = layer(hidden, visual, _layer_adjacency(tuple(EMPIRICAL_RELATION_NAMES)))

    assert torch.isfinite(aux["exclusion_beta"])
    assert aux["exclusion_beta"] > 0
    assert torch.all(aux["exclusion_score_contribution"] <= 0)


def test_tegar_uses_dynamic_relation_gate_and_exposes_final_score_aux(tmp_path) -> None:
    model = TEGAR(
        kg_path=_write_kg(tmp_path, include_semantic=True),
        hidden_dim=6,
        visual_dim=4,
        num_layers=2,
        dropout=0.0,
        use_pairnorm=False,
        exclusion_mode="negative",
    )
    hidden = torch.randn(2, 3, 6)
    visual = torch.randn(2, 4)
    output, aux = model(hidden, visual, return_aux=True)

    assert model.relation_names == (*EMPIRICAL_RELATION_NAMES, "semantic_affinity")
    assert output.shape == hidden.shape
    assert aux["gate_values"].shape == (2, 2, 4)
    assert aux["exclusion_evidence"].shape == (2, 3)
    assert aux["exclusion_gate"].shape == (2, 1)
    assert aux["exclusion_score_contribution"].shape == (2, 3)
    assert aux["exclusion_evidence_history"].shape == (2, 2, 3)
    assert aux["exclusion_gate_history"].shape == (2, 2, 1)
    assert aux["exclusion_score_contribution_history"].shape == (2, 2, 3)
    assert aux["exclusion_beta"] > 0
    assert torch.all(aux["exclusion_evidence"] >= 0)
    assert torch.all(aux["exclusion_score_contribution"] <= 0)


def test_legacy_kg_falls_back_to_three_relations(tmp_path) -> None:
    model = TEGAR(
        kg_path=_write_kg(tmp_path, include_semantic=False),
        hidden_dim=4,
        visual_dim=3,
        num_layers=1,
        dropout=0.0,
        use_pairnorm=False,
    )
    _, aux = model(torch.randn(2, 3, 4), torch.randn(2, 3), return_aux=True)

    assert model.relation_names == EMPIRICAL_RELATION_NAMES
    assert aux["gate_values"].shape == (2, 1, 3)


def test_relation_ablation_cannot_remove_semantic_affinity(tmp_path) -> None:
    kg_path = _write_kg(tmp_path, include_semantic=True)
    model = TEGAR(
        kg_path=kg_path,
        hidden_dim=4,
        visual_dim=3,
        num_layers=1,
        disabled_relations=("often_cooccur",),
    )
    adjacency = model._adjacency()

    assert torch.count_nonzero(adjacency["often_cooccur"]) == 0
    assert torch.count_nonzero(adjacency["semantic_affinity"]) > 0
    with pytest.raises(ValueError, match="must remain active"):
        TEGAR(
            kg_path=kg_path,
            hidden_dim=4,
            visual_dim=3,
            num_layers=1,
            disabled_relations=("semantic_affinity",),
        )


def test_disabled_relation_shared_gate_row_has_exact_zero_gradient(tmp_path) -> None:
    model = TEGAR(
        kg_path=_write_kg(tmp_path, include_semantic=True),
        hidden_dim=6,
        visual_dim=4,
        num_layers=2,
        dropout=0.0,
        use_pairnorm=False,
        disabled_relations=("hierarchical",),
    )
    hidden = torch.randn(3, 3, 6)
    visual = torch.randn(3, 4)
    output, aux = model(hidden, visual, return_aux=True)
    disabled_index = model.relation_names.index("hierarchical")
    active_index = model.relation_names.index("often_cooccur")

    assert torch.count_nonzero(aux["gate_values"][..., disabled_index]) == 0
    loss = output.square().sum() + aux["gate_values"][..., active_index].sum()
    loss.backward()

    for layer in model.layers:
        assert layer.gate.weight.grad is not None
        assert layer.gate.bias.grad is not None
        assert torch.equal(
            layer.gate.weight.grad[disabled_index],
            torch.zeros_like(layer.gate.weight.grad[disabled_index]),
        )
        assert torch.equal(
            layer.gate.bias.grad[disabled_index],
            torch.zeros_like(layer.gate.bias.grad[disabled_index]),
        )
        assert torch.count_nonzero(layer.gate.weight.grad[active_index]) > 0


def test_parameter_matched_homogeneous_control_is_exact_and_uses_union_graph(tmp_path) -> None:
    kg_path = _write_kg(tmp_path, include_semantic=True)
    kwargs = {
        "kg_path": kg_path,
        "hidden_dim": 7,
        "visual_dim": 5,
        "num_layers": 2,
        "dropout": 0.0,
        "use_pairnorm": False,
    }
    typed = TEGAR(**kwargs)
    control = ParameterMatchedHomogeneousGAT(**kwargs)
    report = parameter_count_report(typed, control)

    assert report["reference"] == report["control"]
    assert report["difference"] == 0
    assert report["relative_difference"] == 0.0

    control_adjacency = control._adjacency()
    first = control_adjacency[control.relation_names[0]]
    for relation_name in control.relation_names[1:]:
        assert torch.equal(first, control_adjacency[relation_name])
    assert first[0, 2] == 1 and first[2, 0] == 1

    hidden = torch.randn(2, 3, 7)
    visual = torch.randn(2, 5)
    output, aux = control(hidden, visual, return_aux=True)
    assert output.shape == hidden.shape
    assert aux["gate_values"].shape == (2, 2, 4)
    assert control.legacy_latent_exclusion is False
    assert control.exclusion_mode == typed.exclusion_mode


@pytest.mark.parametrize("initial_activation", [0.5, 0.2, 0.8])
def test_gate_bias_realizes_requested_initial_activation(initial_activation: float) -> None:
    layer = TEGARLayer(
        hidden_dim=5,
        visual_dim=4,
        dropout=0.0,
        gate_initial_activation=initial_activation,
    )
    hidden = torch.randn(3, 3, 5)
    visual = torch.randn(3, 4)
    _, aux = layer(hidden, visual, _layer_adjacency(tuple(EMPIRICAL_RELATION_NAMES)))

    assert torch.count_nonzero(layer.gate.weight) == 0
    assert torch.allclose(
        aux["gate_values"],
        torch.full_like(aux["gate_values"], initial_activation),
        atol=1e-7,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "initial_activation",
    [0.0, 1.0, -0.1, 1.1, float("inf"), float("nan")],
)
def test_invalid_gate_initial_activation_is_rejected(initial_activation: float) -> None:
    with pytest.raises(ValueError, match="gate_initial_activation must be finite and in"):
        TEGARLayer(
            hidden_dim=4,
            visual_dim=3,
            gate_initial_activation=initial_activation,
        )


def test_gate_controls_preserve_parameters_and_fixed_gate_never_updates() -> None:
    kwargs = {
        "hidden_dim": 5,
        "visual_dim": 4,
        "dropout": 0.0,
        "relation_names": (*EMPIRICAL_RELATION_NAMES, "semantic_affinity"),
        "gate_initial_activation": 0.3,
    }
    learned = TEGARLayer(**kwargs, gate_control="learned")
    fixed = TEGARLayer(**kwargs, gate_control="fixed")
    shuffled = TEGARLayer(**kwargs, gate_control="shuffled", gate_shuffle_seed=73)
    assert count_trainable_parameters(learned) == count_trainable_parameters(fixed)
    assert count_trainable_parameters(learned) == count_trainable_parameters(shuffled)

    hidden = torch.randn(2, 3, 5)
    visual = torch.randn(2, 4)
    adjacency = {
        name: torch.eye(3)
        for name in kwargs["relation_names"]
    }
    before_weight = fixed.gate.weight.detach().clone()
    before_bias = fixed.gate.bias.detach().clone()
    output, aux = fixed(hidden, visual, adjacency)
    assert torch.allclose(
        aux["gate_values"],
        torch.full_like(aux["gate_values"], 0.3),
        atol=1e-7,
        rtol=0.0,
    )
    optimizer = torch.optim.AdamW(fixed.parameters(), lr=0.1, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    output.square().sum().backward()
    assert fixed.gate.weight.grad is None
    assert fixed.gate.bias.grad is None
    optimizer.step()
    assert torch.equal(fixed.gate.weight, before_weight)
    assert torch.equal(fixed.gate.bias, before_bias)


def test_shuffled_gate_is_seeded_nonidentity_and_routes_gradients() -> None:
    relation_names = (*EMPIRICAL_RELATION_NAMES, "semantic_affinity")
    kwargs = {
        "hidden_dim": 5,
        "visual_dim": 4,
        "dropout": 0.0,
        "relation_names": relation_names,
        "gate_initial_activation": 0.4,
    }
    learned = TEGARLayer(**kwargs, gate_control="learned")
    shuffled_a = TEGARLayer(**kwargs, gate_control="shuffled", gate_shuffle_seed=73)
    shuffled_b = TEGARLayer(**kwargs, gate_control="shuffled", gate_shuffle_seed=73)
    shuffled_a.load_state_dict(learned.state_dict(), strict=True)
    shuffled_b.load_state_dict(learned.state_dict(), strict=True)
    assert shuffled_a.gate_permutation == shuffled_b.gate_permutation
    assert shuffled_a.gate_permutation != tuple(range(len(relation_names)))
    assert sorted(shuffled_a.gate_permutation) == list(range(len(relation_names)))

    hidden = torch.randn(3, 3, 5)
    visual = torch.randn(3, 4)
    adjacency = {name: torch.eye(3) for name in relation_names}
    _, learned_aux = learned(hidden, visual, adjacency)
    _, shuffled_aux_a = shuffled_a(hidden, visual, adjacency)
    _, shuffled_aux_b = shuffled_b(hidden, visual, adjacency)
    expected = learned_aux["gate_values"][:, list(shuffled_a.gate_permutation)]
    assert torch.equal(shuffled_aux_a["gate_values"], expected)
    assert torch.equal(shuffled_aux_a["gate_values"], shuffled_aux_b["gate_values"])

    shuffled_a.zero_grad(set_to_none=True)
    weights = torch.arange(1, len(relation_names) + 1, dtype=hidden.dtype)
    (shuffled_aux_a["gate_values"] * weights).sum().backward()
    assert shuffled_a.gate.weight.grad is not None
    assert shuffled_a.gate.bias.grad is not None
    assert all(
        torch.count_nonzero(shuffled_a.gate.weight.grad[index]) > 0
        for index in range(len(relation_names))
    )


@pytest.mark.parametrize("control", ["unknown", "random_from_outer_validation"])
def test_gate_control_rejects_unregistered_or_data_dependent_modes(control: str) -> None:
    with pytest.raises(ValueError, match="Unsupported gate_control"):
        TEGARLayer(hidden_dim=5, visual_dim=4, gate_control=control)


def test_legacy_negative_exclusion_flag_maps_to_positive_control() -> None:
    layer = TEGARLayer(hidden_dim=4, visual_dim=3, negative_exclusion=False)
    assert layer.exclusion_mode == "positive"


def test_invalid_exclusion_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported exclusion_mode"):
        TEGARLayer(hidden_dim=4, visual_dim=3, exclusion_mode="legacy_signed")


@pytest.mark.parametrize("invalid_init", [0.0, -0.1, float("inf"), float("nan")])
def test_invalid_exclusion_beta_initialization_is_rejected(invalid_init: float) -> None:
    with pytest.raises(ValueError, match="must be finite and > 0"):
        TEGARLayer(hidden_dim=4, visual_dim=3, exclusion_beta_init=invalid_init)
