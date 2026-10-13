from __future__ import annotations

import json

import torch

from models.controlled_baselines import FeatureTEGAR, GCNZAnchor


def _write_inputs(tmp_path):
    labels = ["a", "b", "c"]
    kg_path = tmp_path / "kg.json"
    kg_path.write_text(
        json.dumps(
            {
                "nodes": labels,
                "node_count": 3,
                "label_to_index": {label: idx for idx, label in enumerate(labels)},
                "label_frequency": {label: 1 for label in labels},
                "edges": {
                    "often_cooccur": [["a", "b"]],
                    "statistical_exclusion": [["b", "c"]],
                    "hierarchical": [["a", "c"]],
                    "semantic_affinity": [["a", "c"], ["b", "c"]],
                },
            }
        ),
        encoding="utf-8",
    )
    anchors_path = tmp_path / "anchors.pt"
    torch.save({"anchors": torch.randn(3, 8), "label_names": labels}, anchors_path)
    return kg_path, anchors_path


def test_feature_tegar_reports_embedding_drift(tmp_path) -> None:
    kg_path, anchors_path = _write_inputs(tmp_path)
    model = FeatureTEGAR(
        num_labels=3,
        kg_path=str(kg_path),
        text_anchors_path=str(anchors_path),
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    logits, aux = model(torch.randn(4, 8), return_aux=True)
    assert logits.shape == (4, 3)
    assert aux["embedding_drift"].shape == (4, 3)
    assert torch.all(aux["embedding_drift"] >= -1e-6)


def test_gcnz_anchor_generates_classifier_weights(tmp_path) -> None:
    kg_path, anchors_path = _write_inputs(tmp_path)
    model = GCNZAnchor(
        num_labels=3,
        kg_path=str(kg_path),
        text_anchors_path=str(anchors_path),
        hidden_dim=4,
        num_layers=2,
        dropout=0.0,
    )
    logits, aux = model(torch.randn(5, 8), return_aux=True)
    assert logits.shape == (5, 3)
    assert aux["classifier_weights"].shape == (3, 8)
    assert torch.allclose(aux["classifier_weights"].norm(dim=-1), torch.ones(3), atol=1e-5)
